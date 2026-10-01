"""The owner's digest edits: saved as a diff over digest.yaml, back to the file on reset,
and the Settings -> Digest API (ADR-0120, loop-proof plan D4)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.settings import SETTINGS_DB_NAME, SettingsStore
from iris_harness.kernel.governance.devices import DeviceService
from iris_harness.runtime.handlers.brief_prefs import (
    BriefPreferences,
    load_brief_prefs,
    save_brief_prefs,
)
from iris_harness.server.iris_api.main import create_app
from iris_harness.services.digest import edits
from iris_harness.services.digest.settings import load_digest_settings

YAML = """\
time: "07:00"
channel: all
sections: [bills_due, todays_events, focus, email_summary, news, learned_yesterday]
locked_sections: [learned_yesterday]
news_topics: [AI, world]
news_sources: []
focus_categories: [email/personal]
focus_limit: 5
"""


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "digest.yaml").write_text(YAML, encoding="utf-8")
    return tmp_path, config_dir


def _saved(data_dir: Path) -> Any:
    return SettingsStore(db_path=data_dir / SETTINGS_DB_NAME).get("digest", "config")


def test_an_edit_is_saved_as_a_diff_and_the_digest_reads_it(dirs) -> None:
    data_dir, config_dir = dirs

    after = edits.update(
        data_dir,
        {
            "time": "6:30",
            "news_sources": ["https://www.CBSNews.com/latest"],
            "focus_limit": "3",
            "focus_per_account": "2",
        },
        actor="d",
        config_dir=config_dir,
    )

    assert (after.time, after.news_sources, after.focus_limit, after.focus_per_account) == (
        "06:30",
        ("cbsnews.com",),
        3,
        2,
    )
    assert _saved(data_dir) == {
        "time": "06:30",
        "news_sources": ["cbsnews.com"],
        "focus_limit": 3,
        "focus_per_account": 2,
    }
    assert load_digest_settings(data_dir, config_dir) == after


def test_editing_back_to_the_file_clears_it_and_reset_restores(dirs) -> None:
    data_dir, config_dir = dirs
    edits.update(data_dir, {"news_topics": ["AI", "Chicago"]}, actor="d", config_dir=config_dir)
    edits.update(data_dir, {"news_topics": "AI, world"}, actor="d", config_dir=config_dir)
    assert _saved(data_dir) is None

    edits.update(data_dir, {"channel": "telegram,web"}, actor="d", config_dir=config_dir)
    back = edits.reset(data_dir, actor="d", config_dir=config_dir)

    assert back.channel == "all"
    assert _saved(data_dir) is None
    store = SettingsStore(db_path=data_dir / SETTINGS_DB_NAME)
    assert [c.action for c in store.history(section="digest")] == ["reset", "set", "reset", "set"]


def test_sections_alone_means_exactly_these_and_off_alone_keeps_the_rest(dirs) -> None:
    data_dir, config_dir = dirs

    after = edits.update(
        data_dir, {"sections": ["news", "bills_due"]}, actor="d", config_dir=config_dir
    )
    assert after.sections == ("news", "bills_due", "learned_yesterday")  # the footer stays
    assert _saved(data_dir)["sections_off"] == ["todays_events", "focus", "email_summary"]

    after = edits.update(
        data_dir,
        {"sections": ["news", "bills_due", "focus"], "sections_off": ["todays_events"]},
        actor="d",
        config_dir=config_dir,
    )
    # email_summary is in neither list: it comes back where the file puts it.
    assert after.sections == ("news", "bills_due", "focus", "email_summary", "learned_yesterday")

    after = edits.update(data_dir, {"sections_off": ["news"]}, actor="d", config_dir=config_dir)
    assert after.sections == ("bills_due", "focus", "email_summary", "learned_yesterday")


@pytest.mark.parametrize(
    "change",
    [
        {"time": "7am"},
        {"time": "24:00"},
        {"focus_limit": 0},
        {"focus_per_account": 0},
        {"focus_per_account": "five"},
        {"news_sources": ["not a domain"]},
        {"news_topics": ["x" * 81]},
        {"channel": "tele gram"},
        {"sections": ["no_such_section"]},
        {"sections": ["news"], "sections_off": ["news"]},
        {"sections": ["news", "news"]},
        {"sections_off": ["learned_yesterday"]},
        {"section_config": {"news": {"line_cap": 0}}},
        {"section_config": {"news": "five"}},
        {"news_language": "english"},
        {"colour": "blue"},
        {},
    ],
)
def test_bad_edits_are_refused_and_nothing_is_saved(dirs, change: dict[str, Any]) -> None:
    data_dir, config_dir = dirs
    with pytest.raises(edits.DigestEditError):
        edits.update(data_dir, change, actor="d", config_dir=config_dir)
    assert _saved(data_dir) is None


def test_a_line_cap_is_saved_per_section(dirs) -> None:
    data_dir, config_dir = dirs

    after = edits.update(
        data_dir,
        {"section_config": {"news": {"line_cap": "4"}, "bills_due": {"within_days": 7}}},
        actor="d",
        config_dir=config_dir,
    )

    assert after.section_config == {"news": {"line_cap": 4}, "bills_due": {"within_days": 7}}


def test_chat_and_the_app_write_the_same_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """configure_brief (via brief_prefs) and Settings -> Digest are one store."""
    monkeypatch.chdir(Path(__file__).resolve().parents[5])  # the shipped config/digest.yaml
    prefs = BriefPreferences(enabled_sections=("portfolio", "bills_due")).with_section_config(
        "bills_due", within_days=7
    )

    save_brief_prefs(tmp_path, prefs)

    digest = load_digest_settings(tmp_path)
    assert digest.sections[:2] == ("portfolio", "bills_due")
    assert not {"news_ai", "news_global", "news_local"} & set(digest.sections)
    assert digest.sections[-1] == "learned_yesterday"  # never omitted
    # Chat's knobs over the file's (the file's other sections keep theirs).
    assert digest.section_config["bills_due"] == {"within_days": 7}
    assert digest.section_config["news_local"] == {"line_cap": 3}
    saved = SettingsStore(db_path=tmp_path / SETTINGS_DB_NAME).get("digest", "config")
    assert saved["section_config"] == {"bills_due": {"within_days": 7}}
    history = SettingsStore(db_path=tmp_path / SETTINGS_DB_NAME).history(section="digest")
    assert history[0].actor == "chat:configure_brief"
    assert load_brief_prefs(tmp_path).enabled_sections == digest.sections

    save_brief_prefs(tmp_path, BriefPreferences(section_config=digest.section_config))
    assert load_brief_prefs(tmp_path).enabled_sections is None  # back to the full brief


# -- the API ------------------------------------------------------------------------------


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirs) -> Iterator[SimpleNamespace]:
    data_dir, config_dir = dirs
    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("IRIS_TZ", "America/Chicago")
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    devices = DeviceService()
    runtime = SimpleNamespace(config_dir=config_dir, data_dir=data_dir)
    app = create_app(runtime=runtime, auto_start_runtime=False)  # type: ignore[arg-type]
    with TestClient(app, base_url="http://iris.test") as client:
        yield SimpleNamespace(
            client=client,
            data_dir=data_dir,
            owner=_pair(devices, "control"),
            reader=_pair(devices, "read"),
        )


def _pair(devices: DeviceService, scope: str) -> dict[str, str]:
    code = devices.start_pairing(scope=scope, actor="service")
    token = devices.claim(code=code.code, name=f"{scope} phone", kind="browser").token
    return {"Authorization": f"Bearer {token}"}


def test_the_api_reads_changes_and_resets_the_digest(api: SimpleNamespace) -> None:
    body = api.client.get("/digest/config", headers=api.reader).json()
    assert body["fields"]["time"] == "07:00"
    assert body["fields"]["sections_off"] == []
    assert body["changed"] == []
    assert body["timezone"] == "America/Chicago"
    assert body["all_sections"][0] == "bills_due"

    r = api.client.patch(
        "/digest/config",
        json={"time": "06:45", "sections": ["bills_due", "news"]},
        headers=api.owner,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fields"]["time"] == "06:45"
    assert body["fields"]["sections"] == ["bills_due", "news", "learned_yesterday"]
    assert body["fields"]["sections_off"] == ["todays_events", "focus", "email_summary"]
    assert body["locked_sections"] == ["learned_yesterday"]
    assert set(body["changed"]) == {"time", "sections", "sections_off"}
    assert body["all_sections"][:3] == ["bills_due", "news", "learned_yesterday"]  # then off
    assert load_digest_settings(api.data_dir, api.data_dir / "config").time == "06:45"

    back = api.client.delete("/digest/config", headers=api.owner).json()
    assert back["fields"]["time"] == "07:00"
    assert back["changed"] == []


def test_the_api_refuses_bad_values_and_read_devices(api: SimpleNamespace) -> None:
    assert (
        api.client.patch("/digest/config", json={"time": "25:00"}, headers=api.owner).status_code
        == 422
    )
    assert api.client.patch("/digest/config", json={}, headers=api.owner).status_code == 422
    assert (
        api.client.patch("/digest/config", json={"time": "06:00"}, headers=api.reader).status_code
        == 403
    )
    assert _saved(api.data_dir) is None


# -- off by default, and the news language (owner feedback on PR 2) ------------------------

OFF_YAML = YAML + "sections_off: [top_repos]\nnews_language: en\n"


@pytest.fixture
def off_dirs(tmp_path: Path) -> tuple[Path, Path]:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "digest.yaml").write_text(OFF_YAML, encoding="utf-8")
    return tmp_path, config_dir


def test_a_section_off_in_the_file_is_listed_off_and_is_no_change(off_dirs) -> None:
    data_dir, config_dir = off_dirs
    view = edits.payload(data_dir, config_dir)
    assert "top_repos" not in view["fields"]["sections"]
    assert view["fields"]["sections_off"] == ["top_repos"]
    assert view["file"]["sections_off"] == ["top_repos"]
    assert "top_repos" in view["all_sections"]
    assert view["changed"] == []


def test_the_owner_turns_a_default_off_section_on_and_off_again(off_dirs) -> None:
    data_dir, config_dir = off_dirs
    on = ["bills_due", "todays_events", "focus", "email_summary", "top_repos", "news"]

    after = edits.update(
        data_dir, {"sections": on, "sections_off": []}, actor="d", config_dir=config_dir
    )
    assert after.sections == (*on, "learned_yesterday")
    assert edits.payload(data_dir, config_dir)["changed"] == ["sections", "sections_off"]

    after = edits.update(
        data_dir, {"sections_off": ["top_repos"]}, actor="d", config_dir=config_dir
    )
    assert "top_repos" not in after.sections
    assert _saved(data_dir) is None  # back to the file: nothing saved


def test_the_news_language_is_a_code_or_any(off_dirs) -> None:
    data_dir, config_dir = off_dirs
    assert load_digest_settings(data_dir, config_dir).news_language == "en"
    after = edits.update(data_dir, {"news_language": " JA "}, actor="d", config_dir=config_dir)
    assert after.news_language == "ja"
    after = edits.update(data_dir, {"news_language": "any"}, actor="d", config_dir=config_dir)
    assert after.news_language == "any"
    edits.update(data_dir, {"news_language": "en"}, actor="d", config_dir=config_dir)
    assert _saved(data_dir) is None


# -- digest v5: the news groups, the local area, per-section knobs over the file's ---------

NEWS_YAML = (
    YAML.replace("news, learned_yesterday", "news_ai, news_local, learned_yesterday")
    + """section_config:
  news_ai: {line_cap: 3}
news_groups:
  news_ai: {title: AI / Tech, topics: [AI, technology]}
  news_local: {title: "Local — {news_local_area}", topics: ["{news_local_area}"]}
news_local_area: St. Louis
renamed_sections:
  news: [news_ai, news_local]
"""
)


@pytest.fixture
def news_dirs(tmp_path: Path) -> tuple[Path, Path]:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "digest.yaml").write_text(NEWS_YAML, encoding="utf-8")
    return tmp_path, config_dir


def test_the_local_area_and_a_groups_topics_are_the_owners_to_change(news_dirs) -> None:
    from iris_harness.services.digest.settings import news_group_title, news_group_topics

    data_dir, config_dir = news_dirs
    after = edits.update(
        data_dir,
        {
            "news_local_area": "  Kansas   City ",
            "news_groups": {"news_local": {"topics": ["{news_local_area}", "Missouri"]}},
        },
        actor="d",
        config_dir=config_dir,
    )

    assert after.news_local_area == "Kansas City"
    assert news_group_title(after, "news_local") == "Local — Kansas City"
    assert news_group_topics(after, "news_local") == ("Kansas City", "Missouri")
    assert news_group_topics(after, "news_ai") == ("AI", "technology")  # untouched: the file's
    # Only what differs from the file is saved: the one group's topics.
    assert _saved(data_dir) == {
        "news_local_area": "Kansas City",
        "news_groups": {"news_local": {"topics": ["{news_local_area}", "Missouri"]}},
    }
    body = edits.payload(data_dir, config_dir)
    assert body["section_titles"] == {"news_ai": "AI / Tech", "news_local": "Local — Kansas City"}
    assert {"news_local_area", "news_groups"} <= set(body["changed"])

    # Back to the file's topics: nothing of the group is saved.
    edits.update(
        data_dir,
        {"news_groups": {"news_local": {"topics": ["{news_local_area}"]}}},
        actor="d",
        config_dir=config_dir,
    )
    assert _saved(data_dir) == {"news_local_area": "Kansas City"}


@pytest.mark.parametrize(
    "change",
    [
        {"news_local_area": "   "},
        {"news_local_area": "{area}"},
        {"news_local_area": "x" * 81},
        {"news_groups": {"news_sports": {"topics": ["cricket"]}}},  # no such group
        {"news_groups": {"news_ai": {"topics": ["{city} news"]}}},  # not a placeholder
        {"news_groups": {"news_ai": {"topics": ["AI {"]}}},
        {"news_groups": {"news_ai": {"colour": "red"}}},
        {"news_groups": {"news_ai": ["AI"]}},
        {"news_groups": ["news_ai"]},
    ],
)
def test_a_bad_area_or_news_group_is_refused(news_dirs, change: dict[str, Any]) -> None:
    data_dir, config_dir = news_dirs
    with pytest.raises(edits.DigestEditError):
        edits.update(data_dir, change, actor="d", config_dir=config_dir)
    assert _saved(data_dir) is None


def test_a_line_cap_is_per_section_over_the_files(news_dirs) -> None:
    data_dir, config_dir = news_dirs
    view = edits.payload(data_dir, config_dir)["fields"]
    assert view["section_config"] == {"news_ai": {"line_cap": 3}}

    # The app sends the whole mapping; only the section that differs is saved.
    after = edits.update(
        data_dir,
        {"section_config": {"news_ai": {"line_cap": 3}, "news_local": {"line_cap": 2}}},
        actor="d",
        config_dir=config_dir,
    )
    assert after.section_config == {"news_ai": {"line_cap": 3}, "news_local": {"line_cap": 2}}
    assert _saved(data_dir) == {"section_config": {"news_local": {"line_cap": 2}}}

    # An empty mapping clears the file's cap for that section ("every line").
    after = edits.update(
        data_dir, {"section_config": {"news_ai": {}}}, actor="d", config_dir=config_dir
    )
    assert after.section_config == {"news_ai": {}}
    assert _saved(data_dir) == {"section_config": {"news_ai": {}}}


def test_a_save_over_old_names_writes_todays(news_dirs) -> None:
    data_dir, config_dir = news_dirs
    SettingsStore(db_path=data_dir / SETTINGS_DB_NAME).set(
        "digest", "config", {"sections": ["news", "bills_due"]}, old=None, actor="d"
    )
    assert edits.payload(data_dir, config_dir)["fields"]["sections"][:2] == [
        "news_ai",
        "news_local",
    ]

    edits.update(data_dir, {"time": "06:15"}, actor="d", config_dir=config_dir)

    saved = _saved(data_dir)
    assert saved["time"] == "06:15"
    assert saved["sections"][:3] == ["news_ai", "news_local", "bills_due"]
