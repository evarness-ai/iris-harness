"""Digest settings: digest.yaml defaults, the owner's saved changes over them, the
one-time brief_prefs.json move, and IRIS_TZ (loop-proof plan D4/D5, ADR-0120)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from iris_harness.foundation.settings import SETTINGS_DB_NAME, SettingsStore
from iris_harness.services.digest.settings import (
    MIGRATION_KEY,
    MORE_GROUP,
    SETTINGS_SECTION,
    DigestGroup,
    DigestSettings,
    group_sections,
    iris_timezone,
    load_defaults,
    load_digest_settings,
    merge_sections,
    news_group_title,
    news_group_topics,
    register_section_knob_validator,
)

REPO_CONFIG = Path(__file__).resolve().parents[5] / "config"

YAML = """\
time: "07:00"
channel: all
sections: [bills_due, todays_events, focus, email_summary, news, learned_yesterday]
locked_sections: [learned_yesterday]
news_topics: [AI, world]
focus_categories: [email/personal, email/finance]
focus_limit: 5
brief_prefs_migration:
  new_sections: [focus, learned_yesterday]
  renamed: {ai_news: news, global_news: news}
"""

# The owner's real brief_prefs.json shape (2026-09-25): every manifest section on, bills
# first (configure_brief "add dues" from the full brief), and the mis-saved "images".
# ai_news and global_news are one `news` section now.
OWNER_PREFS = {
    "enabled_sections": [
        "bills_due",
        "todays_events",
        "email_summary",
        "ai_news",
        "global_news",
    ],
    "section_order": ["bills_due"],
    "section_config": {
        "bills_due": {"categories": ["images"], "within_days": 7},
        "global_news": {"line_cap": 5},
    },
}

# The brief's tool slots once the digest's content lands (PR 2 stream D).
MANIFEST_SLOTS = {
    "todays_plan",
    "todays_events",
    "due_today",
    "overdue",
    "open_tasks",
    "bills_due",
    "unusual_spend",
    "open_followups",
    "resolved_followups",
    "email_summary",
    "focus",
    "needs_reply",
    "judged_yesterday",
    "missed_reminders",
    "reminders",
    "active_items",
    "routines",
    "top_repos",
    "news_ai",
    "news_global",
    "news_local",
    "stocks",
    "indexes",
    "portfolio",
    "learned_yesterday",
}


@pytest.fixture(autouse=True)
def _a_plugin_keeps_categories_to_its_vocabulary() -> Iterator[None]:
    """The dues section's ``categories`` knob is checked by a validator the finance
    plugin registers (core/SDK boundary plan, PR 2); these tests register a stand-in so
    the migration drops the owner's mis-saved ``["images"]`` as it always did."""
    register_section_knob_validator("categories", lambda: ("insurance", "rent", "loan"))
    yield
    register_section_knob_validator("categories", None)


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "digest.yaml").write_text(YAML, encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir, config_dir


def _store(data_dir: Path) -> SettingsStore:
    return SettingsStore(db_path=data_dir / SETTINGS_DB_NAME)


# -- defaults -------------------------------------------------------------------------


def test_the_shipped_file_lists_the_sections_in_group_order_and_the_footer_at_the_end() -> None:
    settings = load_defaults(REPO_CONFIG).settings

    assert (settings.enabled, settings.time, settings.channel) == (True, "07:00", "all")
    assert settings.sections[0] == "todays_plan"
    assert settings.sections[-1] == "learned_yesterday"
    # The sections in group order, then the ungrouped ones: the flat list reads the
    # way the grouped digest does.
    grouped = [s for g in settings.groups for s in g.sections]
    assert list(settings.sections[: len(grouped)]) == grouped
    assert settings.sections.index("focus") > settings.sections.index("todays_events")
    assert settings.news_topics == ("AI", "world")
    assert settings.news_sources == ()
    assert (settings.focus_limit, settings.focus_per_account) == (10, 5)
    assert all(c.startswith("email/") for c in settings.focus_categories)


def test_the_shipped_file_names_every_tool_slot_of_the_morning_brief() -> None:
    defaults = load_defaults(REPO_CONFIG)
    assert set(defaults.settings.sections) | set(defaults.sections_off) == MANIFEST_SLOTS
    assert defaults.locked_sections == ("learned_yesterday",)
    assert defaults.renamed == {
        "news": ("news_ai", "news_global", "news_local"),
        "ai_news": ("news_ai",),
        "global_news": ("news_global",),
    }


def test_the_shipped_file_leaves_trending_repos_off_and_news_in_english() -> None:
    """Owner feedback on PR 2: no git repos unless asked for, English news only."""
    defaults = load_defaults(REPO_CONFIG)
    assert defaults.sections_off == ("top_repos",)
    assert "top_repos" not in defaults.settings.sections
    assert defaults.dropped_sections == ("top_repos",)
    assert defaults.settings.news_language == "en"


def test_no_file_gives_the_built_in_defaults(tmp_path: Path) -> None:
    assert load_digest_settings(tmp_path, tmp_path) == DigestSettings()


def test_a_broken_file_gives_the_built_ins_and_a_bad_field_only_loses_itself(
    dirs, caplog: pytest.LogCaptureFixture
) -> None:
    data_dir, config_dir = dirs
    (config_dir / "digest.yaml").write_text("time: [", encoding="utf-8")
    assert load_digest_settings(data_dir, config_dir) == DigestSettings()

    (config_dir / "digest.yaml").write_text('time: "25:99"\nfocus_limit: 3\n', encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        settings = load_digest_settings(data_dir, config_dir)
    assert (settings.time, settings.focus_limit) == ("07:00", 3)
    assert "ignoring time" in caplog.text


# -- saved changes -----------------------------------------------------------------------


def test_saved_changes_are_laid_over_the_file_and_a_bad_one_is_ignored(dirs) -> None:
    data_dir, config_dir = dirs
    _store(data_dir).set(
        SETTINGS_SECTION,
        "config",
        {"time": "06:30", "news_sources": ["cbsnews.com"], "focus_limit": 999},
        old=None,
        actor="d",
    )

    settings = load_digest_settings(data_dir, config_dir)

    assert settings.time == "06:30"
    assert settings.news_sources == ("cbsnews.com",)
    assert settings.focus_limit == 5  # out of range: the file's value
    assert settings.news_topics == ("AI", "world")


def test_a_broken_store_gives_the_file(dirs) -> None:
    data_dir, config_dir = dirs
    (data_dir / SETTINGS_DB_NAME).mkdir()  # not a database

    settings = load_digest_settings(data_dir, config_dir)

    assert settings.sections[0] == "bills_due"


def test_a_section_new_to_the_owner_slots_in_after_the_one_it_follows() -> None:
    defaults = ("bills_due", "todays_events", "focus", "email_summary", "learned_yesterday")

    merged = merge_sections(defaults, ["email_summary", "bills_due"], ["todays_events"])

    assert merged == ("email_summary", "bills_due", "focus", "learned_yesterday")
    assert merge_sections(defaults, None, None) == defaults
    assert merge_sections(defaults, [], None) == defaults  # nothing on and nothing off: new


def test_the_footer_is_always_shown_and_always_last(dirs) -> None:
    data_dir, config_dir = dirs
    _store(data_dir).set(
        SETTINGS_SECTION,
        "config",
        {"sections": ["learned_yesterday", "news"], "sections_off": ["learned_yesterday"]},
        old=None,
        actor="d",
    )

    settings = load_digest_settings(data_dir, config_dir)

    assert settings.sections[-1] == "learned_yesterday"
    assert settings.sections.count("learned_yesterday") == 1


# -- brief_prefs.json, once -------------------------------------------------------------


def test_the_owners_brief_prefs_move_once_and_the_images_glitch_is_dropped(dirs) -> None:
    data_dir, config_dir = dirs
    (data_dir / "brief_prefs.json").write_text(json.dumps(OWNER_PREFS), encoding="utf-8")

    settings = load_digest_settings(data_dir, config_dir)

    # Every section on, the file's order, the two new sections in their place.
    assert settings.sections == (
        "bills_due",
        "todays_events",
        "focus",
        "email_summary",
        "news",
        "learned_yesterday",
    )
    assert settings.section_config == {"bills_due": {"within_days": 7}, "news": {"line_cap": 5}}
    store = _store(data_dir)
    marker = store.get(SETTINGS_SECTION, MIGRATION_KEY)
    assert marker["status"] == "migrated"
    assert marker["dropped"] == ["bills_due.categories:images"]
    assert (data_dir / "brief_prefs.json").exists()  # left in place, never read again

    rows = len(store.history())
    (data_dir / "brief_prefs.json").write_text(
        json.dumps({"enabled_sections": ["ai_news"]}), encoding="utf-8"
    )
    assert load_digest_settings(data_dir, config_dir) == settings
    assert len(store.history()) == rows


def test_a_section_the_old_file_left_out_is_off_but_a_new_one_is_on(dirs) -> None:
    data_dir, config_dir = dirs
    prefs = {"enabled_sections": ["ai_news", "bills_due"], "section_order": ["ai_news"]}
    (data_dir / "brief_prefs.json").write_text(json.dumps(prefs), encoding="utf-8")

    settings = load_digest_settings(data_dir, config_dir)

    # news (was ai_news) first because the owner put it there; the new sections are on.
    assert settings.sections == ("news", "bills_due", "focus", "learned_yesterday")


def test_an_unreadable_old_file_is_marked_and_skipped(dirs) -> None:
    data_dir, config_dir = dirs
    (data_dir / "brief_prefs.json").write_text("{not json", encoding="utf-8")

    settings = load_digest_settings(data_dir, config_dir)

    assert settings.sections[0] == "bills_due"
    assert _store(data_dir).get(SETTINGS_SECTION, MIGRATION_KEY)["status"] == "unreadable"


def test_no_old_file_writes_nothing(dirs) -> None:
    data_dir, config_dir = dirs

    load_digest_settings(data_dir, config_dir)

    assert not (data_dir / SETTINGS_DB_NAME).exists()


def test_the_owners_saved_fields_win_over_the_old_file(dirs) -> None:
    data_dir, config_dir = dirs
    saved = {"time": "06:00", "section_config": {"bills_due": {"within_days": 3}}}
    _store(data_dir).set(SETTINGS_SECTION, "config", saved, old=None, actor="d")
    prefs = {**OWNER_PREFS, "enabled_sections": ["ai_news"], "section_order": []}
    (data_dir / "brief_prefs.json").write_text(json.dumps(prefs), encoding="utf-8")

    settings = load_digest_settings(data_dir, config_dir)

    assert settings.time == "06:00"
    assert settings.section_config == {"bills_due": {"within_days": 3}}
    assert settings.sections == ("focus", "news", "learned_yesterday")  # from the file


# -- IRIS_TZ ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "UTC"), ("", "UTC"), ("America/Chicago", "America/Chicago"), ("Mars/Base", "UTC")],
)
def test_iris_timezone(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: str) -> None:
    if value is None:
        monkeypatch.delenv("IRIS_TZ", raising=False)
    else:
        monkeypatch.setenv("IRIS_TZ", value)
    assert iris_timezone().key == expected


# -- top_repos: off by default, and off after the move (owner feedback on PR 2) -----------

OFF_YAML = (
    YAML.replace(
        "  renamed: {ai_news: news, global_news: news}\n",
        "  renamed: {ai_news: news, global_news: news}\n  dropped_sections: [top_repos]\n",
    )
    + "sections_off: [top_repos]\n"
)


def test_a_section_off_in_the_file_is_known_but_not_shown(tmp_path: Path) -> None:
    (tmp_path / "digest.yaml").write_text(OFF_YAML, encoding="utf-8")
    defaults = load_defaults(tmp_path)
    assert defaults.sections_off == ("top_repos",)
    assert "top_repos" not in defaults.settings.sections
    assert defaults.dropped_sections == ("top_repos",)


def test_the_move_leaves_top_repos_off_though_the_old_file_had_it_on(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "digest.yaml").write_text(OFF_YAML, encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    prefs = {**OWNER_PREFS, "enabled_sections": [*OWNER_PREFS["enabled_sections"], "top_repos"]}
    (data_dir / "brief_prefs.json").write_text(json.dumps(prefs), encoding="utf-8")

    settings = load_digest_settings(data_dir, config_dir)

    assert "top_repos" not in settings.sections
    assert settings.sections == (
        "bills_due",
        "todays_events",
        "focus",
        "email_summary",
        "news",
        "learned_yesterday",
    )
    marker = _store(data_dir).get(SETTINGS_SECTION, MIGRATION_KEY)
    assert marker["dropped"] == ["bills_due.categories:images", "top_repos"]


def test_the_owners_real_prefs_against_the_shipped_file_drop_top_repos(tmp_path: Path) -> None:
    """The owner's brief_prefs.json turns every section on, top_repos included."""
    # Written before the news split: ai_news and global_news, no local news (nor the
    # Overdue line or Missed reminders, which came later still).
    old = MANIFEST_SLOTS - {
        "focus",
        "learned_yesterday",
        "missed_reminders",
        "news_ai",
        "news_global",
        "news_local",
        "overdue",
    }
    (tmp_path / "brief_prefs.json").write_text(
        json.dumps({"enabled_sections": sorted(old | {"ai_news", "global_news"})}),
        encoding="utf-8",
    )
    settings = load_digest_settings(tmp_path, REPO_CONFIG)
    assert "top_repos" not in settings.sections
    assert set(settings.sections) == MANIFEST_SLOTS - {"top_repos"}


def test_brief_prefs_waits_for_digest_yaml(tmp_path: Path) -> None:
    """Without digest.yaml the renames are unknown: no move, no marker — it runs later."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "brief_prefs.json").write_text(
        json.dumps({"enabled_sections": ["bills_due", "ai_news"]}), encoding="utf-8"
    )
    missing = tmp_path / "no-config"

    load_digest_settings(data_dir, missing)
    store = _store(data_dir)
    assert store.get(SETTINGS_SECTION, MIGRATION_KEY) is None

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "digest.yaml").write_text(YAML, encoding="utf-8")
    settings = load_digest_settings(data_dir, config_dir)
    assert store.get(SETTINGS_SECTION, MIGRATION_KEY) is not None
    assert "ai_news" not in settings.sections


# -- digest v5: groups, channels, the three news groups (prototype signed off 2026-09-25) --


def test_the_shipped_file_groups_the_sections_and_says_how_each_channel_renders() -> None:
    settings = load_defaults(REPO_CONFIG).settings

    assert [(g.id, g.title, g.icon) for g in settings.groups] == [
        ("today", "Today", "☀️"),
        ("money", "Money", "💳"),
        ("inbox", "Inbox", "📬"),
        ("markets", "Markets", "📈"),
        ("news", "News", "📰"),
    ]
    today, money, inbox, markets, news = settings.groups
    assert today.sections == (
        "todays_plan",
        "todays_events",
        "due_today",
        "overdue",
        "missed_reminders",
        "reminders",
        "open_tasks",
    )
    assert money.sections == ("bills_due", "unusual_spend", "portfolio")
    assert inbox.sections == (
        "needs_reply",
        "focus",
        "judged_yesterday",
        "email_summary",
        "open_followups",
        "resolved_followups",
    )
    assert markets.sections == ("indexes", "stocks")
    assert news.sections == ("news_ai", "news_global", "news_local")
    assert today.push == "Events {todays_events} · Due today {due_today} · Tasks {open_tasks}"
    assert (money.push, inbox.push) == ("Bills {bills_due}", "Focus {focus}")
    assert markets.push == news.push == ""
    assert settings.footer_sections == ("learned_yesterday",)
    grouped = {s for g in settings.groups for s in g.sections}
    assert set(settings.sections) - grouped == {"active_items", "routines", "learned_yesterday"}

    assert settings.channels == {
        "web": {"layout": "grouped", "empty": "fold", "links": "source"},
        "telegram": {
            "layout": "message_per_group",
            "section": "card",
            "news": "expandable_card",
            "empty": "hide",
            "links": "source",
            "buttons": ["full_digest", "settings"],
            "rule": "",
        },
        "web_push": {"layout": "headline", "lines": 3, "exclude_groups": ["news", "markets"]},
    }


def test_the_shipped_news_is_three_groups_of_three_lines_with_st_louis_local() -> None:
    settings = load_defaults(REPO_CONFIG).settings

    assert settings.news_local_area == "St. Louis"
    assert news_group_title(settings, "news_ai") == "AI / Tech"
    assert news_group_title(settings, "news_global") == "Global"
    assert news_group_title(settings, "news_local") == "Local — St. Louis"
    assert news_group_topics(settings, "news_ai") == ("AI", "technology")
    assert news_group_topics(settings, "news_global") == ("world",)
    assert news_group_topics(settings, "news_local") == ("St. Louis",)
    for slot in ("news_ai", "news_global", "news_local"):
        assert settings.section_config[slot] == {"line_cap": 3}
    assert "news" not in settings.section_config
    other = DigestSettings(news_groups=settings.news_groups, news_local_area="Chennai")
    assert news_group_title(other, "news_local") == "Local — Chennai"
    assert news_group_topics(other, "news_local") == ("Chennai",)
    assert news_group_topics(other, "news_sports") == ()


def test_group_sections_puts_rendered_sections_under_their_groups() -> None:
    settings = load_defaults(REPO_CONFIG).settings
    rendered = [
        "bills_due",
        "learned_yesterday",
        "news_local",
        "todays_events",
        "top_repos",
        "focus",
        "news_ai",
        "routines",
    ]

    grouped = group_sections(settings, rendered)

    # The file's group order; within a group the render (owner's) order; no empty
    # group; the ungrouped last under "More"; the footer in none.
    assert [(g.id, names) for g, names in grouped] == [
        ("today", ["todays_events"]),
        ("money", ["bills_due"]),
        ("inbox", ["focus"]),
        ("news", ["news_local", "news_ai"]),
        ("more", ["top_repos", "routines"]),
    ]
    assert grouped[-1][0] is MORE_GROUP
    assert (MORE_GROUP.title, MORE_GROUP.push) == ("More", "")
    assert group_sections(settings, ["learned_yesterday"]) == []
    # No groups in the file (the built-ins): everything but the footer is "More".
    assert group_sections(DigestSettings(), ["a", "learned_yesterday", "b"]) == [
        (MORE_GROUP, ["a", "b"])
    ]


def test_a_bad_group_loses_only_itself(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "digest.yaml").write_text(
        YAML + """groups:
  - {id: one, title: One, sections: [bills_due, focus, learned_yesterday], push: "{focus} new"}
  - {id: two, title: Two, icon: "2", sections: [focus, news], push: "{bills_due} due"}
  - {id: "Bad Id", title: Bad, sections: [todays_events]}
  - {id: three, sections: [todays_events]}
  - not a mapping
channels:
  web: {layout: grouped}
  "Not A Name": {layout: x}
  telegram: just text
""",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        settings = load_defaults(tmp_path).settings

    assert settings.groups == (
        DigestGroup(id="one", title="One", sections=("bills_due", "focus"), push="{focus} new"),
        # focus is the first group's; a push over a section outside the group is dropped
        DigestGroup(id="two", title="Two", icon="2", sections=("news",)),
    )
    assert settings.channels == {"web": {"layout": "grouped"}}
    assert "ignoring a group" in caplog.text


def test_a_saved_news_section_loads_as_the_three_news_groups(tmp_path: Path) -> None:
    """The owner's settings.db from before the split names ``news``: it reads as the
    three groups, in its place, and its old line cap gives way to the file's 3 each."""
    _store(tmp_path).set(
        SETTINGS_SECTION,
        "config",
        {
            "sections": ["bills_due", "news", "focus"],
            "sections_off": ["todays_events"],
            "section_config": {"news": {"line_cap": 5}, "bills_due": {"within_days": 7}},
        },
        old=None,
        actor="d",
    )

    settings = load_digest_settings(tmp_path, REPO_CONFIG)

    at = settings.sections.index("news_ai")
    assert settings.sections[at : at + 3] == ("news_ai", "news_global", "news_local")
    # In the saved news's place: after bills_due, before focus.
    assert settings.sections.index("bills_due") < at < settings.sections.index("focus")
    assert "news" not in settings.sections and "todays_events" not in settings.sections
    assert settings.section_config["bills_due"] == {"within_days": 7}
    assert settings.section_config["news_global"] == {"line_cap": 3}
    assert "news" not in settings.section_config


def test_a_saved_news_off_turns_all_three_off(tmp_path: Path) -> None:
    _store(tmp_path).set(
        SETTINGS_SECTION, "config", {"sections_off": ["news", "top_repos"]}, old=None, actor="d"
    )

    settings = load_digest_settings(tmp_path, REPO_CONFIG)

    assert not {"news", "news_ai", "news_global", "news_local"} & set(settings.sections)


def test_the_old_ai_and_global_news_move_to_their_own_groups(tmp_path: Path) -> None:
    (tmp_path / "brief_prefs.json").write_text(json.dumps(OWNER_PREFS), encoding="utf-8")

    settings = load_digest_settings(tmp_path, REPO_CONFIG)

    # ai_news -> news_ai, global_news -> news_global (with its line cap); local is new, on.
    assert {"news_ai", "news_global", "news_local"} <= set(settings.sections)
    assert settings.section_config["news_global"] == {"line_cap": 5}
    assert settings.section_config["news_ai"] == {"line_cap": 3}
    assert settings.sections[0] == "bills_due"  # the owner put bills first


# -- section knob validators (core/SDK boundary plan, PR 2) -----------------------------


def test_a_knob_no_plugin_registered_is_kept_as_saved(dirs) -> None:
    """The core names no knob: with no validator (no finance plugin), categories stay."""
    register_section_knob_validator("categories", None)
    data_dir, config_dir = dirs
    (data_dir / "brief_prefs.json").write_text(json.dumps(OWNER_PREFS), encoding="utf-8")

    settings = load_digest_settings(data_dir, config_dir)

    assert settings.section_config["bills_due"] == {"categories": ["images"], "within_days": 7}
    assert _store(data_dir).get(SETTINGS_SECTION, MIGRATION_KEY)["dropped"] == []


def test_a_registered_knob_keeps_its_good_values_and_drops_the_rest(dirs) -> None:
    data_dir, config_dir = dirs
    prefs = {
        **OWNER_PREFS,
        "section_config": {"bills_due": {"categories": ["Insurance", "images"]}},
    }
    (data_dir / "brief_prefs.json").write_text(json.dumps(prefs), encoding="utf-8")

    settings = load_digest_settings(data_dir, config_dir)

    assert settings.section_config["bills_due"] == {"categories": ["insurance"]}
    marker = _store(data_dir).get(SETTINGS_SECTION, MIGRATION_KEY)
    assert marker["dropped"] == ["bills_due.categories:images"]
