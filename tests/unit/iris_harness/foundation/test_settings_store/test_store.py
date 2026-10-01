"""The settings store keeps overrides and their history on disk (ADR-0120)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.foundation.settings import SETTINGS_DB_NAME, SettingsStore


@pytest.fixture
def store(tmp_path: Path) -> SettingsStore:
    return SettingsStore(db_path=tmp_path / SETTINGS_DB_NAME)


def test_an_empty_store_answers_reads_without_creating_its_file(store: SettingsStore) -> None:
    assert store.get("heartbeat", "email_sweep") is None
    assert store.section("heartbeat") == {}
    assert store.history() == []
    assert not store.db_path.exists()


def test_set_saves_the_override_and_records_the_change(store: SettingsStore) -> None:
    change = store.set(
        "heartbeat", "email_sweep", {"schedule": "interval:300"}, old={"a": 1}, actor="device:x"
    )

    assert store.get("heartbeat", "email_sweep") == {"schedule": "interval:300"}
    assert store.section("heartbeat") == {"email_sweep": {"schedule": "interval:300"}}
    assert change.action == "set"
    assert change.old == {"a": 1}
    assert change.new == {"schedule": "interval:300"}
    assert [c.as_dict() for c in store.history()] == [change.as_dict()]


def test_history_can_record_a_fuller_value_than_the_stored_diff(store: SettingsStore) -> None:
    change = store.set(
        "heartbeat",
        "email_sweep",
        {"enabled": False},
        old={"schedule": "interval:600", "enabled": True},
        new={"schedule": "interval:600", "enabled": False},
        actor="service",
    )

    assert store.get("heartbeat", "email_sweep") == {"enabled": False}
    assert change.new == {"schedule": "interval:600", "enabled": False}
    assert store.history()[0].new == {"schedule": "interval:600", "enabled": False}


def test_clear_drops_the_override_and_records_a_reset(store: SettingsStore) -> None:
    store.set("heartbeat", "wiki_lint", {"enabled": False}, old=True, actor="service")

    change = store.clear("heartbeat", "wiki_lint", old=False, new=True, actor="device:y")

    assert store.get("heartbeat", "wiki_lint") is None
    assert change.action == "reset"
    assert [c.action for c in store.history()] == ["reset", "set"]


def test_values_survive_a_new_store_on_the_same_file(tmp_path: Path) -> None:
    """A restart or a fresh container opens the same file on the volume."""
    path = tmp_path / "vol" / SETTINGS_DB_NAME
    SettingsStore(db_path=path).set("heartbeat", "x", {"enabled": True}, old=None, actor="s")

    reopened = SettingsStore(db_path=path)

    assert reopened.get("heartbeat", "x") == {"enabled": True}
    assert len(reopened.history()) == 1


def test_history_filters_by_section_and_limits_newest_first(store: SettingsStore) -> None:
    for i in range(3):
        store.set("heartbeat", f"h{i}", i, old=None, actor="s")
    store.set("flags", "IRIS_X", True, old=False, actor="s")

    assert [c.key for c in store.history(section="heartbeat")] == ["h2", "h1", "h0"]
    assert [c.key for c in store.history(limit=2)] == ["IRIS_X", "h2"]
    assert len(store.history(limit=0)) == 4


def test_default_path_follows_the_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path))
    assert SettingsStore().db_path == tmp_path / SETTINGS_DB_NAME
