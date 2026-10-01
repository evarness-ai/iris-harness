"""Curated per-agent settings writes (ADR-0074 §4), saved in the settings store (ADR-0120)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from iris_harness.foundation.settings import SETTINGS_DB_NAME, SettingsStore
from iris_harness.foundation.settings.catalog import SettingDeclaration
from iris_harness.foundation.settings.env_overrides import ENV_SECTION, _reset_for_tests
from iris_harness.runtime.agent_settings_store import (
    apply_overrides_to_env,
    load_overrides,
    override_path,
    set_toggle,
    toggles_for_agent,
)
from iris_harness.runtime.plugin_host.manifest import PluginManifest
from iris_harness.runtime.settings_catalog import installed_catalog, registry_catalog

ROOT = Path(__file__).resolve().parents[5]


@pytest.fixture(scope="module")
def installed():
    """This install's catalog, read from the real profile + plugin manifests: the agent
    switches come from the plugins (ADR-0120), not a list in the core. Which switches
    each domain plugin declares is pinned beside those plugins
    (tests/unit/iris_personal/plugins/test_personal_profile/)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("IRIS_PROFILE", "personal-assistant")
        return installed_catalog(ROOT / "config")


def _switch(agent: str, *, applies: str = "next_run", **extra: object) -> SettingDeclaration:
    return SettingDeclaration(
        kind="bool",
        default=True,
        applies=applies,  # type: ignore[arg-type]
        label=f"{agent} switch",
        description=f"A switch of the {agent} agent.",
        tab="agents",
        agent=agent,
        **extra,  # type: ignore[arg-type]
    )


@pytest.fixture(scope="module")
def catalog():
    """A catalog with plugins that declare agent switches the way the domain plugins do
    (a next-run switch, a restart switch, a guarded switch), built from inline manifests
    so ``set_toggle`` is exercised in any tree."""
    plugins = [
        PluginManifest(
            name="finance_workflows",
            settings={
                "IRIS_FINANCE_AUTO_INGEST": _switch("finance"),
                "IRIS_FINANCE_DUES_FROM_EMAIL": _switch("finance"),
            },
        ),
        PluginManifest(
            name="email_workflows",
            settings={"IRIS_EMAIL_SEMANTIC_SEARCH": _switch("email", applies="restart")},
        ),
        PluginManifest(
            name="calendar",
            settings={
                "IRIS_CALENDAR_AUTO_APPROVE_INVITES": _switch(
                    "calendar", guarded=True, guard_reason="it accepts invites for the owner"
                )
            },
        ),
    ]
    return registry_catalog(
        SimpleNamespace(plugins=lambda: [SimpleNamespace(manifest=m) for m in plugins])
    )


@pytest.fixture
def store(tmp_path: Path) -> SettingsStore:
    _reset_for_tests()
    return SettingsStore(db_path=tmp_path / "data" / SETTINGS_DB_NAME)


def test_the_two_switches_nothing_read_are_gone(installed) -> None:
    """ADR-0120 review: "Organize plans" and "Encrypted document vault" were shown in
    the app but no code read them. Removed at the owner's request."""
    assert installed.get("IRIS_FM_ORGANIZE") is None
    assert installed.get("IRIS_FM_VAULT") is None


def test_a_guarded_agent_switch_is_refused_here(
    tmp_path: Path, store: SettingsStore, catalog
) -> None:
    """Only Settings asks for the owner's confirm, so the agent path refuses a guard."""
    assert "IRIS_CALENDAR_AUTO_APPROVE_INVITES" in toggles_for_agent("calendar", catalog)
    with pytest.raises(ValueError, match="guarded"):
        set_toggle("IRIS_CALENDAR_AUTO_APPROVE_INVITES", True, catalog=catalog, store=store)
    assert store.history() == []


def test_set_toggle_saves_in_the_store_and_hot_applies(
    tmp_path: Path, store: SettingsStore, monkeypatch: pytest.MonkeyPatch, catalog
) -> None:
    monkeypatch.delenv("IRIS_FINANCE_AUTO_INGEST", raising=False)
    res = set_toggle(
        "IRIS_FINANCE_AUTO_INGEST", False, catalog=catalog, store=store, actor="device:x"
    )

    assert res["enabled"] is False
    assert res["applies"] == "next run"
    assert res["restart_required"] is False
    assert os.environ["IRIS_FINANCE_AUTO_INGEST"] == "0"
    assert store.get(ENV_SECTION, "IRIS_FINANCE_AUTO_INGEST") == "0"
    assert store.history()[0].actor == "device:x"
    assert not override_path(tmp_path).exists()  # nothing written into the config dir


def test_set_toggle_restart_flag(tmp_path: Path, store: SettingsStore, catalog) -> None:
    res = set_toggle("IRIS_EMAIL_SEMANTIC_SEARCH", True, catalog=catalog, store=store)
    assert res["applies"] == "restart" and res["restart_required"] is True


def test_set_toggle_rejects_unknown_key(tmp_path: Path, store: SettingsStore, catalog) -> None:
    with pytest.raises(ValueError):
        set_toggle("IRIS_NOT_A_TOGGLE", True, catalog=catalog, store=store)


def test_a_saved_toggle_applies_at_the_next_start(
    tmp_path: Path, store: SettingsStore, monkeypatch: pytest.MonkeyPatch, catalog
) -> None:
    set_toggle("IRIS_FINANCE_AUTO_INGEST", False, catalog=catalog, store=store)
    set_toggle("IRIS_FINANCE_DUES_FROM_EMAIL", True, catalog=catalog, store=store)

    monkeypatch.delenv("IRIS_FINANCE_AUTO_INGEST", raising=False)
    monkeypatch.delenv("IRIS_FINANCE_DUES_FROM_EMAIL", raising=False)
    apply_overrides_to_env(tmp_path, store=SettingsStore(db_path=store.db_path))

    assert os.environ["IRIS_FINANCE_AUTO_INGEST"] == "0"
    assert os.environ["IRIS_FINANCE_DUES_FROM_EMAIL"] == "1"


def test_an_old_override_file_is_imported_once_and_set_aside(
    tmp_path: Path, store: SettingsStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    override_path(tmp_path).write_text(
        yaml.safe_dump({"toggles": {"IRIS_FINANCE_AUTO_INGEST": False}}), encoding="utf-8"
    )
    monkeypatch.delenv("IRIS_FINANCE_AUTO_INGEST", raising=False)

    apply_overrides_to_env(tmp_path, store=store)

    assert os.environ["IRIS_FINANCE_AUTO_INGEST"] == "0"
    assert store.get(ENV_SECTION, "IRIS_FINANCE_AUTO_INGEST") == "0"
    assert store.history()[0].actor == "migration"
    assert not override_path(tmp_path).exists()
    assert override_path(tmp_path).with_suffix(".yaml.migrated").exists()


def test_the_store_wins_over_an_old_file_it_already_has(
    tmp_path: Path, store: SettingsStore
) -> None:
    store.set(ENV_SECTION, "IRIS_FINANCE_AUTO_INGEST", "1", old=None, actor="device:x")
    override_path(tmp_path).write_text(
        yaml.safe_dump({"toggles": {"IRIS_FINANCE_AUTO_INGEST": False}}), encoding="utf-8"
    )
    apply_overrides_to_env(tmp_path, store=store)
    assert store.get(ENV_SECTION, "IRIS_FINANCE_AUTO_INGEST") == "1"


def test_load_overrides_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_overrides(tmp_path) == {}
