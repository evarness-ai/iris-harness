"""The owner's health-watch edits: live at the next tick, and back after a reload (ADR-0120)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.settings import SETTINGS_DB_NAME, SettingsStore
from iris_harness.kernel.governance.devices import DeviceService
from iris_harness.server.iris_api.main import create_app
from iris_harness.services.health import watch_edits
from iris_harness.services.health.incidents import IncidentStore
from iris_harness.services.health.watch import (
    HealthWatcher,
    install_watcher,
    load_watch_config,
)

YAML = "confirm_ticks: 2\nnotify:\n  renotify_hours: 12\n  recovered: true\n"


@pytest.fixture
def cfg(tmp_path: Path) -> tuple[Path, SettingsStore]:
    (tmp_path / "health_watch.yaml").write_text(YAML, encoding="utf-8")
    return tmp_path, SettingsStore(db_path=tmp_path / SETTINGS_DB_NAME)


def test_an_edit_is_saved_as_a_diff_and_comes_back_after_a_reload(cfg) -> None:
    config_dir, store = cfg
    file_config = load_watch_config(config_dir, with_edits=False)

    new = watch_edits.update(
        file_config,
        file_config,
        store,
        {"confirm_ticks": "5", "notify_recovered": "off"},
        actor="d",
    )

    assert (new.confirm_ticks, new.notify_recovered) == (5, False)
    assert store.get("health_watch", "config") == {"confirm_ticks": 5, "notify_recovered": False}
    again = load_watch_config(config_dir, settings=SettingsStore(db_path=store.db_path))
    assert (again.confirm_ticks, again.notify_recovered, again.renotify_hours) == (5, False, 12)


def test_editing_back_to_the_file_clears_it_and_reset_restores(cfg) -> None:
    config_dir, store = cfg
    file_config = load_watch_config(config_dir, with_edits=False)
    changed = watch_edits.update(file_config, file_config, store, {"max_attempts": 4}, actor="d")

    back = watch_edits.reset(file_config, changed, store, actor="d")

    assert back.max_attempts == file_config.max_attempts
    assert store.get("health_watch", "config") is None


@pytest.mark.parametrize(
    "change",
    [{"confirm_ticks": 0}, {"renotify_hours": 1000}, {"notify_recovered": "maybe"}, {"ignore": []}],
)
def test_bad_edits_are_refused(cfg, change: dict[str, object]) -> None:
    config_dir, store = cfg
    file_config = load_watch_config(config_dir, with_edits=False)
    with pytest.raises(watch_edits.WatchEditError):
        watch_edits.update(file_config, file_config, store, change, actor="d")
    assert store.history() == []


def test_a_saved_value_that_no_longer_fits_falls_back_to_the_file(cfg) -> None:
    config_dir, store = cfg
    store.set(
        "health_watch", "config", {"confirm_ticks": 999, "max_attempts": 3}, old=None, actor="d"
    )

    config = load_watch_config(config_dir, settings=store)

    assert config.confirm_ticks == 2
    assert config.max_attempts == 3


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cfg) -> Iterator[SimpleNamespace]:
    config_dir, store = cfg
    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    watcher = HealthWatcher(
        store=IncidentStore(tmp_path / "health.db"),
        config=load_watch_config(config_dir, settings=store),
    )
    install_watcher(watcher)
    devices = DeviceService()
    runtime = SimpleNamespace(config_dir=config_dir, data_dir=tmp_path)
    app = create_app(runtime=runtime, auto_start_runtime=False)  # type: ignore[arg-type]
    with TestClient(app, base_url="http://iris.test") as client:
        yield SimpleNamespace(
            client=client,
            watcher=watcher,
            owner=_pair(devices, "control"),
            reader=_pair(devices, "read"),
        )
    install_watcher(None)


def _pair(devices: DeviceService, scope: str) -> dict[str, str]:
    code = devices.start_pairing(scope=scope, actor="service")
    token = devices.claim(code=code.code, name=f"{scope} phone", kind="browser").token
    return {"Authorization": f"Bearer {token}"}


def _field(body: dict, name: str) -> dict:
    return next(f for f in body["fields"] if f["name"] == name)


def test_the_api_changes_the_running_watcher_at_once(api: SimpleNamespace) -> None:
    body = api.client.get("/health/watch/config", headers=api.reader).json()
    assert _field(body, "confirm_ticks") == {
        "name": "confirm_ticks",
        "kind": "int",
        "min": 1,
        "max": 60,
        "value": 2,
        "file": 2,
        "changed": False,
    }
    assert body["running"] is True

    r = api.client.patch("/health/watch/config", json={"confirm_ticks": 4}, headers=api.owner)
    assert r.status_code == 200, r.text
    assert _field(r.json(), "confirm_ticks")["changed"] is True
    assert api.watcher.config.confirm_ticks == 4  # the next tick reads this

    back = api.client.delete("/health/watch/config", headers=api.owner)
    assert _field(back.json(), "confirm_ticks")["value"] == 2
    assert api.watcher.config.confirm_ticks == 2


def test_bad_values_and_read_devices_are_refused(api: SimpleNamespace) -> None:
    assert (
        api.client.patch(
            "/health/watch/config", json={"renotify_hours": 0}, headers=api.owner
        ).status_code
        == 422
    )
    assert api.client.patch("/health/watch/config", json={}, headers=api.owner).status_code == 422
    assert (
        api.client.patch(
            "/health/watch/config", json={"max_attempts": 1}, headers=api.reader
        ).status_code
        == 403
    )
    assert api.watcher.config.max_attempts == 2
