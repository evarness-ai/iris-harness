"""Editing heartbeats over the API, and the history of edits (ADR-0120).

Wired on the real ``create_app``: the write guard, paired devices and the settings
store are the real ones; only the runtime is a stand-in holding a real scheduler.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.foundation.settings import SETTINGS_DB_NAME, SettingsStore
from iris_harness.kernel.governance.devices import DeviceService, PairedDevice
from iris_harness.server.iris_api.main import create_app
from iris_harness.services.heartbeat import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatScheduler,
    HeartbeatStatus,
)


def _ok(definition: HeartbeatDefinition) -> HeartbeatRun:
    return HeartbeatRun(
        name=definition.name, status=HeartbeatStatus.SUCCESS, finished_at=datetime.now(UTC)
    )


@pytest.fixture
def aps() -> Iterator[BackgroundScheduler]:
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.start(paused=True)
    yield scheduler
    scheduler.shutdown(wait=False)


@pytest.fixture
def runtime(tmp_path: Path, aps: BackgroundScheduler) -> SimpleNamespace:
    heartbeats = HeartbeatScheduler(
        scheduler=aps,
        handlers={"sweep": _ok, "photos": _ok},
        settings=SettingsStore(db_path=tmp_path / SETTINGS_DB_NAME),
    )
    heartbeats.register_all(
        [
            HeartbeatDefinition(
                name="email_sweep",
                handler="sweep",
                schedule="interval:600",
                description="Fetch new mail.",
            ),
            HeartbeatDefinition(
                name="filemanager_photos", handler="photos", schedule="30 4 * * *", enabled=False
            ),
        ]
    )
    return SimpleNamespace(heartbeats=heartbeats, data_dir=tmp_path)


@pytest.fixture
def devices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DeviceService:
    # The app builds its own DeviceService from IRIS_HOME; the test meets it there.
    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    return DeviceService()


@pytest.fixture
def client(runtime: SimpleNamespace, devices: DeviceService) -> Iterator[TestClient]:
    app = create_app(runtime=runtime, auto_start_runtime=False)  # type: ignore[arg-type]
    with TestClient(app, base_url="http://iris.test") as c:
        yield c


def _pair(devices: DeviceService, scope: str) -> dict[str, str]:
    code = devices.start_pairing(scope=scope, actor="service")
    paired: PairedDevice = devices.claim(code=code.code, name=f"{scope} phone", kind="browser")
    return {"Authorization": f"Bearer {paired.token}"}


def test_list_shows_every_heartbeat_with_its_default_and_next_run(client: TestClient) -> None:
    body = client.get("/heartbeat", headers=auth_headers()).json()

    rows = {row["name"]: row for row in body["heartbeats"]}
    assert set(rows) == {"email_sweep", "filemanager_photos"}
    sweep = rows["email_sweep"]
    assert sweep["schedule_text"] == "every 10 min"
    assert sweep["default_schedule"] == "interval:600"
    assert sweep["overridden"] is False
    assert sweep["runnable"] is True
    assert sweep["next_run_at"] is not None
    assert rows["filemanager_photos"]["enabled"] is False
    assert rows["filemanager_photos"]["next_run_at"] is None
    assert body["timezone"]


def test_a_control_device_changes_a_schedule_live(
    client: TestClient, devices: DeviceService, aps: BackgroundScheduler
) -> None:
    owner = _pair(devices, "control")

    response = client.patch("/heartbeat/email_sweep", json={"schedule": "every"}, headers=owner)
    assert response.status_code == 422

    response = client.patch(
        "/heartbeat/email_sweep", json={"schedule": "interval:300"}, headers=owner
    )

    assert response.status_code == 200
    row = response.json()
    assert row["schedule_text"] == "every 5 min"
    assert row["overridden"] is True
    assert aps.get_job("email_sweep").trigger.interval.total_seconds() == 300


def test_a_control_device_turns_a_disabled_heartbeat_on(
    client: TestClient, devices: DeviceService, aps: BackgroundScheduler
) -> None:
    response = client.patch(
        "/heartbeat/filemanager_photos", json={"enabled": True}, headers=_pair(devices, "control")
    )

    assert response.status_code == 200
    assert response.json()["enabled"] is True
    assert response.json()["next_run_at"] is not None
    assert aps.get_job("filemanager_photos") is not None


def test_a_read_device_and_the_bare_secret_may_not_edit(
    client: TestClient, devices: DeviceService, aps: BackgroundScheduler
) -> None:
    for headers in (_pair(devices, "read"), auth_headers()):
        patched = client.patch("/heartbeat/email_sweep", json={"enabled": False}, headers=headers)
        reset = client.delete("/heartbeat/email_sweep/override", headers=headers)
        assert patched.status_code == 403
        assert reset.status_code == 403
    assert aps.get_job("email_sweep") is not None


def test_the_secret_may_edit_when_the_operator_opted_in(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
    response = client.patch(
        "/heartbeat/email_sweep", json={"enabled": False}, headers=auth_headers()
    )
    assert response.status_code == 200
    assert response.json()["enabled"] is False


def test_errors_say_what_is_wrong(client: TestClient, devices: DeviceService) -> None:
    owner = _pair(devices, "control")

    assert client.patch("/heartbeat/nope", json={"enabled": True}, headers=owner).status_code == 404
    assert client.delete("/heartbeat/nope/override", headers=owner).status_code == 404
    assert client.patch("/heartbeat/email_sweep", json={}, headers=owner).status_code == 422
    unknown_field = client.patch("/heartbeat/email_sweep", json={"handler": "x"}, headers=owner)
    assert unknown_field.status_code == 422
    too_fast = client.patch(
        "/heartbeat/email_sweep", json={"schedule": "interval:5"}, headers=owner
    )
    assert too_fast.status_code == 422
    assert "30 seconds" in too_fast.json()["detail"]


def test_reset_and_history_name_the_device(
    client: TestClient, devices: DeviceService, runtime: SimpleNamespace
) -> None:
    owner = _pair(devices, "control")
    client.patch("/heartbeat/email_sweep", json={"schedule": "interval:1800"}, headers=owner)

    reset = client.delete("/heartbeat/email_sweep/override", headers=owner)

    assert reset.status_code == 200
    assert reset.json()["schedule"] == "interval:600"
    assert reset.json()["overridden"] is False
    history = client.get("/settings/history?section=heartbeat", headers=owner).json()
    assert [c["action"] for c in history["changes"]] == ["reset", "set"]
    assert {c["actor_name"] for c in history["changes"]} == {"control phone"}
    assert history["changes"][1]["new"] == {"schedule": "interval:1800", "enabled": True}
    assert (runtime.data_dir / SETTINGS_DB_NAME).exists()
