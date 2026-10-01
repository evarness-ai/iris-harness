"""Paired-device tokens on the real iris_api app (ADR-0117).

The pure policy is covered in ``foundation/test_auth``; this file is the wiring:
iris_api passes a verifier and the other services do not, the write guard reads the
principal, and ``/health`` stopped being an open path.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.kernel.governance.devices import DeviceService, PairedDevice
from iris_harness.server.iris_api.main import create_app

# A gated write whose handler answers without a runtime: what is under test is
# whether the request gets past the guard, not what the route does.
_GATED_WRITE = "/governance/approvals/no-such-approval/respond"
_UNGATED_WRITE = "/rag/search"


@pytest.fixture
def devices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DeviceService:
    # The app builds its own DeviceService from the default path, so the test and
    # the app meet in the same file only through IRIS_HOME.
    monkeypatch.setenv("IRIS_HOME", str(tmp_path))
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
    return DeviceService()


@pytest.fixture
def client(devices: DeviceService) -> Iterator[TestClient]:
    app = create_app(runtime=SimpleNamespace(tool_service=None), auto_start_runtime=False)
    with TestClient(app, base_url="http://iris.test") as c:
        yield c


def _pair(devices: DeviceService, scope: str) -> PairedDevice:
    code = devices.start_pairing(scope=scope, actor="service")
    return devices.claim(code=code.code, name=f"{scope} phone", kind="browser")


def _bearer(paired: PairedDevice) -> dict[str, str]:
    return {"Authorization": f"Bearer {paired.token}"}


def _respond(client: TestClient, headers: dict[str, str]) -> int:
    return client.post(_GATED_WRITE, json={"status": "approved"}, headers=headers).status_code


# ``/health`` builds a real snapshot, which probes the developer's running stack and
# Ollama; the network guard fails any test that reaches them (PR #596).
@pytest.mark.usefixtures("offline_services")
def test_health_snapshot_needs_a_token_and_the_probe_does_not(client: TestClient) -> None:
    assert client.get("/healthz").status_code == 200
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers=auth_headers()).status_code != 401


def test_device_token_reads_by_header_and_by_cookie(
    client: TestClient, devices: DeviceService
) -> None:
    reader = _pair(devices, "read")
    assert client.get("/capabilities").status_code == 401
    assert client.get("/capabilities", headers=_bearer(reader)).status_code == 200
    by_cookie = client.get("/capabilities", headers={"Cookie": f"iris_device={reader.token}"})
    assert by_cookie.status_code == 200


def test_read_device_is_refused_gated_writes_and_control_is_not(
    client: TestClient, devices: DeviceService
) -> None:
    reader, owner = _pair(devices, "read"), _pair(devices, "control")

    refused = client.post(_GATED_WRITE, json={"status": "approved"}, headers=_bearer(reader))
    assert refused.status_code == 403
    assert refused.json() == {"detail": "this device is paired read-only"}

    # Past the guard, the route itself answers: there is no such approval.
    assert _respond(client, _bearer(owner)) == 404
    assert _respond(client, auth_headers()) == 404


def test_read_device_keeps_the_ungated_surface(client: TestClient, devices: DeviceService) -> None:
    reader = _pair(devices, "read")
    resp = client.post(_UNGATED_WRITE, json={"query": "x"}, headers=_bearer(reader))
    assert resp.status_code not in (401, 403)


def test_a_control_device_no_longer_needs_the_global_switch(
    client: TestClient, devices: DeviceService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This used to assert the opposite — the switch won over a paired device.
    It was written while the switch stood in for authentication. A paired device
    *is* that authentication, and requiring both made the owner's own phone
    read-only on the deployment that can only be reached by pairing."""
    owner = _pair(devices, "control")
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES")
    allowed = client.post(_GATED_WRITE, json={"status": "approved"}, headers=_bearer(owner))
    assert allowed.status_code == 404  # past the guard; no such approval


def test_revoking_a_device_logs_it_out_on_the_next_request(
    client: TestClient, devices: DeviceService
) -> None:
    owner = _pair(devices, "control")
    assert client.get("/capabilities", headers=_bearer(owner)).status_code == 200
    devices.revoke(owner.device.device_id, actor="service")
    assert client.get("/capabilities", headers=_bearer(owner)).status_code == 401
    cookie = {"Cookie": f"iris_device={owner.token}"}
    assert client.get("/capabilities", headers=cookie).status_code == 401


def test_cookie_writes_are_same_origin_only(client: TestClient, devices: DeviceService) -> None:
    owner = _pair(devices, "control")
    cookie = {"Cookie": f"iris_device={owner.token}"}
    assert _respond(client, {**cookie, "Origin": "http://iris.test"}) == 404
    assert _respond(client, {**cookie, "Origin": "https://evil.example"}) == 403
    assert _respond(client, cookie) == 403
    # The ungated surface is a write too: chat from a hostile page is still CSRF.
    cross = client.post(
        _UNGATED_WRITE, json={"query": "x"}, headers={**cookie, "Origin": "https://evil.example"}
    )
    assert cross.status_code == 403


def test_governor_and_evaluator_do_not_accept_device_tokens(devices: DeviceService) -> None:
    from iris_harness.server.evaluator.main import create_app as evaluator_app
    from iris_harness.server.governor.main import create_app as governor_app

    owner = _pair(devices, "control")
    for build in (evaluator_app, governor_app):
        with TestClient(build()) as service:
            paths = [r.path for r in service.app.routes if getattr(r, "methods", None)]  # type: ignore[attr-defined]
            guarded = next(p for p in paths if p not in ("/healthz", "/health") and "{" not in p)
            for headers in (_bearer(owner), {"Cookie": f"iris_device={owner.token}"}):
                assert service.get(guarded, headers=headers).status_code == 401, guarded


# ── A paired device is the authentication the switch stood in for ───────────
#
# IRIS_WEBUI_ALLOW_WRITES existed because the server could not tell who was
# asking. A paired device says exactly that, so the switch no longer applies to
# one; it still governs the shared secret, which names a process, not a person.


@pytest.fixture
def writes_off(devices: DeviceService, monkeypatch: pytest.MonkeyPatch) -> DeviceService:
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    return devices


def test_control_device_writes_with_the_switch_off(
    client: TestClient, writes_off: DeviceService
) -> None:
    """The cloud console is reached only by paired devices; requiring the switch
    as well made the owner's own phone read-only."""
    owner = _pair(writes_off, "control")
    assert _respond(client, _bearer(owner)) == 404  # past the guard


def test_read_device_is_still_refused_with_the_switch_off(
    client: TestClient, writes_off: DeviceService
) -> None:
    reader = _pair(writes_off, "read")
    refused = client.post(_GATED_WRITE, json={"status": "approved"}, headers=_bearer(reader))
    assert refused.status_code == 403
    # The scope is the reason, not the switch — the message must not send the
    # owner off to set an env var that would change nothing.
    assert refused.json() == {"detail": "this device is paired read-only"}


def test_shared_secret_still_obeys_the_switch(
    client: TestClient, writes_off: DeviceService
) -> None:
    """A secret says which process is calling, never which person, so it is not
    the authentication the gate was waiting for."""
    refused = client.post("/tasks", json={"title": "x"}, headers=auth_headers())
    assert refused.status_code == 403
    assert "IRIS_WEBUI_ALLOW_WRITES" in refused.json()["detail"]


def test_the_one_write_the_shared_secret_may_make_is_answering_an_approval(
    client: TestClient, writes_off: DeviceService
) -> None:
    """The exception (owner's decision, 2026-09-21): the channel gateway answers a
    Telegram tap with the secret. The person is identified by Telegram — the chat
    allowlist and the tapping user's id, carried as the actor — not by the secret."""
    passed = client.post(_GATED_WRITE, json={"status": "approved"}, headers=auth_headers())
    assert passed.status_code == 404  # through the gate; there is just no such approval


def test_capabilities_answers_for_the_caller_not_the_process(
    client: TestClient, writes_off: DeviceService
) -> None:
    """The console hides its write controls from this field. Reporting the
    process-wide switch made a control device render itself read-only."""
    owner, reader = _pair(writes_off, "control"), _pair(writes_off, "read")

    assert client.get("/capabilities", headers=_bearer(owner)).json()["writes_enabled"] is True
    assert client.get("/capabilities", headers=_bearer(reader)).json()["writes_enabled"] is False
    assert client.get("/capabilities", headers=auth_headers()).json()["writes_enabled"] is False


def test_capabilities_read_device_stays_false_with_the_switch_on(
    client: TestClient, devices: DeviceService
) -> None:
    # `devices` sets the switch on; the read device must not be told it may write.
    reader = _pair(devices, "read")
    assert client.get("/capabilities", headers=_bearer(reader)).json()["writes_enabled"] is False
    assert client.get("/capabilities", headers=auth_headers()).json()["writes_enabled"] is True
