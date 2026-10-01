"""The reconnect API on the real IRIS API app: who may call what, and the callback.

The writes (``start``, the client upload) are gated like every control write; the
callback is the one route that answers without a bearer token or device cookie, and
only by its one-time ``state``. ``IRIS_WEBUI_ALLOW_WRITES`` is unset unless a test says
otherwise: a default install.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.kernel.governance.devices import DeviceService, DeviceStore
from iris_harness.kernel.governance.vault import credentials
from iris_harness.server.iris_api.main import create_app
from iris_harness.services.health import service as health_service
from iris_harness.services.health.models import (
    CheckKind,
    HealthCheck,
    HealthSnapshot,
    HealthState,
)
from iris_personal.connections import google

from .fake_google import CLIENT_SECRET, DESKTOP_CLIENT_JSON, WEB_CLIENT_JSON, FakeGoogle

OWNER = "you@example.com"
START = "/api/v1/connections/google/start"
CALLBACK = "/api/v1/connections/google/callback"
CLIENT = "/api/v1/connections/google/client"


@pytest.fixture
def client(
    fake: FakeGoogle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    service = DeviceService(store=DeviceStore(db_path=tmp_path / "devices.db"))
    app = create_app(runtime=SimpleNamespace(), auto_start_runtime=False, device_service=service)
    with TestClient(app, base_url="http://iris.test") as c:
        yield c


def _device(client: TestClient, scope: str) -> dict[str, str]:
    code = client.post(
        "/api/v1/devices/pair/start", json={"scope": scope}, headers=auth_headers()
    ).json()["code"]
    claimed = client.post(
        "/api/v1/devices/pair/claim", json={"code": code, "name": "Phone", "kind": "app"}
    ).json()
    client.cookies.clear()
    return {"Authorization": f"Bearer {claimed['token']}"}


def _callback(client: TestClient, params: dict[str, str]) -> tuple[int, str, dict[str, str]]:
    """Google's redirect: a bare browser, no header, no cookie."""
    client.cookies.clear()
    resp = client.get(CALLBACK, params=params, follow_redirects=False)
    return resp.status_code, resp.headers.get("location", ""), dict(resp.headers)


def test_a_control_phone_reconnects_end_to_end(client: TestClient, fake: FakeGoogle) -> None:
    phone = _device(client, "control")
    assert client.put(CLIENT, json={"client_json": WEB_CLIENT_JSON}, headers=phone).is_success

    started = client.post(START, json={"provider": "gmail", "account": OWNER}, headers=phone)
    assert started.status_code == 200, started.text
    assert started.json()["expires_in"] == 600

    status, location, headers = _callback(
        client, fake.consent(started.json()["auth_url"], as_email=OWNER)
    )
    assert status == 303
    assert location == (
        "/settings?connect=connected&provider=gmail&account=you%40example.com#connections"
    )
    assert headers["cache-control"] == "no-store"
    assert headers["referrer-policy"] == "no-referrer"
    assert credentials.load_token("gmail", OWNER) is not None


def test_the_callback_lands_every_failure_on_connections_too(
    client: TestClient, fake: FakeGoogle
) -> None:
    google.save_client(WEB_CLIENT_JSON)
    phone = _device(client, "control")

    url = client.post(START, json={"provider": "gmail", "account": OWNER}, headers=phone)
    _, location, _ = _callback(client, fake.consent(url.json()["auth_url"], as_email="w@x.io"))
    q = parse_qs(urlsplit(location).query)
    assert q["connect"] == ["wrong_account"] and q["approved"] == ["w@x.io"]

    url = client.post(START, json={"provider": "gmail", "account": OWNER}, headers=phone)
    params = FakeGoogle.cancel(url.json()["auth_url"])
    assert "connect=cancelled" in _callback(client, params)[1]
    assert "connect=expired" in _callback(client, params)[1]  # used once already


def test_writes_are_gated_a_read_only_phone_is_refused(client: TestClient) -> None:
    google.save_client(WEB_CLIENT_JSON)
    phone = _device(client, "read")
    for method, path, body in (
        ("POST", START, {"provider": "gmail", "account": OWNER}),
        ("PUT", CLIENT, {"client_json": WEB_CLIENT_JSON}),
    ):
        refused = client.request(method, path, json=body, headers=phone)
        assert refused.status_code == 403, (method, path)
        assert refused.json() == {"detail": "this device is paired read-only"}
    # Reading the setup is fine for a read-only phone: it carries no secret.
    assert client.get(CLIENT, headers=phone).status_code == 200


def test_the_service_secret_needs_the_writes_switch(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    google.save_client(WEB_CLIENT_JSON)
    body = {"provider": "gmail", "account": OWNER}
    assert client.post(START, json=body, headers=auth_headers()).status_code == 403
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
    assert client.post(START, json=body, headers=auth_headers()).status_code == 200


def test_only_the_callback_get_is_open_and_only_exactly_it(client: TestClient) -> None:
    client.cookies.clear()
    assert client.get(CALLBACK, follow_redirects=False).status_code == 303  # expired
    for method, path in (
        ("POST", CALLBACK),
        ("GET", f"{CALLBACK}/"),
        ("GET", f"{CALLBACK}x"),
        ("GET", START),
        ("GET", CLIENT),
        ("POST", START),
        ("GET", "/memory/facts"),
    ):
        assert client.request(method, path).status_code == 401, (method, path)
    # The Host-header trick (#649) cannot borrow the open path for another route.
    smuggled = client.get("/memory/facts", headers={"host": f"iris.test{CALLBACK}?"})
    assert smuggled.status_code == 400


def test_start_explains_what_is_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    phone = _device(client, "control")
    body = {"provider": "gmail", "account": OWNER}
    no_client = client.post(START, json=body, headers=phone)
    assert no_client.status_code == 409 and "Web application client" in no_client.text
    google.save_client(WEB_CLIENT_JSON)
    monkeypatch.delenv("IRIS_PUBLIC_URL")
    no_url = client.post(START, json=body, headers=phone)
    assert no_url.status_code == 409 and "IRIS_PUBLIC_URL" in no_url.text
    assert client.get(CLIENT, headers=phone).json()["public_url_set"] is False


def test_the_client_upload_never_echoes_the_secret(client: TestClient) -> None:
    phone = _device(client, "control")
    desktop = client.put(CLIENT, json={"client_json": DESKTOP_CLIENT_JSON}, headers=phone)
    assert desktop.status_code == 422 and "Desktop client" in desktop.text
    stored = client.put(CLIENT, json={"client_json": WEB_CLIENT_JSON}, headers=phone)
    assert stored.json()["configured"] is True
    assert stored.json()["redirect_uri_listed"] is True
    for resp in (stored, client.get(CLIENT, headers=phone)):
        assert CLIENT_SECRET not in resp.text


def test_no_token_in_any_response_body_or_header(client: TestClient, fake: FakeGoogle) -> None:
    phone = _device(client, "control")
    seen = [client.put(CLIENT, json={"client_json": WEB_CLIENT_JSON}, headers=phone)]
    for as_email in (OWNER, "w@x.io"):
        started = client.post(START, json={"provider": "gmail", "account": OWNER}, headers=phone)
        seen.append(started)
        client.cookies.clear()
        seen.append(
            client.get(
                CALLBACK,
                params=fake.consent(started.json()["auth_url"], as_email=as_email),
                follow_redirects=False,
            )
        )
    assert fake.issued
    for resp in seen:
        dumped = resp.text + repr(dict(resp.headers))
        for secret in (*fake.issued, CLIENT_SECRET):
            assert secret not in dumped


def test_health_connectors_rows_name_their_reconnect_route(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = google.with_reconnect(
        [HealthCheck("Gmail", CheckKind.CREDENTIAL, HealthState.RED, "revoked", subject=OWNER)],
        "gmail",
    )
    # Through monkeypatch, so the process-wide cache is put back after the test.
    monkeypatch.setattr(
        health_service, "_cached", HealthSnapshot(checks=tuple(rows), sampled_at="now")
    )
    body = client.get("/health/connectors", headers=auth_headers()).json()
    row = body["connectors"][0]
    # The fields every consumer read before are unchanged; the new ones are additive.
    assert row["target"] == "Gmail" and row["subject"] == OWNER and row["state"] == "red"
    assert row["fix_url"] == "/settings#connections"
    assert row["reconnect"]["route"] == START and row["reconnect"]["account"] == OWNER
