"""The pairing flow on the real iris_api app: ``/api/v1/devices`` (ADR-0117).

Every test goes through ``create_app`` so the middleware order is part of what is
under test: the claim path's bearer exemption, the write guard's device-admin
exemption and the CSRF rule are wiring, and wiring is what breaks silently.

``IRIS_WEBUI_ALLOW_WRITES`` is deliberately NOT set anywhere in this file: pairing
and revoking must work on a default install.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from iris_harness.foundation.auth import auth_headers
from iris_harness.kernel.governance.devices import (
    CLAIM_FAILURE_LIMIT,
    PAIRING_MAX_ATTEMPTS,
    ClaimThrottle,
    DeviceService,
    DeviceStore,
)
from iris_harness.server.iris_api.device_routes import (
    DEVICE_COOKIE_MAX_AGE,
    is_device_admin_write,
)
from iris_harness.server.iris_api.main import _is_gated_write, create_app

BASE = "/api/v1/devices"
ORIGIN = {"Origin": "http://iris.test"}
DEVICE_FIELDS = {
    "device_id",
    "name",
    "kind",
    "scope",
    "created_at",
    "last_seen_at",
    "revoked_at",
    "current",
}


class _Ledger:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **row: Any) -> int:
        self.rows.append(row)
        return len(self.rows)

    def decisions(self) -> list[str]:
        return [r["decision"] for r in self.rows]


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _default_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)


@pytest.fixture
def ledger() -> _Ledger:
    return _Ledger()


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "devices.db"


@pytest.fixture
def client(db_path: Path, ledger: _Ledger, clock: _Clock) -> Iterator[TestClient]:
    service = DeviceService(
        store=DeviceStore(db_path=db_path), ledger=ledger, throttle=ClaimThrottle(clock=clock)
    )
    app = create_app(runtime=SimpleNamespace(), auto_start_runtime=False, device_service=service)
    with TestClient(app, base_url="http://iris.test") as c:
        yield c


def _start(
    client: TestClient, scope: str = "control", headers: dict[str, str] | None = None
) -> str:
    resp = client.post(
        f"{BASE}/pair/start", json={"scope": scope}, headers=headers or auth_headers()
    )
    assert resp.status_code == 200, resp.text
    return str(resp.json()["code"])


def _claim(
    client: TestClient, code: str, *, kind: str = "app", name: str = "My phone", **kw: Any
) -> Response:
    # A fresh cookie jar per claim: a browser claim must not leak its cookie into
    # the next request of the shared TestClient.
    client.cookies.clear()
    resp = client.post(f"{BASE}/pair/claim", json={"code": code, "name": name, "kind": kind}, **kw)
    client.cookies.clear()
    return resp


def _app_device(
    client: TestClient, scope: str = "control", name: str = "My phone"
) -> tuple[str, dict[str, str]]:
    body = _claim(client, _start(client, scope), name=name).json()
    return body["device"]["device_id"], {"Authorization": f"Bearer {body['token']}"}


def _browser_device(client: TestClient, scope: str = "control") -> tuple[str, dict[str, str]]:
    resp = _claim(client, _start(client, scope), kind="browser", name="My laptop")
    token = resp.headers["set-cookie"].split(";")[0].split("=", 1)[1]
    return resp.json()["device"]["device_id"], {"Cookie": f"iris_device={token}"}


def _cookie_flags(resp: Response) -> set[str]:
    return {part.strip().lower() for part in resp.headers["set-cookie"].split(";")[1:]}


# ── pair/start ───────────────────────────────────────────────────────────────


def test_start_returns_a_code_for_the_service(client: TestClient, ledger: _Ledger) -> None:
    resp = client.post(f"{BASE}/pair/start", json={"scope": "read"}, headers=auth_headers())
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"code", "scope", "expires_at"}
    assert body["scope"] == "read"
    assert len(body["code"]) == 9 and body["code"][4] == "-"
    assert ledger.rows[-1]["decision"] == "pairing_started"
    assert ledger.rows[-1]["payload"]["actor"] == "service"


def test_start_defaults_to_control_with_or_without_a_body(client: TestClient) -> None:
    assert (
        client.post(f"{BASE}/pair/start", json={}, headers=auth_headers()).json()["scope"]
        == "control"
    )
    assert client.post(f"{BASE}/pair/start", headers=auth_headers()).json()["scope"] == "control"


def test_start_rejects_an_unknown_scope(client: TestClient) -> None:
    resp = client.post(f"{BASE}/pair/start", json={"scope": "admin"}, headers=auth_headers())
    assert resp.status_code == 422


def test_a_control_device_can_start_pairing_and_is_the_ledger_actor(
    client: TestClient, ledger: _Ledger
) -> None:
    device_id, bearer = _app_device(client)
    assert client.post(f"{BASE}/pair/start", json={}, headers=bearer).status_code == 200
    assert ledger.rows[-1]["payload"]["actor"] == f"device:{device_id}"


def test_a_read_device_cannot_start_pairing(client: TestClient, ledger: _Ledger) -> None:
    _, bearer = _app_device(client, "read")
    before = ledger.decisions().count("pairing_started")
    resp = client.post(f"{BASE}/pair/start", json={"scope": "read"}, headers=bearer)
    assert resp.status_code == 403
    assert resp.json() == {"detail": "this device is paired read-only"}
    assert ledger.decisions().count("pairing_started") == before


# ── pair/claim ───────────────────────────────────────────────────────────────


def test_an_app_claim_gets_the_token_in_the_body_and_no_cookie(client: TestClient) -> None:
    resp = _claim(client, _start(client, "read"), name="  Owner   iPhone ")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"device", "token"}
    assert body["token"].startswith("irisd_")
    assert "set-cookie" not in resp.headers
    assert set(body["device"]) == DEVICE_FIELDS
    assert body["device"]["name"] == "Owner iPhone"
    assert body["device"]["kind"] == "app" and body["device"]["scope"] == "read"
    assert body["device"]["current"] is True
    # And the token is a working credential.
    me = client.get(f"{BASE}/me", headers={"Authorization": f"Bearer {body['token']}"})
    assert me.status_code == 200


def test_a_browser_claim_gets_an_httponly_cookie_and_no_token_in_the_body(
    client: TestClient,
) -> None:
    resp = _claim(client, _start(client), kind="browser")
    assert resp.status_code == 200
    assert set(resp.json()) == {"device"}
    assert "irisd_" not in resp.text

    set_cookie = resp.headers["set-cookie"]
    assert set_cookie.startswith("iris_device=irisd_")
    flags = _cookie_flags(resp)
    assert {"httponly", "samesite=strict", "path=/"} <= flags
    assert f"max-age={DEVICE_COOKIE_MAX_AGE}" in flags
    assert DEVICE_COOKIE_MAX_AGE == 400 * 24 * 60 * 60
    # Plain http, no proxy header: `Secure` would stop the cookie being stored at all.
    assert "secure" not in flags


def test_the_cookie_is_secure_when_the_request_arrived_over_https(client: TestClient) -> None:
    behind_proxy = _claim(
        client, _start(client), kind="browser", headers={"X-Forwarded-Proto": "https"}
    )
    assert "secure" in _cookie_flags(behind_proxy)

    code = _start(client)
    with TestClient(client.app, base_url="https://iris.test") as https_client:
        direct = https_client.post(
            f"{BASE}/pair/claim", json={"code": code, "name": "x", "kind": "browser"}
        )
    assert "secure" in _cookie_flags(direct)


def test_the_browser_cookie_authenticates(client: TestClient) -> None:
    _, cookie = _browser_device(client)
    me = client.get(f"{BASE}/me", headers=cookie)
    assert me.status_code == 200
    assert me.json()["via"] == "cookie"


def test_claim_needs_no_credential_and_every_other_device_route_does(client: TestClient) -> None:
    code = _start(client)
    assert client.post(f"{BASE}/pair/start", json={}).status_code == 401
    assert client.get(BASE).status_code == 401
    assert client.get(f"{BASE}/me").status_code == 401
    assert client.delete(f"{BASE}/some-id").status_code == 401
    assert _claim(client, code).status_code == 200


def test_claim_works_with_no_secret_header_even_when_a_stale_cookie_rides_along(
    client: TestClient,
) -> None:
    """A browser re-pairing after a revoke still sends its dead cookie."""
    resp = client.post(
        f"{BASE}/pair/claim",
        json={"code": _start(client), "name": "again", "kind": "browser"},
        headers={"Cookie": "iris_device=irisd_dead"},
    )
    assert resp.status_code == 200


def test_every_code_that_does_not_pair_gets_the_same_400(client: TestClient) -> None:
    used = _start(client)
    assert _claim(client, used).status_code == 200
    answers = [
        _claim(client, "ZZZZ-ZZZZ"),  # wrong
        _claim(client, used),  # already used
        _claim(client, ""),  # empty
        _claim(client, "x" * 500),  # malformed
    ]
    for resp in answers:
        assert resp.status_code == 400
        assert resp.json() == {"detail": "pairing code not accepted"}
        assert "set-cookie" not in resp.headers


def test_a_code_voided_by_wrong_guesses_gets_the_same_400(client: TestClient) -> None:
    code = _start(client)
    for _ in range(PAIRING_MAX_ATTEMPTS):
        assert _claim(client, "ZZZZ-ZZZZ").status_code == 400
    resp = _claim(client, code)
    assert resp.status_code == 400
    assert resp.json() == {"detail": "pairing code not accepted"}


@pytest.mark.parametrize(
    "body",
    [
        {"code": "ABCD-EFGH", "name": "x", "kind": "toaster"},
        {"code": "ABCD-EFGH", "name": "", "kind": "app"},
        {"code": "ABCD-EFGH", "name": "   ", "kind": "app"},
        {"code": "ABCD-EFGH", "name": "n" * 65, "kind": "app"},
        {"code": "ABCD-EFGH", "kind": "app"},
    ],
)
def test_a_bad_kind_or_name_is_422_and_is_not_a_failed_claim(
    client: TestClient, body: dict[str, str]
) -> None:
    code = _start(client)
    for _ in range(PAIRING_MAX_ATTEMPTS + 1):
        assert client.post(f"{BASE}/pair/claim", json=body).status_code == 422
    # None of those was charged to the live code.
    assert _claim(client, code).status_code == 200


# ── the global throttle ──────────────────────────────────────────────────────


def _trip(client: TestClient) -> None:
    for _ in range(CLAIM_FAILURE_LIMIT):
        assert _claim(client, "ZZZZ-ZZZZ").status_code == 400


def test_the_throttle_answers_429_with_retry_after(client: TestClient, clock: _Clock) -> None:
    _trip(client)
    clock.now += 20
    resp = _claim(client, "ZZZZ-ZZZZ")
    assert resp.status_code == 429
    assert resp.json() == {"detail": "too many pairing attempts — try again shortly"}
    assert resp.headers["retry-after"] == "40"


def test_while_throttled_a_correct_code_is_not_checked_and_no_attempt_is_charged(
    client: TestClient, clock: _Clock, db_path: Path
) -> None:
    _trip(client)
    code = _start(client)  # minted after the trip: zero attempts against it

    for _ in range(PAIRING_MAX_ATTEMPTS + 2):
        assert _claim(client, "ZZZZ-ZZZZ").status_code == 429
    # The right code is refused too — a 200 here would be the oracle.
    assert _claim(client, code).status_code == 429

    with sqlite3.connect(db_path) as conn:
        attempts = conn.execute(
            "SELECT attempts FROM pairing_codes WHERE used_at IS NULL"
        ).fetchall()
    assert attempts == [(0,)]

    # …and once the window has passed, that same code still pairs.
    clock.now += 61
    assert _claim(client, code).status_code == 200


def test_a_trip_is_ledgered_once_and_says_nothing_secret(
    client: TestClient, clock: _Clock, ledger: _Ledger
) -> None:
    _trip(client)
    for _ in range(5):
        assert _claim(client, "ZZZZ-ZZZZ").status_code == 429
    assert ledger.decisions().count("pairing_throttled") == 1
    row = next(r for r in ledger.rows if r["decision"] == "pairing_throttled")
    assert row["payload"] == {"limit": CLAIM_FAILURE_LIMIT, "window_seconds": 60}
    assert "ZZZZ" not in repr(row)

    clock.now += 61
    _trip(client)
    assert ledger.decisions().count("pairing_throttled") == 2


# ── list / me ────────────────────────────────────────────────────────────────


def test_list_is_newest_first_includes_revoked_and_flags_the_caller(client: TestClient) -> None:
    first_id, first = _app_device(client, name="first")
    second_id, _ = _app_device(client, "read", name="second")
    client.delete(f"{BASE}/{second_id}", headers=auth_headers())

    as_service = client.get(BASE, headers=auth_headers()).json()["devices"]
    assert [d["device_id"] for d in as_service] == [second_id, first_id]
    assert as_service[0]["revoked_at"] is not None and as_service[1]["revoked_at"] is None
    assert [d["current"] for d in as_service] == [False, False]

    as_first = client.get(BASE, headers=first).json()["devices"]
    assert {d["device_id"]: d["current"] for d in as_first} == {first_id: True, second_id: False}


def test_a_read_device_can_list_and_see_itself(client: TestClient) -> None:
    device_id, bearer = _app_device(client, "read")
    assert client.get(BASE, headers=bearer).status_code == 200
    me = client.get(f"{BASE}/me", headers=bearer).json()
    assert me["kind"] == "device" and me["scope"] == "read" and me["via"] == "bearer"
    assert me["device"]["device_id"] == device_id and me["device"]["current"] is True


def test_me_for_the_service_principal(client: TestClient) -> None:
    me = client.get(f"{BASE}/me", headers=auth_headers())
    assert me.json() == {"kind": "service", "scope": "control", "via": "bearer", "device": None}


def test_no_token_or_hash_in_any_list_or_me_response(client: TestClient, db_path: Path) -> None:
    device_id, bearer = _app_device(client)
    token = bearer["Authorization"].removeprefix("Bearer ")
    with sqlite3.connect(db_path) as conn:
        (token_hash,) = conn.execute(
            "SELECT token_hash FROM devices WHERE device_id = ?", (device_id,)
        ).fetchone()

    for headers in (bearer, auth_headers()):
        for path in (BASE, f"{BASE}/me"):
            resp = client.get(path, headers=headers)
            assert token not in resp.text and token_hash not in resp.text
            assert "irisd_" not in resp.text and "hash" not in resp.text
    for device in client.get(BASE, headers=bearer).json()["devices"]:
        assert set(device) == DEVICE_FIELDS


# ── revoke ───────────────────────────────────────────────────────────────────


def test_service_revokes_a_device_and_its_token_stops_working(
    client: TestClient, ledger: _Ledger
) -> None:
    device_id, bearer = _app_device(client)
    assert client.get(f"{BASE}/me", headers=bearer).status_code == 200

    resp = client.delete(f"{BASE}/{device_id}", headers=auth_headers())
    assert resp.status_code == 200
    assert resp.json()["device"]["device_id"] == device_id
    assert resp.json()["device"]["revoked_at"] is not None
    assert client.get(f"{BASE}/me", headers=bearer).status_code == 401

    # Idempotent: 200 again, and still one ledger row.
    assert client.delete(f"{BASE}/{device_id}", headers=auth_headers()).status_code == 200
    assert ledger.decisions().count("revoked") == 1
    assert ledger.rows[-1]["payload"]["actor"] == "service"


def test_revoking_an_unknown_device_is_404(client: TestClient) -> None:
    assert client.delete(f"{BASE}/no-such-device", headers=auth_headers()).status_code == 404


def test_a_control_device_revokes_another(client: TestClient, ledger: _Ledger) -> None:
    owner_id, owner = _app_device(client)
    other_id, other = _app_device(client, "read")
    assert client.delete(f"{BASE}/{other_id}", headers=owner).status_code == 200
    assert client.get(f"{BASE}/me", headers=other).status_code == 401
    assert ledger.rows[-1]["payload"]["actor"] == f"device:{owner_id}"


def test_a_read_device_cannot_revoke_another_nor_probe_for_ids(client: TestClient) -> None:
    other_id, other = _app_device(client)
    _, reader = _app_device(client, "read")
    for target in (other_id, "no-such-device"):
        resp = client.delete(f"{BASE}/{target}", headers=reader)
        assert resp.status_code == 403
        assert resp.json() == {"detail": "this device is paired read-only"}
    assert client.get(f"{BASE}/me", headers=other).status_code == 200


def test_a_read_device_can_revoke_itself(client: TestClient, ledger: _Ledger) -> None:
    device_id, reader = _app_device(client, "read")
    resp = client.delete(f"{BASE}/{device_id}", headers=reader)
    assert resp.status_code == 200
    assert resp.json()["device"]["revoked_at"] is not None
    assert "set-cookie" not in resp.headers  # a bearer client holds no cookie to clear
    assert client.get(f"{BASE}/me", headers=reader).status_code == 401
    assert ledger.rows[-1]["payload"]["actor"] == f"device:{device_id}"


def test_a_browser_signing_out_has_its_cookie_cleared(client: TestClient) -> None:
    device_id, cookie = _browser_device(client, "read")
    resp = client.delete(f"{BASE}/{device_id}", headers={**cookie, **ORIGIN})
    assert resp.status_code == 200
    set_cookie = resp.headers["set-cookie"]
    assert set_cookie.startswith('iris_device="";') or set_cookie.startswith("iris_device=;")
    flags = _cookie_flags(resp)
    assert {"max-age=0", "httponly", "samesite=strict", "path=/"} <= flags
    assert client.get(f"{BASE}/me", headers=cookie).status_code == 401


def test_revoking_someone_else_by_cookie_keeps_your_own_cookie(client: TestClient) -> None:
    _, cookie = _browser_device(client)
    other_id, _ = _app_device(client)
    resp = client.delete(f"{BASE}/{other_id}", headers={**cookie, **ORIGIN})
    assert resp.status_code == 200
    assert "set-cookie" not in resp.headers
    assert client.get(f"{BASE}/me", headers=cookie).status_code == 200


def test_cookie_authenticated_device_writes_are_same_origin_only(client: TestClient) -> None:
    device_id, cookie = _browser_device(client)
    evil = {**cookie, "Origin": "https://evil.example"}
    assert client.post(f"{BASE}/pair/start", json={}, headers=evil).status_code == 403
    assert client.delete(f"{BASE}/{device_id}", headers=evil).status_code == 403
    assert client.delete(f"{BASE}/{device_id}", headers=cookie).status_code == 403
    assert client.get(f"{BASE}/me", headers=cookie).status_code == 200  # not revoked
    assert (
        client.post(f"{BASE}/pair/start", json={}, headers={**cookie, **ORIGIN}).status_code == 200
    )


# ── the write gate ───────────────────────────────────────────────────────────


def test_exactly_the_three_device_writes_are_exempt_from_the_write_gate() -> None:
    assert not _is_gated_write("POST", f"{BASE}/pair/start")
    assert not _is_gated_write("POST", f"{BASE}/pair/claim")
    assert not _is_gated_write("DELETE", f"{BASE}/0b9d6c1e-aaaa-bbbb-cccc-000000000000")
    # Deny-by-default survives: nothing else under the prefix rides along.
    for method, path in [
        ("POST", BASE),
        ("POST", f"{BASE}/"),
        ("POST", f"{BASE}/pair/anything-new"),
        ("POST", f"{BASE}/some-id"),
        ("PUT", f"{BASE}/some-id"),
        ("PATCH", f"{BASE}/some-id"),
        ("DELETE", BASE),
        ("DELETE", f"{BASE}/"),
        ("DELETE", f"{BASE}/some-id/tokens"),
        ("DELETE", "/api/v1/devicesX/some-id"),
        ("DELETE", "/governance/approvals/x"),
    ]:
        assert not is_device_admin_write(method, path), (method, path)
        assert _is_gated_write(method, path), (method, path)


def test_the_rest_of_the_console_is_still_gated_on_a_default_install(client: TestClient) -> None:
    # Not the approval route: that one write the shared secret may make (the channel
    # gateway answering a Telegram tap, owner's decision 2026-09-21).
    resp = client.post("/tasks", json={"title": "x"}, headers=auth_headers())
    assert resp.status_code == 403
    assert "IRIS_WEBUI_ALLOW_WRITES" in resp.json()["detail"]


def test_only_iris_api_opens_the_claim_path() -> None:
    from iris_harness.foundation.auth import DEFAULT_EXEMPT_PATHS, PROBE_ONLY_EXEMPT_PATHS
    from iris_harness.server.iris_api.main import _BEARER_EXEMPT_PATHS

    # /readyz joined 2026-09-26: the container healthcheck's readiness probe. It answers
    # only whether the runtime was built (test_readiness.py), like /healthz.
    assert frozenset({"/healthz", "/readyz", f"{BASE}/pair/claim"}) == _BEARER_EXEMPT_PATHS
    assert frozenset({"/healthz"}) == PROBE_ONLY_EXEMPT_PATHS
    assert frozenset({"/healthz", "/health"}) == DEFAULT_EXEMPT_PATHS


def test_without_the_auth_middleware_the_routes_fail_closed(db_path: Path) -> None:
    """Defence in depth. Through ``create_app`` a credentialed route always has a
    principal; if the middleware were ever dropped or reordered, "no principal" must
    read as 401, not as the service."""
    from fastapi import FastAPI

    from iris_harness.server.iris_api.device_routes import install_device_routes

    service = DeviceService(store=DeviceStore(db_path=db_path), ledger=_Ledger())
    bare = FastAPI()
    install_device_routes(bare, lambda: service)
    with TestClient(bare, base_url="http://iris.test") as c:
        assert c.post(f"{BASE}/pair/start", json={}).status_code == 401
        assert c.get(BASE).status_code == 401
        assert c.get(f"{BASE}/me").status_code == 401
        assert c.delete(f"{BASE}/some-id").status_code == 401
    assert service.list_devices() == []


# ── the shared service ───────────────────────────────────────────────────────


def test_the_default_service_is_built_once_and_shared_by_verifier_and_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No injected service: the app builds its own from IRIS_HOME, lazily, and the
    middleware's verifier sees the device the route just created."""
    monkeypatch.setenv("IRIS_HOME", str(tmp_path))
    app = create_app(runtime=SimpleNamespace(), auto_start_runtime=False)
    with TestClient(app, base_url="http://iris.test") as c:
        assert not list(tmp_path.rglob("devices.db"))  # nothing opened yet
        code = c.post(f"{BASE}/pair/start", json={}, headers=auth_headers()).json()["code"]
        token = c.post(
            f"{BASE}/pair/claim", json={"code": code, "name": "x", "kind": "app"}
        ).json()["token"]
        assert c.get(f"{BASE}/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200
        # Shared throttle: trip it through the route, and the route refuses.
        for _ in range(CLAIM_FAILURE_LIMIT):
            c.post(f"{BASE}/pair/claim", json={"code": "ZZZZ-ZZZZ", "name": "x", "kind": "app"})
        refused = c.post(f"{BASE}/pair/claim", json={"code": "Z", "name": "x", "kind": "app"})
        assert refused.status_code == 429


def test_the_real_ledger_gets_the_rows_and_no_token_or_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No injected ledger: the rows land in the governance ``AuditLog``. Its default
    path is resolved at import (the session's throwaway home), so other tests' rows
    are in there too — hence the per-device ``run_id`` filter."""
    from iris_harness.kernel.governance.audit import AuditLog

    monkeypatch.setenv("IRIS_HOME", str(tmp_path))
    app = create_app(runtime=SimpleNamespace(), auto_start_runtime=False)
    audit = AuditLog()
    started_before = len(audit.query(run_id="device:pairing", decision="pairing_started"))
    with TestClient(app, base_url="http://iris.test") as c:
        code = c.post(f"{BASE}/pair/start", json={}, headers=auth_headers()).json()["code"]
        body = c.post(f"{BASE}/pair/claim", json={"code": code, "name": "x", "kind": "app"}).json()
        device_id = body["device"]["device_id"]
        c.delete(f"{BASE}/{device_id}", headers=auth_headers())

    started = audit.query(run_id="device:pairing", decision="pairing_started")
    assert len(started) == started_before + 1
    mine = audit.query(run_id=f"device:{device_id}")
    assert [r.decision for r in mine] == ["paired", "revoked"]
    dumped = repr(audit.query())
    assert body["token"] not in dumped
    assert code not in dumped and code.replace("-", "") not in dumped
