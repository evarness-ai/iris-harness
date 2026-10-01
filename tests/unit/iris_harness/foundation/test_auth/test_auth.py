"""Bearer-auth policy tests — the security floor every backend service shares.

Covers the pure ``authorize`` policy, the installed middleware behavior on a
real app, and that the evaluator + iris_api services actually enforce it
(middleware installed, not just importable).
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import (
    SERVICE_PRINCIPAL,
    Principal,
    auth_headers,
    authorize,
    bearer_token,
    resolve_principal,
    same_origin,
)
from iris_harness.server.auth import install_bearer_auth

SECRET_ENV = "IRIS_AUTH_SECRET"


# ── authorize(): the pure policy ─────────────────────────────────────────────


def test_exempt_paths_pass_without_header(monkeypatch):
    monkeypatch.setenv(SECRET_ENV, "s3cret")
    assert authorize(None, "/healthz") is None
    assert authorize(None, "/health") is None


def test_valid_token_passes(monkeypatch):
    monkeypatch.setenv(SECRET_ENV, "s3cret")
    assert authorize("Bearer s3cret", "/chat") is None


def test_missing_header_denied_401(monkeypatch):
    monkeypatch.setenv(SECRET_ENV, "s3cret")
    status, _ = authorize(None, "/chat")
    assert status == 401


def test_wrong_token_denied_401(monkeypatch):
    monkeypatch.setenv(SECRET_ENV, "s3cret")
    status, _ = authorize("Bearer wrong", "/chat")
    assert status == 401


def test_unset_secret_fails_closed_503(monkeypatch):
    monkeypatch.delenv(SECRET_ENV, raising=False)
    status, detail = authorize("Bearer anything", "/chat")
    assert status == 503
    assert "IRIS_AUTH_SECRET" in detail


def test_blank_secret_treated_as_unset(monkeypatch):
    monkeypatch.setenv(SECRET_ENV, "   ")
    status, _ = authorize("Bearer anything", "/chat")
    assert status == 503


def test_bearer_token_parsing():
    assert bearer_token("Bearer tok") == "tok"
    assert bearer_token("bearer tok") == "tok"  # scheme is case-insensitive
    assert bearer_token("Basic dXNlcg==") is None
    assert bearer_token("Bearer   ") is None
    assert bearer_token(None) is None


# ── auth_headers(): the client half ──────────────────────────────────────────


def test_auth_headers_round_trip(monkeypatch):
    monkeypatch.setenv(SECRET_ENV, "s3cret")
    headers = auth_headers()
    assert headers == {"Authorization": "Bearer s3cret"}
    assert authorize(headers["Authorization"], "/chat") is None


def test_auth_headers_empty_when_unset(monkeypatch):
    monkeypatch.delenv(SECRET_ENV, raising=False)
    assert auth_headers() == {}


# ── install_bearer_auth(): middleware on a real app ──────────────────────────


def _app() -> FastAPI:
    app = FastAPI()
    install_bearer_auth(app)

    @app.get("/echo")
    def echo() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    return app


def test_middleware_denies_without_header(monkeypatch):
    monkeypatch.setenv(SECRET_ENV, "s3cret")
    with TestClient(_app()) as client:
        resp = client.get("/echo")
    assert resp.status_code == 401
    assert resp.headers["WWW-Authenticate"] == "Bearer"


def test_middleware_allows_with_header(monkeypatch):
    monkeypatch.setenv(SECRET_ENV, "s3cret")
    with TestClient(_app(), headers={"Authorization": "Bearer s3cret"}) as client:
        resp = client.get("/echo")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_middleware_exempts_probe(monkeypatch):
    monkeypatch.setenv(SECRET_ENV, "s3cret")
    with TestClient(_app()) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200


def test_middleware_fails_closed_when_secret_unset(monkeypatch):
    monkeypatch.delenv(SECRET_ENV, raising=False)
    with TestClient(_app(), headers={"Authorization": "Bearer anything"}) as client:
        resp = client.get("/echo")
    assert resp.status_code == 503


def test_middleware_rejects_before_routing(monkeypatch):
    # Auth runs before route resolution: unknown paths answer 401, not 404,
    # so the route table is not enumerable without a token.
    monkeypatch.setenv(SECRET_ENV, "s3cret")
    with TestClient(_app()) as client:
        resp = client.get("/definitely-not-a-route")
    assert resp.status_code == 401


# ── services actually enforce it ─────────────────────────────────────────────


def test_evaluator_service_requires_auth():
    from iris_harness.server.evaluator.main import create_app

    with TestClient(create_app()) as client:
        denied = client.post("/evaluate", json={})
        probe = client.get("/healthz")
        allowed = client.post("/reset/run-1", headers=auth_headers())
    assert denied.status_code == 401
    assert probe.status_code == 200
    assert allowed.status_code == 200


def test_iris_api_service_requires_auth():
    from iris_harness.server.iris_api.main import create_app

    app = create_app(runtime=SimpleNamespace(), auto_start_runtime=False)
    with TestClient(app) as client:
        denied = client.get("/routines")
        probe = client.get("/healthz")
        snapshot = client.get("/health")
    assert denied.status_code == 401
    assert probe.status_code == 200
    # On iris_api `/health` is the System Health snapshot — data, not a probe — so
    # it takes a token like every other route (ADR-0117).
    assert snapshot.status_code == 401


# ── resolve_principal(): secret OR paired device (ADR-0117) ─────────────────

_DEVICES = {"irisd_reader": ("dev-r", "read"), "irisd_owner": ("dev-c", "control")}


def _resolve(monkeypatch, *, secret_set: bool = True, verifier=_DEVICES.get, **overrides):
    if secret_set:
        monkeypatch.setenv(SECRET_ENV, "s3cret")
    else:
        monkeypatch.delenv(SECRET_ENV, raising=False)
    request = {
        "authorization": None,
        "cookie_token": None,
        "method": "GET",
        "path": "/routines",
        "origin": None,
        "host": "iris.example.ts.net",
        "verifier": verifier,
    }
    request.update(overrides)
    return resolve_principal(**request)


def test_secret_resolves_to_the_service_principal(monkeypatch):
    got = _resolve(monkeypatch, authorization="Bearer s3cret")
    assert got == SERVICE_PRINCIPAL
    assert got.kind == "service" and got.can_control


def test_device_bearer_resolves_with_its_own_scope(monkeypatch):
    reader = _resolve(monkeypatch, authorization="Bearer irisd_reader")
    owner = _resolve(monkeypatch, authorization="Bearer irisd_owner")
    assert reader == Principal(kind="device", scope="read", device_id="dev-r", via="bearer")
    assert not reader.can_control
    assert owner == Principal(kind="device", scope="control", device_id="dev-c", via="bearer")
    assert owner.can_control


def test_unknown_token_is_401(monkeypatch):
    assert _resolve(monkeypatch, authorization="Bearer irisd_nobody") == (
        401,
        "missing or invalid bearer token",
    )


def test_no_credential_is_401(monkeypatch):
    got = _resolve(monkeypatch)
    assert isinstance(got, tuple) and got[0] == 401


def test_without_a_verifier_device_tokens_are_refused(monkeypatch):
    """The governor and the evaluator pass no verifier: secret only, header or cookie."""
    by_header = _resolve(monkeypatch, verifier=None, authorization="Bearer irisd_owner")
    by_cookie = _resolve(monkeypatch, verifier=None, cookie_token="irisd_owner")
    assert isinstance(by_header, tuple) and by_header[0] == 401
    assert isinstance(by_cookie, tuple) and by_cookie[0] == 401
    assert _resolve(monkeypatch, verifier=None, authorization="Bearer s3cret") == SERVICE_PRINCIPAL


def test_unset_secret_fails_closed_even_for_a_live_device(monkeypatch):
    got = _resolve(monkeypatch, secret_set=False, authorization="Bearer irisd_owner")
    assert isinstance(got, tuple) and got[0] == 503
    got = _resolve(monkeypatch, secret_set=False, cookie_token="irisd_owner")
    assert isinstance(got, tuple) and got[0] == 503


def test_exempt_path_has_no_principal(monkeypatch):
    assert _resolve(monkeypatch, path="/healthz", exempt_paths=frozenset({"/healthz"})) is None
    got = _resolve(monkeypatch, path="/health", exempt_paths=frozenset({"/healthz"}))
    assert isinstance(got, tuple) and got[0] == 401


def test_cookie_authenticates_a_read(monkeypatch):
    got = _resolve(monkeypatch, cookie_token="irisd_reader")
    assert got == Principal(kind="device", scope="read", device_id="dev-r", via="cookie")


def test_a_bad_bearer_does_not_fall_back_to_a_good_cookie(monkeypatch):
    got = _resolve(monkeypatch, authorization="Bearer wrong", cookie_token="irisd_owner")
    assert isinstance(got, tuple) and got[0] == 401


def test_the_secret_is_not_accepted_as_a_cookie(monkeypatch):
    """The cookie slot is for device tokens only; the secret never rides in a browser."""
    got = _resolve(monkeypatch, verifier=lambda _t: None, cookie_token="s3cret")
    assert isinstance(got, tuple) and got[0] == 401


def test_cookie_write_needs_a_matching_origin(monkeypatch):
    for method in ("POST", "PUT", "PATCH", "DELETE", "post"):
        same = _resolve(
            monkeypatch,
            cookie_token="irisd_owner",
            method=method,
            origin="https://iris.example.ts.net",
        )
        assert isinstance(same, Principal), method
        for origin in (None, "null", "https://evil.example", "https://iris.example.ts.net.evil.io"):
            cross = _resolve(monkeypatch, cookie_token="irisd_owner", method=method, origin=origin)
            assert isinstance(cross, tuple) and cross[0] == 403, (method, origin)


def test_cookie_read_and_bearer_write_skip_the_origin_check(monkeypatch):
    """A cross-site GET cannot read the response, and a bearer header is never
    attached by the browser on its own — neither is a CSRF vector."""
    read = _resolve(monkeypatch, cookie_token="irisd_owner", origin="https://evil.example")
    write = _resolve(
        monkeypatch,
        authorization="Bearer irisd_owner",
        method="POST",
        origin="https://evil.example",
    )
    assert isinstance(read, Principal)
    assert isinstance(write, Principal)


def test_a_revoked_cookie_is_401_not_403(monkeypatch):
    """Identity first: an unknown cookie on a cross-origin write is unauthenticated."""
    got = _resolve(
        monkeypatch, cookie_token="irisd_gone", method="POST", origin="https://evil.example"
    )
    assert isinstance(got, tuple) and got[0] == 401


def test_same_origin():
    assert same_origin("https://iris.example.ts.net", "iris.example.ts.net")
    assert same_origin("http://127.0.0.1:8003", "127.0.0.1:8003")
    assert same_origin("https://IRIS.example.ts.net", "iris.example.ts.net")
    # The scheme differs behind `tailscale serve` (https outside, http inside).
    assert same_origin("https://iris.example.ts.net", "iris.example.ts.net")
    assert not same_origin("http://127.0.0.1:5173", "127.0.0.1:8003")
    assert not same_origin("https://evil.example", "iris.example.ts.net")
    assert not same_origin("null", "iris.example.ts.net")
    assert not same_origin("", "iris.example.ts.net")
    assert not same_origin(None, "iris.example.ts.net")
    assert not same_origin("https://iris.example.ts.net", None)
    assert not same_origin("iris.example.ts.net", "iris.example.ts.net")


# ── the middleware with a verifier ───────────────────────────────────────────


def _device_app() -> FastAPI:
    app = FastAPI()
    install_bearer_auth(app, device_verifier=_DEVICES.get)

    @app.get("/whoami")
    def whoami(request: Request) -> dict[str, str | None]:
        principal = request.state.principal
        return {"kind": principal.kind, "scope": principal.scope, "via": principal.via}

    @app.post("/write")
    def write() -> dict[str, bool]:
        return {"ok": True}

    return app


def test_middleware_sets_the_principal(monkeypatch):
    monkeypatch.setenv(SECRET_ENV, "s3cret")
    with TestClient(_device_app()) as client:
        service = client.get("/whoami", headers={"Authorization": "Bearer s3cret"})
        device = client.get("/whoami", headers={"Authorization": "Bearer irisd_reader"})
        cookie = client.get("/whoami", headers={"Cookie": "iris_device=irisd_owner"})
    assert service.json() == {"kind": "service", "scope": "control", "via": "bearer"}
    assert device.json() == {"kind": "device", "scope": "read", "via": "bearer"}
    assert cookie.json() == {"kind": "device", "scope": "control", "via": "cookie"}


def test_middleware_checks_origin_against_host_on_cookie_writes(monkeypatch):
    monkeypatch.setenv(SECRET_ENV, "s3cret")
    with TestClient(_device_app(), base_url="http://iris.test") as client:
        cookie = {"Cookie": "iris_device=irisd_owner"}
        same = client.post("/write", headers={**cookie, "Origin": "http://iris.test"})
        cross = client.post("/write", headers={**cookie, "Origin": "https://evil.example"})
        bare = client.post("/write", headers=cookie)
    assert same.status_code == 200
    assert cross.status_code == 403
    assert bare.status_code == 403
