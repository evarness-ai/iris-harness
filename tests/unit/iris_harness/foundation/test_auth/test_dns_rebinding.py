"""The Host guard refuses a name the server is not reached by (DNS rebinding).

A page on attacker.example re-resolves its own name to 127.0.0.1 (or the VM's tailnet
address); the owner's browser then sends same-origin requests to IRIS carrying
``Host: attacker.example``. PR #649's guard refused only a malformed Host. These tests
pin the allowlist on top of it: every Host real traffic uses still passes, and any
other name is 400 — on an open route as much as on a protected one.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from iris_harness.server import auth as server_auth
from iris_harness.server.auth import (
    ALLOWED_HOSTS_ENV,
    COMPOSE_SERVICE_HOSTS,
    host_allowed,
    install_bearer_auth,
)

SECRET = "rebinding-test-secret"
ATTACKER = "attacker.example"
REPO = Path(__file__).resolve().parents[5]


def _call(app: Any, method: str, path: str, *, host: str, token: str | None = None) -> int:
    headers = [(b"host", host.encode()), (b"content-type", b"application/json")]
    if token is not None:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8003),
    }
    body = b'{"code": "000000"}' if method == "POST" else b""
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return int(next(m["status"] for m in sent if m["type"] == "http.response.start"))


def _bearer_app() -> FastAPI:
    app = FastAPI()
    install_bearer_auth(app)

    @app.get("/echo")
    def echo() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    return app


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No inherited allowlist, a known machine name, and a fresh refusal log."""
    monkeypatch.setenv("IRIS_AUTH_SECRET", SECRET)
    monkeypatch.delenv(ALLOWED_HOSTS_ENV, raising=False)
    monkeypatch.delenv("IRIS_PUBLIC_URL", raising=False)
    monkeypatch.setattr(server_auth.socket, "gethostname", lambda: "owners-mac.local")
    server_auth._machine_names.cache_clear()
    server_auth._REFUSED_LOGGED.clear()
    yield
    server_auth._machine_names.cache_clear()


# ── what real traffic sends still passes ─────────────────────────────────────


@pytest.mark.parametrize(
    "host",
    [
        # the Mac: CLI, Vite's proxy (changeOrigin), start_iris.sh, the health watch
        "localhost",
        "localhost:8003",
        "LOCALHOST.:8003",
        "127.0.0.1:8003",
        "[::1]:8003",
        # compose healthchecks and the VM's shared namespace
        "127.0.0.1:8080",
        # TestClient
        "testserver",
        # a sibling container in the local compose stack (IRIS_GOVERNOR_BASE_URL)
        "governor:8080",
        "iris-api:8003",
        "channel-gateway:8006",
        # this machine's own name, and its short form
        "owners-mac.local:8003",
        "owners-mac",
        # any IP literal: the tailnet address, a LAN address
        "100.101.102.103:8003",
        "192.168.1.20",
        "[fd7a:115c:a1e0::1]:8003",
    ],
)
def test_hosts_real_traffic_uses_reach_auth(host: str) -> None:
    app = _bearer_app()
    assert _call(app, "GET", "/healthz", host=host) == 200
    assert _call(app, "GET", "/echo", host=host) == 401
    assert _call(app, "GET", "/echo", host=host, token=SECRET) == 200


def test_the_public_url_host_and_its_short_name_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """A harness behind tailscale serve: the browser's Host is the tailnet name, and a
    client on the tailnet (a Vite proxy, say) calls the machine by its short name."""
    monkeypatch.setenv("IRIS_PUBLIC_URL", "https://iris-vm.tail0000.ts.net/")
    app = _bearer_app()
    assert _call(app, "GET", "/healthz", host="iris-vm.tail0000.ts.net") == 200
    assert _call(app, "GET", "/healthz", host="iris-vm:8003") == 200
    assert _call(app, "GET", "/healthz", host=ATTACKER) == 400


# ── any other name is refused, open routes included ──────────────────────────


def test_an_attacker_host_is_refused_on_an_open_route() -> None:
    assert _call(_bearer_app(), "GET", "/healthz", host=ATTACKER) == 400


def test_an_attacker_host_is_refused_even_with_a_valid_token() -> None:
    assert _call(_bearer_app(), "GET", "/echo", host=f"{ATTACKER}:8003", token=SECRET) == 400


@pytest.mark.parametrize("host", ["iris-vm.tail0000.ts.net", "evil.localhost.example", "iris.test"])
def test_names_that_are_not_configured_are_refused(host: str) -> None:
    assert not host_allowed(host)


def test_the_refusal_is_logged_with_the_host_only(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="iris_harness.server.auth")
    app = _bearer_app()
    _call(app, "GET", "/echo", host=ATTACKER, token=SECRET)
    _call(app, "GET", "/echo", host=ATTACKER, token=SECRET)

    lines = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(lines) == 1  # once per name, not once per request
    assert ATTACKER in lines[0]
    assert "IRIS_ALLOWED_HOSTS" in lines[0]
    assert SECRET not in lines[0]
    assert "/echo" not in lines[0]


# ── IRIS_ALLOWED_HOSTS ────────────────────────────────────────────────────────


def test_the_env_list_extends_the_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOWED_HOSTS_ENV, " iris.home.arpa:8443 , https://console.example/ ")
    app = _bearer_app()
    assert _call(app, "GET", "/healthz", host="iris.home.arpa") == 200
    assert _call(app, "GET", "/healthz", host="console.example:443") == 200
    assert _call(app, "GET", "/healthz", host=ATTACKER) == 400


def test_a_star_turns_the_check_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOWED_HOSTS_ENV, "*")
    assert _call(_bearer_app(), "GET", "/healthz", host=ATTACKER) == 200


def test_a_star_does_not_admit_a_malformed_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOWED_HOSTS_ENV, "*")
    assert _call(_bearer_app(), "GET", "/echo", host="127.0.0.1/healthz?") == 400


# ── the compose service names are the compose files' ─────────────────────────


def _services(compose_file: Path) -> set[str]:
    return set(yaml.safe_load(compose_file.read_text())["services"])


def test_the_compose_service_names_are_the_local_stacks_and_the_harness_servers() -> None:
    """docker-compose.yml's services, plus the channel gateway (a harness server the
    local stack does not run). A deployment naming its services otherwise lists them in
    IRIS_ALLOWED_HOSTS: the core knows no deployment's compose file."""
    assert _services(REPO / "docker-compose.yml") | {"channel-gateway"} == COMPOSE_SERVICE_HOSTS


# ── the real services ────────────────────────────────────────────────────────


@pytest.fixture
def iris_api(tmp_path: Path) -> Any:
    from iris_harness.server.iris_api.main import create_app

    return create_app(runtime=SimpleNamespace(data_dir=tmp_path), auto_start_runtime=False)


def test_iris_api_refuses_an_attacker_host_on_open_and_protected_routes(iris_api: Any) -> None:
    from iris_harness.server.iris_api.device_routes import PAIR_CLAIM_PATH

    assert _call(iris_api, "GET", "/healthz", host=ATTACKER) == 400
    assert _call(iris_api, "GET", "/readyz", host=ATTACKER) == 400
    assert _call(iris_api, "POST", PAIR_CLAIM_PATH, host=ATTACKER) == 400
    assert _call(iris_api, "GET", "/rag/documents", host=ATTACKER, token=SECRET) == 400
    # and the same routes still answer on loopback
    assert _call(iris_api, "GET", "/healthz", host="127.0.0.1:8003") == 200
    assert _call(iris_api, "GET", "/rag/documents", host="localhost:8003") == 401


def test_governor_and_evaluator_carry_the_guard() -> None:
    from iris_harness.server.evaluator.main import app as evaluator
    from iris_harness.server.governor.main import app as governor

    for app in (governor, evaluator):
        assert _call(app, "GET", "/healthz", host=ATTACKER) == 400
        assert _call(app, "GET", "/healthz", host="127.0.0.1:8080") == 200


def test_channel_gateway_refuses_an_attacker_host(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_harness.server.channel_gateway.main import create_app

    monkeypatch.setenv("IRIS_API_URL", "http://127.0.0.1:8003")
    app = create_app()
    with TestClient(app, base_url=f"http://{ATTACKER}") as client:
        assert client.get("/health").status_code == 400
        # websocket_connect always joins onto ws://testserver, so the Host is explicit.
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={SECRET}", headers={"host": ATTACKER}) as ws:
                ws.receive_text()
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def test_channel_gateway_ws_still_accepts_an_allowed_host(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_harness.server.channel_gateway.main import create_app

    monkeypatch.setenv("IRIS_API_URL", "http://127.0.0.1:8003")
    with TestClient(create_app()) as client:
        with client.websocket_connect(
            f"/ws?token={SECRET}", headers={"host": "127.0.0.1:8006"}
        ) as ws:
            ws.send_text(json.dumps({"type": "nonsense", "session_id": "s1"}))
            assert json.loads(ws.receive_text())["type"] == "error"
