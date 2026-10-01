"""A Host header cannot move a request onto an exempt path (PYSEC-2026-161).

Starlette 1.0.0 rebuilds ``request.url`` from the Host header, so a request for a
protected route sent with ``Host: 127.0.0.1/healthz?`` had ``request.url.path ==
"/healthz"`` while the router still served the protected route. The bearer check and
the write guard both read ``request.url.path``: an unauthenticated caller could read
any route, and write through any route by borrowing the pairing-claim path, which is
exempt from both. Two layers now stand in the way, and each is tested on its own:

- every security decision reads ``routed_path`` (``scope["path"]``), what routing uses;
- a Host header that is not a plain ``host[:port]`` is refused with 400 first.

Requests are driven as raw ASGI scopes: an HTTP client may refuse or normalise the
malformed header, and the attacker's client would not.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI

from iris_harness.server import auth as server_auth
from iris_harness.server.auth import install_bearer_auth, valid_host_header

SECRET = "host-header-test-secret"
SPOOF_PROBE = "127.0.0.1/healthz?"
SPOOF_PAIR_CLAIM = "127.0.0.1/api/v1/devices/pair/claim?"


def _call(app: Any, method: str, path: str, *, host: str | None, token: str | None = None) -> int:
    headers = [(b"content-type", b"application/json")]
    if host is not None:
        headers.append((b"host", host.encode()))
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
    body = b'{"title": "planted"}' if method == "POST" else b""
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
def _secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_AUTH_SECRET", SECRET)


@pytest.fixture
def no_host_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept any Host, so a test sees the second layer (``routed_path``) alone."""
    monkeypatch.setattr(server_auth, "_VALID_HOST", re.compile(r".*", re.DOTALL))
    monkeypatch.setattr(server_auth, "host_allowed", lambda _value: True)


# ── the Host guard ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "host",
    [
        None,
        "localhost",
        "localhost:8003",
        "127.0.0.1:8003",
        "[::1]:8003",
        "iris-vm.example.ts.net",
        "iris_api:8003",
        "testserver",
    ],
)
def test_plain_hosts_are_accepted(host: str | None) -> None:
    assert valid_host_header(host)


@pytest.mark.parametrize(
    "host",
    [SPOOF_PROBE, SPOOF_PAIR_CLAIM, "a b", "user@host", "host#frag", "host?q", "h:port", ""],
)
def test_hosts_that_can_carry_a_path_are_refused(host: str) -> None:
    assert not valid_host_header(host)


def test_a_spoofed_host_is_refused_before_auth() -> None:
    assert _call(_bearer_app(), "GET", "/echo", host=SPOOF_PROBE) == 400


def test_ordinary_requests_are_unaffected() -> None:
    app = _bearer_app()
    assert _call(app, "GET", "/echo", host="127.0.0.1:8003") == 401
    assert _call(app, "GET", "/echo", host="127.0.0.1:8003", token=SECRET) == 200
    assert _call(app, "GET", "/healthz", host="127.0.0.1:8003") == 200


# ── the bearer check reads the routed path ───────────────────────────────────


@pytest.mark.usefixtures("no_host_guard")
def test_auth_decides_on_the_routed_path_not_the_host() -> None:
    """The regression itself: on starlette 1.0.0 this returned 200 with no token."""
    assert _call(_bearer_app(), "GET", "/echo", host=SPOOF_PROBE) == 401


# ── the real API service: reads and writes ───────────────────────────────────


@pytest.fixture
def iris_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from iris_harness.server.iris_api.main import create_app

    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    return create_app(runtime=SimpleNamespace(data_dir=tmp_path), auto_start_runtime=False)


def test_iris_api_refuses_a_spoofed_host(iris_api: Any) -> None:
    assert _call(iris_api, "GET", "/rag/documents", host=SPOOF_PROBE) == 400
    assert _call(iris_api, "POST", "/tasks", host=SPOOF_PAIR_CLAIM) == 400


@pytest.mark.usefixtures("no_host_guard")
def test_iris_api_read_is_not_unlocked_by_the_host(iris_api: Any) -> None:
    assert _call(iris_api, "GET", "/rag/documents", host=SPOOF_PROBE) == 401


@pytest.mark.usefixtures("no_host_guard")
def test_iris_api_write_cannot_borrow_the_pairing_claim_path(iris_api: Any, tmp_path: Path) -> None:
    """Exempt from auth AND ungated: on 1.0.0 this created a task with no credential."""
    from iris_harness.services.tasks import TaskStore

    assert _call(iris_api, "POST", "/tasks", host=SPOOF_PAIR_CLAIM) == 401
    store = TaskStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    assert store.list() == []


@pytest.mark.usefixtures("no_host_guard")
def test_the_write_gate_reads_the_routed_path_too(iris_api: Any, tmp_path: Path) -> None:
    """A caller who may NOT write (writes off) cannot pass the gate as ``/chat``."""
    from iris_harness.services.tasks import TaskStore

    status = _call(iris_api, "POST", "/tasks", host="127.0.0.1/chat?", token=SECRET)

    assert status == 403
    store = TaskStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    assert store.list() == []
