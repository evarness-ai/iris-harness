"""Plugin-contributed API routes (OSS plan M5.7, track A slice 4).

The owner's rule: every capability has an API. When a capability leaves the core for
a plugin, the plugin registers a router factory in ``iris_harness.runtime.api_routes`` and the
API service mounts whatever is registered — importing no plugin.
"""

from __future__ import annotations

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.runtime.api_routes import (
    clear_api_routers,
    register_api_router,
    registered_api_routers,
)


@pytest.fixture(autouse=True)
def _clean() -> None:
    clear_api_routers()
    yield  # type: ignore[misc]
    clear_api_routers()


def _router(path: str, answer: str) -> APIRouter:
    router = APIRouter()

    @router.get(path)
    def _get() -> dict[str, str]:
        return {"answer": answer}

    return router


def test_registration_is_keyed_so_a_second_runtime_replaces_not_stacks() -> None:
    register_api_router("x", lambda: _router("/x", "one"))
    register_api_router("x", lambda: _router("/x", "two"))
    assert list(registered_api_routers()) == ["x"]


def test_the_api_mounts_every_registered_router() -> None:
    from iris_harness.server.iris_api.main import create_app

    register_api_router("demo", lambda: _router("/demo-plugin-route", "hello"))
    client = TestClient(create_app(auto_start_runtime=False), headers=auth_headers())

    assert client.get("/demo-plugin-route").json() == {"answer": "hello"}


def test_a_factory_that_raises_is_skipped_not_fatal() -> None:
    from iris_harness.server.iris_api.main import create_app

    def _boom() -> APIRouter:
        raise RuntimeError("bad plugin routes")

    register_api_router("broken", _boom)
    register_api_router("fine", lambda: _router("/fine", "still here"))
    client = TestClient(create_app(auto_start_runtime=False), headers=auth_headers())

    assert client.get("/fine").json() == {"answer": "still here"}


def test_nothing_registered_means_no_reminders_route() -> None:
    """With the calendar plugin unmounted the reminder read API is honestly gone."""
    from iris_harness.server.iris_api.main import create_app

    client = TestClient(create_app(auto_start_runtime=False), headers=auth_headers())
    assert client.get("/api/v1/reminders").status_code == 404
    assert client.get("/reminders").status_code == 404  # the retired markdown API


# ── public callbacks (an OAuth redirect the route authenticates itself) ───────


def test_a_public_callback_must_be_an_exact_api_v1_path() -> None:
    from iris_harness.runtime.api_routes import is_public_callback, register_public_callback

    for bad in ("/memory/facts", "/api/v2/x", "/api/v1/x?y=1", "/api/v1/x#y", "api/v1/x"):
        with pytest.raises(ValueError, match="exact path under /api/v1/"):
            register_public_callback(bad)
    register_public_callback("/api/v1/demo/callback")
    assert is_public_callback("GET", "/api/v1/demo/callback")
    assert not is_public_callback("POST", "/api/v1/demo/callback")
    assert not is_public_callback("GET", "/api/v1/demo/callback/")
    clear_api_routers()
    assert not is_public_callback("GET", "/api/v1/demo/callback")


def test_a_declared_callback_answers_without_a_credential_and_nothing_else_does() -> None:
    from iris_harness.runtime.api_routes import register_public_callback
    from iris_harness.server.iris_api.main import create_app

    register_api_router("demo", lambda: _router("/api/v1/demo/callback", "back"))
    register_api_router("demo2", lambda: _router("/api/v1/demo/other", "closed"))
    client = TestClient(create_app(auto_start_runtime=False))
    assert client.get("/api/v1/demo/callback").status_code == 401  # not declared yet

    # Declared after the app is built -- as a plugin does, inside the lifespan.
    register_public_callback("/api/v1/demo/callback")
    assert client.get("/api/v1/demo/callback").json() == {"answer": "back"}
    assert client.get("/api/v1/demo/other").status_code == 401
    assert client.post("/api/v1/demo/callback").status_code == 401


def test_the_access_log_keeps_a_public_callbacks_path_and_drops_its_query() -> None:
    import logging

    from iris_harness.runtime.api_routes import is_public_callback, register_public_callback
    from iris_harness.server.auth import PublicCallbackQueryRedactor

    register_public_callback("/api/v1/demo/callback")
    redactor = PublicCallbackQueryRedactor(is_public_callback)

    def line(method: str, path: str) -> str:
        record = logging.LogRecord(
            "uvicorn.access",
            logging.INFO,
            __file__,
            1,
            '%s - "%s %s HTTP/%s" %d',
            ("127.0.0.1:5", method, path, "1.1", 303),
            None,
        )
        assert redactor.filter(record)
        return record.getMessage()

    assert line("GET", "/api/v1/demo/callback?state=S3CRET&code=4/C0DE") == (
        '127.0.0.1:5 - "GET /api/v1/demo/callback?<redacted> HTTP/1.1" 303'
    )
    # Anything else keeps its query: only a declared callback's query is a credential.
    assert "q=1" in line("GET", "/api/v1/demo/other?q=1")
    assert "state=x" in line("POST", "/api/v1/demo/callback?state=x")
