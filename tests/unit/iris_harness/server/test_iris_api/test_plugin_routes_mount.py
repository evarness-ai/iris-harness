"""Plugin routes registered while the lifespan builds the runtime are mounted.

Regression (found in the PR 3a demo, 2026-09-25): ``create_app`` mounted plugin
routers once, at app build — but under uvicorn the runtime (and so every plugin's
``setup()``, where routers are registered) is built later, in the lifespan. Every
plugin route (reminders, finance senders) answered 404 on a real server.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.runtime import api_routes
from iris_harness.server.iris_api import main


def _route_paths(routes: Any) -> Any:
    """Every route path, looking inside included routers.

    FastAPI 0.141 stopped flattening ``include_router`` into ``app.routes``: each include
    is one ``_IncludedRouter`` entry holding the router as ``original_router``. Older
    releases listed the routes themselves; this counts the same way on both.
    """
    for route in routes:
        inner = getattr(route, "original_router", None)
        if inner is not None:
            yield from _route_paths(inner.routes)
        else:
            yield getattr(route, "path", "")


def _router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/v1/plugin-probe")
    def probe() -> dict[str, Any]:
        return {"ok": True}

    return router


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    api_routes.clear_api_routers()
    yield
    api_routes.clear_api_routers()


def test_a_router_registered_during_the_runtime_build_is_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build_runtime() -> Any:
        # What a plugin's setup() does while the runtime is built.
        api_routes.register_api_router("probe", _router)
        return SimpleNamespace(tracer=None)

    monkeypatch.setattr(main, "build_runtime", build_runtime)
    app = main.create_app(auto_start_runtime=False)

    with TestClient(app, headers=auth_headers()) as client:
        assert client.get("/api/v1/plugin-probe").json() == {"ok": True}


def test_each_plugin_router_is_mounted_once() -> None:
    api_routes.register_api_router("probe", _router)
    app = main.create_app(runtime=SimpleNamespace(tracer=None), auto_start_runtime=False)  # type: ignore[arg-type]
    main._mount_plugin_routes(app)  # the lifespan's second pass

    assert list(_route_paths(app.routes)).count("/api/v1/plugin-probe") == 1
