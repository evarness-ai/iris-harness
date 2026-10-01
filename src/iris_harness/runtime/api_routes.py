"""Plugin-contributed API routes — the registry the API service mounts from.

The owner's rule is that every capability has an API, never a UI-only surface. A
capability that leaves the core for a plugin has to keep its API, and the API service
(``src/iris_harness/server/iris_api``) must not import the plugin to get it. So the core keeps a
registry: a plugin registers a router *factory* under a key in its ``setup()``, and
``create_app`` mounts every registered router after its own routes.

Same shape as ``email.providers`` and ``health.service.register_check_provider``: a
keyed core registry, not a seventh registration kind. Keyed, so a process that builds a
second runtime replaces the factory instead of mounting the routes twice. A factory
runs at app build, with no arguments — a plugin whose routes need runtime state closes
over the services it was given at ``setup()``.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any

from iris_harness.foundation.process_state import track_globals

_lock = Lock()
_factories: dict[str, Callable[[], Any]] = {}
_public_callbacks: set[str] = set()

# A public callback may only live under the versioned API prefix. The core's own
# routes (``/memory``, ``/chat``, ``/health`` ...) sit outside it, so a plugin cannot
# open one of those by declaring it; it can only open a path it serves itself.
_PUBLIC_CALLBACK_PREFIX = "/api/v1/"


def register_api_router(key: str, factory: Callable[[], Any]) -> None:
    """Install (or replace) the router factory under ``key``.

    ``factory()`` returns a ``fastapi.APIRouter``; it is called once per app build.
    """
    with _lock:
        _factories[key] = factory


def registered_api_routers() -> dict[str, Callable[[], Any]]:
    with _lock:
        return dict(_factories)


def register_public_callback(path: str) -> None:
    """Let browsers reach ``GET path`` without a bearer token or device cookie.

    For a redirect back from a third party (an OAuth provider's consent page). The
    browser arrives from the provider's site, so a ``SameSite=Strict`` device cookie is
    not sent and a bearer header never is: the route cannot be authenticated the usual
    way. It is safe to open only because the route authenticates the request itself,
    with a one-time secret it issued to an authenticated caller earlier (an OAuth
    ``state``) — a plugin that declares a path takes on that duty.

    Exact path, ``GET`` only, and only under ``/api/v1/``. Registering is the plugin's
    declaration; the API service reads it per request, so a callback registered while
    the runtime is built (after the app) is honoured.
    """
    if not path.startswith(_PUBLIC_CALLBACK_PREFIX) or "?" in path or "#" in path:
        raise ValueError(f"a public callback must be an exact path under /api/v1/: {path!r}")
    with _lock:
        _public_callbacks.add(path)


def is_public_callback(method: str, path: str) -> bool:
    """Whether ``method path`` is a declared public callback (see above)."""
    if method.upper() != "GET":
        return False
    with _lock:
        return path in _public_callbacks


def clear_api_routers() -> None:
    """Drop every factory and public callback (tests)."""
    with _lock:
        _factories.clear()
        _public_callbacks.clear()


__all__ = [
    "clear_api_routers",
    "is_public_callback",
    "register_api_router",
    "register_public_callback",
    "registered_api_routers",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_factories", "_public_callbacks")
