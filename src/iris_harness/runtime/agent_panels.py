"""Agent "key stores" panels — the registry the agent dashboard reads from (ADR-0074).

``GET /agents/{name}`` shows a panel of what an agent keeps (finance: accounts and a
statement inventory). What goes in it is the owning plugin's knowledge, and the API
service must not import a plugin to get it, so a plugin registers a panel builder
under its agent's name in ``setup()`` and the dashboard calls it. An agent with no
panel shows none.

Same shape as :mod:`iris_harness.runtime.api_routes`: a keyed core registry, so a
process that builds a second runtime replaces the builder instead of adding one.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any

from iris_harness.foundation.process_state import track_globals

_lock = Lock()
_builders: dict[str, Callable[[], dict[str, Any]]] = {}


def register_agent_panel(agent: str, build: Callable[[], dict[str, Any]]) -> None:
    """Install (or replace) the panel builder for ``agent``.

    ``build()`` runs per dashboard request and returns the panel's JSON body; a
    plugin closes over the services it was given at ``setup()``.
    """
    with _lock:
        _builders[agent] = build


def agent_panel(agent: str) -> dict[str, Any] | None:
    """The panel for ``agent``, or ``None`` when no plugin registered one."""
    with _lock:
        build = _builders.get(agent)
    return build() if build is not None else None


def clear_agent_panels() -> None:
    """Drop every builder (tests)."""
    with _lock:
        _builders.clear()


__all__ = ["agent_panel", "clear_agent_panels", "register_agent_panel"]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_builders")
