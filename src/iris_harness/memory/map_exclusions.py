"""Names a plugin says the memory Map must never draw (ADR-0119, cleanup plan decision 9).

A conversation summary's "Referenced:" line lists what the chat mentioned, and every
"check my email" chat mentions the senders it listed. So a shop that mails the owner
every day recurs in every such summary, and the Map drew it as if it were someone the
owner knows. The core cannot tell a shop from a friend — it knows no email, and no
vocabulary (ADR-0115 decision 2). The plugin that owns the mail can: it has already
classified every message.

So the core keeps a registry, the same shape as ``runtime.api_routes`` and
``health.service.register_check_provider``: a plugin installs a provider under a key
at ``setup()``, and the Map asks every provider for names when it draws a summary's
mentions. A provider returns raw names; the Map folds them the way it folds a mention,
so "Walgreens Co." and "Walgreens" are one name. Keyed, so a second runtime in the same
process replaces the provider instead of adding it twice.

What an exclusion does NOT do: hide a memory entity. A fact the owner confirmed is drawn
by the statements pass whatever any plugin says; an exclusion only stops a summary
mention from adding a node or an edge of its own.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from threading import Lock

from iris_harness.foundation.process_state import track_globals

logger = logging.getLogger(__name__)

_lock = Lock()
_providers: dict[str, Callable[[], Iterable[str]]] = {}


def register_map_exclusions(key: str, provider: Callable[[], Iterable[str]]) -> None:
    """Install (or replace) the provider under ``key``.

    ``provider()`` returns the names to leave off the Map. It is called each time the
    Map is drawn, so a provider that reads a store caches its own answer.
    """
    with _lock:
        _providers[key] = provider


def clear_map_exclusions() -> None:
    """Drop every provider (tests)."""
    with _lock:
        _providers.clear()


def excluded_names(fold: Callable[[str], str]) -> frozenset[str]:
    """Every provider's names, folded with ``fold`` and lower-cased.

    A provider that raises contributes nothing: a broken plugin must not take the Map
    down, and drawing a shop is a smaller harm than drawing nothing.
    """
    with _lock:
        providers = list(_providers.items())
    names: set[str] = set()
    for key, provider in providers:
        try:
            raw = list(provider())
        except Exception:  # plugin code
            logger.warning("map exclusion provider %r failed", key, exc_info=True)
            continue
        for name in raw:
            folded = fold(str(name)).strip().lower()
            if folded:
                names.add(folded)
    return frozenset(names)


__all__ = ["clear_map_exclusions", "excluded_names", "register_map_exclusions"]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_providers")
