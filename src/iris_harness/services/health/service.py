"""Snapshot caching + runtime flags for System Health (ADR-0069 slice 3).

The ``health_tick`` heartbeat ``refresh()``es the cached ``HealthSnapshot`` in the
background; the ``system_health`` tool, ``GET /health``, and the CLI read it via
``current_snapshot()`` (building fresh on a cold cache). One process holds the
runtime, the API, and the ReAct loop, so a module-level cache is shared by all
three — no need to thread state through the runtime object.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextvars import ContextVar
from threading import Lock

from iris_harness.foundation.process_state import track_globals
from iris_harness.services.health.checks import build_snapshot
from iris_harness.services.health.models import HealthCheck, HealthSnapshot
from iris_harness.services.heartbeat.diagnostics import HeartbeatDiagnostic

logger = logging.getLogger(__name__)

_lock = Lock()
_cached: HealthSnapshot | None = None

# Extra check providers keyed by name (e.g. "plugins" → the plugin registry's
# per-plugin verdicts). The runtime registers its provider at build time;
# ``refresh`` folds the checks into every snapshot. Keyed, not appended, so a
# process that builds a second runtime (the playground does) replaces the
# provider instead of reporting every plugin twice. Module-level for the same
# reason the cache is: one process, shared.
_check_providers: dict[str, Callable[[], list[HealthCheck]]] = {}

_TRUTHY = {"1", "true", "yes", "on"}


def register_check_provider(key: str, provider: Callable[[], list[HealthCheck]]) -> None:
    """Install (or replace) the provider under ``key``; its checks join every snapshot."""
    with _lock:
        _check_providers[key] = provider


def clear_check_providers() -> None:
    """Drop all extra providers (tests)."""
    with _lock:
        _check_providers.clear()


def _extra_checks() -> list[HealthCheck]:
    with _lock:
        providers = list(_check_providers.values())
    checks: list[HealthCheck] = []
    for provider in providers:
        try:
            checks.extend(provider())
        except Exception:  # a broken provider must not break health itself
            logger.warning("health check provider %r failed", provider, exc_info=True)
    return checks


def _flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in _TRUTHY


def health_enabled() -> bool:
    """Master switch for the subsystem (default on; ADR-0069 §4)."""
    return _flag("IRIS_HEALTH_ENABLED", default=True)


# The probe choice of the refresh in progress. Plugin check providers take no
# arguments and read ``net_probe_enabled()``; without this they would answer from the
# env flag alone, so ``refresh(net_probe=True)`` — the health watch's diagnosis and
# ``GET /health/connectors?live=true`` — silently skipped every plugin-owned
# credential (Gmail, Calendar, Drive). Found by simulating a revoked Gmail token.
_refresh_net_probe: ContextVar[bool | None] = ContextVar("health_refresh_net_probe", default=None)


def net_probe_enabled() -> bool:
    """Whether credential checks should live-probe now.

    Inside a ``refresh`` this is that refresh's choice; outside one it is the opt-in
    ``IRIS_HEALTH_NET_PROBE`` flag (default off; the only new egress).
    """
    requested = _refresh_net_probe.get()
    if requested is not None:
        return requested
    return _flag("IRIS_HEALTH_NET_PROBE", default=False)


def store_snapshot(snapshot: HealthSnapshot) -> None:
    """Publish a snapshot to the shared cache (used by the heartbeat + tests)."""
    global _cached
    with _lock:
        _cached = snapshot


def cached_snapshot() -> HealthSnapshot | None:
    """The last published snapshot, or None if none has been taken yet."""
    with _lock:
        return _cached


def refresh(
    *,
    net_probe: bool | None = None,
    heartbeat_diagnostics: list[HeartbeatDiagnostic] | None = None,
) -> HealthSnapshot:
    """Build a fresh snapshot and publish it to the cache."""
    use_net = net_probe_enabled() if net_probe is None else net_probe
    token = _refresh_net_probe.set(use_net)
    try:
        snapshot = build_snapshot(net_probe=use_net, heartbeat_diagnostics=heartbeat_diagnostics)
        extra = _extra_checks()
    finally:
        _refresh_net_probe.reset(token)
    if extra:
        snapshot = HealthSnapshot(
            checks=tuple(snapshot.checks) + tuple(extra), sampled_at=snapshot.sampled_at
        )
    store_snapshot(snapshot)
    return snapshot


def current_snapshot(
    *,
    net_probe: bool | None = None,
    heartbeat_diagnostics: list[HeartbeatDiagnostic] | None = None,
) -> HealthSnapshot:
    """Return the cached snapshot, building (and caching) one on a cold cache."""
    cached = cached_snapshot()
    if cached is not None:
        return cached
    return refresh(net_probe=net_probe, heartbeat_diagnostics=heartbeat_diagnostics)


__all__ = [
    "cached_snapshot",
    "clear_check_providers",
    "current_snapshot",
    "register_check_provider",
    "health_enabled",
    "net_probe_enabled",
    "refresh",
    "store_snapshot",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_cached", "_check_providers")
