"""``health_tick`` heartbeat handler (ADR-0069 slice 3).

Refreshes the cached ``HealthSnapshot`` in the background so the ``system_health``
tool, ``GET /health``, and the Web UI banner read a recent snapshot without each
re-probing. Local + zero-egress by default; honors the ``IRIS_HEALTH_ENABLED``
kill-switch and the opt-in ``IRIS_HEALTH_NET_PROBE`` (slice 5). After each refresh
the health watch (ADR-0116), when one is installed, repairs and notifies.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from iris_harness.services.health.models import HealthSnapshot
from iris_harness.services.heartbeat.diagnostics import HeartbeatDiagnostic
from iris_harness.services.heartbeat.models import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatStatus,
)
from iris_harness.services.heartbeat.scheduler import HeartbeatHandler

logger = logging.getLogger(__name__)


def build_health_tick_handler(
    *,
    heartbeat_diagnostics_provider: Callable[[], list[HeartbeatDiagnostic]] | None = None,
) -> HeartbeatHandler:
    """Return the heartbeat handler that refreshes the health snapshot cache."""

    def handler(definition: HeartbeatDefinition) -> HeartbeatRun:
        from iris_harness.services.health.service import health_enabled, refresh

        if not health_enabled():
            return HeartbeatRun(
                name=definition.name,
                status=HeartbeatStatus.SKIPPED,
                output="health disabled (IRIS_HEALTH_ENABLED=0)",
            )
        try:
            diagnostics = (
                heartbeat_diagnostics_provider() if heartbeat_diagnostics_provider else None
            )
            snapshot = (
                refresh(heartbeat_diagnostics=diagnostics) if diagnostics is not None else refresh()
            )
        except Exception as exc:  # noqa: BLE001 — a probe failure must not kill the tick
            return HeartbeatRun(
                name=definition.name,
                status=HeartbeatStatus.FAILED,
                error=str(exc),
            )
        output = f"{snapshot.worst().value}: {snapshot.summary()}"
        watch_events = _watch(snapshot)
        if watch_events:
            output += " | watch: " + "; ".join(watch_events)
        return HeartbeatRun(name=definition.name, status=HeartbeatStatus.SUCCESS, output=output)

    return handler


def _watch(snapshot: HealthSnapshot) -> list[str]:
    """Hand the fresh snapshot to the installed health watch (ADR-0116), if any."""
    from iris_harness.services.health.watch import current_watcher

    watcher = current_watcher()
    if watcher is None:
        return []
    try:
        return watcher.observe(snapshot)
    except Exception:  # the watch must never fail the refresh
        logger.warning("health watch pass failed", exc_info=True)
        return []


__all__ = ["build_health_tick_handler"]
