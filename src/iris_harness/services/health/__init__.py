"""System Health — a harness capability (ADR-0069).

Builds a ``HealthSnapshot`` from service/credential/hardware ``HealthCheck``s and
exposes the red subset via ``alerts()``. The agent (``system_health`` tool), the
CLI, and the Web UI are interchangeable renderers over the same snapshot — no
health logic lives in any one channel.

Slice 1 ships services + hardware checks and the agent tool. Credentials, the
opt-in network probe, the ``health_tick`` heartbeat, and ``GET /health`` follow.
"""

from __future__ import annotations

from iris_harness.services.health.checks import build_snapshot, hardware_check, service_checks
from iris_harness.services.health.models import (
    CheckKind,
    HealthCheck,
    HealthSnapshot,
    HealthState,
    alerts,
)


def render_text(snapshot: HealthSnapshot) -> str:
    """Render a snapshot as a compact, agent-readable report.

    Used by the ``system_health`` ReAct tool so chat/CLI can answer "is anything
    broken?" without a UI. The Web UI renders the same snapshot graphically.
    """
    lines = [f"System health [{snapshot.worst().value}] — {snapshot.summary()}"]
    for check in snapshot.checks:
        loc = f" ({check.endpoint})" if check.endpoint else ""
        lines.append(f"- {check.target} [{check.state.value}] {check.detail}{loc}")

    active = alerts(snapshot)
    if active:
        lines.append("Needs attention:")
        for check in active:
            fix = f" — fix: {check.action}" if check.action else ""
            lines.append(f"- {check.target}: {check.detail}{fix}")
    else:
        lines.append("No active alerts.")
    return "\n".join(lines)


__all__ = [
    "CheckKind",
    "HealthCheck",
    "HealthSnapshot",
    "HealthState",
    "alerts",
    "build_snapshot",
    "hardware_check",
    "render_text",
    "service_checks",
]
