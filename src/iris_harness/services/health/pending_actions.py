"""Health → Action Center adapter (ADR-0073 §2b, slice 4).

Synthesizes pending actions from the live Health snapshot's red, actionable
checks (ADR-0069). This is a **read-time projection** — nothing is persisted, so
a synthesized action exists exactly as long as its red check does and disappears
when the next probe recovers. The Action Center unions these with the persisted
finance/agent action-tasks; the Health screen remains the diagnostic view.

Health remediations are CLI commands (e.g. ``iris auth gmail login``), so they map
to display-only ``copy_command`` actions — the user runs them locally and there is
no server-side invoke lifecycle.
"""

from __future__ import annotations

import logging
from typing import Any

from iris_harness.services.health.models import HealthCheck, HealthSnapshot, HealthState
from iris_harness.services.tasks import TaskAction
from iris_harness.services.tasks.pending_actions import PendingAction

logger = logging.getLogger(__name__)


def health_alert_dims(check: HealthCheck) -> dict[str, str]:
    """Surface-feedback key for a health alert: the probe identity, not its
    transient detail. So "this alert isn't useful" suppresses that probe."""
    return {"kind": check.kind.value, "target": check.target}


def _suppressed(feedback_store: Any, check: HealthCheck) -> bool:
    if feedback_store is None:
        return False
    try:
        return bool(
            feedback_store.should_suppress("system", "health_alert", health_alert_dims(check))
        )
    except Exception:  # suppression must never hide a real outage by erroring loudly
        logger.debug("health suppression check failed for %s", check.target, exc_info=True)
        return False


def health_pending_actions(
    snapshot: HealthSnapshot, *, feedback_store: Any = None
) -> list[PendingAction]:
    """Red, actionable Health checks as synthesized pending actions.

    ``feedback_store`` (the surface-feedback spine) drops alerts the user marked
    "not useful"; when omitted a default ``learning.db`` store is used so the
    Action Center honors feedback by default (issue 0028).
    """
    if feedback_store is None:
        from iris_harness.services.learning.suppression import SurfaceFeedbackStore

        feedback_store = SurfaceFeedbackStore()
        feedback_store.ensure_schema()
    out: list[PendingAction] = []
    for c in snapshot.checks:
        if c.state is not HealthState.RED or not c.action:
            continue
        if _suppressed(feedback_store, c):
            continue
        out.append(
            PendingAction(
                id=f"health:{c.kind.value}:{c.target}",
                origin="health",
                source_kind="system-health",
                title=f"{c.target}: {c.detail}",
                description=c.detail,
                action=TaskAction(
                    kind="copy_command",
                    label="Run remediation command",
                    command=c.action,
                ),
                created_at=None,
            )
        )
    return out
