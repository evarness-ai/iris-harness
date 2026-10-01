"""Surface-feedback suppression of health alerts (issue 0028)."""

from __future__ import annotations

from pathlib import Path

from iris_harness.services.health.models import CheckKind, HealthCheck, HealthSnapshot, HealthState
from iris_harness.services.health.pending_actions import health_alert_dims, health_pending_actions
from iris_harness.services.learning.suppression import NOT_USEFUL, SurfaceFeedbackStore


def _snapshot() -> HealthSnapshot:
    return HealthSnapshot(
        checks=(
            HealthCheck(
                target="gmail",
                kind=CheckKind.CREDENTIAL,
                state=HealthState.RED,
                detail="token revoked",
                action="iris auth gmail login",
            ),
        ),
        sampled_at="2026-06-27T00:00:00+00:00",
    )


def test_health_alert_suppressed_by_feedback(tmp_path: Path) -> None:
    feedback = SurfaceFeedbackStore(db_path=tmp_path / "learning.db")
    feedback.ensure_schema()
    snap = _snapshot()

    # Surfaced before any feedback.
    assert len(health_pending_actions(snap, feedback_store=feedback)) == 1

    # User marks this probe's alert not useful → suppressed.
    feedback.record("system", "health_alert", health_alert_dims(snap.checks[0]), NOT_USEFUL)
    assert health_pending_actions(snap, feedback_store=feedback) == []
