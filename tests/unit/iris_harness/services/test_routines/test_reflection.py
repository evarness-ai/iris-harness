"""Tests for the routines reflection analyzer (routines learning loop)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from iris_harness.services.digest.settings import DigestSettings
from iris_harness.services.learning.store import LearningMetricsStore
from iris_harness.services.routines.models import RoutineApprovalStatus, create_routine_spec
from iris_harness.services.routines.reflection import reflect_on_routines
from iris_harness.services.routines.seeded import build_morning_digest_spec


def _store(tmp_path: Path) -> LearningMetricsStore:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    return store


def _run(store: LearningMetricsStore, rid: str, *, success: bool) -> None:
    store.record_signal(
        source="routine",
        metric_name="routine_run",
        value=1.0 if success else 0.0,
        success=success,
        metadata={"routine_id": rid},
    )


def _spec(*, status: RoutineApprovalStatus = RoutineApprovalStatus.APPROVED, last_run=None):  # type: ignore[no-untyped-def]
    spec = create_routine_spec(
        title="Morning Brief",
        goal="daily brief",
        schedule="interval:3600",
        template="skill_brief",
        approval_status=status,
    )
    return spec.model_copy(update={"last_run_at": last_run})


def test_failing_routine_is_flagged(tmp_path: Path) -> None:
    store = _store(tmp_path)
    spec = _spec(last_run=datetime.now(UTC))
    for _ in range(3):
        _run(store, spec.id, success=False)
    for _ in range(2):
        _run(store, spec.id, success=True)  # 3/5 = 60% failure over 5 runs
    refs = reflect_on_routines(store, [spec], min_runs=5, failure_rate_threshold=0.5)
    assert len(refs) == 1
    assert refs[0].kind == "failing"
    assert refs[0].routine_id == spec.id


def test_clean_routine_not_flagged(tmp_path: Path) -> None:
    store = _store(tmp_path)
    spec = _spec(last_run=datetime.now(UTC))
    for _ in range(6):
        _run(store, spec.id, success=True)
    refs = reflect_on_routines(store, [spec], min_runs=5, failure_rate_threshold=0.5)
    assert refs == []


def test_below_min_runs_not_failing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    spec = _spec(last_run=datetime.now(UTC))
    for _ in range(4):
        _run(store, spec.id, success=False)  # all fail but only 4 < min 5
    refs = reflect_on_routines(store, [spec], min_runs=5, failure_rate_threshold=0.5)
    assert refs == []


def test_stale_routine_is_flagged(tmp_path: Path) -> None:
    store = _store(tmp_path)
    spec = _spec(last_run=datetime.now(UTC) - timedelta(days=40))
    refs = reflect_on_routines(store, [spec], stale_days=30)
    assert len(refs) == 1
    assert refs[0].kind == "stale"


def test_never_run_approved_routine_is_stale(tmp_path: Path) -> None:
    store = _store(tmp_path)
    spec = _spec(last_run=None)
    refs = reflect_on_routines(store, [spec], stale_days=30)
    assert len(refs) == 1 and refs[0].kind == "stale"


def test_seeded_digest_is_never_proposed_for_retirement(tmp_path: Path) -> None:
    """ADR-0122 §3: the seeded morning-digest is never auto-retired — not even proposed."""
    store = _store(tmp_path)
    digest = build_morning_digest_spec(DigestSettings(), ZoneInfo("America/Chicago"))
    old = digest.model_copy(update={"last_run_at": datetime.now(UTC) - timedelta(days=40)})
    assert reflect_on_routines(store, [digest, old], stale_days=30) == []


def test_draft_routine_not_flagged_stale(tmp_path: Path) -> None:
    store = _store(tmp_path)
    spec = _spec(status=RoutineApprovalStatus.DRAFT, last_run=None)  # not executable
    assert reflect_on_routines(store, [spec], stale_days=30) == []
