"""The routines reflection tick: propose via HITL queue, never auto-apply."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from iris_harness.runtime.self_learning_loop import SelfLearningLoop
from iris_harness.services.learning.signals import LearningSignalCollector
from iris_harness.services.learning.store import LearningMetricsStore
from iris_harness.services.routines.models import (
    RoutineApprovalRequestStatus,
    RoutineApprovalStatus,
    create_routine_spec,
)
from iris_harness.services.routines.store import RoutineStore


def _runtime(tmp_path: Path) -> Any:
    learning_store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    learning_store.ensure_schema()
    routine_store = RoutineStore(tmp_path / "routines.db")
    return SimpleNamespace(
        learning_store=learning_store,
        routine_store=routine_store,
        signal_collector=LearningSignalCollector(learning_store),
    )


def _failing_routine(rt: Any) -> str:
    spec = create_routine_spec(
        title="Flaky Brief",
        goal="g",
        schedule="interval:3600",
        template="skill_brief",
        approval_status=RoutineApprovalStatus.APPROVED,
    ).model_copy(update={"last_run_at": datetime.now(UTC)})
    rt.routine_store.save(spec)
    for _ in range(5):
        rt.learning_store.record_signal(
            source="routine",
            metric_name="routine_run",
            value=0.0,
            success=False,
            metadata={"routine_id": spec.id},
        )
    return spec.id


def test_disabled_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("IRIS_ROUTINE_REFLECTION", raising=False)
    rt = _runtime(tmp_path)
    _failing_routine(rt)
    assert SelfLearningLoop(rt)._reflect_on_routines() == 0
    assert rt.routine_store.list_approval_requests() == []


def test_creates_proposal_when_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("IRIS_ROUTINE_REFLECTION", "1")
    rt = _runtime(tmp_path)
    rid = _failing_routine(rt)

    created = SelfLearningLoop(rt)._reflect_on_routines()
    assert created == 1
    requests = rt.routine_store.list_approval_requests(status=RoutineApprovalRequestStatus.PENDING)
    assert len(requests) == 1
    assert requests[0].routine_id == rid
    assert requests[0].metadata.get("source") == "reflection"
    # never mutated the routine itself
    assert rt.routine_store.load(rid).approval_status == RoutineApprovalStatus.APPROVED
    # the proposal is recorded as a learning signal
    assert rt.learning_store.recent_signals(metric_name="routine_reflection")


def test_dedups_existing_proposal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("IRIS_ROUTINE_REFLECTION", "1")
    rt = _runtime(tmp_path)
    _failing_routine(rt)
    assert SelfLearningLoop(rt)._reflect_on_routines() == 1
    # second pass: a pending reflection request already exists -> no duplicate
    assert SelfLearningLoop(rt)._reflect_on_routines() == 0
    assert (
        len(rt.routine_store.list_approval_requests(status=RoutineApprovalRequestStatus.PENDING))
        == 1
    )
