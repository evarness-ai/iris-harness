"""Tests for MissionEngine execution and crash recovery."""

from __future__ import annotations

from pathlib import Path

from iris_harness.services.missions import (
    Mission,
    MissionEngine,
    MissionStatus,
    MissionStep,
    MissionStore,
    StepStatus,
)


def _engine(tmp_path: Path, **handlers: object) -> MissionEngine:
    store = MissionStore(tmp_path / "missions.db")
    return MissionEngine(store, handlers=dict(handlers))  # type: ignore[arg-type]


def test_run_completes_all_steps_in_order(tmp_path: Path) -> None:
    executed: list[str] = []

    def handler(mission: Mission, step: MissionStep) -> StepStatus:
        executed.append(step.name)
        step.output = f"done:{step.name}"
        return StepStatus.COMPLETED

    engine = _engine(tmp_path, h=handler)
    mission = engine.create("m", "h", ["a", "b", "c"])
    result = engine.run(mission)

    assert executed == ["a", "b", "c"]
    assert result.status is MissionStatus.COMPLETED
    assert result.cursor == 3
    assert all(s.status is StepStatus.COMPLETED for s in result.steps)
    assert all(s.output.startswith("done:") for s in result.steps)


def test_run_records_failure_when_handler_raises(tmp_path: Path) -> None:
    def handler(mission: Mission, step: MissionStep) -> StepStatus:
        if step.name == "boom":
            raise RuntimeError("crashed")
        return StepStatus.COMPLETED

    engine = _engine(tmp_path, h=handler)
    mission = engine.create("m", "h", ["ok", "boom", "never"])
    result = engine.run(mission)

    assert result.status is MissionStatus.FAILED
    assert result.steps[0].status is StepStatus.COMPLETED
    assert result.steps[1].status is StepStatus.FAILED
    assert "crashed" in result.steps[1].error
    assert result.steps[2].status is StepStatus.PENDING


def test_run_marks_failed_when_handler_returns_failed(tmp_path: Path) -> None:
    def handler(mission: Mission, step: MissionStep) -> StepStatus:
        return StepStatus.FAILED

    engine = _engine(tmp_path, h=handler)
    mission = engine.create("m", "h", ["only"])
    result = engine.run(mission)
    assert result.status is MissionStatus.FAILED


def test_run_marks_failed_when_handler_unknown(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    mission = engine.create("m", "missing", ["a"])
    result = engine.run(mission)
    assert result.status is MissionStatus.FAILED
    assert result.metadata["errors"] == ["no handler registered for 'missing'"]


def test_resume_pending_continues_from_cursor(tmp_path: Path) -> None:
    store = MissionStore(tmp_path / "missions.db")
    # Simulate a crash: step 0 completed, cursor at 1, status RUNNING
    mission = Mission(
        name="m",
        handler="h",
        steps=[
            MissionStep(name="a", status=StepStatus.COMPLETED),
            MissionStep(name="b"),
            MissionStep(name="c"),
        ],
        status=MissionStatus.RUNNING,
        cursor=1,
    )
    store.save(mission)

    executed: list[str] = []

    def handler(m: Mission, step: MissionStep) -> StepStatus:
        executed.append(step.name)
        return StepStatus.COMPLETED

    engine = MissionEngine(store, handlers={"h": handler})
    resumed = engine.resume_pending()

    assert len(resumed) == 1
    assert executed == ["b", "c"]
    assert resumed[0].status is MissionStatus.COMPLETED


def test_cancel_marks_active_mission_cancelled(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    mission = engine.create("m", "h", ["a"])
    cancelled = engine.cancel(mission.id)
    assert cancelled is not None
    assert cancelled.status is MissionStatus.CANCELLED


def test_cancel_does_not_overwrite_completed(tmp_path: Path) -> None:
    def handler(m: Mission, step: MissionStep) -> StepStatus:
        return StepStatus.COMPLETED

    engine = _engine(tmp_path, h=handler)
    mission = engine.create("m", "h", ["a"])
    engine.run(mission)
    cancelled = engine.cancel(mission.id)
    assert cancelled is not None
    assert cancelled.status is MissionStatus.COMPLETED


def test_run_skips_already_completed_steps(tmp_path: Path) -> None:
    """If steps before the cursor are already done, they shouldn't be re-run."""
    executed: list[str] = []

    def handler(m: Mission, step: MissionStep) -> StepStatus:
        executed.append(step.name)
        return StepStatus.COMPLETED

    engine = _engine(tmp_path, h=handler)
    mission = engine.create("m", "h", ["a", "b"])
    mission.steps[0].status = StepStatus.COMPLETED
    mission.cursor = 0  # cursor not yet advanced; engine should skip past completed step
    engine.store.save(mission)

    engine.run(mission)
    assert executed == ["b"]
