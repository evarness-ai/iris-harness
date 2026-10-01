"""Unit tests for the learning retriever module."""

from __future__ import annotations

from pathlib import Path

import pytest

# The retriever reads rows the coding agent writes; without the writer there is
# nothing to test (the coding agent is not part of the harness release).
pytest.importorskip("iris_code")

from iris_code.models import (
    CodingTask,
    PersonaName,
    PersonaRewardSignal,
    ProjectScope,
    RewardMetricName,
    RewardSummary,
    TaskStatus,
    TaskType,
    TrackerState,
)
from iris_code.task_db import (
    initialize_coding_task_db,
    persist_coding_task_snapshot,
)
from iris_harness.services.learning.retriever import (
    PersonaLearningSignal,
    compute_persona_retry_summary,
    load_persona_learning_signals,
)


def _make_task(task_id: str = "task-1") -> CodingTask:
    return CodingTask(
        id=task_id,
        description="Test task",
        source="user",
        task_type=TaskType.FEATURE,
        scope=ProjectScope.PLATFORM,
        status=TaskStatus.TESTING,
        tracker_state=TrackerState(
            scope=ProjectScope.PLATFORM,
            current_phase=TaskStatus.TESTING,
            current_persona="tester",
            next_action="run tests",
            verification_state="pending",
        ),
    )


def _make_reward_summary(
    task_id: str,
    *,
    developer_score: float = 1.0,
    tester_score: float = 1.0,
    iterations_to_pass: int | None = None,
    persona_retry_map: dict[str, int] | None = None,
) -> RewardSummary:
    return RewardSummary(
        task_id=task_id,
        signals=(
            PersonaRewardSignal(
                persona_name=PersonaName.DEVELOPER,
                metric_name=RewardMetricName.DEVELOPER_CORRECTNESS,
                score=developer_score,
                evidence="test evidence for developer",
            ),
            PersonaRewardSignal(
                persona_name=PersonaName.TESTER,
                metric_name=RewardMetricName.FAILURE_LOCALIZATION_QUALITY,
                score=tester_score,
                evidence="test evidence for tester",
            ),
        ),
        summary="test reward summary",
        iterations_to_pass=iterations_to_pass,
        persona_retry_map=persona_retry_map or {},
    )


def test_load_persona_learning_signals_returns_empty_list_for_no_db(tmp_path: Path) -> None:
    signals = load_persona_learning_signals(tmp_path)

    assert signals == []


def test_load_persona_learning_signals_returns_empty_list_for_no_tasks(tmp_path: Path) -> None:
    initialize_coding_task_db(tmp_path)

    signals = load_persona_learning_signals(tmp_path)

    assert signals == []


def test_load_persona_learning_signals_extracts_signals_from_task(tmp_path: Path) -> None:
    initialize_coding_task_db(tmp_path)
    task = _make_task("task-42").model_copy(
        update={
            "reward_summary": _make_reward_summary(
                "task-42",
                developer_score=0.8,
                tester_score=1.0,
                iterations_to_pass=3,
                persona_retry_map={"developer": 2},
            )
        }
    )
    persist_coding_task_snapshot(tmp_path, task)

    signals = load_persona_learning_signals(tmp_path)

    assert len(signals) >= 2
    developer_signal = next(s for s in signals if s.persona_name == "developer")
    assert developer_signal.task_id == "task-42"
    assert developer_signal.score == pytest.approx(0.8)
    assert developer_signal.retry_count == 2
    assert developer_signal.iterations_to_pass == 3
    assert "developer" in developer_signal.lesson


def test_load_persona_learning_signals_filters_by_persona(tmp_path: Path) -> None:
    initialize_coding_task_db(tmp_path)
    task = _make_task("task-55").model_copy(
        update={
            "reward_summary": _make_reward_summary(
                "task-55",
                developer_score=1.0,
                tester_score=0.5,
                persona_retry_map={"developer": 1},
            )
        }
    )
    persist_coding_task_snapshot(tmp_path, task)

    signals = load_persona_learning_signals(tmp_path, persona="tester")

    assert all(s.persona_name == "tester" for s in signals)
    assert len(signals) >= 1


def test_load_persona_learning_signals_respects_limit(tmp_path: Path) -> None:
    initialize_coding_task_db(tmp_path)
    for i in range(5):
        task = _make_task(f"task-limit-{i}").model_copy(
            update={"reward_summary": _make_reward_summary(f"task-limit-{i}")}
        )
        persist_coding_task_snapshot(tmp_path, task)

    signals = load_persona_learning_signals(tmp_path, limit=3)

    assert len(signals) <= 3


def test_load_persona_learning_signals_with_no_iteration_data(tmp_path: Path) -> None:
    initialize_coding_task_db(tmp_path)
    task = _make_task("task-bare").model_copy(
        update={"reward_summary": _make_reward_summary("task-bare")}  # no iteration data
    )
    persist_coding_task_snapshot(tmp_path, task)

    signals = load_persona_learning_signals(tmp_path)

    found = next(s for s in signals if s.task_id == "task-bare")
    assert found.iterations_to_pass is None
    assert found.retry_count == 0


def test_compute_persona_retry_summary_aggregates_correctly() -> None:
    signals = [
        PersonaLearningSignal(
            task_id="t1",
            persona_name="developer",
            metric_name="developer_correctness",
            score=1.0,
            retry_count=2,
            iterations_to_pass=3,
            lesson="passed after retries",
        ),
        PersonaLearningSignal(
            task_id="t2",
            persona_name="developer",
            metric_name="developer_correctness",
            score=0.5,
            retry_count=0,
            iterations_to_pass=1,
            lesson="first pass",
        ),
        PersonaLearningSignal(
            task_id="t1",
            persona_name="tester",
            metric_name="failure_localization_quality",
            score=0.8,
            retry_count=1,
            iterations_to_pass=3,
            lesson="tester localized failure",
        ),
    ]

    summary = compute_persona_retry_summary(signals)

    assert "developer" in summary
    assert "tester" in summary
    assert summary["developer"]["avg_retry_count"] == pytest.approx(1.0)  # (2+0)/2
    assert summary["developer"]["avg_score"] == pytest.approx(0.75)  # (1.0+0.5)/2
    assert summary["developer"]["sample_count"] == 2.0
    assert summary["tester"]["avg_retry_count"] == pytest.approx(1.0)
    assert summary["tester"]["avg_score"] == pytest.approx(0.8)


def test_compute_persona_retry_summary_returns_empty_for_no_signals() -> None:
    result = compute_persona_retry_summary([])

    assert result == {}
