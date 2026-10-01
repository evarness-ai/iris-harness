"""Unit tests for :mod:`iris_harness.services.learning.engine`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from iris_harness.services.learning.engine import LearningEngine
from iris_harness.services.learning.experiment_loop import ExperimentLoop
from iris_harness.services.learning.store import LearningMetricsStore
from iris_harness.services.learning.strategies import (
    StrategyDefinition,
    StrategyEvaluation,
    StrategyTrigger,
    StrategyVariant,
)


def _strategy() -> StrategyDefinition:
    return StrategyDefinition(
        name="bump_threshold",
        domain="intent_routing",
        metric="intent_confidence",
        baseline_min_samples=3,
        trigger=StrategyTrigger(below=0.6),
        variant=StrategyVariant(
            description="bump",
            config_changes={"threshold": 0.7},
            rollback={"threshold": 0.6},
        ),
        evaluation=StrategyEvaluation(window_hours=1, min_improvement_pct=5.0),
    )


def _engine(tmp_path: Path) -> tuple[LearningEngine, LearningMetricsStore]:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    loop = ExperimentLoop(min_improvement_pct=5.0)
    engine = LearningEngine(loop=loop, store=store, strategies=[_strategy()])
    return engine, store


def test_tick_skips_when_baseline_too_small(tmp_path: Path) -> None:
    engine, store = _engine(tmp_path)
    store.record_signal(source="chat", metric_name="intent_confidence", value=0.4, success=True)
    report = engine.tick()
    assert report.started == ()
    assert "bump_threshold" in report.skipped


def test_tick_starts_experiment_when_trigger_fires(tmp_path: Path) -> None:
    engine, store = _engine(tmp_path)
    for _ in range(5):
        store.record_signal(source="chat", metric_name="intent_confidence", value=0.4, success=True)

    report = engine.tick()
    assert len(report.started) == 1
    experiment_id = report.started[0]
    persisted = store.list_experiments(strategy_name="bump_threshold")
    assert [e.id for e in persisted] == [experiment_id]
    assert persisted[0].status == "running"


def test_tick_evaluates_after_window(tmp_path: Path) -> None:
    engine, store = _engine(tmp_path)
    for _ in range(5):
        store.record_signal(source="chat", metric_name="intent_confidence", value=0.4, success=True)
    start = datetime.now(UTC)
    engine.tick(now=start)

    # Add post-experiment measurements that show improvement.
    for _ in range(5):
        store.record_signal(source="chat", metric_name="intent_confidence", value=0.9, success=True)

    later = start + timedelta(hours=2)
    report = engine.tick(now=later)
    assert len(report.evaluated) == 1
    finalised = store.list_experiments()
    assert finalised[0].status in {"kept", "discarded"}
