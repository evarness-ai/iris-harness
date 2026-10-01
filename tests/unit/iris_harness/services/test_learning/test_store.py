"""Unit tests for :mod:`iris_harness.services.learning.store`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from iris_harness.services.learning.models import Experiment
from iris_harness.services.learning.store import LearningMetricsStore


def _make_store(tmp_path: Path) -> LearningMetricsStore:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    return store


def test_record_and_recent_signals(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.record_signal(
        source="chat",
        metric_name="intent_confidence",
        value=0.42,
        success=True,
        latency_ms=12.5,
        metadata={"intent": "system", "agent_type": "system"},
    )
    store.record_signal(
        source="chat",
        metric_name="intent_confidence",
        value=0.83,
        success=True,
    )
    rows = store.recent_signals(metric_name="intent_confidence", limit=10)
    assert len(rows) == 2
    assert rows[0].value in {0.42, 0.83}
    assert rows[1].value in {0.42, 0.83}
    by_value = {row.value: row for row in rows}
    assert by_value[0.42].metadata["intent"] == "system"
    assert by_value[0.42].latency_ms == 12.5


def test_aggregate_metric_window(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    now = datetime.now(UTC)
    store.record_signal(
        source="chat",
        metric_name="m",
        value=1.0,
        success=True,
        ts=now - timedelta(hours=2),
    )
    store.record_signal(
        source="chat",
        metric_name="m",
        value=0.0,
        success=False,
        ts=now - timedelta(minutes=5),
    )

    overall = store.aggregate_metric(metric_name="m", now=now)
    assert overall.sample_count == 2
    assert overall.average == 0.5
    assert overall.success_rate == 0.5

    windowed = store.aggregate_metric(metric_name="m", window=timedelta(minutes=30), now=now)
    assert windowed.sample_count == 1
    assert windowed.average == 0.0


def test_experiment_upsert_round_trip(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    created = datetime.now(UTC)
    experiment = Experiment(
        id="exp-1",
        domain="intent_routing",
        hypothesis="strategy:demo",
        variant_description="bump threshold",
        config_changes={"threshold": 0.7},
        baseline_metric=0.5,
        current_metric=None,
        status="pending",
        created_at=created,
        rollback_config={"threshold": 0.6},
    )
    store.upsert_experiment(experiment, strategy_name="demo")
    listed = store.list_experiments(strategy_name="demo")
    assert [e.id for e in listed] == ["exp-1"]

    updated = Experiment(
        id="exp-1",
        domain=experiment.domain,
        hypothesis=experiment.hypothesis,
        variant_description=experiment.variant_description,
        config_changes=experiment.config_changes,
        baseline_metric=experiment.baseline_metric,
        current_metric=0.6,
        status="kept",
        created_at=experiment.created_at,
        started_at=created,
        evaluated_at=created + timedelta(hours=1),
        rollback_config=experiment.rollback_config,
    )
    store.upsert_experiment(updated)
    kept = store.list_experiments(status="kept")
    assert kept[0].current_metric == 0.6
    assert kept[0].status == "kept"
