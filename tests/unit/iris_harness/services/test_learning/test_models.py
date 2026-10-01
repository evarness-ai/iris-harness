from __future__ import annotations

from datetime import datetime

from iris_harness.services.learning.models import ExperimentResult, IterationRecord, MetricSnapshot


def test_iteration_record() -> None:
    timestamp = datetime.now().timestamp()
    record = IterationRecord("iter_1", ["metric_1", "metric_2"], timestamp)

    assert record.iteration_id == "iter_1"
    assert record.metrics == ["metric_1", "metric_2"]
    assert record.timestamp == timestamp


def test_experiment_result() -> None:
    timestamp = datetime.now().timestamp()
    result = ExperimentResult("exp_1", True, "details", timestamp)

    assert result.experiment_id == "exp_1"
    assert result.success is True
    assert result.details == "details"
    assert result.timestamp == timestamp


def test_metric_snapshot() -> None:
    timestamp = datetime.now().timestamp()
    snapshot = MetricSnapshot("snap_1", ["metric_1", "metric_2"], timestamp)

    assert snapshot.snapshot_id == "snap_1"
    assert snapshot.metrics == ["metric_1", "metric_2"]
    assert snapshot.timestamp == timestamp
