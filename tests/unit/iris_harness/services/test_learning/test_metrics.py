from __future__ import annotations

from iris_harness.services.learning.metrics import LearningMetricsCollector


def test_learning_metrics_collector() -> None:
    collector = LearningMetricsCollector()
    collector.record_metric("task_completed", 1.23, True)
    metrics = collector.get_metrics()
    assert len(metrics) == 1
    assert metrics[0]["outcome"] == "task_completed"
    assert metrics[0]["latency"] == 1.23
    assert metrics[0]["success"] is True


def test_clear_metrics() -> None:
    collector = LearningMetricsCollector()
    collector.record_metric("task_completed", 1.23, True)
    collector.clear_metrics()
    assert collector.get_metrics() == []
