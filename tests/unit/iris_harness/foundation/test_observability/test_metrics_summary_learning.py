"""The learning health section of summarize_llm_metrics (§4.4 meta-observability)."""

from __future__ import annotations

from pathlib import Path

from iris_harness.foundation.observability.metrics_summary import summarize_llm_metrics
from iris_harness.services.learning.store import LearningMetricsStore


def test_summary_merges_learning_health(tmp_path: Path) -> None:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    store.record_signal(source="chat", metric_name="intent_confidence", value=0.9, success=True)
    store.increment_counter("signals_dropped_total", by=1)

    summary = summarize_llm_metrics(log_dir=tmp_path / "no-logs", learning_store=store)

    learning = summary["learning"]
    assert learning["signals_recorded_total"] == 1
    assert learning["signals_dropped_total"] == 1
    assert learning["signal_volume"]["intent_confidence"] == 1
    assert "process" in learning  # in-process counters always present


def test_summary_learning_section_without_store(tmp_path: Path) -> None:
    summary = summarize_llm_metrics(log_dir=tmp_path / "no-logs")
    # Always present, never depends on a store being passed (P2 always-on floor).
    assert "learning" in summary
    assert "process" in summary["learning"]
