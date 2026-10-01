"""Unit tests for :mod:`iris_harness.services.learning.signals`."""

from __future__ import annotations

from pathlib import Path

from iris_harness.services.learning.signals import (
    LearningSignalCollector,
    reset_signal_health,
    signal_health,
)
from iris_harness.services.learning.store import LearningMetricsStore


def test_record_response_writes_signals(tmp_path: Path) -> None:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    collector = LearningSignalCollector(store)

    collector.record_response(
        intent="system",
        agent_type="system",
        intent_confidence=0.7,
        has_errors=False,
        latency_ms=42.0,
    )

    confidence_rows = store.recent_signals(metric_name="intent_confidence")
    error_rows = store.recent_signals(metric_name="response_has_errors")
    assert len(confidence_rows) == 1
    assert confidence_rows[0].metadata["intent"] == "system"
    assert len(error_rows) == 1
    assert error_rows[0].value == 0.0
    assert error_rows[0].success is True


def test_record_response_handles_missing_confidence(tmp_path: Path) -> None:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    collector = LearningSignalCollector(store)

    collector.record_response(
        intent="email",
        agent_type="email",
        intent_confidence=None,
        has_errors=True,
    )
    assert store.recent_signals(metric_name="intent_confidence") == []
    error_rows = store.recent_signals(metric_name="response_has_errors")
    assert error_rows[0].value == 1.0
    assert error_rows[0].success is False


def test_record_response_persists_correlation(tmp_path: Path) -> None:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    collector = LearningSignalCollector(store)

    collector.record_response(
        intent="system",
        agent_type="general",
        intent_confidence=0.6,
        has_errors=False,
        session_id="sess-9",
        turn_id="turn-9",
        trace_id="trace-9",
        span_id="span-9",
        resolved_tier="tier2",
    )

    [row] = store.recent_signals(metric_name="intent_confidence")
    assert row.session_id == "sess-9"
    assert row.turn_id == "turn-9"
    assert row.trace_id == "trace-9"
    assert row.span_id == "span-9"
    assert row.resolved_tier == "tier2"
    # resolved_agent defaults to agent_type when the caller doesn't override it.
    assert row.resolved_agent == "general"


class _ExplodingStore:
    """Store whose signal writes always fail, but counters succeed."""

    def __init__(self) -> None:
        self.dropped = 0

    def record_signal(self, **_kwargs: object) -> int:
        raise RuntimeError("disk full")

    def increment_counter(self, name: str, *, by: int = 1) -> int:
        self.dropped += by
        return self.dropped


def test_failed_record_is_counted_not_silent() -> None:
    reset_signal_health()
    store = _ExplodingStore()
    collector = LearningSignalCollector(store)  # type: ignore[arg-type]

    collector.record_response(
        intent="system",
        agent_type="system",
        intent_confidence=0.5,
        has_errors=False,
    )

    # The drop is visible: in-process counter + durable store counter both bumped.
    assert signal_health()["signals_dropped"] == 1
    assert store.dropped == 1
    reset_signal_health()


def test_record_escalation_shadow(tmp_path: Path) -> None:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    collector = LearningSignalCollector(store)

    collector.record_escalation_shadow(
        verdict={"action": "escalate", "diagnosis": "capability_gap", "confidence": 0.8},
        has_errors=False,
        session_id="sess-1",
        turn_id="turn-1",
        trace_id="tr-1",
    )
    [row] = store.recent_signals(metric_name="escalation_shadow")
    assert row.value == 1.0  # would act (action != accept)
    assert row.success is True
    assert row.turn_id == "turn-1"
    assert row.metadata["action"] == "escalate"
    assert row.metadata["has_errors"] is False


def test_record_escalation_shadow_accept_is_zero(tmp_path: Path) -> None:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    collector = LearningSignalCollector(store)
    collector.record_escalation_shadow(
        verdict={"action": "accept", "diagnosis": "acceptable", "confidence": 0.9},
        has_errors=False,
    )
    [row] = store.recent_signals(metric_name="escalation_shadow")
    assert row.value == 0.0  # accept = would not act


def test_record_metric_generic(tmp_path: Path) -> None:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    collector = LearningSignalCollector(store)

    collector.record_metric(
        metric_name="task_completed",
        value=1.0,
        success=True,
        session_id="s1",
        turn_id="t1",
        resolved_tier="tier1",
        metadata={"intent": "general"},
    )
    [row] = store.recent_signals(metric_name="task_completed")
    assert row.value == 1.0
    assert row.success is True
    assert row.resolved_tier == "tier1"
    assert row.metadata["intent"] == "general"


def test_record_feedback_writes_user_feedback_signal(tmp_path: Path) -> None:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    collector = LearningSignalCollector(store)

    collector.record_feedback(
        sentiment="down",
        session_id="s1",
        trace_id="s1~0",
        rating=2,
        note="  not what I meant  ",
        intent="finance",
        agent_type="finance",
    )

    [row] = store.recent_signals(metric_name="user_feedback")
    assert row.value == -1.0
    assert row.success is False
    assert row.session_id == "s1" and row.trace_id == "s1~0"
    assert row.metadata["sentiment"] == "down"
    assert row.metadata["rating"] == 2
    assert row.metadata["note"] == "not what I meant"  # trimmed
    assert row.metadata["intent"] == "finance"


def test_record_feedback_thumbs_up_is_positive(tmp_path: Path) -> None:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    collector = LearningSignalCollector(store)

    collector.record_feedback(sentiment="UP", session_id="s2")  # case-insensitive

    [row] = store.recent_signals(metric_name="user_feedback")
    assert row.value == 1.0 and row.success is True
    assert row.metadata["sentiment"] == "up"
    assert "note" not in row.metadata and "rating" not in row.metadata
