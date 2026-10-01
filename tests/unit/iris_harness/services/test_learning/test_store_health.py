"""Tests for the L1 observability backbone on :class:`LearningMetricsStore`.

Covers correlation columns, the migration path for pre-existing DBs, durable
meta counters, per-metric volume, and the combined health summary
(learning-observability.md §4.1, §4.4).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from iris_harness.services.learning.store import LearningMetricsStore


def _store(tmp_path: Path) -> LearningMetricsStore:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    return store


def test_record_signal_persists_correlation_fields(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_signal(
        source="chat",
        metric_name="intent_confidence",
        value=0.9,
        success=True,
        session_id="sess-1",
        turn_id="turn-1",
        trace_id="abc123",
        span_id="def456",
        resolved_tier="tier1",
        resolved_agent="general",
    )
    [row] = store.recent_signals(metric_name="intent_confidence")
    assert row.session_id == "sess-1"
    assert row.turn_id == "turn-1"
    assert row.trace_id == "abc123"
    assert row.span_id == "def456"
    assert row.resolved_tier == "tier1"
    assert row.resolved_agent == "general"


def test_ensure_schema_migrates_legacy_signals_table(tmp_path: Path) -> None:
    # Simulate a pre-L1 DB: signals table without the correlation columns.
    db_path = tmp_path / "learning.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE signals (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,"
            " source TEXT NOT NULL, metric_name TEXT NOT NULL, value REAL NOT NULL,"
            " success INTEGER NOT NULL, latency_ms REAL, metadata_json TEXT)"
        )
        conn.execute(
            "INSERT INTO signals(ts, source, metric_name, value, success)"
            " VALUES ('2026-01-01T00:00:00+00:00', 'chat', 'intent_confidence', 0.5, 1)"
        )

    store = LearningMetricsStore(db_path=db_path)
    store.ensure_schema()  # must ALTER the new columns in without dropping the row

    [legacy] = store.recent_signals(metric_name="intent_confidence")
    assert legacy.value == 0.5
    assert legacy.session_id is None  # legacy row backfills as NULL, not an error
    # New writes carry correlation on the migrated table.
    store.record_signal(
        source="chat",
        metric_name="intent_confidence",
        value=0.8,
        success=True,
        turn_id="t-2",
    )
    rows = store.recent_signals(metric_name="intent_confidence")
    assert any(r.turn_id == "t-2" for r in rows)


def test_counters_are_durable_and_monotonic(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.counter("signals_dropped_total") == 0
    assert store.increment_counter("signals_dropped_total") == 1
    assert store.increment_counter("signals_dropped_total", by=3) == 4

    # Survives a fresh handle to the same DB (durable, not in-process).
    reopened = LearningMetricsStore(db_path=tmp_path / "learning.db")
    assert reopened.counter("signals_dropped_total") == 4
    assert reopened.all_counters()["signals_dropped_total"] == 4


def test_metric_volume_and_health_summary(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for _ in range(3):
        store.record_signal(
            source="chat", metric_name="response_has_errors", value=0.0, success=True
        )
    store.record_signal(source="chat", metric_name="intent_confidence", value=0.7, success=True)
    store.increment_counter("signals_dropped_total", by=2)

    volume = store.metric_volume()
    assert volume["response_has_errors"] == 3
    assert volume["intent_confidence"] == 1

    health = store.health_summary()
    assert health["signals_recorded_total"] == 4
    assert health["signals_dropped_total"] == 2
    # drop_rate = dropped / (recorded + dropped) = 2 / 6
    assert abs(health["drop_rate"] - (2 / 6)) < 1e-9
    assert health["signal_volume"]["response_has_errors"] == 3
