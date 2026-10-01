"""Tests for the loop-closer (ADR-0069 #4, slice 3): promote + re-measure.

A recommendation is promoted to a tracked experiment that captures a baseline and
is re-measured against the same metric over time. These cover target resolution,
baseline capture, lower-is-better inversion, and the keep/discard outcome past the
window.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from iris_harness.services.learning.analyst import MetricTarget
from iris_harness.services.learning.intelligence import build_intelligence
from iris_harness.services.learning.promote import (
    TARGET_KEY,
    improvement_pct,
    promote_recommendation,
    remeasure_promoted_experiments,
    resolve_target_value,
)
from iris_harness.services.learning.store import LearningMetricsStore

_NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


def _store(tmp_path: Path) -> LearningMetricsStore:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    return store


def _completed(
    store: LearningMetricsStore, *, intent: str, tier: str, clean: bool, ts: datetime
) -> None:
    store.record_signal(
        source="chat",
        metric_name="task_completed",
        value=1.0 if clean else 0.0,
        success=clean,
        metadata={"intent": intent},
        resolved_tier=tier,
        ts=ts,
    )


def _save_analysis(store: LearningMetricsStore, *, target: dict | None) -> None:
    store.save_analysis(
        {
            "generated_at": _NOW.isoformat(),
            "model": "m",
            "summary": "s",
            "recommendations": [
                {
                    "title": "Route email up",
                    "finding": "email@tier1 completion low",
                    "action": "start email at tier2",
                    "evidence": ["email@tier1"],
                    "confidence": "high",
                    "target": target,
                }
            ],
        }
    )


def test_resolve_cell_and_global_metrics(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _completed(store, intent="email", tier="tier1", clean=True, ts=_NOW)
    _completed(store, intent="email", tier="tier1", clean=False, ts=_NOW)
    report = build_intelligence(store, now=_NOW)

    cell = resolve_target_value(
        report, MetricTarget(metric="completion_rate", intent="email", tier="tier1")
    )
    assert cell == 0.5
    glob = resolve_target_value(report, MetricTarget(metric="drop_rate"))
    assert glob == 0.0
    missing = resolve_target_value(
        report, MetricTarget(metric="completion_rate", intent="x", tier="y")
    )
    assert missing is None


def test_improvement_inverts_lower_is_better() -> None:
    # completion_rate: higher is better -> raw improvement.
    assert improvement_pct(baseline=0.5, current=0.6, metric="completion_rate") > 0
    # correction_rate: lower is better -> a drop is an improvement.
    assert improvement_pct(baseline=0.4, current=0.2, metric="correction_rate") > 0
    assert improvement_pct(baseline=0.2, current=0.4, metric="correction_rate") < 0


def test_promote_captures_baseline_and_persists_pending(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _completed(store, intent="email", tier="tier1", clean=True, ts=_NOW)
    _completed(store, intent="email", tier="tier1", clean=False, ts=_NOW)
    _save_analysis(store, target={"metric": "completion_rate", "intent": "email", "tier": "tier1"})

    result = promote_recommendation(store, index=1, experiment_id="e1", now=_NOW)
    assert result is not None
    assert result.measurable is True
    exp = result.experiment
    assert exp.status == "pending"
    assert exp.baseline_metric == 0.5
    assert exp.config_changes[TARGET_KEY]["metric"] == "completion_rate"
    # Persisted to the ledger.
    [stored] = store.list_experiments()
    assert stored.id == "e1"


def test_promote_out_of_range_and_no_analysis(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert promote_recommendation(store, index=1, now=_NOW) is None  # no analysis
    _save_analysis(store, target=None)
    assert promote_recommendation(store, index=5, now=_NOW) is None  # out of range


def test_promote_without_target_is_unmeasurable_but_tracked(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _save_analysis(store, target=None)
    result = promote_recommendation(store, index=1, experiment_id="e2", now=_NOW)
    assert result is not None
    assert result.measurable is False
    assert result.experiment.baseline_metric == 0.0
    assert TARGET_KEY not in result.experiment.config_changes


def test_remeasure_keeps_on_improvement_past_window(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # Baseline: 50% completion for email@tier1.
    _completed(store, intent="email", tier="tier1", clean=True, ts=_NOW)
    _completed(store, intent="email", tier="tier1", clean=False, ts=_NOW)
    _save_analysis(store, target={"metric": "completion_rate", "intent": "email", "tier": "tier1"})
    promote_recommendation(store, index=1, experiment_id="e1", now=_NOW, evaluation_window_hours=24)

    # The change "worked": later traffic is all clean -> higher completion. Use a
    # wide window so older baseline rows are still counted alongside new ones.
    later = _NOW + timedelta(hours=30)
    for _ in range(8):
        _completed(store, intent="email", tier="tier1", clean=True, ts=later)

    updated = remeasure_promoted_experiments(store, now=later)
    assert updated == 1
    [exp] = store.list_experiments()
    assert exp.current_metric is not None and exp.current_metric > 0.5
    assert exp.status == "kept"


def test_remeasure_evaluating_before_window(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _completed(store, intent="email", tier="tier1", clean=True, ts=_NOW)
    _save_analysis(store, target={"metric": "completion_rate", "intent": "email", "tier": "tier1"})
    promote_recommendation(store, index=1, experiment_id="e1", now=_NOW, evaluation_window_hours=48)

    # Within the window: measured but not yet judged.
    soon = _NOW + timedelta(hours=1)
    remeasure_promoted_experiments(store, now=soon)
    [exp] = store.list_experiments()
    assert exp.status == "evaluating"
    assert exp.evaluated_at is None


def test_remeasure_ignores_non_recommendation_experiments(tmp_path: Path) -> None:
    from iris_harness.services.learning.models import Experiment

    store = _store(tmp_path)
    store.upsert_experiment(
        Experiment(
            id="strat-1",
            domain="routing",
            hypothesis="strategy:foo",
            variant_description="v",
            config_changes={},  # no TARGET_KEY -> skipped
            baseline_metric=0.4,
            created_at=_NOW,
        )
    )
    assert remeasure_promoted_experiments(store, now=_NOW + timedelta(hours=1)) == 0
    [exp] = store.list_experiments()
    assert exp.status == "pending"  # untouched
