"""Tests for the deterministic learning-intelligence layer (ADR-0069 #4).

The intelligence report is pure measurement: these assert the aggregates are
computed correctly from correlated signals (escalation precision against the
has_errors proxy, the per-(intent, tier) outcome matrix, signal integrity) and
that no interpretation/thresholding leaks in.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from iris_harness.services.learning.intelligence import build_intelligence, render_text
from iris_harness.services.learning.store import LearningMetricsStore


def _store(tmp_path: Path) -> LearningMetricsStore:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    return store


_NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


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


def _shadow(
    store: LearningMetricsStore, *, would_act: bool, has_errors: bool, ts: datetime
) -> None:
    store.record_signal(
        source="chat",
        metric_name="escalation_shadow",
        value=1.0 if would_act else 0.0,
        success=not has_errors,
        metadata={"action": "escalate" if would_act else "accept", "has_errors": has_errors},
        ts=ts,
    )


def test_outcome_matrix_groups_by_intent_and_tier(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _completed(store, intent="email", tier="tier1", clean=True, ts=_NOW)
    _completed(store, intent="email", tier="tier1", clean=False, ts=_NOW)
    _completed(store, intent="coding", tier="tier2", clean=True, ts=_NOW)

    report = build_intelligence(store, now=_NOW)
    cells = {(c.intent, c.tier): c for c in report.matrix}

    assert cells[("email", "tier1")].samples == 2
    assert cells[("email", "tier1")].completion_rate == 0.5
    assert cells[("coding", "tier2")].completion_rate == 1.0


def test_matrix_overlays_tokens_correction_and_reuse(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _completed(store, intent="email", tier="tier1", clean=True, ts=_NOW)
    store.record_signal(
        source="chat",
        metric_name="turn_tokens",
        value=900.0,
        success=True,
        metadata={"intent": "email"},
        resolved_tier="tier1",
        ts=_NOW,
    )
    store.record_signal(
        source="chat",
        metric_name="user_correction",
        value=1.0,
        success=False,
        metadata={"intent": "email"},
        resolved_tier="tier1",
        ts=_NOW,
    )
    store.record_signal(
        source="chat",
        metric_name="downstream_reuse",
        value=1.0,
        success=True,
        metadata={"intent": "email"},
        resolved_tier="tier1",
        ts=_NOW,
    )

    cell = {(c.intent, c.tier): c for c in build_intelligence(store, now=_NOW).matrix}[
        ("email", "tier1")
    ]
    assert cell.avg_tokens == 900.0
    assert cell.correction_rate == 1.0
    assert cell.correction_samples == 1
    assert cell.reuse_count == 1


def test_escalation_precision_is_share_of_would_act_that_erred(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # 3 would-act: 2 actually erred -> precision 2/3. 1 accept (ignored by precision).
    _shadow(store, would_act=True, has_errors=True, ts=_NOW)
    _shadow(store, would_act=True, has_errors=True, ts=_NOW)
    _shadow(store, would_act=True, has_errors=False, ts=_NOW)
    _shadow(store, would_act=False, has_errors=False, ts=_NOW)

    acc = build_intelligence(store, now=_NOW).accuracy
    assert acc.escalation_shadow_total == 4
    assert acc.escalation_would_act == 3
    assert acc.escalation_would_act_rate == 0.75
    assert acc.escalation_precision == 2 / 3


def test_escalation_precision_none_without_would_act_samples(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _shadow(store, would_act=False, has_errors=False, ts=_NOW)
    acc = build_intelligence(store, now=_NOW).accuracy
    assert acc.escalation_would_act == 0
    assert acc.escalation_precision is None


def test_window_excludes_old_signals(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _completed(store, intent="email", tier="tier1", clean=True, ts=_NOW)
    _completed(store, intent="email", tier="tier1", clean=True, ts=_NOW - timedelta(days=30))

    report = build_intelligence(store, window=timedelta(days=7), now=_NOW)
    cell = {(c.intent, c.tier): c for c in report.matrix}[("email", "tier1")]
    assert cell.samples == 1


def test_drop_rate_surfaces_signal_integrity(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _completed(store, intent="email", tier="tier1", clean=True, ts=_NOW)
    store.increment_counter("signals_dropped_total", by=1)

    acc = build_intelligence(store, now=_NOW).accuracy
    assert acc.signals_dropped_total == 1
    assert 0.0 < acc.drop_rate < 1.0


def test_empty_store_is_safe(tmp_path: Path) -> None:
    store = _store(tmp_path)
    report = build_intelligence(store, now=_NOW)
    assert report.matrix == ()
    assert report.accuracy.escalation_precision is None
    assert "no measured turns" in render_text(report)


def test_as_dict_round_trips_shape(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _completed(store, intent="email", tier="tier1", clean=True, ts=_NOW)
    payload = build_intelligence(store, now=_NOW).as_dict()
    assert set(payload) == {
        "sampled_at",
        "window_hours",
        "accuracy",
        "matrix",
        "experiments",
        "feedback",
    }
    assert payload["matrix"][0]["intent"] == "email"
    assert "escalation_precision" in payload["accuracy"]


def _feedback(store: LearningMetricsStore, *, up: bool, intent: str, ts: datetime) -> None:
    store.record_signal(
        source="chat",
        metric_name="user_feedback",
        value=1.0 if up else -1.0,
        success=up,
        metadata={"sentiment": "up" if up else "down", "intent": intent},
        ts=ts,
    )


def test_intelligence_aggregates_user_feedback(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _feedback(store, up=True, intent="email", ts=_NOW)
    _feedback(store, up=True, intent="email", ts=_NOW)
    _feedback(store, up=False, intent="finance", ts=_NOW)

    report = build_intelligence(store, now=_NOW)
    fb = report.feedback
    assert (fb.total, fb.positive, fb.negative) == (3, 2, 1)
    by = {f.intent: f for f in fb.by_intent}
    assert by["email"].satisfaction == 1.0
    assert by["finance"].positive == 0 and by["finance"].negative == 1
    # Serialized for the analyst + UI; lowest-satisfaction intent surfaced in text.
    assert report.as_dict()["feedback"]["satisfaction"] == 0.6667
    assert "User feedback: 67% satisfied" in render_text(report)
    assert "lowest: finance 0%" in render_text(report)


def test_intelligence_no_feedback_is_empty(tmp_path: Path) -> None:
    report = build_intelligence(_store(tmp_path), now=_NOW)
    assert report.feedback.total == 0
    assert report.as_dict()["feedback"]["satisfaction"] is None
    assert "User feedback:" not in render_text(report)
