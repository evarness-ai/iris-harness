"""Tests for the escalation priors analyzer (ADR-0068 L5)."""

from __future__ import annotations

from pathlib import Path

from iris_harness.services.learning.escalation_priors import recommend_start_tiers
from iris_harness.services.learning.store import LearningMetricsStore


def _store(tmp_path: Path) -> LearningMetricsStore:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    return store


def _shadow(
    store: LearningMetricsStore, *, intent: str, action: str, target: str = "tier2"
) -> None:
    store.record_signal(
        source="chat",
        metric_name="escalation_shadow",
        value=1.0 if action != "accept" else 0.0,
        success=True,
        metadata={"intent": intent, "action": action, "target_tier": target},
    )


def test_recommends_when_escalate_rate_clears_threshold(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for _ in range(4):
        _shadow(store, intent="research", action="escalate", target="tier2")
    for _ in range(1):
        _shadow(store, intent="research", action="accept")
    # 4/5 = 0.8 escalate rate, 5 samples
    recs = recommend_start_tiers(
        store,
        current_tiers={"research": "tier1"},
        min_samples=5,
        escalate_rate_threshold=0.5,
    )
    assert recs == {"research": "tier2"}


def test_no_recommendation_below_threshold(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for _ in range(1):
        _shadow(store, intent="chat", action="escalate")
    for _ in range(4):
        _shadow(store, intent="chat", action="accept")
    # 1/5 = 0.2 < 0.5
    recs = recommend_start_tiers(
        store, current_tiers={"chat": "tier1"}, min_samples=5, escalate_rate_threshold=0.5
    )
    assert recs == {}


def test_no_recommendation_below_min_samples(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for _ in range(3):
        _shadow(store, intent="rare", action="escalate")
    recs = recommend_start_tiers(
        store, current_tiers={"rare": "tier1"}, min_samples=10, escalate_rate_threshold=0.5
    )
    assert recs == {}


def test_skips_noop_when_already_at_target(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for _ in range(5):
        _shadow(store, intent="code", action="escalate", target="tier2")
    recs = recommend_start_tiers(
        store,
        current_tiers={"code": "tier2"},  # already starts at the escalate target
        min_samples=5,
        escalate_rate_threshold=0.5,
    )
    assert recs == {}


def test_picks_dominant_target_tier(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for _ in range(4):
        _shadow(store, intent="hard", action="escalate", target="tier3")
    for _ in range(2):
        _shadow(store, intent="hard", action="escalate", target="tier2")
    recs = recommend_start_tiers(
        store, current_tiers={"hard": "tier1"}, min_samples=5, escalate_rate_threshold=0.5
    )
    assert recs == {"hard": "tier3"}  # the more frequent escalate target
