"""Tests for the context-health snapshot dataclasses (ADR-0081)."""

from __future__ import annotations

from iris_harness.agent.context_health import BudgetSplit, ContextHealth, WindowHealth


def _window(current: int, budget: int = 1000, ratio: float = 0.8) -> WindowHealth:
    return WindowHealth(
        budget_tokens=budget,
        current_tokens=current,
        compaction_ratio=ratio,
        last_compaction=None,
    )


def test_fill_pct_and_near_full() -> None:
    assert _window(400).fill_pct == 0.4
    assert _window(400).near_full is False
    assert _window(850).near_full is True  # >= 0.8 * 1000
    assert _window(800).near_full is True


def test_zero_budget_is_safe() -> None:
    w = _window(50, budget=0)
    assert w.fill_pct == 0.0
    assert w.near_full is False


def test_window_as_dict_rounds_fill() -> None:
    d = _window(333).as_dict()
    assert d["fill_pct"] == 0.333
    assert d["near_full"] is False
    assert d["last_compaction"] is None


def test_context_health_composes_and_serializes() -> None:
    health = ContextHealth(
        session_id="s1",
        window=_window(850),
        budgets=BudgetSplit(
            transcript_budget=450,
            memory_budget=350,
            last_context_tokens=1200,
            last_transcript_evicted=80,
        ),
        suppression={"total_feedback": 3, "active_suppressions": 1, "by_subsystem": {"email": 1}},
    )
    d = health.as_dict()
    assert d["session_id"] == "s1"
    assert d["window"]["near_full"] is True
    assert d["budgets"]["last_transcript_evicted"] == 80
    assert d["suppression"]["active_suppressions"] == 1
