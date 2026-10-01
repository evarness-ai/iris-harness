"""ADR-0077 P3: the in-loop context-budget controller (admission + eviction).

Admission caps the injected memory block (evict lowest-priority, pin the user profile);
eviction caps the running transcript (drop oldest, pin freshest). The controller derives
both budgets from the model window so they scale on a model swap.
"""

from __future__ import annotations

from iris_harness.agent.context_budget import (
    ContextBudgetController,
    admit_memory_parts,
    trim_history,
)


def test_trim_history_drops_oldest_pins_freshest() -> None:
    hist = ["A" * 400, "B" * 400, "C" * 40]  # ~100, 100, 10 tokens
    kept, evicted = trim_history(hist, 60)
    assert kept[-1] == hist[-1]  # freshest pinned
    assert evicted > 0


def test_trim_history_noop_under_budget() -> None:
    hist = ["x", "y"]
    assert trim_history(hist, 10_000) == (hist, 0)


def test_admit_memory_evicts_lowest_priority_first() -> None:
    parts = ["PROFILE" * 50, "EPISODIC" * 50, "SIGNALS" * 50]  # high -> low priority
    kept, evicted = admit_memory_parts(parts, budget_tokens=120)
    assert kept[0] == parts[0]  # profile retained
    assert parts[-1] not in kept  # lowest-priority block dropped first
    assert evicted > 0


def test_admit_memory_pins_profile_even_if_over_budget() -> None:
    parts = ["PROFILE" * 200]  # single huge profile block, over any small budget
    kept, evicted = admit_memory_parts(parts, budget_tokens=10, pin_first=True)
    assert kept == parts  # never silently drop identity
    assert evicted == 0


def test_admit_memory_noop_under_budget() -> None:
    parts = ["a", "b"]
    assert admit_memory_parts(parts, 10_000) == (parts, 0)


def test_controller_splits_window_budget() -> None:
    ctrl = ContextBudgetController(1000, transcript_fraction=0.45, memory_fraction=0.35)
    assert ctrl.transcript_budget == 450
    assert ctrl.memory_budget == 350
    # Budgets scale with the window (model swap to a bigger context).
    big = ContextBudgetController(4000)
    assert big.transcript_budget > ctrl.transcript_budget
    assert big.memory_budget > ctrl.memory_budget


def test_controller_methods_delegate() -> None:
    ctrl = ContextBudgetController(200)  # transcript=90, memory=70
    hist = ["Z" * 800, "Y" * 40]
    kept, ev = ctrl.trim_transcript(hist)
    assert kept[-1] == hist[-1] and ev > 0
    parts = ["P" * 200, "Q" * 200]
    mkept, mev = ctrl.admit_memory(parts)
    assert mkept[0] == parts[0] and mev > 0
