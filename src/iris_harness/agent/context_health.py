"""Context-health snapshot — one view of how the harness manages its context budget.

The "brain" keeps its working context bounded through several mechanisms that each live
in a different place: the conversation compactor (ADR-0079) bounds the multi-turn history
to ~80% of the model window; the in-loop ``ContextBudgetController`` (ADR-0077 P3) splits a
per-turn budget between the running transcript and the durable-memory block, evicting the
oldest/lowest-value parts; and the surface-feedback spine (ADR-0078) suppresses proactively-
surfaced noise. Individually they log; together they had no single surface.

This module is the aggregation point. It is pure — the runtime gathers the live primitives
(current history tokens, the window-derived budgets, the last compaction event, the
suppression roll-up) and hands them here to be composed into one ``ContextHealth`` snapshot
for the CLI / API / web. No I/O, no LLM, fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WindowHealth:
    """The multi-turn conversation window (compactor side)."""

    budget_tokens: int  # the chat tier's window-derived prompt budget
    current_tokens: int  # estimated tokens of the live history + summary this session
    compaction_ratio: float  # compaction fires at/above this fraction of the budget
    last_compaction: dict[str, Any] | None  # most recent compaction event, or None

    @property
    def fill_pct(self) -> float:
        return self.current_tokens / self.budget_tokens if self.budget_tokens else 0.0

    @property
    def near_full(self) -> bool:
        """At/above the compaction line — the next turn is likely to compact."""
        return self.fill_pct >= self.compaction_ratio

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget_tokens": self.budget_tokens,
            "current_tokens": self.current_tokens,
            "fill_pct": round(self.fill_pct, 3),
            "compaction_ratio": self.compaction_ratio,
            "near_full": self.near_full,
            "last_compaction": self.last_compaction,
        }


@dataclass(frozen=True)
class BudgetSplit:
    """The per-turn in-loop budget (P3 ContextBudgetController side)."""

    transcript_budget: int  # tokens reserved for the running ReAct transcript
    memory_budget: int  # tokens reserved for the durable-memory block
    last_context_tokens: int | None  # size of the last assembled prompt, if a turn has run
    last_transcript_evicted: int | None  # tokens evicted from the transcript last turn

    def as_dict(self) -> dict[str, Any]:
        return {
            "transcript_budget": self.transcript_budget,
            "memory_budget": self.memory_budget,
            "last_context_tokens": self.last_context_tokens,
            "last_transcript_evicted": self.last_transcript_evicted,
        }


@dataclass(frozen=True)
class ContextHealth:
    """A composed snapshot of the harness's context management for one session."""

    session_id: str
    window: WindowHealth
    budgets: BudgetSplit
    suppression: dict[str, Any]  # SuppressionSummary.as_dict()

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "window": self.window.as_dict(),
            "budgets": self.budgets.as_dict(),
            "suppression": self.suppression,
        }


__all__ = ["BudgetSplit", "ContextHealth", "WindowHealth"]
