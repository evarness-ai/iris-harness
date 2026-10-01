"""ADR-0077 P3 — the in-loop context-budget controller.

The unified ReAct loop pulls two kinds of context into every prompt:

  - **Durable memory** injected per turn (identity, user profile, episodic patterns,
    learning signals, recent turns) — the "inject context" side.
  - The **running transcript** (Thought/Action/Observation blocks) that grows each
    iteration — the "evict context" side.

Left unbounded, both pollute the window: stale observations crowd out the system prompt,
and a fat memory block crowds out the transcript. This controller is the single in-loop
owner of that budget. It is *kernel-audited* (the loop emits its token deltas into the
``PRE_LLM_CALL`` hook), not kernel-owned — lighter than a mandatory pre-assembly hook,
and promotable to one later if pre-assembly veto is ever needed (ADR-0077, user choice).

Both admission (memory) and eviction (transcript) are pure functions here so they unit-
test in isolation; the controller composes them and reports telemetry.

What is "context" vs "memory": MEMORY is durable (facts/signals/episodic in the stores,
selected by the MemoryRetriever); CONTEXT is the ephemeral per-turn assembly of that memory
+ the live transcript into one prompt. This controller governs the *context* — it never
deletes memory, only decides how much of it enters this turn's prompt.
"""

from __future__ import annotations

from iris_harness.llm.budget import estimate_tokens


def trim_history(history: list[str], budget_tokens: int) -> tuple[list[str], int]:
    """Cap the running ReAct transcript at ``budget_tokens`` (eviction side).

    The transcript is purely additive across iterations — every block is re-sent in the
    next prompt — so a multi-tool loop accumulates stale observations. Drop the OLDEST
    blocks first, always pinning the most recent one (the freshest grounded observation
    the model is reasoning over). Returns ``(kept, evicted_tokens)``.
    """
    if budget_tokens <= 0 or len(history) <= 1:
        return history, 0
    used = sum(estimate_tokens(h) for h in history)
    if used <= budget_tokens:
        return history, 0
    kept = list(history)
    evicted = 0
    while used > budget_tokens and len(kept) > 1:
        ev = estimate_tokens(kept.pop(0))
        used -= ev
        evicted += ev
    return kept, evicted


def admit_memory_parts(
    parts: list[str], budget_tokens: int, *, pin_first: bool = True
) -> tuple[list[str], int]:
    """Admit memory blocks within ``budget_tokens`` (admission side).

    ``parts`` is the priority-ordered list of memory blocks the prompt builder assembled
    (highest-value first: user profile, active items, episodic, behaviour, summary, recent
    turns, learning hints). When the block exceeds the budget, evict from the LOWEST-value
    END first; ``pin_first`` keeps the user-profile block (index 0) even if it alone is over
    budget — identity is never dropped silently. Returns ``(kept, evicted_tokens)``.
    """
    if budget_tokens <= 0 or not parts:
        return parts, 0
    used = sum(estimate_tokens(p) for p in parts)
    if used <= budget_tokens:
        return parts, 0
    kept = list(parts)
    evicted = 0
    floor = 1 if pin_first else 0
    while used > budget_tokens and len(kept) > floor:
        ev = estimate_tokens(kept.pop())  # lowest-priority block is last
        used -= ev
        evicted += ev
    return kept, evicted


def fill_recent_turns(turns: list[str], budget_tokens: int, *, min_turns: int = 2) -> list[str]:
    """Take conversation lines from the NEWEST end until ``budget_tokens`` is used.

    Replaces the fixed ``recent_turns[-3:]`` tail in the ReAct prompt, which showed
    1.5 exchanges however much window was free: the loop kept a window sized in
    thousands of tokens and then handed the model three lines of it.

    Always returns at least ``min_turns`` lines when that many exist (a single huge
    turn must not starve continuity), and keeps chronological order.
    """
    if not turns:
        return []
    if budget_tokens <= 0:
        return turns[-min_turns:]
    kept_rev: list[str] = []
    used = 0
    for line in reversed(turns):
        tok = estimate_tokens(line)
        if kept_rev and used + tok > budget_tokens and len(kept_rev) >= min_turns:
            break
        kept_rev.append(line)
        used += tok
    return list(reversed(kept_rev))


class ContextBudgetController:
    """Single in-loop owner of the per-turn context budget (admission + eviction).

    Splits a total token budget (derived from the model's context window) between the
    durable-memory block and the running transcript, leaving the remainder for the system
    prompt + tool catalog + the model's response. Pure aside from the split arithmetic, so
    it is fully unit-testable; the loop calls :meth:`trim_transcript` / :meth:`admit_memory`
    and folds the returned token deltas into the kernel audit.
    """

    def __init__(
        self,
        total_budget: int,
        *,
        transcript_fraction: float = 0.45,
        memory_fraction: float = 0.35,
    ) -> None:
        self.total_budget = max(0, total_budget)
        self._transcript_fraction = transcript_fraction
        self._memory_fraction = memory_fraction

    @property
    def transcript_budget(self) -> int:
        return int(self.total_budget * self._transcript_fraction)

    @property
    def memory_budget(self) -> int:
        return int(self.total_budget * self._memory_fraction)

    def trim_transcript(self, history: list[str]) -> tuple[list[str], int]:
        return trim_history(history, self.transcript_budget)

    def admit_memory(self, parts: list[str]) -> tuple[list[str], int]:
        return admit_memory_parts(parts, self.memory_budget)
