"""ADR-0077: ReAct context guardrail + the cross-domain finance/email prompt nudge.

The running ReAct transcript was purely additive — every Thought/Action/Observation
block re-entered the next prompt, so a multi-tool loop grew context unbounded. The
guardrail caps the transcript at a token budget, evicting oldest blocks while pinning
the freshest. The cross-domain nudge is added only when both the finance and email
surfaces are on the loop, so it never bloats unrelated turns.
"""

from __future__ import annotations

from iris_harness.agent.agentic_core import (
    AgenticCore,
    AgenticCoreConfig,
    ToolSpec,
    _build_react_prompt,
    _trim_react_history,
)

_FIN = ToolSpec("finance_lookup", "money", lambda _a: "")
_INBOX = ToolSpec("search_inbox", "email", lambda _a: "")
_WEB = ToolSpec("research", "web", lambda _a: "")


def test_trim_drops_oldest_pins_freshest() -> None:
    history = ["A" * 400, "B" * 400, "C" * 400, "D" * 40]  # ~100,100,100,10 tokens
    kept, evicted = _trim_react_history(history, budget_tokens=120)
    assert kept[-1] == history[-1]  # freshest block always pinned
    assert evicted > 0
    assert sum(len(h) // 4 for h in kept) <= 120 or len(kept) == 1


def test_trim_noop_under_budget() -> None:
    history = ["short", "blocks"]
    kept, evicted = _trim_react_history(history, budget_tokens=10_000)
    assert kept == history
    assert evicted == 0


def test_trim_disabled_with_zero_budget() -> None:
    history = ["A" * 4000, "B" * 4000]
    kept, evicted = _trim_react_history(history, budget_tokens=0)
    assert kept == history and evicted == 0


def test_apply_context_budget_uses_config() -> None:
    core = AgenticCore(config=AgenticCoreConfig(history_token_budget=50), tools=[])
    long_history = ["X" * 400, "Y" * 400, "Z" * 40]
    trimmed, evicted = core._apply_context_budget(long_history)
    assert evicted > 0
    assert trimmed[-1] == long_history[-1]


def test_apply_context_budget_off_by_default() -> None:
    core = AgenticCore(config=AgenticCoreConfig(), tools=[])  # budget None
    history = ["X" * 4000, "Y" * 4000]
    trimmed, evicted = core._apply_context_budget(history)
    assert trimmed == history and evicted == 0


# The finance plugin's own guidance (its nudge present, and naming no institution) is
# asserted beside the plugin: tests/unit/iris_personal/plugins/test_finance_workflows/
# test_loop_guidance.py. The core's half stays here: it authors no routing prose.


def test_cross_domain_nudge_absent_when_the_core_authors_nothing() -> None:
    # A bare finance_lookup (no declared guidance) gets no routing prose from the core.
    prompt = _build_react_prompt(
        "what do I owe", tools=[_FIN, _INBOX], history=[], memory_context=None
    )
    assert "finance_lookup first" not in prompt
