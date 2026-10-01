"""A tool result raises the run's classification, on both ReAct paths (design §6.5).

The regression: the label was derived once from the user's question and cached for the
whole run, so "wondering how my day looks like ?" stayed `public` while `daily_plan`
put the user's inbox into the prompt. Every later egress_gate decision was made against
that stale label. Nothing fired PostToolUse at all, so there was no point at which it
could be corrected.

Both loops are covered here on purpose. `chat` and `chat_stream` share the turn
pipeline but `AgenticCore` still has two ReAct loops, and a governance behaviour that
lands on one is a hole in the other.
"""

from __future__ import annotations

from typing import Any

from iris_harness.agent.agentic_core import AgenticCore, AgenticCoreConfig, ToolSpec
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.plugins.output_classifier import OutputClassifierHook

_INBOX = "inbox: jordan1.kp@example.com, jordankpatel@example.com — 57 new messages"


class _RecordingEgress:
    """Stands in for egress_gate: records the label each LLM call was decided against."""

    name = "recording_egress"
    hook_point = HookPoint.PRE_LLM_CALL
    priority = 50

    def __init__(self) -> None:
        self.seen: list[str | None] = []

    async def __call__(self, ctx: HookContext) -> HookDecision:
        self.seen.append(ctx.classification)
        return HookDecision(outcome="allow", reason="test")


class _PublicClassify:
    """The question is harmless; only the tool result is not."""

    name = "public_classify"
    hook_point = HookPoint.PRE_CLASSIFY
    priority = 10

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="test", set_classification="public")


def _scripted_llm() -> Any:
    """Calls the tool once, then answers."""
    replies = iter(
        [
            "Thought: I should check the day.\nAction: daily_plan\nAction Input: {}",
            "Thought: done.\nFinal Answer: here is your day",
        ]
    )

    def call(prompt: str) -> str:
        return next(replies)

    return call


def _core(egress: _RecordingEgress) -> AgenticCore:
    kernel = GovernanceKernel()
    kernel.register(_PublicClassify())
    kernel.register(OutputClassifierHook())
    kernel.register(egress)
    kernel.init_lock()
    return AgenticCore(
        config=AgenticCoreConfig(max_iterations=4, timeout_seconds=30),
        llm_call=_scripted_llm(),
        tools=[ToolSpec(name="daily_plan", description="day aggregator", call=lambda a: _INBOX)],
        kernel=kernel,
        target_tier="tier_1",
        agent_type="chat",
    )


def test_the_sync_loop_egresses_under_the_raised_label() -> None:
    egress = _RecordingEgress()
    _core(egress).run("wondering how my day looks like ?")

    assert len(egress.seen) >= 2, egress.seen
    # The first call is before any tool ran; the question really was public.
    assert egress.seen[0] == "public"
    # Every call after the inbox entered the prompt sees the truth.
    assert all(label == "personal" for label in egress.seen[1:]), egress.seen


def test_the_streaming_loop_egresses_under_the_raised_label() -> None:
    egress = _RecordingEgress()
    list(_core(egress).run_stream("wondering how my day looks like ?"))

    assert len(egress.seen) >= 2, egress.seen
    assert egress.seen[0] == "public"
    assert all(label == "personal" for label in egress.seen[1:]), egress.seen
