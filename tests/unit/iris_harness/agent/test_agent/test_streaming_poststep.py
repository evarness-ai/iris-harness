"""ADR-0107 — the evaluator on the streaming loop, and what the user sees when it halts.

The owner's decision:

- enforce on streaming only the signals with **no local equivalent** —
  classification_violation, cost_budget, goal_drift, semantic loop_detect;
- leave runaway-loop containment to `run_stream`'s own `max_iterations` /
  `stall_limit` / malformed-step guards rather than duplicating it in the evaluator;
- when the harness interrupts a task, **summarise what ran, say why, and stop** —
  the user decides what happens next.
"""

from __future__ import annotations

import pytest

from iris_harness.agent.agentic_core import (
    STREAMING_ENFORCED_SIGNALS,
    AgenticCore,
    AgenticCoreConfig,
    ReactStep,
    ToolSpec,
    _completed_work,
    _enforced_signal_names,
)
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookPoint
from iris_harness.kernel.governance.hooks.types import HookDecision


def _decision(outcome: str, *signals: tuple[str, str]) -> HookDecision:
    """A decision carrying `(signal_name, verdict)` pairs, as the evaluator records them."""
    return HookDecision(
        outcome=outcome,  # type: ignore[arg-type]
        reason="evaluator: test",
        audit_metadata={
            "signals": [
                {"name": name, "verdict": verdict, "severity": "warn", "reason": "…"}
                for name, verdict in signals
            ]
        },
    )


# ── which signals streaming enforces ──────────────────────────────────────────


def test_the_enforced_set_is_exactly_the_signals_without_a_local_equivalent() -> None:
    assert STREAMING_ENFORCED_SIGNALS == {
        "classification_violation",
        "cost_budget",
        "goal_drift",
        "loop_detect",
    }
    # The counting signals stay out: run_stream's own guards already contain these.
    for local in ("step_cap", "action_repeat", "tool_failure_streak"):
        assert local not in STREAMING_ENFORCED_SIGNALS


@pytest.mark.parametrize(
    "signal", ["classification_violation", "cost_budget", "goal_drift", "loop_detect"]
)
def test_an_enforced_signal_is_detected(signal: str) -> None:
    decision = _decision("deny", (signal, "halt"))
    assert _enforced_signal_names(decision, STREAMING_ENFORCED_SIGNALS) == [signal]


@pytest.mark.parametrize("signal", ["step_cap", "action_repeat", "tool_failure_streak"])
def test_a_locally_guarded_signal_does_not_enforce(signal: str) -> None:
    """The duplicate features the owner asked us to reuse rather than re-enforce."""
    decision = _decision("deny", (signal, "halt"))
    assert _enforced_signal_names(decision, STREAMING_ENFORCED_SIGNALS) == []


def test_a_non_terminal_verdict_from_an_enforced_signal_does_not_halt() -> None:
    decision = _decision("deny", ("cost_budget", "warn"))
    assert _enforced_signal_names(decision, STREAMING_ENFORCED_SIGNALS) == []


def test_a_mixed_decision_enforces_on_the_enforced_one_only() -> None:
    decision = _decision("deny", ("step_cap", "halt"), ("goal_drift", "halt"))
    assert _enforced_signal_names(decision, STREAMING_ENFORCED_SIGNALS) == ["goal_drift"]


def test_malformed_metadata_never_halts() -> None:
    """Signal metadata is read structurally; anything unexpected fails open to the
    local guards rather than halting a run on a parse accident."""
    assert _enforced_signal_names(HookDecision(outcome="deny", reason="x"), frozenset({"a"})) == []
    bad = HookDecision(outcome="deny", reason="x", audit_metadata={"signals": "nope"})
    assert _enforced_signal_names(bad, frozenset({"a"})) == []


# ── what the user sees ────────────────────────────────────────────────────────


def test_completed_work_lists_only_steps_that_ran_a_tool() -> None:
    steps = [
        ReactStep(thought="thinking", action=None),
        ReactStep(thought="t", action="research", action_input={}, observation="6 results"),
        ReactStep(thought="t", action="read_file", action_input={}, observation=None),
    ]
    assert _completed_work(steps) == [
        "1. research - 6 results",
        "2. read_file - no result",
    ]


def test_a_long_observation_is_previewed_not_dumped() -> None:
    steps = [ReactStep(thought="t", action="research", action_input={}, observation="x" * 500)]
    line = _completed_work(steps)[0]
    assert line.endswith("...")
    assert len(line) < 200


def test_the_interruption_message_summarises_stops_and_defers_to_the_user() -> None:
    steps = [
        ReactStep(thought="t", action="research", action_input={}, observation="6 vendor results")
    ]
    msg = AgenticCore._evaluator_block_message(_decision("deny", ("cost_budget", "halt")), steps)

    assert "stopped partway" in msg.lower()
    assert "What I did before stopping:" in msg
    assert "research - 6 vendor results" in msg
    assert "Why I stopped:" in msg
    assert "Nothing further has run" in msg  # and it does not retry on its own
    assert "how you'd like to proceed" in msg  # the user decides


def test_the_message_works_when_nothing_ran_yet() -> None:
    msg = AgenticCore._evaluator_block_message(_decision("deny", ("goal_drift", "halt")))
    assert "What I did before stopping:" not in msg
    assert "Why I stopped:" in msg


# ── the streaming loop honours it ─────────────────────────────────────────────


class _HaltOnSignal:
    """A real PostStep hook whose verdict names ``signal``, as the evaluator's does."""

    name = "halt_on_signal"
    hook_point = HookPoint.POST_STEP
    priority = 10

    def __init__(self, signal: str) -> None:
        self._signal = signal

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return _decision("deny", (self._signal, "halt"))


def _kernel_halting_on(signal: str) -> GovernanceKernel:
    kernel = GovernanceKernel(audit_log=None)
    kernel.register(_HaltOnSignal(signal))
    kernel.init_lock()
    return kernel


class _ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def __call__(self, _prompt: str) -> str:
        if not self._responses:
            return "Thought: done\nFinal Answer: finished"
        return self._responses.pop(0)


def _research_tool() -> ToolSpec:
    return ToolSpec(name="research", description="search", call=lambda _a: "6 vendor results")


def _core(signal: str, responses: list[str]) -> AgenticCore:
    return AgenticCore(
        config=AgenticCoreConfig(max_iterations=4),
        llm_call=_ScriptedLLM(responses),
        tools=[_research_tool()],
        kernel=_kernel_halting_on(signal),
        agent_type="chat",
    )


_TOOL_STEP = 'Thought: looking\nAction: research\nAction Input: {"q": "vendors"}'


def test_streaming_halts_and_summarises_on_an_enforced_signal() -> None:
    core = _core("cost_budget", [_TOOL_STEP, _TOOL_STEP])

    chunks = list(core.run_stream("find vendors"))
    text = [c for c in chunks if isinstance(c, str)]
    meta = [c for c in chunks if isinstance(c, dict)]

    assert text, "the halt must reach the user"
    assert "stopped partway" in text[-1].lower()
    assert "research - 6 vendor results" in text[-1]  # what ran, before it stopped
    assert "how you'd like to proceed" in text[-1]  # the user decides
    assert meta and meta[-1]["reason"] == "evaluator_cost_budget"
    assert meta[-1]["success"] is False


def test_streaming_does_not_halt_on_a_locally_guarded_signal() -> None:
    """step_cap halting here would be a second mechanism stopping the same run at a
    different threshold from `max_iterations` — the duplication the owner ruled out."""
    core = _core("step_cap", [_TOOL_STEP, _TOOL_STEP, _TOOL_STEP, _TOOL_STEP])

    chunks = list(core.run_stream("find vendors"))
    text = [c for c in chunks if isinstance(c, str)]
    meta = [c for c in chunks if isinstance(c, dict)]

    assert not any("stopped partway" in c.lower() for c in text)
    # It ends on a local guard instead, never on the evaluator.
    assert meta and not str(meta[-1]["reason"]).startswith("evaluator_")


def test_the_sync_loop_summarises_the_same_way() -> None:
    """Coverage rule: both loops interrupt with the same explanation."""
    core = _core("goal_drift", [_TOOL_STEP, _TOOL_STEP])

    trace = core.run("find vendors")

    assert trace.halted_by == "evaluator"
    assert "stopped partway" in trace.final_answer.lower()
    assert "research - 6 vendor results" in trace.final_answer


# ── the halt message names the run ────────────────────────────────────────────
#
# Found from a real halted web turn (session `web-6d670ccd`, run `b7c928e2`): the
# message closed with `iris run resume <run_id>` as a literal placeholder, so the one
# fact needed to act on it was the one fact withheld — and the command it named does
# not re-enter the loop on any channel anyway.


def test_the_interruption_message_states_the_run_id() -> None:
    msg = AgenticCore._evaluator_block_message(
        _decision("require_approval", ("goal_drift", "require_approval")),
        (),
        run_id="b7c928e2-5a09-41ee-b771-d0322391c3b3",
    )
    assert "b7c928e2-5a09-41ee-b771-d0322391c3b3" in msg
    assert "<run_id>" not in msg


def test_the_interruption_message_does_not_promise_a_resume_that_does_not_exist() -> None:
    """`iris run resume` verifies side effects against the ledger; it does not
    re-enter the loop. Advising it read as a resume path that was never there."""
    msg = AgenticCore._evaluator_block_message(
        _decision("deny", ("cost_budget", "halt")), (), run_id="r-1"
    )
    assert "iris run resume" not in msg
    assert "how you'd like to proceed" in msg  # the channel-neutral action survives


def test_the_message_omits_the_run_line_when_there_is_no_run_id() -> None:
    msg = AgenticCore._evaluator_block_message(_decision("deny", ("goal_drift", "halt")))
    assert "Run ID:" not in msg
    assert "Nothing further has run" in msg
