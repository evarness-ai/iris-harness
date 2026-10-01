"""Phase 3 chat resume: PostStep halt → checkpoint → resume_from_checkpoint."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.agentic_core import AgenticCore, AgenticCoreConfig, ToolSpec
from iris_harness.kernel.governance import (
    GovernanceKernel,
    HookContext,
    HookDecision,
    HookPoint,
)
from iris_harness.memory.state import CheckpointStore

# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------


class _ScriptedLLM:
    """Yields each scripted response on successive calls."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self._responses:
            return "Thought: idle\nFinal Answer: done"
        return self._responses.pop(0)


class _HaltOnStep:
    """PostStep hook that halts when step_id reaches ``halt_at``."""

    name = "halt_on_step"
    hook_point = HookPoint.POST_STEP
    priority = 10

    def __init__(self, halt_at: int) -> None:
        self.halt_at = halt_at

    async def __call__(self, ctx: HookContext) -> HookDecision:
        if ctx.step_id is not None and ctx.step_id >= self.halt_at:
            return HookDecision(
                outcome="deny",
                reason=f"halted at step {ctx.step_id}",
                severity="warn",
            )
        return HookDecision(outcome="allow", reason="ok")


@pytest.fixture()
def checkpoint_store(tmp_path: Path) -> Iterator[CheckpointStore]:
    yield CheckpointStore(db_path=tmp_path / "checkpoints.db")


def _kernel_with_halt(halt_at: int) -> GovernanceKernel:
    kernel = GovernanceKernel(audit_log=None)
    kernel.register(_HaltOnStep(halt_at=halt_at))
    kernel.init_lock()
    return kernel


def _echo_tool() -> ToolSpec:
    def call(args: dict[str, Any]) -> str:
        return f"observed: {args.get('q', '?')}"

    return ToolSpec(name="echo", description="echo", call=call)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_writes_checkpoint_on_evaluator_halt(checkpoint_store: CheckpointStore) -> None:
    llm = _ScriptedLLM(
        [
            'Thought: try the tool\nAction: echo\nAction Input: {"q": "first"}',
            'Thought: try again\nAction: echo\nAction Input: {"q": "second"}',
        ]
    )
    core = AgenticCore(
        config=AgenticCoreConfig(max_iterations=10),
        llm_call=llm,
        tools=[_echo_tool()],
        kernel=_kernel_with_halt(halt_at=0),
        checkpoint_store=checkpoint_store,
        agent_type="chat",
    )

    trace = core.run("hello")

    assert trace.halted_by == "evaluator"
    assert "halted at step 0" in (trace.halt_reason or "")
    assert trace.checkpoint_id is not None
    # ADR-0107: an interrupted run summarises what ran and defers to the user,
    # rather than returning a bare verdict.
    assert "stopped partway" in trace.final_answer.lower()
    assert "halted by the evaluator" in trace.final_answer
    assert "echo" in trace.final_answer  # the work done before stopping
    assert trace.success is False

    cp = checkpoint_store.get_latest(trace.run_id)
    assert cp.signal == "halt"
    payload = cp.payload
    assert payload["query"] == "hello"
    assert payload["iteration"] == 1
    assert len(payload["steps"]) == 1
    assert payload["steps"][0]["action"] == "echo"


def test_resume_from_checkpoint_continues_loop(checkpoint_store: CheckpointStore) -> None:
    # First run halts at step 0.
    llm_first = _ScriptedLLM(
        [
            'Thought: A\nAction: echo\nAction Input: {"q": "x"}',
        ]
    )
    core_first = AgenticCore(
        config=AgenticCoreConfig(max_iterations=10),
        llm_call=llm_first,
        tools=[_echo_tool()],
        kernel=_kernel_with_halt(halt_at=0),
        checkpoint_store=checkpoint_store,
    )
    first_trace = core_first.run("hello")
    assert first_trace.checkpoint_id is not None
    checkpoint = checkpoint_store.get_latest(first_trace.run_id)

    # Second run resumes — kernel now lets everything through, LLM returns a
    # final answer on the next step.
    llm_second = _ScriptedLLM(["Thought: wrapping up\nFinal Answer: all done"])
    permissive = GovernanceKernel(audit_log=None)
    permissive.init_lock()
    core_second = AgenticCore(
        config=AgenticCoreConfig(max_iterations=10),
        llm_call=llm_second,
        tools=[_echo_tool()],
        kernel=permissive,
        checkpoint_store=checkpoint_store,
    )

    resumed = core_second.resume_from_checkpoint(checkpoint)

    assert resumed.run_id == first_trace.run_id
    assert resumed.success is True
    assert resumed.final_answer == "all done"
    # The replayed step is preserved in the new trace.
    assert len(resumed.steps) == 2
    assert resumed.steps[0].action == "echo"
    assert resumed.steps[1].final_answer == "all done"
    # The continuation invoked the LLM exactly once for the new step.
    assert len(llm_second.calls) == 1


def test_run_without_kernel_or_store_is_unchanged() -> None:
    """Legacy callers (no kernel, no store) keep the pre-Phase-3 behavior."""
    llm = _ScriptedLLM(["Thought: easy\nFinal Answer: hi back"])
    core = AgenticCore(
        config=AgenticCoreConfig(max_iterations=3),
        llm_call=llm,
        tools=[],
    )
    trace = core.run("hi")
    assert trace.success is True
    assert trace.final_answer == "hi back"
    assert trace.halted_by is None
    assert trace.checkpoint_id is None
    assert trace.run_id  # a uuid was still generated for tracing


def test_post_step_require_approval_writes_pending_checkpoint(
    checkpoint_store: CheckpointStore,
) -> None:
    class _RequireApproval:
        name = "needs_approval"
        hook_point = HookPoint.POST_STEP
        priority = 10

        async def __call__(self, ctx: HookContext) -> HookDecision:
            return HookDecision(outcome="require_approval", reason="check with human")

    kernel = GovernanceKernel(audit_log=None)
    kernel.register(_RequireApproval())
    kernel.init_lock()

    llm = _ScriptedLLM(['Thought: A\nAction: echo\nAction Input: {"q": "x"}'])
    core = AgenticCore(
        llm_call=llm,
        tools=[_echo_tool()],
        kernel=kernel,
        checkpoint_store=checkpoint_store,
    )
    trace = core.run("hi")

    assert trace.halted_by == "evaluator"
    assert trace.checkpoint_id is not None
    cp = checkpoint_store.get_latest(trace.run_id)
    assert cp.signal == "require_approval"
    assert "paused for approval" in trace.final_answer
