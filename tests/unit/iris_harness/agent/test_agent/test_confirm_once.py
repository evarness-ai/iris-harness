"""The loop hands PreToolUse the run's own ask_user evidence (loop plan, decision 8) and
the tool's own declaration (ADR-0110): the kernel is told no tool name.

A write tool under ``confirm_once_tools`` is turned back with "ask first" until the
run has proposed that write and then paused on ``ask_user``; after the user's answer
the resumed run carries both steps, so the writes that follow pass without asking
again. A clarifying question asked before any write was proposed is not consent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from iris_harness.agent.agentic_core import (
    ASK_USER_ACTION,
    AgenticCore,
    AgenticCoreConfig,
    ReactStep,
    ToolSpec,
    _run_asked_user,
    resume_seed_from_checkpoint,
)
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.plugins import ToolPolicyHook
from iris_harness.memory.state import CheckpointStore

_ASK = (
    "Thought: I should confirm the list first\n"
    'Action: ask_user\nAction Input: {"question": "Create a reminder for rent on 2026-09-30?"}'
)
_CLARIFY = (
    "Thought: unclear what they mean\n"
    'Action: ask_user\nAction Input: {"question": "What task are you checking for 5 PM?"}'
)
_WRITE = (
    "Thought: creating it\n"
    'Action: create_reminder\nAction Input: {"task": "pay rent", "date": "2026-09-30"}'
)
_WRITE_2 = (
    "Thought: and the next one\n"
    'Action: create_reminder\nAction Input: {"task": "pay AT&T", "date": "2026-10-05"}'
)
_FINAL = "Thought: done\nFinal Answer: Reminder set."


class _AllowHook:
    priority: int = 10

    def __init__(self, name: str, hook_point: HookPoint) -> None:
        self.name = name
        self.hook_point = hook_point

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="test")


class _Writes:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def tool(self) -> ToolSpec:
        def _call(args: dict[str, Any]) -> str:
            self.calls.append(dict(args))
            return "Reminder set: pay rent"

        # ADR-0110: the declaration rides on the ToolSpec; the kernel is told no name.
        return ToolSpec(
            name="create_reminder", description="write", call=_call, effect="write", confirm="once"
        )


class _ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def __call__(self, prompt: str) -> str:
        return self._responses.pop(0) if self._responses else _FINAL


def _kernel(tmp_path: Path) -> GovernanceKernel:
    kernel = GovernanceKernel(audit_log=AuditLog(db_path=tmp_path / "audit.db"))
    kernel.register(_AllowHook("allow_classify", HookPoint.PRE_CLASSIFY))
    kernel.register(_AllowHook("allow_llm", HookPoint.PRE_LLM_CALL))
    kernel.register(ToolPolicyHook())
    kernel.init_lock()
    return kernel


def _core(
    tmp_path: Path, responses: list[str], writes: _Writes, *, store: CheckpointStore | None = None
) -> AgenticCore:
    return AgenticCore(
        config=AgenticCoreConfig(max_iterations=6, allow_ask_user=True),
        llm_call=_ScriptedLLM(responses),
        tools=[writes.tool()],
        kernel=_kernel(tmp_path),
        checkpoint_store=store,
        session_id="s1",
        agent_type="chat",
    )


def test_run_asked_user_reads_the_step_list() -> None:
    write = ReactStep(thought="t", action="create_reminder")
    ask = ReactStep(thought="t", action=ASK_USER_ACTION)
    assert _run_asked_user([], "create_reminder") is False
    assert _run_asked_user([write], "create_reminder") is False
    # Only an ask AFTER the write was proposed is the confirmation.
    assert _run_asked_user([write, ask], "create_reminder") is True
    assert _run_asked_user([ask, write], "create_reminder") is False
    assert _run_asked_user([ask], "create_reminder") is False
    # Consent to one write is not consent to another.
    assert _run_asked_user([write, ask], "delete_event") is False


def test_a_write_before_asking_is_turned_back_and_never_runs(tmp_path: Path) -> None:
    writes = _Writes()
    core = _core(tmp_path, [_WRITE, _FINAL], writes)

    trace = core.run("remind me about rent")

    assert writes.calls == []
    blocked = trace.steps[0].observation or ""
    assert blocked.startswith("Request needs approval by governance")
    assert "ask_user" in blocked
    assert trace.final_answer == "Reminder set."


def test_writes_pass_after_the_user_answers(tmp_path: Path) -> None:
    store = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    writes = _Writes()
    # Proposed (held), then asked: the confirmation the held call asked for.
    paused = _core(tmp_path, [_WRITE, _ASK], writes, store=store).run("remind me about rent")
    assert paused.halted_by == "ask_user"
    assert writes.calls == []

    seed = resume_seed_from_checkpoint(store.get_latest(paused.run_id), user_reply="yes, go ahead")
    resumed = _core(tmp_path, [_WRITE, _WRITE_2, _FINAL], writes, store=store).run_from_seed(seed)

    # Both writes ran (a fan-out of two), and the run asked exactly once.
    assert [c["task"] for c in writes.calls] == ["pay rent", "pay AT&T"]
    assert resumed.final_answer == "Reminder set."


def test_a_clarifying_question_is_not_consent_to_write(tmp_path: Path) -> None:
    """The 2026-09-15 session: "how about 5 PM?" -> "What task…?" -> "Python training"
    went straight to a reminder, because any earlier ask_user counted as consent."""
    store = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    writes = _Writes()
    paused = _core(tmp_path, [_CLARIFY], writes, store=store).run("how about 5 PM?")
    assert paused.halted_by == "ask_user"

    seed = resume_seed_from_checkpoint(
        store.get_latest(paused.run_id), user_reply="Python training"
    )
    resumed = _core(tmp_path, [_WRITE, _FINAL], writes, store=store).run_from_seed(seed)

    assert writes.calls == []
    held = next(s for s in resumed.steps if s.action == "create_reminder")
    assert (held.observation or "").startswith("Request needs approval by governance")


def test_streaming_loop_supplies_the_same_evidence(tmp_path: Path) -> None:
    writes = _Writes()
    core = _core(tmp_path, [_WRITE, _FINAL], writes)
    chunks = list(core.run_stream("remind me about rent"))
    assert writes.calls == []
    assert any("ask_user" in str(getattr(c, "text", c)) for c in chunks)
