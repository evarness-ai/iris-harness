"""A destructive call with no way to approve it is refused (ADR-0118).

A destructive call runs only as the pinned item of an approved approval
(test_destructive_approval_loop.py). Where no approval can be taken, the loop refuses
it: with no governance kernel (before governance is consulted), and with a kernel that
lacks the approval hook (governance allows it, but nobody approved it). It is refused
after alias resolution, so a near-miss name cannot slip past. Nothing runs and nothing
is recorded as changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from iris_harness.agent.agentic_core import (
    AgenticCore,
    AgenticCoreConfig,
    ToolSpec,
    _build_react_prompt,
)
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.plugins import ToolPolicyHook

_FINAL = "Thought: done\nFinal Answer: I did not delete anything."


def _call(name: str) -> str:
    return f'Thought: delete it\nAction: {name}\nAction Input: {{"id": "m1"}}'


class _AllowHook:
    priority: int = 10

    def __init__(self, name: str, hook_point: HookPoint) -> None:
        self.name = name
        self.hook_point = hook_point

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="test")


class _ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def __call__(self, prompt: str) -> str:
        return self._responses.pop(0) if self._responses else _FINAL


class _Trash:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def tool(self) -> ToolSpec:
        def _call(args: dict[str, Any]) -> str:
            self.calls.append(dict(args))
            return "trashed"

        return ToolSpec(
            name="trash_email",
            description="Move an email to the trash.",
            call=_call,
            effect="destructive",
            confirm="approval",
        )


def _kernel(tmp_path: Path) -> GovernanceKernel:
    kernel = GovernanceKernel(audit_log=AuditLog(db_path=tmp_path / "audit.db"))
    kernel.register(_AllowHook("allow_classify", HookPoint.PRE_CLASSIFY))
    kernel.register(_AllowHook("allow_llm", HookPoint.PRE_LLM_CALL))
    kernel.register(ToolPolicyHook())
    kernel.init_lock()
    return kernel


def _core(trash: _Trash, responses: list[str], kernel: GovernanceKernel | None) -> AgenticCore:
    return AgenticCore(
        config=AgenticCoreConfig(max_iterations=4),
        llm_call=_ScriptedLLM(responses),
        tools=[trash.tool()],
        kernel=kernel,
        session_id="s1",
        agent_type="chat",
    )


def test_a_destructive_call_is_refused_and_never_runs(tmp_path: Path) -> None:
    trash = _Trash()
    trace = _core(trash, [_call("trash_email"), _FINAL], _kernel(tmp_path)).run("delete it")

    assert trash.calls == []
    assert (trace.steps[0].observation or "").startswith("Refused: 'trash_email'")
    assert "Nothing was changed" in (trace.steps[0].observation or "")
    assert trace.effects_executed == []  # it did not run, so nothing changed


def test_the_refusal_holds_with_governance_off() -> None:
    trash = _Trash()
    trace = _core(trash, [_call("trash_email"), _FINAL], kernel=None).run("delete it")
    assert trash.calls == []
    assert (trace.steps[0].observation or "").startswith("Refused:")


def test_a_near_miss_name_is_refused_too(tmp_path: Path) -> None:
    trash = _Trash()
    trace = _core(trash, [_call("trash_email_now"), _FINAL], _kernel(tmp_path)).run("delete it")
    assert trash.calls == []
    assert (trace.steps[0].observation or "").startswith("Refused: 'trash_email'")


def test_the_streaming_loop_refuses_it_too(tmp_path: Path) -> None:
    trash = _Trash()
    chunks = list(_core(trash, [_call("trash_email"), _FINAL], _kernel(tmp_path)).run_stream("x"))
    assert trash.calls == []
    final = [c for c in chunks if isinstance(c, dict)][-1]
    assert final["effects_executed"] == []
    assert "Refused:" in str(final["trace"])


def test_the_prompt_marks_a_destructive_tool() -> None:
    prompt = _build_react_prompt("delete it", [_Trash().tool()], [], None)
    assert "trash_email (DELETES or OVERWRITES the user's data" in prompt
    # The confirm-once rule is for writes; a destructive tool alone does not bring it.
    assert "ask the user ONCE" not in prompt
