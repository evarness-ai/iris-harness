"""The loop reports what it changed (ADR-0118 decision 5).

Escalation re-runs a turn, so it has to know whether the turn wrote anything. The
loop records the declared effect of every non-read tool it invoked, on both the
sync and the streaming path, before the call runs (a write that raised may still
have changed something). A write governance held back never ran, so it does not
count.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from iris_harness.agent.agentic_core import AgenticCore, AgenticCoreConfig, ToolSpec
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.plugins import ToolPolicyHook

_READ = 'Thought: look\nAction: lookup\nAction Input: {"q": "rent"}'
_WRITE = 'Thought: save\nAction: save_note\nAction Input: {"text": "rent"}'
_FINAL = "Thought: done\nFinal Answer: Done."


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


def _kernel(tmp_path: Path) -> GovernanceKernel:
    kernel = GovernanceKernel(audit_log=AuditLog(db_path=tmp_path / "audit.db"))
    kernel.register(_AllowHook("allow_classify", HookPoint.PRE_CLASSIFY))
    kernel.register(_AllowHook("allow_llm", HookPoint.PRE_LLM_CALL))
    kernel.register(ToolPolicyHook())
    kernel.init_lock()
    return kernel


def _tools(*, confirm: str = "never", fail: bool = False) -> list[ToolSpec]:
    def _save(args: dict[str, Any]) -> str:
        if fail:
            raise RuntimeError("disk full")
        return "saved"

    return [
        ToolSpec(name="lookup", description="read", call=lambda a: "rent is due"),
        ToolSpec(
            name="save_note", description="write", call=_save, effect="write", confirm=confirm
        ),
    ]


def _core(tmp_path: Path, responses: list[str], **tool_kw: Any) -> AgenticCore:
    return AgenticCore(
        config=AgenticCoreConfig(max_iterations=6),
        llm_call=_ScriptedLLM(responses),
        tools=_tools(**tool_kw),
        kernel=_kernel(tmp_path),
        session_id="s1",
        agent_type="chat",
    )


def _stream_effects(core: AgenticCore, query: str) -> object:
    final = [c for c in core.run_stream(query) if isinstance(c, dict)][-1]
    return final["effects_executed"]


def test_a_read_only_run_reports_nothing_changed(tmp_path: Path) -> None:
    assert _core(tmp_path, [_READ, _FINAL]).run("rent?").effects_executed == []
    assert _stream_effects(_core(tmp_path, [_READ, _FINAL]), "rent?") == []


def test_an_executed_write_is_reported_on_both_paths(tmp_path: Path) -> None:
    assert _core(tmp_path, [_READ, _WRITE, _FINAL]).run("note it").effects_executed == ["write"]
    assert _stream_effects(_core(tmp_path, [_READ, _WRITE, _FINAL]), "note it") == ["write"]


def test_a_write_that_raised_still_counts(tmp_path: Path) -> None:
    trace = _core(tmp_path, [_WRITE, _FINAL], fail=True).run("note it")
    assert "Tool error" in (trace.steps[0].observation or "")
    assert trace.effects_executed == ["write"]


def test_a_write_governance_held_back_does_not_count(tmp_path: Path) -> None:
    # confirm: once without asking first: the tool policy turns it back unrun.
    trace = _core(tmp_path, [_WRITE, _FINAL], confirm="once").run("note it")
    assert (trace.steps[0].observation or "").startswith("Request needs approval by governance")
    assert trace.effects_executed == []
