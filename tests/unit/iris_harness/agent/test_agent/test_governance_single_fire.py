"""Governance fires exactly once per ReAct LLM call (no double firing).

Phase 1 instrumentation found the ReAct path fired PRE_CLASSIFY +
PRE_LLM_CALL twice per prompt: once from ``AgenticCore._loop`` and again
inside ``CodingLLMClient.invoke``. The fix routes the ReAct client through
``governance_handled_upstream=True``; these tests pin the single-fire
behavior end to end against a real kernel + audit log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from iris_harness.agent.agentic_core import AgenticCore, AgenticCoreConfig
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.llm.client import CodingLLMClient, CodingLLMConfig


class _AllowHook:
    priority: int = 10

    def __init__(self, name: str, hook_point: HookPoint) -> None:
        self.name = name
        self.hook_point = hook_point

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="test")


class _FinalAnswerMessage:
    content = "Thought: simple.\nFinal Answer: ok"
    tool_calls = None
    additional_kwargs: dict[str, Any] = {}
    usage_metadata = None
    response_metadata: dict[str, Any] = {}


class _StubModel:
    def invoke(self, messages: Any, **kwargs: Any) -> _FinalAnswerMessage:
        return _FinalAnswerMessage()


def _kernel(tmp_path: Path) -> tuple[GovernanceKernel, AuditLog]:
    log = AuditLog(db_path=tmp_path / "audit.db")
    kernel = GovernanceKernel(audit_log=log)
    kernel.register(_AllowHook("allow_classify", HookPoint.PRE_CLASSIFY))
    kernel.register(_AllowHook("allow_llm", HookPoint.PRE_LLM_CALL))
    kernel.init_lock()
    return kernel, log


def _react_llm_call() -> Any:
    """Mirror bootstrap._make_react_handler._llm_call wiring."""
    client = CodingLLMClient(
        CodingLLMConfig(provider="ollama", model="stub", tier_name="tier1"),
        model_factory=lambda **kwargs: _StubModel(),
        governance_handled_upstream=True,
    )

    def call(prompt: str) -> str:
        return client.invoke(system_prompt="", user_prompt=prompt)

    return call


def test_react_run_fires_each_hook_point_exactly_once(tmp_path: Path) -> None:
    kernel, log = _kernel(tmp_path)
    core = AgenticCore(
        config=AgenticCoreConfig(max_iterations=3, timeout_seconds=30),
        llm_call=_react_llm_call(),
        tools=[],
        kernel=kernel,
        target_tier="tier_1",
        agent_type="system",
    )

    trace = core.run("say ok")

    assert trace.success is True
    rows = log.query()
    llm_rows = [r for r in rows if r.hook_point == "pre_llm_call"]
    classify_rows = [r for r in rows if r.hook_point == "pre_classify"]
    # One LLM call in the loop -> exactly one pre_llm_call and one
    # pre_classify. Before the fix this was 2 + 2 (core + client).
    assert len(llm_rows) == 1, [(r.hook_point, r.agent_type) for r in rows]
    assert len(classify_rows) == 1
    # And the single firing is the core's, carrying its agent_type.
    assert llm_rows[0].agent_type == "system"


def test_client_opt_out_disables_internal_hooks_even_with_env(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_ENABLED", "1")
    client = CodingLLMClient(
        CodingLLMConfig(provider="ollama", model="stub", tier_name="tier2"),
        model_factory=lambda **kwargs: _StubModel(),
        governance_handled_upstream=True,
    )
    assert client._governance_kernel is None
    # Invocation works and produces content without any kernel.
    assert "ok" in client.invoke(system_prompt="", user_prompt="hi")


def test_default_client_still_governs_itself(tmp_path: Path) -> None:
    """The opt-out is opt-in: a default client keeps its internal hooks."""
    kernel, log = _kernel(tmp_path)
    client = CodingLLMClient(
        CodingLLMConfig(provider="ollama", model="stub", tier_name="tier2"),
        model_factory=lambda **kwargs: _StubModel(),
        governance_kernel=kernel,
    )
    client.invoke(system_prompt="", user_prompt="hi")
    rows = log.query()
    assert [r.hook_point for r in rows] == ["pre_classify", "pre_llm_call"]
