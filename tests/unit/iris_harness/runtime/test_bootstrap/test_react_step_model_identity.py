"""A ReAct step's audit row names the model it called, and the router is asked once (#105).

The loop's ``PRE_LLM_CALL`` row used to carry a model snapshot taken when the loop was
built, while each step's call asked the tier router again. When the router's answer moves
between steps (a governor downshift under host pressure, a runtime tier edit) the row
named a model that was not the one called, and every step cost a second resolve (a second
``governor.acquire``, a second arbiter eviction). The step now resolves its model once and
the row and the call share that resolution.

Driven through the real handler and the real ``AgenticCore`` (``chat`` and ``chat_stream``
share it); only the tier router (it answers a different model every time it is asked) and
the model transport are stubs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import iris_harness.llm.client as llm_client
from iris_harness.agent.agent_executor import AgentTask
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.llm.client import CodingLLMClient, CodingLLMConfig
from iris_harness.runtime.handlers import react
from iris_harness.tools.skills.registry import SkillRegistry


class _Allow:
    priority = 10

    def __init__(self, point: HookPoint) -> None:
        self.name = f"allow_{point.value}"
        self.hook_point = point

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="test")


class _MovingRouter:
    """Answers a different model every time it is asked, and counts the questions."""

    def __init__(self) -> None:
        self.resolves = 0

    def get_llm_config(self, intent: str) -> CodingLLMConfig:
        self.resolves += 1
        return CodingLLMConfig(
            provider="ollama",
            model=f"model-{self.resolves}",
            base_url="http://localhost",
            api_key_env="STUB",
            temperature=0.0,
            max_tokens=64,
            timeout_seconds=10,
        )


class _EmptyRegistry(SkillRegistry):
    def __init__(self) -> None:
        super().__init__(repo_root=Path("/nonexistent"))

    def discover(self) -> tuple[Any, ...]:
        return ()


def _turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, stream: bool, steps: int
) -> tuple[list[str], list[str], int]:
    """Run a turn of ``steps`` model calls; (models named on rows, models called, resolves)."""
    audit = AuditLog(db_path=tmp_path / f"audit-{steps}-{stream}.db")
    kernel = GovernanceKernel(audit_log=audit)
    for point in (HookPoint.PRE_CLASSIFY, HookPoint.PRE_LLM_CALL):
        kernel.register(_Allow(point))
    kernel.init_lock()
    monkeypatch.setattr("iris_harness.kernel.governance.kernel_from_env", lambda: kernel)
    monkeypatch.setattr(llm_client, "kernel_from_env", lambda: kernel)

    called: list[str] = []
    replies = ["Thought: x\nAction: no_such_tool\nAction Input: {}"] * (steps - 1) + [
        "Thought: done.\nFinal Answer: ok"
    ]

    def fake_invoke(self: CodingLLMClient, **_kw: Any) -> str:
        called.append(self.config.model)
        return replies[len(called) - 1]

    monkeypatch.setattr(CodingLLMClient, "invoke", fake_invoke)

    router = _MovingRouter()
    handler, stream_handler = react._make_react_handler(router, _EmptyRegistry())  # type: ignore[arg-type]
    task = AgentTask(query="say ok", agent_type="system")
    if stream:
        "".join(c for c in stream_handler(task) if isinstance(c, str))
    else:
        handler(task)
    rows = [r for r in audit.query() if r.hook_point == "pre_llm_call"]
    return [json.loads(r.payload_json)["model"] for r in rows], called, router.resolves


@pytest.mark.parametrize("stream", [False, True], ids=["chat", "chat_stream"])
def test_the_row_names_the_called_model_and_the_router_is_asked_once_per_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stream: bool
) -> None:
    named1, called1, resolves1 = _turn(monkeypatch, tmp_path, stream=stream, steps=1)
    named3, called3, resolves3 = _turn(monkeypatch, tmp_path, stream=stream, steps=3)

    assert len(called1) == 1 and len(called3) == 3
    assert named1 == called1  # each row names the model its step called
    assert named3 == called3

    # (The resolve count is pinned on its own below: a stable router would hide a mismatch.)


@pytest.mark.parametrize("stream", [False, True], ids=["chat", "chat_stream"])
def test_each_extra_step_costs_one_resolve_not_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stream: bool
) -> None:
    _, called1, resolves1 = _turn(monkeypatch, tmp_path, stream=stream, steps=1)
    _, called3, resolves3 = _turn(monkeypatch, tmp_path, stream=stream, steps=3)
    # One governor.acquire / arbiter eviction per step, not a second one for the call.
    assert resolves3 - resolves1 == len(called3) - len(called1)
