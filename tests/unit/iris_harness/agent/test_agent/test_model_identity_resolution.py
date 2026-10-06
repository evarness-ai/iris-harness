"""The step's model lookup for its audit row may fail softly, but only softly (#105).

``AgenticCore(model_identity=...)`` asks which model the next step calls so its
``PRE_LLM_CALL`` row can name it. A recoverable lookup failure (an unknown tier, an
unreadable config) costs the row its ``model``, logs one warning and leaves the turn
running; a programming error (a ``TypeError``) is not swallowed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.agentic_core import AgenticCore, AgenticCoreConfig
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.audit import AuditLog


class _Allow:
    priority = 10

    def __init__(self, point: HookPoint) -> None:
        self.name = f"allow_{point.value}"
        self.hook_point = point

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="test")


def _core(tmp_path: Path, identity: Any) -> tuple[AgenticCore, AuditLog]:
    log = AuditLog(db_path=tmp_path / "audit.db")
    kernel = GovernanceKernel(audit_log=log)
    for point in (HookPoint.PRE_CLASSIFY, HookPoint.PRE_LLM_CALL):
        kernel.register(_Allow(point))
    kernel.init_lock()
    core = AgenticCore(
        config=AgenticCoreConfig(max_iterations=3, timeout_seconds=30),
        llm_call=lambda prompt: "Thought: done.\nFinal Answer: ok",
        tools=[],
        kernel=kernel,
        target_tier="tier_1",
        agent_type="system",
        model_identity=identity,
    )
    return core, log


@pytest.mark.parametrize("error", [KeyError("tier9"), OSError("unreadable"), ValueError("bad")])
def test_a_recoverable_lookup_failure_costs_the_row_its_model_not_the_turn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, error: Exception
) -> None:
    def broken() -> tuple[str, str]:
        raise error

    core, log = _core(tmp_path, broken)
    with caplog.at_level(logging.WARNING, logger="iris_harness.agent.agentic_core"):
        trace = core.run("say ok")

    assert trace.success is True
    rows = [r for r in log.query() if r.hook_point == "pre_llm_call"]
    assert len(rows) == 1
    payload = json.loads(rows[0].payload_json)
    assert "model" not in payload and "provider" not in payload
    warnings = [r for r in caplog.records if "model for the audit row" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.parametrize("error", [TypeError("wrong config type"), AttributeError("no model")])
def test_a_programming_error_is_not_swallowed(tmp_path: Path, error: Exception) -> None:
    def buggy() -> tuple[str, str]:
        raise error

    core, _ = _core(tmp_path, buggy)
    with pytest.raises(type(error)):
        core.run("say ok")
