"""ADR-0118 through a real runtime: halt, answer, resume — only the model is scripted.

The runtime is built for real (governance on by default, the approval queue and its
router wired by ``build_default_kernel``), a destructive tool joins the loop through
the plugin registry the way a plugin's would, and the owner answers through
``respond_to_approval`` — the one capability the API, the CLI and Telegram all call —
with the runtime as the resumer, so the resumed turn runs the real turn pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.agent_executor import AgentTask
from iris_harness.agent.agentic_core import ToolSpec
from iris_harness.kernel.governance.approvals import ApprovalQueue
from iris_harness.kernel.governance.approvals.service import respond_to_approval
from iris_harness.runtime import build_runtime
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginStatus

pytestmark = pytest.mark.integration

_TRASH = 'Thought: trash them\nAction: purge_notes\nAction Input: {"ids": ["m1", "m2"]}'


def _model(self: Any, *, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
    """Proposes the delete once; after the approval is settled, says what happened."""
    if "The owner approved. Results:" in user_prompt:
        return "Thought: done\nFinal Answer: Trashed the two promos."
    if "The owner rejected this" in user_prompt:
        return "Thought: ok\nFinal Answer: Understood, I did not delete them."
    if "purge_notes" in user_prompt:
        return _TRASH
    return "Thought: nothing to do\nFinal Answer: ok"


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from iris_harness.llm.client import CodingLLMClient

    monkeypatch.delenv("IRIS_GOVERNANCE_ENABLED", raising=False)  # on by default
    monkeypatch.setattr(CodingLLMClient, "invoke", _model)
    config_dir, data_dir = tmp_path / "config", tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()
    runtime = build_runtime(
        config_dir=config_dir, data_dir=data_dir, use_background_scheduler=False
    )

    trashed: list[list[str]] = []

    def _trash(args: dict[str, Any]) -> str:
        trashed.append(list(args["ids"]))
        return f"trashed {len(args['ids'])}"

    runtime.plugin_registry.add_plugin(
        PluginRecord(name="mailbox", source="test", status=PluginStatus.LOADED)
    )
    runtime.plugin_registry.add_tool(
        "mailbox",
        ToolSpec(
            name="purge_notes",
            description="Remove notes. Takes a list of ids.",
            call=_trash,
            effect="destructive",
            confirm="approval",
        ),
    )
    return runtime, trashed


def _halt(runtime: Any) -> Any:
    result = runtime.agent_executor.execute(
        AgentTask(query="trash the two promos", agent_type="system", session_id="web-s1")
    )
    approval_id = result.metadata.get("pending_approval_id")
    assert approval_id, result.output
    return result, approval_id


@pytest.mark.minilm
def test_the_turn_halts_on_an_approval_the_owner_can_answer(world: Any) -> None:
    runtime, trashed = world
    result, approval_id = _halt(runtime)

    assert trashed == []
    assert "needs your approval, so nothing has changed yet" in result.output
    row = ApprovalQueue().get(approval_id)
    assert row is not None and row.status == "pending"
    assert row.checkpoint_id  # linked, so answering can resume it
    assert '"ids": ["m1", "m2"]' in row.context_summary


@pytest.mark.minilm
def test_approving_resumes_the_turn_and_runs_the_pinned_call(world: Any) -> None:
    runtime, trashed = world
    _result, approval_id = _halt(runtime)

    outcome = respond_to_approval(
        approval_id, status="approved", actor="test:owner", resumer=runtime
    )

    assert outcome.resumed is True
    assert trashed == [["m1", "m2"]]
    assert "Trashed the two promos." in outcome.detail


@pytest.mark.minilm
def test_rejecting_resumes_the_turn_and_runs_nothing(world: Any) -> None:
    runtime, trashed = world
    _result, approval_id = _halt(runtime)

    outcome = respond_to_approval(
        approval_id, status="rejected", actor="test:owner", resumer=runtime
    )

    assert outcome.resumed is True
    assert trashed == []
    assert "did not delete them" in outcome.detail
