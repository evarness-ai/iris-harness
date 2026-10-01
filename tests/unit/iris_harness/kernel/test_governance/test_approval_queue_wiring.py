"""The approval queue, actually connected to the evaluator.

``EvaluatorHook.__init__`` has accepted ``approval_queue`` and ``channel_router``
since the queue shipped, and neither ``build_default_kernel`` site ever passed one. So
``self._approval_queue`` was always None, the ``require_approval`` branch never ran,
``approvals.db`` was never created, and a halt that said "paused for approval" had no
approval behind it to grant. These tests pin the three facts that were missing:

1. a ``require_approval`` verdict enqueues a row and notifies a channel;
2. the row carries the channel the **turn** came from, not a hardcoded "cli";
3. the row is linked to the checkpoint that answering it resumes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from iris_harness.agent.agentic_core import AgenticCore, AgenticCoreConfig, ToolSpec
from iris_harness.kernel.governance.approvals.queue import ApprovalQueue
from iris_harness.kernel.governance.approvals.store import ApprovalRow, ApprovalStore
from iris_harness.kernel.governance.evaluator import EvaluatorHook, EvaluatorRegistry
from iris_harness.kernel.governance.evaluator.types import SignalResult, StepRecord
from iris_harness.kernel.governance.kernel import GovernanceKernel
from iris_harness.memory.state import CheckpointStore


class _AlwaysApproval:
    """A signal that asks for approval on the first step and every step after."""

    name = "goal_drift"
    priority = 10

    def __call__(self, step: StepRecord, *, state: dict[str, Any]) -> SignalResult:
        return SignalResult(
            name=self.name,
            verdict="require_approval",
            reason="thought drifted from original task",
            severity="warn",
        )


class _RecordingRouter:
    """Stands in for ChannelRouter to record what would have been delivered."""

    def __init__(self) -> None:
        self.notified: list[ApprovalRow] = []

    def notify(self, approval: ApprovalRow) -> None:
        self.notified.append(approval)


def _core(
    tmp_path: Path, *, origin_channel: str = "console"
) -> tuple[AgenticCore, ApprovalStore, _RecordingRouter]:
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    queue = ApprovalQueue(store=store)
    router = _RecordingRouter()

    registry = EvaluatorRegistry()
    registry.register(_AlwaysApproval())
    registry.init_lock()
    kernel = GovernanceKernel(audit_log=None)
    kernel.register(EvaluatorHook(registry=registry, approval_queue=queue, channel_router=router))
    kernel.init_lock()

    core = AgenticCore(
        config=AgenticCoreConfig(max_iterations=3),
        llm_call=lambda _p: 'Thought: look it up\nAction: echo\nAction Input: {"q": "x"}',
        tools=[ToolSpec(name="echo", description="echo", call=lambda a: "observed")],
        kernel=kernel,
        checkpoint_store=CheckpointStore(db_path=tmp_path / "checkpoints.db"),
        session_id="web-6d670ccd",
        agent_type="system",
        origin_channel=origin_channel,
        link_approval_checkpoint=queue.set_checkpoint,
    )
    return core, store, router


# ── the connection itself ─────────────────────────────────────────────────────


def test_a_require_approval_halt_enqueues_an_approval(tmp_path: Path) -> None:
    core, store, _ = _core(tmp_path)

    core.run("analyse what is happening around the world")

    rows = store.list_pending()
    assert len(rows) == 1
    assert rows[0].signal == "goal_drift"
    assert "drifted" in rows[0].context_summary


def test_the_halt_notifies_a_channel(tmp_path: Path) -> None:
    core, _, router = _core(tmp_path)

    core.run("analyse what is happening around the world")

    assert len(router.notified) == 1


def test_the_halt_message_now_carries_the_approval_id(tmp_path: Path) -> None:
    """The `Approval ID:` line has been in the message all along, behind an
    `approval_request_id is not None` that nothing could ever satisfy."""
    core, store, _ = _core(tmp_path)

    trace = core.run("analyse what is happening around the world")

    approval_id = store.list_pending()[0].approval_id
    assert "Approval ID:" in (trace.final_answer or "")
    assert approval_id in (trace.final_answer or "")


# ── where it gets delivered ───────────────────────────────────────────────────


def test_the_row_records_the_channel_the_turn_came_from(tmp_path: Path) -> None:
    core, store, _ = _core(tmp_path, origin_channel="web")

    core.run("analyse what is happening around the world")

    assert store.list_pending()[0].channel == "web"


def test_a_caller_that_declares_no_channel_still_gets_the_old_default(
    tmp_path: Path,
) -> None:
    """Tests, the sandbox loop, anything driving the kernel directly."""
    core, store, _ = _core(tmp_path, origin_channel="")

    core.run("analyse what is happening around the world")

    assert store.list_pending()[0].channel == "cli"


# ── the link that makes approving actionable ───────────────────────────────────


def test_the_row_is_linked_to_the_checkpoint_that_resumes_it(tmp_path: Path) -> None:
    """`hook.py` enqueues at PostStep, before the checkpoint exists, and passes
    checkpoint_id=None. Nothing ever came back to fill it in, so every approval ever
    written pointed at nothing."""
    core, store, _ = _core(tmp_path, origin_channel="web")

    trace = core.run("analyse what is happening around the world")

    row = store.list_pending()[0]
    assert row.checkpoint_id is not None
    assert row.checkpoint_id == trace.checkpoint_id
    assert row.checkpoint_id.startswith(f"{trace.run_id}:")


def test_the_streaming_loop_links_it_too(tmp_path: Path) -> None:
    """`run_stream` answers the first task of every turn, so a link only the sync
    loop wrote would be a link the web never got."""
    core, store, _ = _core(tmp_path, origin_channel="web")

    list(core.run_stream("analyse what is happening around the world"))

    row = store.list_pending()[0]
    assert row.checkpoint_id is not None
    assert row.channel == "web"


def test_a_run_with_no_checkpoint_store_still_records_the_decision(tmp_path: Path) -> None:
    """Losing the link costs the one-click resume, never the approval itself."""
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    queue = ApprovalQueue(store=store)
    registry = EvaluatorRegistry()
    registry.register(_AlwaysApproval())
    registry.init_lock()
    kernel = GovernanceKernel(audit_log=None)
    kernel.register(EvaluatorHook(registry=registry, approval_queue=queue))
    kernel.init_lock()
    core = AgenticCore(
        config=AgenticCoreConfig(max_iterations=3),
        llm_call=lambda _p: 'Thought: t\nAction: echo\nAction Input: {"q": "x"}',
        tools=[ToolSpec(name="echo", description="echo", call=lambda a: "observed")],
        kernel=kernel,
        checkpoint_store=None,
        link_approval_checkpoint=queue.set_checkpoint,
    )

    core.run("do the thing")

    row = store.list_pending()[0]
    assert row.checkpoint_id is None  # nothing to link to


# ── the session, so a lapse can be announced where it was noticed ─────────────


def test_the_row_records_the_session_the_halt_belongs_to(tmp_path: Path) -> None:
    """Needed by the timeout sweep: a notice has to land somewhere, and deriving the
    session from the nullable `checkpoint_id` only works for approvals that got linked."""
    core, store, _ = _core(tmp_path, origin_channel="web")

    core.run("analyse what is happening around the world")

    assert store.list_pending()[0].session_id == "web-6d670ccd"


def test_a_run_with_no_session_leaves_it_null(tmp_path: Path) -> None:
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    queue = ApprovalQueue(store=store)
    registry = EvaluatorRegistry()
    registry.register(_AlwaysApproval())
    registry.init_lock()
    kernel = GovernanceKernel(audit_log=None)
    kernel.register(EvaluatorHook(registry=registry, approval_queue=queue))
    kernel.init_lock()
    core = AgenticCore(
        config=AgenticCoreConfig(max_iterations=3),
        llm_call=lambda _p: 'Thought: t\nAction: echo\nAction Input: {"q": "x"}',
        tools=[ToolSpec(name="echo", description="echo", call=lambda a: "observed")],
        kernel=kernel,
        session_id=None,
    )

    core.run("do the thing")

    assert store.list_pending()[0].session_id is None
