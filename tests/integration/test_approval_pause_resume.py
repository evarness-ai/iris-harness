"""Integration test for AC-5: evaluator approval pause-resume flow (story 12.gov-4.9).

Tests the end-to-end path: evaluator signal trips → approval enqueued →
ApprovalGate polls → background task approves/rejects → gate resolves.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from iris_harness.kernel.governance.approvals.gate import (
    ApprovalGate,
    ApprovalRejectedError,
    ApprovalTimedOutError,
)
from iris_harness.kernel.governance.approvals.queue import ApprovalQueue
from iris_harness.kernel.governance.approvals.store import ApprovalStore


async def test_approval_pause_resume_flow(tmp_path: Path) -> None:
    """AC-5: evaluator trips goal_drift; approval is enqueued; gate awaits; run resumes."""
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    queue = ApprovalQueue(store=store)
    gate = ApprovalGate(store=store)

    # Simulate EvaluatorHook enqueuing (as if it fired PostStep)
    approval_id = queue.enqueue(
        "run-resume-1",
        None,
        "goal_drift",
        "Goal drift > 0.65 detected",
        channel="cli",
    )

    # Simulate async background approval
    async def _approve_later() -> None:
        await asyncio.sleep(0.05)
        store.respond(approval_id, status="approved", actor="cli:test")

    asyncio.create_task(_approve_later())

    row = await gate.await_approval(approval_id, poll_interval=0.02)
    assert row.status == "approved"
    assert row.response_actor == "cli:test"
    assert row.run_id == "run-resume-1"
    assert row.signal == "goal_drift"


async def test_approval_pause_resume_rejected(tmp_path: Path) -> None:
    """AC-5 variant: operator rejects the approval; gate raises ApprovalRejectedError."""
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    queue = ApprovalQueue(store=store)
    gate = ApprovalGate(store=store)

    approval_id = queue.enqueue(
        "run-resume-2",
        None,
        "loop_detect",
        "Loop detected at step 8",
        channel="cli",
    )

    async def _reject_later() -> None:
        await asyncio.sleep(0.05)
        store.respond(approval_id, status="rejected", actor="cli:operator")

    asyncio.create_task(_reject_later())

    with pytest.raises(ApprovalRejectedError) as exc_info:
        await gate.await_approval(approval_id, poll_interval=0.02)

    assert exc_info.value.approval_id == approval_id
    assert exc_info.value.run_id == "run-resume-2"


async def test_approval_pause_resume_timeout(tmp_path: Path) -> None:
    """AC-5 variant: caller timeout expires before any human action."""
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    queue = ApprovalQueue(store=store)
    gate = ApprovalGate(store=store)

    # Enqueue with a long DB timeout so it won't auto-expire
    approval_id = queue.enqueue(
        "run-resume-3",
        None,
        "step_cap",
        "Step cap reached",
        channel="cli",
        timeout_minutes=60,
    )

    with pytest.raises(ApprovalTimedOutError) as exc_info:
        await gate.await_approval(approval_id, poll_interval=0.02, timeout=0.1)

    assert exc_info.value.approval_id == approval_id
    assert exc_info.value.run_id == "run-resume-3"


async def test_multiple_approval_requests_independent(tmp_path: Path) -> None:
    """Multiple approvals in one run are tracked independently."""
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    queue = ApprovalQueue(store=store)
    gate = ApprovalGate(store=store)

    aid1 = queue.enqueue("run-multi", None, "goal_drift", "drift 1", channel="cli")
    aid2 = queue.enqueue("run-multi", None, "step_cap", "cap 2", channel="cli")

    async def _approve_both() -> None:
        await asyncio.sleep(0.03)
        store.respond(aid1, status="approved", actor="cli:tester")
        await asyncio.sleep(0.02)
        store.respond(aid2, status="rejected", actor="cli:tester")

    asyncio.create_task(_approve_both())

    row1 = await gate.await_approval(aid1, poll_interval=0.01)
    assert row1.status == "approved"

    with pytest.raises(ApprovalRejectedError):
        await gate.await_approval(aid2, poll_interval=0.01)
