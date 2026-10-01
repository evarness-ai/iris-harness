"""DestructiveApprovalHook on its own (ADR-0118). The loop-level flow is in
tests/unit/iris_harness/agent/test_agent/test_destructive_approval_loop.py."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from iris_harness.kernel.governance import HookContext, HookPoint
from iris_harness.kernel.governance.approvals import ApprovalQueue
from iris_harness.kernel.governance.plugins import DestructiveApprovalHook


class _Router:
    def __init__(self) -> None:
        self.notified: list[Any] = []

    def notify(self, row: Any) -> None:
        self.notified.append(row)


def _ctx(effect: str = "destructive", **metadata: Any) -> HookContext:
    return HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="run-1",
        agent_type="chat",
        payload={"tool_name": "trash_email", "args": {"ids": ["m1"]}},
        metadata={"tool_effect": effect, "resumable": True, **metadata},
    )


def _fire(hook: DestructiveApprovalHook, ctx: HookContext) -> Any:
    return asyncio.run(hook(ctx))


def test_other_effects_pass_straight_through(tmp_path: Path) -> None:
    hook = DestructiveApprovalHook(approval_queue=ApprovalQueue(db_path=tmp_path / "a.db"))
    for effect in ("read", "write"):
        assert _fire(hook, _ctx(effect)).outcome == "allow"


def test_a_request_is_queued_delivered_and_returned_as_a_halt(tmp_path: Path) -> None:
    queue, router = ApprovalQueue(db_path=tmp_path / "a.db"), _Router()
    hook = DestructiveApprovalHook(approval_queue=queue, channel_router=router)  # type: ignore[arg-type]

    decision = _fire(hook, _ctx(origin_channel="telegram", session_id="telegram:42"))

    assert decision.outcome == "require_approval"
    assert decision.approval_request_id is not None
    row = queue.get(str(decision.approval_request_id))
    assert row is not None and row.channel == "telegram" and row.session_id == "telegram:42"
    assert [r.approval_id for r in router.notified] == [row.approval_id]  # the phone hears


def test_a_claim_on_an_unanswered_approval_is_denied(tmp_path: Path) -> None:
    queue = ApprovalQueue(db_path=tmp_path / "a.db")
    hook = DestructiveApprovalHook(approval_queue=queue)
    pending = str(_fire(hook, _ctx()).approval_request_id)

    decision = _fire(hook, _ctx(approved_by=pending))

    assert decision.outcome == "deny"
    assert "is pending" in decision.reason


def test_a_claim_on_an_approved_row_that_pins_this_call_is_allowed(tmp_path: Path) -> None:
    queue = ApprovalQueue(db_path=tmp_path / "a.db")
    hook = DestructiveApprovalHook(approval_queue=queue)
    approval_id = str(_fire(hook, _ctx()).approval_request_id)
    queue.respond(approval_id, status="approved", actor="owner")

    assert _fire(hook, _ctx(approved_by=approval_id)).outcome == "allow"


def test_a_destructive_approval_stays_answerable_for_an_hour(tmp_path: Path) -> None:
    """Answered from the phone, away from the chat: an hour, not the queue's 10 minutes."""
    from datetime import datetime

    queue = ApprovalQueue(db_path=tmp_path / "a.db")
    decision = _fire(DestructiveApprovalHook(approval_queue=queue), _ctx())
    row = queue.get(str(decision.approval_request_id))
    assert row is not None
    window = datetime.fromisoformat(row.timeout_at) - datetime.fromisoformat(row.requested_at)
    assert window.total_seconds() == 60 * 60
