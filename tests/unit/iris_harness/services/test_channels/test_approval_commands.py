"""Answering an approval from Telegram: a tap on its buttons, or a typed command.

Until 2026-09-21 a typed ``/approve <id>`` reached the chat model as ordinary text: the
handler that parsed it had no caller. These pin the replacement end to end at the
poller — the update comes in, the command is answered before chat, the queue records
who answered — over a real approval queue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris_harness.kernel.governance.approvals import ApprovalCard, ApprovalItem, ApprovalQueue
from iris_harness.services.channels import ChannelMessage, DeliveryReceipt, DeliveryStatus
from iris_harness.services.channels.approval_commands import (
    ApprovalSummary,
    LocalApprovalBackend,
    approval_keyboard,
    handle_approval_command,
)
from iris_harness.services.channels.connectors.telegram_poller import TelegramPoller


class _FakeConnector:
    def __init__(self) -> None:
        self.sent: list[ChannelMessage] = []
        self.answered: list[str] = []
        self.cleared: list[tuple[str, str]] = []

    def send(self, message: ChannelMessage) -> DeliveryReceipt:
        self.sent.append(message)
        return DeliveryReceipt(channel="telegram", status=DeliveryStatus.SENT)

    def answer_callback(self, callback_query_id: str, *, text: str = "") -> None:
        self.answered.append(callback_query_id)

    def clear_inline_keyboard(self, chat_id: str, message_id: str) -> None:
        self.cleared.append((chat_id, message_id))


@pytest.fixture()
def queue(tmp_path: Path) -> ApprovalQueue:
    return ApprovalQueue(db_path=tmp_path / "approvals.db")


def _destructive(queue: ApprovalQueue) -> str:
    return queue.enqueue(
        "run-1",
        None,
        "Trash 2 emails",
        "ctx",
        items=(ApprovalItem.of("trash_email", {"ids": ["m1", "m2"]}),),
        card=ApprovalCard(title="Trash 2 emails", lines=("a", "b"), undo_tool="restore_email"),
    )


def _evaluator(queue: ApprovalQueue) -> str:
    return queue.enqueue("run-2", None, "step_cap", "Too many steps")


def _poller(conn: _FakeConnector, queue: ApprovalQueue, chat: list[str]) -> TelegramPoller:
    backend = LocalApprovalBackend(queue=queue)
    return TelegramPoller(
        bot_token="t",
        connector=conn,  # type: ignore[arg-type]
        chat_handler=lambda text, sid, _audience: chat.append(text) or "chat reply",
        command_handler=lambda text, uid: handle_approval_command(
            text, uid, backend, allowed_users=frozenset()
        ),
        allowed_chat_ids=frozenset({"555"}),
    )


def _tap(poller: TelegramPoller, data: str, *, user: int = 42) -> None:
    poller._handle_update(
        {
            "callback_query": {
                "id": "cbq",
                "data": data,
                "from": {"id": user},
                "message": {"message_id": 7, "chat": {"id": 555}},
            }
        }
    )


def _type(poller: TelegramPoller, text: str, *, user: int = 42) -> None:
    poller._handle_update({"message": {"text": text, "from": {"id": user}, "chat": {"id": 555}}})


# --- the buttons ------------------------------------------------------------------


def test_a_destructive_approval_asks_before_the_action_runs(queue: ApprovalQueue) -> None:
    approval_id = _destructive(queue)
    [[first, reject]] = approval_keyboard(approval_id, ApprovalSummary("Trash 2 emails", True))
    assert first == {"text": "Trash 2 emails", "callback_data": f"/ask {approval_id}"}
    assert reject == {"text": "Reject", "callback_data": f"/reject {approval_id}"}
    # Telegram caps callback_data at 64 bytes.
    assert len(first["callback_data"].encode()) <= 64


def test_an_evaluator_approval_is_one_tap(queue: ApprovalQueue) -> None:
    approval_id = _evaluator(queue)
    [[first, _]] = approval_keyboard(approval_id, ApprovalSummary("step_cap", False))
    assert first == {"text": "Approve", "callback_data": f"/approve {approval_id}"}


# --- through the poller -------------------------------------------------------------


def test_tapping_the_action_then_yes_approves_it(queue: ApprovalQueue) -> None:
    conn, chat = _FakeConnector(), []
    approval_id = _destructive(queue)
    poller = _poller(conn, queue, chat)
    try:
        _tap(poller, f"/ask {approval_id}")
        confirm = conn.sent[-1]
        assert confirm.body.startswith("Trash 2 emails? IRIS will run exactly the call")
        yes = confirm.metadata["inline_keyboard"][0][0]
        assert yes["callback_data"] == f"/approve {approval_id}"
        assert queue.get(approval_id).status == "pending"  # the first tap runs nothing

        _tap(poller, yes["callback_data"])
    finally:
        poller.stop()

    row = queue.get(approval_id)
    assert row.status == "approved" and row.response_actor == "telegram:42"
    assert conn.sent[-1].body.startswith("Approved: Trash 2 emails.")
    assert chat == []  # never reached the chat model
    assert conn.answered == ["cbq", "cbq"] and conn.cleared == [("555", "7")] * 2


def test_cancel_changes_nothing_and_brings_the_buttons_back(queue: ApprovalQueue) -> None:
    conn, chat = _FakeConnector(), []
    approval_id = _destructive(queue)
    poller = _poller(conn, queue, chat)
    try:
        _tap(poller, f"/cancel {approval_id}")
    finally:
        poller.stop()
    assert queue.get(approval_id).status == "pending"
    assert conn.sent[-1].metadata["inline_keyboard"][0][0]["callback_data"] == f"/ask {approval_id}"


def test_reject_is_one_tap_and_says_nothing_changed(queue: ApprovalQueue) -> None:
    conn, chat = _FakeConnector(), []
    approval_id = _destructive(queue)
    poller = _poller(conn, queue, chat)
    try:
        _tap(poller, f"/reject {approval_id}")
    finally:
        poller.stop()
    assert queue.get(approval_id).status == "rejected"
    assert conn.sent[-1].body.startswith("Rejected. Nothing was changed.")


def test_a_typed_command_is_answered_not_sent_to_chat(queue: ApprovalQueue) -> None:
    """The bug: this used to reach the chat model as ordinary text."""
    conn, chat = _FakeConnector(), []
    approval_id = _evaluator(queue)
    poller = _poller(conn, queue, chat)
    try:
        _type(poller, f"/approve {approval_id}")
    finally:
        poller.stop()
    assert chat == []
    assert queue.get(approval_id).status == "approved"


def test_ordinary_text_and_other_taps_still_go_to_chat(queue: ApprovalQueue) -> None:
    conn, chat = _FakeConnector(), []
    poller = _poller(conn, queue, chat)
    try:
        _type(poller, "what's on today?")
        _tap(poller, "approve")  # an in-chat confirmation (ADR-0076), not a queue approval
    finally:
        poller.stop()
    assert chat == ["what's on today?", "approve"]


def test_an_answered_approval_says_so(queue: ApprovalQueue) -> None:
    conn, chat = _FakeConnector(), []
    approval_id = _evaluator(queue)
    queue.respond(approval_id, status="approved", actor="web")
    poller = _poller(conn, queue, chat)
    try:
        _tap(poller, f"/approve {approval_id}")
    finally:
        poller.stop()
    assert "not waiting any more" in conn.sent[-1].body


def test_an_unlisted_user_cannot_answer(queue: ApprovalQueue) -> None:
    approval_id = _evaluator(queue)
    reply = handle_approval_command(
        f"/approve {approval_id}",
        "evil",
        LocalApprovalBackend(queue=queue),
        allowed_users=frozenset({"42"}),
    )
    assert reply is not None and "permission" in reply.text
    assert queue.get(approval_id).status == "pending"


def test_an_unlisted_chat_is_dropped_before_any_command(queue: ApprovalQueue) -> None:
    conn, chat = _FakeConnector(), []
    approval_id = _evaluator(queue)
    poller = _poller(conn, queue, chat)
    try:
        poller._handle_update(
            {
                "callback_query": {
                    "id": "cbq",
                    "data": f"/approve {approval_id}",
                    "from": {"id": 42},
                    "message": {"message_id": 7, "chat": {"id": 999}},
                }
            }
        )
    finally:
        poller.stop()
    assert queue.get(approval_id).status == "pending"


def test_a_failing_backend_is_said_not_swallowed(queue: ApprovalQueue) -> None:
    class _Broken:
        def summary(self, approval_id: str) -> Any:
            raise RuntimeError("api down")

        def respond(self, *a: Any) -> Any:
            raise AssertionError

    conn, chat = _FakeConnector(), []
    poller = TelegramPoller(
        bot_token="t",
        connector=conn,  # type: ignore[arg-type]
        chat_handler=lambda text, sid, _audience: chat.append(text) or "x",
        command_handler=lambda text, uid: handle_approval_command(
            text, uid, _Broken(), allowed_users=frozenset()
        ),
        allowed_chat_ids=frozenset({"555"}),
    )
    try:
        _tap(poller, "/approve 3f9c0d2a-0000-4000-8000-000000000001")
    finally:
        poller.stop()
    assert chat == []
    assert "Could not answer that approval" in conn.sent[-1].body
