"""Approving from Telegram, end to end: the message IRIS sends carries buttons, and
tapping them answers the approval (story 12.gov-4.8; ADR-0118 step 4 follow-up).

The message comes from the real ``TelegramApprovalChannel``; the taps go through the
real ``TelegramPoller``, whose command handler answers over a real queue. Only the
Telegram HTTP connector is faked. This replaces tests that called a command handler
directly — one nothing in the running system called, so a typed ``/approve`` used to
reach the chat model instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from iris_harness.kernel.governance.approvals import ApprovalCard, ApprovalItem, ApprovalQueue
from iris_harness.services.channels.approval_commands import (
    LocalApprovalBackend,
    handle_approval_command,
)
from iris_harness.services.channels.approval_delivery import TelegramApprovalChannel
from iris_harness.services.channels.connectors.telegram_poller import TelegramPoller
from iris_harness.services.channels.models import ChannelMessage, DeliveryReceipt, DeliveryStatus


class _Connector:
    def __init__(self) -> None:
        self.sent: list[ChannelMessage] = []

    def send(self, message: ChannelMessage) -> DeliveryReceipt:
        self.sent.append(message)
        return DeliveryReceipt(channel="telegram", status=DeliveryStatus.SENT)

    def answer_callback(self, callback_query_id: str, *, text: str = "") -> None:
        pass

    def clear_inline_keyboard(self, chat_id: str, message_id: str) -> None:
        pass


def _tap(poller: TelegramPoller, button: dict[str, str]) -> None:
    poller._handle_update(
        {
            "callback_query": {
                "id": "cbq",
                "data": button["callback_data"],
                "from": {"id": 42},
                "message": {"message_id": 1, "chat": {"id": 123456}},
            }
        }
    )


def _setup(tmp_path: Path) -> tuple[ApprovalQueue, _Connector, TelegramPoller, list[str]]:
    queue = ApprovalQueue(db_path=tmp_path / "approvals.db")
    connector, chat = _Connector(), []
    backend = LocalApprovalBackend(queue=queue)
    poller = TelegramPoller(
        bot_token="t",
        connector=connector,  # type: ignore[arg-type]
        chat_handler=lambda text, sid, _audience: chat.append(text) or "chat",
        command_handler=lambda text, uid: handle_approval_command(
            text, uid, backend, allowed_users=frozenset()
        ),
        allowed_chat_ids=frozenset({"123456"}),
    )
    return queue, connector, poller, chat


def _notify(queue: ApprovalQueue, connector: _Connector, approval_id: str) -> Any:
    row = queue.get(approval_id)
    assert row is not None
    TelegramApprovalChannel(connector=connector, chat_id="123456").notify(row)  # type: ignore[arg-type]
    return connector.sent[-1]


def test_an_evaluator_approval_is_approved_with_one_tap(tmp_path: Path) -> None:
    queue, connector, poller, chat = _setup(tmp_path)
    approval_id = queue.enqueue("run-1", None, "step_cap", "Approve me please")
    message = _notify(queue, connector, approval_id)
    approve, reject = message.metadata["inline_keyboard"][0]
    assert (approve["text"], reject["text"]) == ("Approve", "Reject")
    try:
        _tap(poller, approve)
    finally:
        poller.stop()
    row = queue.get(approval_id)
    assert row is not None and row.status == "approved"
    assert row.response_actor == "telegram:42"
    assert chat == []


def test_a_destructive_approval_takes_the_action_then_yes(tmp_path: Path) -> None:
    queue, connector, poller, chat = _setup(tmp_path)
    approval_id = queue.enqueue(
        "run-2",
        None,
        "Trash 2 emails",
        "ctx",
        items=(ApprovalItem.of("trash_email", {"ids": ["m1", "m2"]}),),
        card=ApprovalCard(
            title="Trash 2 emails", lines=("Deals", "Sale"), undo_tool="restore_email"
        ),
    )
    message = _notify(queue, connector, approval_id)
    assert "Tap a button below" in message.body
    action, _reject = message.metadata["inline_keyboard"][0]
    assert action["text"] == "Trash 2 emails"
    try:
        _tap(poller, action)
        assert queue.get(approval_id).status == "pending"  # type: ignore[union-attr]
        yes = connector.sent[-1].metadata["inline_keyboard"][0][0]
        assert yes["text"] == "Yes: Trash 2 emails"
        _tap(poller, yes)
    finally:
        poller.stop()
    assert queue.get(approval_id).status == "approved"  # type: ignore[union-attr]
    assert chat == []


def test_reject_with_one_tap(tmp_path: Path) -> None:
    queue, connector, poller, _chat = _setup(tmp_path)
    approval_id = queue.enqueue("run-3", None, "step_cap", "Reject me")
    _approve, reject = _notify(queue, connector, approval_id).metadata["inline_keyboard"][0]
    try:
        _tap(poller, reject)
    finally:
        poller.stop()
    assert queue.get(approval_id).status == "rejected"  # type: ignore[union-attr]


def test_a_timeout_notice_has_no_buttons(tmp_path: Path) -> None:
    queue, connector, _poller, _chat = _setup(tmp_path)
    approval_id = queue.enqueue("run-4", None, "step_cap", "late")
    row = queue.get(approval_id)
    assert row is not None
    TelegramApprovalChannel(connector=connector, chat_id="123456").notify_timeout(row)  # type: ignore[arg-type]
    assert "inline_keyboard" not in connector.sent[-1].metadata
