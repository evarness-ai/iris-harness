"""ADR-0076 — Telegram inline approve/reject for in-chat confirmations.

The poller's update dispatch + reply rendering are exercised directly (no network):
a callback_query taps resolve as a chat decision; a reply carries inline buttons
when the session has a pending confirmation."""

from __future__ import annotations

from typing import Any

from iris_harness.services.channels import ChannelMessage, DeliveryReceipt, DeliveryStatus
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


def _poller(
    connector: _FakeConnector,
    handler: Any,
    options: Any,
    *,
    allowed: frozenset[str] = frozenset({"555", "9"}),
) -> TelegramPoller:
    # The poller fails closed without an allowlist, so the fixture declares the
    # chat ids these tests send from; the fail-closed tests pass an empty set.
    return TelegramPoller(
        bot_token="t",
        connector=connector,  # type: ignore[arg-type]
        chat_handler=handler,
        confirmation_options=options,
        allowed_chat_ids=allowed,
    )


def test_callback_resolves_decision_and_acks() -> None:
    conn = _FakeConnector()
    calls: list[tuple[str, str]] = []
    poller = _poller(
        conn,
        lambda text, sid, _audience: calls.append((text, sid)) or "Approved — added it.",
        lambda _sid: None,  # resolved → no more buttons
    )
    try:
        poller._handle_update(
            {
                "callback_query": {
                    "id": "cbq1",
                    "data": "approve",
                    "message": {"message_id": 7, "chat": {"id": 555}},
                }
            }
        )
    finally:
        poller.stop()

    assert calls == [("approve", "telegram:555")]  # routed as a chat decision
    assert conn.cleared == [("555", "7")]  # original buttons removed
    assert conn.answered == ["cbq1"]  # spinner cleared
    assert any("Approved" in m.body for m in conn.sent)


def test_reply_carries_inline_buttons_when_pending() -> None:
    conn = _FakeConnector()
    poller = _poller(
        conn,
        lambda text, sid, _audience: "Create 'Meeting' and invite 1? approve/reject",
        lambda _sid: ["approve", "reject"],  # a confirmation is pending
    )
    try:
        poller._handle_update(
            {"message": {"text": "schedule a meeting with bob@x.com at 3pm", "chat": {"id": 555}}}
        )
    finally:
        poller.stop()

    assert conn.sent
    keyboard = conn.sent[-1].metadata.get("inline_keyboard")
    assert keyboard == [
        [
            {"text": "Approve", "callback_data": "approve"},
            {"text": "Reject", "callback_data": "reject"},
        ]
    ]


def test_reply_has_no_buttons_when_nothing_pending() -> None:
    conn = _FakeConnector()
    poller = _poller(conn, lambda text, sid, _audience: "Here's your schedule.", lambda _sid: None)
    try:
        poller._handle_update({"message": {"text": "what's on tomorrow?", "chat": {"id": 9}}})
    finally:
        poller.stop()
    assert conn.sent
    assert "inline_keyboard" not in conn.sent[-1].metadata


# ── Fail-closed allowlist (security floor, phase 0) ─────────────────────────


def test_message_dropped_when_no_allowlist_configured() -> None:
    conn = _FakeConnector()
    calls: list[tuple[str, str]] = []
    poller = _poller(
        conn,
        lambda text, sid, _audience: calls.append((text, sid)) or "reply",
        lambda _sid: None,
        allowed=frozenset(),
    )
    try:
        poller._handle_update({"message": {"text": "hello", "chat": {"id": 42}}})
    finally:
        poller.stop()
    assert calls == []  # handler never invoked
    assert conn.sent == []  # nothing sent back


def test_message_dropped_when_chat_not_in_allowlist() -> None:
    conn = _FakeConnector()
    calls: list[tuple[str, str]] = []
    poller = _poller(
        conn,
        lambda text, sid, _audience: calls.append((text, sid)) or "reply",
        lambda _sid: None,
        allowed=frozenset({"555"}),
    )
    try:
        poller._handle_update({"message": {"text": "hello", "chat": {"id": 666}}})
    finally:
        poller.stop()
    assert calls == []
    assert conn.sent == []


def test_callback_dropped_when_no_allowlist_configured() -> None:
    conn = _FakeConnector()
    calls: list[tuple[str, str]] = []
    poller = _poller(
        conn,
        lambda text, sid, _audience: calls.append((text, sid)) or "done",
        lambda _sid: ["approve", "reject"],
        allowed=frozenset(),
    )
    try:
        poller._handle_update(
            {
                "callback_query": {
                    "id": "cbq9",
                    "data": "approve",
                    "message": {"message_id": 3, "chat": {"id": 42}},
                }
            }
        )
    finally:
        poller.stop()
    assert calls == []  # approval NOT resolved
    assert conn.cleared == []  # buttons left in place
    assert conn.answered == ["cbq9"]  # spinner still cleared for UX
