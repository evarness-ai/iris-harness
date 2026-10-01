"""The optional per-user Telegram allowlist (security review, 2026-09-26).

The chat allowlist admits a chat; in a group that is everyone in it, so any member could
drive the agent or tap an approval button. ``TELEGRAM_ALLOWED_USER_IDS`` narrows it to
people. Unset, the poller behaves exactly as before. No network: updates are fed to the
poller's dispatch directly.
"""

from __future__ import annotations

from typing import Any

import pytest

from iris_harness.services.channels import ChannelMessage, DeliveryReceipt, DeliveryStatus
from iris_harness.services.channels.connectors.telegram_poller import (
    TelegramPoller,
    allowed_user_ids_from_env,
)

GROUP = "-100200"
OWNER = "42"
STRANGER = "77"


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
    handled: list[tuple[str, str]], users: frozenset[str]
) -> tuple[TelegramPoller, _FakeConnector]:
    connector = _FakeConnector()
    poller = TelegramPoller(
        bot_token="t",
        connector=connector,  # type: ignore[arg-type]
        chat_handler=lambda text, sid, _audience: handled.append((text, sid)) or "ok",
        allowed_chat_ids=frozenset({GROUP}),
        allowed_user_ids=users,
    )
    return poller, connector


def _message(sender: str | None) -> dict[str, Any]:
    message: dict[str, Any] = {"text": "what is on today", "chat": {"id": int(GROUP)}}
    if sender is not None:
        message["from"] = {"id": int(sender)}
    return {"message": message}


def _tap(sender: str) -> dict[str, Any]:
    return {
        "callback_query": {
            "id": "cbq1",
            "data": "approve",
            "from": {"id": int(sender)},
            "message": {"message_id": 7, "chat": {"id": int(GROUP)}},
        }
    }


def test_without_a_user_list_the_chat_allowlist_alone_decides() -> None:
    handled: list[tuple[str, str]] = []
    poller, _ = _poller(handled, frozenset())
    try:
        poller._handle_update(_message(STRANGER))
        poller._handle_update(_message(None))
    finally:
        poller.stop()
    assert len(handled) == 2


def test_with_a_user_list_only_those_users_are_served() -> None:
    handled: list[tuple[str, str]] = []
    poller, connector = _poller(handled, frozenset({OWNER}))
    try:
        poller._handle_update(_message(STRANGER))
        poller._handle_update(_message(None))  # no sender id: refused, not waved through
        assert handled == []
        assert connector.sent == []
        poller._handle_update(_message(OWNER))
    finally:
        poller.stop()
    assert handled == [("what is on today", f"telegram:{GROUP}")]


def test_a_button_tap_from_another_group_member_does_nothing() -> None:
    handled: list[tuple[str, str]] = []
    poller, connector = _poller(handled, frozenset({OWNER}))
    try:
        poller._handle_update(_tap(STRANGER))
        assert handled == []
        assert connector.cleared == []  # the buttons stay for the owner
        assert connector.answered == ["cbq1"]  # the stranger's spinner still stops
        poller._handle_update(_tap(OWNER))
    finally:
        poller.stop()
    assert handled == [("approve", f"telegram:{GROUP}")]
    assert connector.cleared == [(GROUP, "7")]


def test_the_env_reader_parses_csv_and_treats_blank_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TELEGRAM_ALLOWED_USER_IDS", raising=False)
    assert allowed_user_ids_from_env() == frozenset()
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "  ")
    assert allowed_user_ids_from_env() == frozenset()
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "42, 43,,")
    assert allowed_user_ids_from_env() == frozenset({"42", "43"})
