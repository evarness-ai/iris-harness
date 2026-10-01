"""Who reads a Telegram answer (ADR-0125): the owner in a private chat, others in a group.

The poller reads ``chat.type`` and hands the audience to the chat handler (the gateway
bridge forwards it: ``test_telegram_gateway.py``). No network: updates go straight to the
poller's dispatch.
"""

from __future__ import annotations

from typing import Any

import pytest

from iris_harness.services.channels import ChannelMessage, DeliveryReceipt, DeliveryStatus
from iris_harness.services.channels.connectors.telegram_poller import (
    TelegramPoller,
    audience_of_chat,
)

CHAT = "-100200"


class _Connector:
    def __init__(self) -> None:
        self.sent: list[ChannelMessage] = []

    def send(self, message: ChannelMessage) -> DeliveryReceipt:
        self.sent.append(message)
        return DeliveryReceipt(channel="telegram", status=DeliveryStatus.SENT)

    def answer_callback(self, callback_query_id: str, *, text: str = "") -> None:
        return None

    def clear_inline_keyboard(self, chat_id: str, message_id: str) -> None:
        return None


def _poller(seen: list[tuple[str, str, str]]) -> TelegramPoller:
    return TelegramPoller(
        bot_token="t",
        connector=_Connector(),  # type: ignore[arg-type]
        chat_handler=lambda text, sid, audience: seen.append((text, sid, audience)) or "ok",
        allowed_chat_ids=frozenset({CHAT}),
        thinking_delay=60,
    )


@pytest.mark.parametrize(
    ("chat_type", "audience"),
    [("private", "owner"), ("group", "other"), ("supergroup", "other"), (None, "other")],
)
def test_the_chat_type_decides_the_audience(chat_type: str | None, audience: str) -> None:
    chat: dict[str, Any] = {"id": int(CHAT)}
    if chat_type is not None:
        chat["type"] = chat_type
    seen: list[tuple[str, str, str]] = []
    _poller(seen)._handle_update({"message": {"text": "what is on today", "chat": chat}})
    assert seen == [("what is on today", f"telegram:{CHAT}", audience)]
    assert audience_of_chat(chat) == audience


def test_a_button_tap_in_a_group_is_answered_to_the_group() -> None:
    seen: list[tuple[str, str, str]] = []
    _poller(seen)._handle_callback(
        {
            "id": "cbq1",
            "data": "approve",
            "message": {"message_id": 7, "chat": {"id": int(CHAT), "type": "group"}},
        }
    )
    assert [a for _, _, a in seen] == ["other"]
