"""A slow Telegram turn is answered late, not dropped (2026-09-25).

The VM gave up at 90s and replied "timed out" while the answer arrived 7s later. The
poller now says it is still working and posts the answer when it lands. Timeouts are
shrunk to fractions of a second; the handler blocks on an event the test controls.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from iris_harness.services.channels import ChannelMessage, DeliveryReceipt, DeliveryStatus
from iris_harness.services.channels.connectors.telegram_poller import TelegramPoller


class _Connector:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, message: ChannelMessage) -> DeliveryReceipt:
        self.sent.append(message.body)
        return DeliveryReceipt(channel="telegram", status=DeliveryStatus.SENT)


def _update(text: str) -> dict[str, Any]:
    return {"message": {"text": text, "chat": {"id": 555}, "from": {"id": 1}}}


def _poller(conn: _Connector, handler: Any, *, late: float) -> TelegramPoller:
    return TelegramPoller(
        bot_token="t",
        connector=conn,  # type: ignore[arg-type]
        chat_handler=handler,
        allowed_chat_ids=frozenset({"555"}),
        handler_timeout=0.2,
        late_answer_limit=late,
        thinking_delay=60,
    )


def _wait_for(pred: Any, timeout: float = 3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_a_slow_answer_is_posted_when_it_lands() -> None:
    conn = _Connector()
    release = threading.Event()

    def slow(text: str, session: str, audience: str) -> str:
        release.wait(2)
        return "Today's inbox is busy: 91 new messages."

    poller = _poller(conn, slow, late=3.0)
    try:
        poller._handle_update(_update("How is my inbox today"))
        assert conn.sent == [
            "⏳ Still working on it — I'll send the answer here as soon as it's ready."
        ]
        release.set()
        assert _wait_for(lambda: len(conn.sent) == 2)
        assert conn.sent[1] == (
            "About “How is my inbox today”:\n\nToday's inbox is busy: 91 new messages."
        )
    finally:
        release.set()
        poller.stop()


def test_an_answer_that_never_comes_is_reported_once_the_limit_passes() -> None:
    conn = _Connector()
    release = threading.Event()
    poller = _poller(conn, lambda t, s, _a: release.wait(5) and "late", late=0.5)
    try:
        poller._handle_update(_update("What's my Discover bill"))
        assert _wait_for(lambda: len(conn.sent) == 2)
        assert conn.sent[1].startswith("⚠️ No answer after")
    finally:
        release.set()
        poller.stop()


def test_a_fast_answer_is_unchanged() -> None:
    conn = _Connector()
    poller = _poller(conn, lambda t, s, _a: "Hello!", late=3.0)
    try:
        poller._handle_update(_update("hi"))
        assert conn.sent == ["Hello!"]
    finally:
        poller.stop()
