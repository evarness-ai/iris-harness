"""The first-chat welcome on Telegram (ADR-0127): on `/start` or the first message.

The poller asks its opener (the gateway's bridge → ``POST /chat/welcome``; the runtime's
own poller → ``runtime.welcome``) until the harness has answered once. A new welcome is
sent first; `/start` asks nothing more, any other first message is then answered as usual.
No network: updates go straight to the poller's dispatch.
"""

from __future__ import annotations

from typing import Any

import httpx

from iris_harness.server.channel_gateway.telegram import TelegramIrisChatBridge
from iris_harness.services.channels import ChannelMessage, DeliveryReceipt, DeliveryStatus
from iris_harness.services.channels.connectors.telegram_poller import TelegramPoller

CHAT = "4242"


class _Connector:
    def __init__(self) -> None:
        self.sent: list[ChannelMessage] = []

    def send(self, message: ChannelMessage) -> DeliveryReceipt:
        self.sent.append(message)
        return DeliveryReceipt(channel="telegram", status=DeliveryStatus.SENT)


def _poller(opener: Any, chats: list[str]) -> tuple[TelegramPoller, _Connector]:
    connector = _Connector()
    poller = TelegramPoller(
        bot_token="t",
        connector=connector,  # type: ignore[arg-type]
        chat_handler=lambda text, _sid, _aud: chats.append(text) or f"answer to {text}",
        opener=opener,
        allowed_chat_ids=frozenset({CHAT}),
        thinking_delay=60,
    )
    return poller, connector


def _message(text: str) -> dict[str, Any]:
    return {"message": {"text": text, "chat": {"id": int(CHAT), "type": "private"}}}


def test_start_on_a_fresh_install_gets_the_welcome_and_nothing_else() -> None:
    asked: list[str] = []
    chats: list[str] = []
    poller, connector = _poller(lambda audience: asked.append(audience) or "WELCOME", chats)

    poller._handle_update(_message("/start"))

    assert asked == ["owner"]
    assert [m.body for m in connector.sent] == ["WELCOME"]
    assert chats == []


def test_a_first_message_gets_the_welcome_then_its_answer() -> None:
    chats: list[str] = []
    poller, connector = _poller(lambda _a: "WELCOME", chats)

    poller._handle_update(_message("what can you do?"))

    assert [m.body for m in connector.sent] == ["WELCOME", "answer to what can you do?"]


def test_once_the_harness_has_answered_it_is_not_asked_again() -> None:
    asked: list[str] = []
    chats: list[str] = []
    poller, connector = _poller(lambda a: asked.append(a) or None, chats)  # ran before

    poller._handle_update(_message("/start"))
    poller._handle_update(_message("hello"))

    assert asked == ["owner"]
    # Not new: `/start` goes to chat as it always did, and nothing extra is sent.
    assert chats == ["/start", "hello"]
    assert [m.body for m in connector.sent] == ["answer to /start", "answer to hello"]


def test_an_api_out_of_reach_is_asked_again_and_never_stops_the_chat() -> None:
    calls: list[int] = []

    def flaky(_audience: str) -> str | None:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("refused")
        return "WELCOME"

    chats: list[str] = []
    poller, connector = _poller(flaky, chats)
    poller._handle_update(_message("hi"))
    poller._handle_update(_message("hi again"))

    assert [m.body for m in connector.sent] == ["answer to hi", "WELCOME", "answer to hi again"]


def _bridge(body: dict[str, Any], seen: list[httpx.Request]) -> TelegramIrisChatBridge:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=body)

    return TelegramIrisChatBridge(
        iris_api_url="http://iris-api.test",
        client=httpx.Client(transport=httpx.MockTransport(handle)),
    )


def test_the_gateway_bridge_asks_the_api() -> None:
    seen: list[httpx.Request] = []
    bridge = _bridge({"created": True, "response": "WELCOME"}, seen)
    assert bridge.welcome("other") == "WELCOME"
    assert seen[0].url.path == "/chat/welcome"
    assert seen[0].read() == b'{"channel":"telegram","audience":"other"}'
    assert _bridge({"created": False, "response": "WELCOME"}, []).welcome() is None


def test_the_runtime_poller_asks_the_harness_in_process(monkeypatch: Any, tmp_path: Any) -> None:
    from pathlib import Path

    from iris_harness.runtime import build_runtime

    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    config_dir = Path(__file__).resolve().parents[5] / "config"
    runtime = build_runtime(
        config_dir=config_dir, data_dir=tmp_path / "data", use_background_scheduler=False
    )
    opener = runtime.welcome.new_text_for("telegram")
    first = opener("owner")
    assert first and "Call trace" in first
    assert opener("owner") is None
    recorded = runtime.welcome.recorded()
    assert recorded is not None and recorded.created is False


def test_the_gateway_wires_its_bridge_as_the_poller_s_opener(monkeypatch: Any) -> None:
    from iris_harness.server.channel_gateway.telegram import build_telegram_runtime_from_env

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    idle = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    runtime = build_telegram_runtime_from_env(
        iris_api_url="http://iris-api.test",
        upstream_client=idle,
        telegram_poll_client=idle,
        telegram_send_client=idle,
    )
    assert runtime is not None
    poller = runtime._poller  # type: ignore[attr-defined]
    assert poller._opener == runtime._bridge.welcome  # type: ignore[attr-defined]
    runtime.stop()
