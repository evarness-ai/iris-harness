"""Tests for TelegramConnector using httpx.MockTransport."""

from __future__ import annotations

import json

import httpx

from iris_harness.services.channels import ChannelMessage, DeliveryStatus
from iris_harness.services.channels.connectors import TelegramConnector


def _client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler)


def test_telegram_send_success() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 42}})

    client = _client(httpx.MockTransport(handler))
    connector = TelegramConnector(bot_token="abc", client=client)
    receipt = connector.send(ChannelMessage(recipient="123", body="hi"))

    assert receipt.status is DeliveryStatus.SENT
    assert receipt.message_id == "42"
    assert "/botabc/sendMessage" in str(captured["url"])
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["chat_id"] == "123"
    assert payload["text"] == "hi"


def test_telegram_uses_default_chat_id_when_recipient_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    connector = TelegramConnector(
        bot_token="abc",
        default_chat_id="default",
        client=_client(httpx.MockTransport(handler)),
    )
    receipt = connector.send(ChannelMessage(recipient="", body="x"))
    assert receipt.status is DeliveryStatus.SENT


def test_telegram_returns_failed_when_no_chat_id_available() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not be called")

    connector = TelegramConnector(bot_token="abc", client=_client(httpx.MockTransport(handler)))
    receipt = connector.send(ChannelMessage(recipient="", body="x"))
    assert receipt.status is DeliveryStatus.FAILED
    assert "chat_id" in receipt.error


def test_telegram_handles_rate_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="too many requests")

    connector = TelegramConnector(bot_token="abc", client=_client(httpx.MockTransport(handler)))
    receipt = connector.send(ChannelMessage(recipient="123", body="hi"))
    assert receipt.status is DeliveryStatus.RATE_LIMITED


def test_telegram_handles_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server boom")

    connector = TelegramConnector(bot_token="abc", client=_client(httpx.MockTransport(handler)))
    receipt = connector.send(ChannelMessage(recipient="123", body="hi"))
    assert receipt.status is DeliveryStatus.FAILED
    assert "HTTP 500" in receipt.error


def test_telegram_handles_transport_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down")

    connector = TelegramConnector(bot_token="abc", client=_client(httpx.MockTransport(handler)))
    receipt = connector.send(ChannelMessage(recipient="123", body="hi"))
    assert receipt.status is DeliveryStatus.FAILED
    assert "ConnectError" in receipt.error


def test_telegram_passes_parse_mode_metadata() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    connector = TelegramConnector(bot_token="abc", client=_client(httpx.MockTransport(handler)))
    connector.send(
        ChannelMessage(recipient="123", body="*bold*", metadata={"parse_mode": "Markdown"})
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["parse_mode"] == "Markdown"


def test_telegram_requires_bot_token() -> None:
    try:
        TelegramConnector(bot_token="")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ── inline buttons + callback helpers (ADR-0076) ─────────────────────────────


def test_telegram_send_attaches_inline_keyboard() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    connector = TelegramConnector(bot_token="abc", client=_client(httpx.MockTransport(handler)))
    kb = [[{"text": "Approve", "callback_data": "approve"}]]
    connector.send(
        ChannelMessage(recipient="123", body="confirm?", metadata={"inline_keyboard": kb})
    )
    assert captured["body"]["reply_markup"] == {"inline_keyboard": kb}  # type: ignore[index]


def test_telegram_send_passes_url_buttons_with_the_html_body() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 7}})

    connector = TelegramConnector(bot_token="abc", client=_client(httpx.MockTransport(handler)))
    kb = [[{"text": "📄 Full digest", "url": "https://iris.example/digest/abc"}]]
    receipt = connector.send(
        ChannelMessage(
            recipient="123",
            body="<blockquote><b>Stocks</b>\n• AAA</blockquote>",
            metadata={"parse_mode": "HTML", "inline_keyboard": kb},
        )
    )
    assert receipt.status is DeliveryStatus.SENT
    [payload] = captured
    assert payload["parse_mode"] == "HTML"
    assert payload["reply_markup"] == {"inline_keyboard": kb}


def test_telegram_resends_without_buttons_it_refuses() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        if "reply_markup" in body:
            return httpx.Response(
                400, json={"ok": False, "description": "Bad Request: BUTTON_URL_INVALID"}
            )
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 8}})

    connector = TelegramConnector(bot_token="abc", client=_client(httpx.MockTransport(handler)))
    kb = [[{"text": "📄 Full digest", "url": "http://localhost/digest/abc"}]]
    receipt = connector.send(
        ChannelMessage(recipient="123", body="digest", metadata={"inline_keyboard": kb})
    )
    assert receipt.status is DeliveryStatus.SENT and receipt.message_id == "8"
    assert len(captured) == 2 and "reply_markup" not in captured[1]


def test_telegram_other_400s_are_not_retried() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json={"ok": False, "description": "Bad Request: can't parse"})

    connector = TelegramConnector(bot_token="abc", client=_client(httpx.MockTransport(handler)))
    kb = [[{"text": "x", "url": "https://iris.example"}]]
    receipt = connector.send(
        ChannelMessage(recipient="123", body="<b", metadata={"inline_keyboard": kb})
    )
    assert receipt.status is DeliveryStatus.FAILED and len(calls) == 1


def test_telegram_answer_callback_hits_endpoint() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True})

    connector = TelegramConnector(bot_token="abc", client=_client(httpx.MockTransport(handler)))
    connector.answer_callback("cbq1")
    assert str(seen["url"]).endswith("/answerCallbackQuery")
    assert seen["body"]["callback_query_id"] == "cbq1"  # type: ignore[index]
