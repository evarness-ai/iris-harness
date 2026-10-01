"""Tests for the channel gateway WebSocket client."""

from __future__ import annotations

import json
from typing import Any

import pytest

from iris_harness.services.channels.connectors import ws_client
from iris_harness.services.channels.connectors.ws_client import GatewayClient, GatewayClientConfig


class FakeConnection:
    """Small async WebSocket stand-in for client unit tests."""

    def __init__(self, frames: list[str | bytes] | None = None) -> None:
        self._frames = list(frames or [])
        self.sent: list[str] = []

    def __aiter__(self) -> FakeConnection:
        return self

    async def __anext__(self) -> str | bytes:
        if not self._frames:
            raise StopAsyncIteration
        return self._frames.pop(0)

    async def send(self, payload: str) -> None:
        self.sent.append(payload)


class FakeConnectContext:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.exited = False

    async def __aenter__(self) -> FakeConnection:
        return self.connection

    async def __aexit__(self, *_exc: object) -> None:
        self.exited = True


async def test_stream_sends_auth_header_replies_to_ping_and_yields_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection(
        [
            json.dumps({"type": "ping", "ts": 12.5}),
            json.dumps({"type": "token", "text": "hello"}),
        ]
    )
    context = FakeConnectContext(connection)
    captured: dict[str, Any] = {}

    def fake_connect(url: str, **kwargs: Any) -> FakeConnectContext:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return context

    monkeypatch.setattr(ws_client.websockets, "connect", fake_connect)

    client = GatewayClient(GatewayClientConfig(url="ws://gateway.test/ws", token="secret"))
    stream = client.stream()
    try:
        frame = await anext(stream)
    finally:
        await stream.aclose()

    assert frame == {"type": "token", "text": "hello"}
    assert captured["url"] == "ws://gateway.test/ws"
    assert captured["kwargs"]["additional_headers"] == [("Authorization", "Bearer secret")]
    assert captured["kwargs"]["open_timeout"] == 10
    assert json.loads(connection.sent[0]) == {"type": "pong", "ts": 12.5}
    assert context.exited is True


async def test_send_chat_requires_live_connection() -> None:
    client = GatewayClient(GatewayClientConfig(url="ws://gateway.test/ws", token="secret"))

    with pytest.raises(RuntimeError, match="gateway not connected"):
        await client.send_chat("hi")


async def test_send_chat_writes_session_and_model_hints() -> None:
    connection = FakeConnection()
    client = GatewayClient(
        GatewayClientConfig(url="ws://gateway.test/ws", token="secret", session_id="resume-me")
    )
    client._conn = connection

    await client.send_chat(
        "continue",
        preferred_model="google/gemma-4-e4b",
        provider_profile="lmstudio",
        router_model="executor",
        strict=True,
    )

    assert json.loads(connection.sent[0]) == {
        "type": "chat",
        "session_id": "resume-me",
        "message": "continue",
        "preferred_model": "google/gemma-4-e4b",
        "provider_profile": "lmstudio",
        "router_model": "executor",
        "strict": True,
    }


def test_jittered_backoff_stays_non_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ws_client.random, "uniform", lambda _low, _high: -10.0)

    assert ws_client._jittered(1.0) == 0.0


def test_jittered_backoff_applies_positive_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ws_client.random, "uniform", lambda _low, high: high)

    assert ws_client._jittered(4.0) == 5.0
