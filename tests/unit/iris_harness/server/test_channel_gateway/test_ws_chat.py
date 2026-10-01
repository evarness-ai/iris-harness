"""Tests for the channel gateway WebSocket endpoint."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from iris_harness.server.channel_gateway import main as gateway_main
from iris_harness.server.channel_gateway.main import create_app


@pytest.fixture(autouse=True)
def _enable_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_AUTH_SECRET", "test-secret")
    monkeypatch.setenv("IRIS_API_URL", "http://iris-api.test")


def _mock_iris_api(
    events: list[dict[str, Any]] | None = None,
    *,
    seen_payloads: list[dict[str, Any]] | None = None,
) -> httpx.AsyncClient:
    """Return an httpx client whose /chat/stream emits ``events`` as NDJSON."""
    payload_events = (
        events
        if events is not None
        else [
            {"event": "token", "text": "hi"},
            {
                "event": "done",
                "response": "hello",
                "intent": "chat",
                "agent_type": "system",
                "sources": [],
                "has_errors": False,
                "error_summary": None,
                "metadata": {},
            },
        ]
    )
    body = ("\n".join(json.dumps(e) for e in payload_events) + "\n").encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/stream"
        body_in = json.loads(request.content.decode("utf-8"))
        if seen_payloads is not None:
            seen_payloads.append(body_in)
        assert body_in["message"]
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/x-ndjson"},
        )

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


# ---------------------------------------------------------------------------
# Handshake / auth
# ---------------------------------------------------------------------------


def test_ws_rejects_handshake_without_token() -> None:
    app = create_app(http_client=_mock_iris_api())
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws") as ws:
                ws.receive_text()


def test_ws_rejects_when_shared_secret_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_AUTH_SECRET", raising=False)
    app = create_app(http_client=_mock_iris_api())
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws?token=test-secret") as ws:
                ws.receive_text()


def test_ws_rejects_wrong_bearer() -> None:
    app = create_app(http_client=_mock_iris_api())
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws", headers={"Authorization": "Bearer wrong"}) as ws:
                ws.receive_text()


def test_ws_accepts_correct_bearer_via_header() -> None:
    app = create_app(http_client=_mock_iris_api())
    with TestClient(app) as client:
        with client.websocket_connect("/ws", headers={"Authorization": "Bearer test-secret"}) as ws:
            ws.send_text(json.dumps({"type": "chat", "session_id": "s1", "message": "hi"}))
            hello = json.loads(ws.receive_text())
            assert hello == {"type": "hello", "session_id": "s1"}


def test_ws_accepts_correct_token_via_query_param() -> None:
    app = create_app(http_client=_mock_iris_api())
    with TestClient(app) as client:
        with client.websocket_connect("/ws?token=test-secret") as ws:
            ws.send_text(json.dumps({"type": "chat", "session_id": "s1", "message": "hi"}))
            assert json.loads(ws.receive_text())["type"] == "hello"


def test_ws_token_is_compared_in_constant_time(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both the header and the query form go through hmac.compare_digest, the house
    # pattern in foundation/auth.py, never a plain `!=`.
    compared: list[tuple[bytes, bytes]] = []
    real = gateway_main.hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        compared.append((a, b))
        return real(a, b)

    monkeypatch.setattr(gateway_main.hmac, "compare_digest", spy)
    app = create_app(http_client=_mock_iris_api())
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws?token=wrong") as ws:
                ws.receive_text()
        with client.websocket_connect("/ws", headers={"Authorization": "Bearer test-secret"}):
            pass
    assert compared == [(b"wrong", b"test-secret"), (b"test-secret", b"test-secret")]


def test_query_token_never_reaches_the_logs(caplog: pytest.LogCaptureFixture) -> None:
    # Our ingress line records the routed path only; uvicorn logs the handshake with
    # its query string, so its loggers carry a redaction filter.
    app = create_app(http_client=_mock_iris_api())
    with caplog.at_level(logging.DEBUG):
        with TestClient(app) as client:
            with client.websocket_connect("/ws?token=test-secret") as ws:
                ws.send_text(json.dumps({"type": "chat", "session_id": "s1", "message": "hi"}))
                ws.receive_text()
        # The exact call uvicorn makes on an accepted handshake.
        logging.getLogger("uvicorn.error").info(
            '%s - "WebSocket %s" [accepted]', "127.0.0.1:5000", "/ws?token=test-secret&x=1"
        )
        logging.getLogger("uvicorn.access").info('"GET /ws?access_token=test-secret" 403')

    rendered = [record.getMessage() for record in caplog.records]
    assert not [line for line in rendered if "test-secret" in line]
    assert any("/ws?token=[redacted]&x=1" in line for line in rendered)
    assert any("INGRESS WS /ws" in line for line in rendered)


# ---------------------------------------------------------------------------
# Chat proxying
# ---------------------------------------------------------------------------


def test_chat_frames_are_forwarded_from_iris_api() -> None:
    app = create_app(
        http_client=_mock_iris_api(
            [
                {"event": "token", "text": "hel"},
                {"event": "token", "text": "lo"},
                {
                    "event": "done",
                    "response": "hello",
                    "intent": "chat",
                    "agent_type": "system",
                    "sources": [],
                    "has_errors": False,
                    "error_summary": None,
                    "metadata": {},
                },
            ]
        )
    )
    with TestClient(app) as client:
        with client.websocket_connect("/ws?token=test-secret") as ws:
            ws.send_text(json.dumps({"type": "chat", "session_id": "s1", "message": "hi"}))
            frames = [json.loads(ws.receive_text()) for _ in range(4)]
    types = [f["type"] for f in frames]
    assert types == ["hello", "token", "token", "done"]
    assert frames[1]["text"] == "hel"
    assert frames[3]["response"] == "hello"


def test_chat_payload_round_trips_session_and_model_hints() -> None:
    seen_payloads: list[dict[str, Any]] = []
    app = create_app(http_client=_mock_iris_api(seen_payloads=seen_payloads))
    with TestClient(app) as client:
        with client.websocket_connect("/ws?token=test-secret") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "chat",
                        "session_id": "resume-me",
                        "message": "continue",
                        "preferred_model": "google/gemma-4-e4b",
                        "provider_profile": "lmstudio",
                        "router_model": "executor",
                        "strict": True,
                    }
                )
            )
            frames = [json.loads(ws.receive_text()) for _ in range(3)]

    assert [frame["type"] for frame in frames] == ["hello", "token", "done"]
    assert seen_payloads == [
        {
            "message": "continue",
            "session_id": "resume-me",
            # The gateway forwards its origin channel (defaults to "web" when the
            # WS frame doesn't specify one) so routine delivery defaults correctly.
            "channel": "web",
            "preferred_model": "google/gemma-4-e4b",
            "provider_profile": "lmstudio",
            "router_model": "executor",
            "strict": True,
        }
    ]


def test_chat_with_unknown_type_returns_error_frame() -> None:
    app = create_app(http_client=_mock_iris_api())
    with TestClient(app) as client:
        with client.websocket_connect("/ws?token=test-secret") as ws:
            ws.send_text(json.dumps({"type": "bogus"}))
            frame = json.loads(ws.receive_text())
    assert frame["type"] == "error"
    assert "unknown_type" in frame["error"]


def test_chat_with_missing_message_returns_error() -> None:
    app = create_app(http_client=_mock_iris_api())
    with TestClient(app) as client:
        with client.websocket_connect("/ws?token=test-secret") as ws:
            ws.send_text(json.dumps({"type": "chat", "session_id": "s1"}))
            frame = json.loads(ws.receive_text())
    assert frame["type"] == "error"
    assert frame["error"] == "missing_message"


def test_chat_with_invalid_json_returns_error_frame() -> None:
    app = create_app(http_client=_mock_iris_api())
    with TestClient(app) as client:
        with client.websocket_connect("/ws?token=test-secret") as ws:
            ws.send_text("not-json")
            frame = json.loads(ws.receive_text())
    assert frame["type"] == "error"
    assert frame["error"] == "invalid_json"


def test_bad_upstream_chunk_surfaces_error_and_stream_continues() -> None:
    done_event = {
        "event": "done",
        "response": "hello",
        "intent": "chat",
        "agent_type": "system",
        "sources": [],
        "has_errors": False,
        "error_summary": None,
        "metadata": {},
    }
    body = b"not-json\n" + json.dumps(done_event).encode("utf-8") + b"\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    app = create_app(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with TestClient(app) as client:
        with client.websocket_connect("/ws?token=test-secret") as ws:
            ws.send_text(json.dumps({"type": "chat", "session_id": "s1", "message": "hi"}))
            frames = [json.loads(ws.receive_text()) for _ in range(3)]

    assert frames[0] == {"type": "hello", "session_id": "s1"}
    assert frames[1] == {"type": "error", "error": "bad_upstream_chunk"}
    assert frames[2]["type"] == "done"


def test_upstream_error_surfaces_as_error_frame() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    bad_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(http_client=bad_http)
    with TestClient(app) as client:
        with client.websocket_connect("/ws?token=test-secret") as ws:
            ws.send_text(json.dumps({"type": "chat", "session_id": "s1", "message": "hi"}))
            assert json.loads(ws.receive_text())["type"] == "hello"
            frame = json.loads(ws.receive_text())
    assert frame["type"] == "error"
    assert frame["error"] == "upstream_500"


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_endpoint_reports_service_status() -> None:
    app = create_app(http_client=_mock_iris_api())
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "channel_gateway"
    assert body["iris_api"] == "http://iris-api.test"
    assert body["auth_required"] is True
