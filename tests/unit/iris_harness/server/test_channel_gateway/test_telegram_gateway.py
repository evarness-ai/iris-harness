"""Tests for Telegram integration in the channel gateway service."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from iris_harness.server.channel_gateway.main import create_app
from iris_harness.server.channel_gateway.telegram import (
    TelegramIrisChatBridge,
    build_telegram_runtime_from_env,
)


class FakeTelegramRuntime:
    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        self.starts += 1
        self._running = True

    def stop(self, *, join_timeout: float = 5.0) -> None:
        del join_timeout
        self.stops += 1
        self._running = False


def _mock_async_iris_api() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"event":"done","response":"ok"}\n')

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _mock_sync_iris_api(
    events: list[dict[str, Any]],
    *,
    seen_payloads: list[dict[str, Any]] | None = None,
) -> httpx.Client:
    body = ("\n".join(json.dumps(event) for event in events) + "\n").encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/stream"
        if seen_payloads is not None:
            seen_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_gateway_lifespan_starts_and_stops_telegram_runtime() -> None:
    runtime = FakeTelegramRuntime()
    app = create_app(
        http_client=_mock_async_iris_api(),
        telegram_runtime_factory=lambda: runtime,
    )

    with TestClient(app) as client:
        body = client.get("/health").json()
        assert body["telegram"] == {"configured": True, "running": True}
        assert runtime.starts == 1

    assert runtime.stops == 1
    assert runtime.running is False


def test_telegram_bridge_returns_final_done_response() -> None:
    seen_payloads: list[dict[str, Any]] = []
    bridge = TelegramIrisChatBridge(
        iris_api_url="http://iris-api.test",
        provider_profile="lmstudio",
        preferred_model="google/gemma-4-e4b",
        router_model="executor",
        strict=True,
        client=_mock_sync_iris_api(
            [
                {"event": "token", "text": "hel"},
                {"event": "token", "text": "lo"},
                {"event": "done", "response": "hello from iris"},
            ],
            seen_payloads=seen_payloads,
        ),
    )

    reply = bridge.reply("Say hi", "telegram:123")

    assert reply == "hello from iris"
    assert seen_payloads == [
        {
            "message": "Say hi",
            "session_id": "telegram:123",
            # The bridge declares its origin channel so routine delivery defaults
            # back to Telegram rather than console.
            "channel": "telegram",
            # Who reads the answer: the owner, unless the poller said a group does.
            "audience": "owner",
            "preferred_model": "google/gemma-4-e4b",
            "provider_profile": "lmstudio",
            "router_model": "executor",
            "strict": True,
        }
    ]


def test_telegram_bridge_falls_back_to_token_stream_without_done_response() -> None:
    bridge = TelegramIrisChatBridge(
        iris_api_url="http://iris-api.test",
        client=_mock_sync_iris_api(
            [
                {"event": "token", "text": "hel"},
                {"event": "token", "text": "lo"},
            ]
        ),
    )

    assert bridge.reply("Say hi", "telegram:123") == "hello"


def test_telegram_bridge_reports_upstream_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"offline")

    bridge = TelegramIrisChatBridge(
        iris_api_url="http://iris-api.test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert bridge.reply("hi", "telegram:123") == (
        "IRIS API returned HTTP 503. Please check the API logs."
    )


def test_build_telegram_runtime_from_env_returns_none_without_token(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    assert build_telegram_runtime_from_env(iris_api_url="http://iris-api.test") is None


def test_build_telegram_runtime_refuses_to_start_without_allowlist(monkeypatch) -> None:
    # Fail closed: a bot token with no allowlist must NOT start a poller that
    # would answer any Telegram user who finds the bot.
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_IDS", raising=False)

    assert build_telegram_runtime_from_env(iris_api_url="http://iris-api.test") is None


def test_build_telegram_runtime_from_env_uses_default_chat_as_allowlist(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_IDS", raising=False)

    runtime = build_telegram_runtime_from_env(
        iris_api_url="http://iris-api.test",
        upstream_client=_mock_sync_iris_api([{"event": "done", "response": "ok"}]),
        telegram_poll_client=httpx.Client(transport=httpx.MockTransport(lambda request: None)),
        telegram_send_client=httpx.Client(transport=httpx.MockTransport(lambda request: None)),
    )

    assert runtime is not None
    assert runtime._poller._allowed == frozenset({"123"})
    # TELEGRAM_ALLOWED_USER_IDS unset: no per-user check, exactly as before.
    assert runtime._poller._allowed_users == frozenset()
    runtime.stop()


def test_build_telegram_runtime_passes_the_optional_user_allowlist(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "42,43")

    runtime = build_telegram_runtime_from_env(
        iris_api_url="http://iris-api.test",
        upstream_client=_mock_sync_iris_api([{"event": "done", "response": "ok"}]),
        telegram_poll_client=httpx.Client(transport=httpx.MockTransport(lambda request: None)),
        telegram_send_client=httpx.Client(transport=httpx.MockTransport(lambda request: None)),
    )

    assert runtime is not None
    assert runtime._poller._allowed_users == frozenset({"42", "43"})
    runtime.stop()


# --- approvals from Telegram, answered through the API (owner, 2026-09-21) -----------

_APPROVAL = "3f9c0d2a-0000-4000-8000-000000000001"


def _approvals_api(posted: list[dict[str, Any]], *, respond_status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/governance/approvals":
            return httpx.Response(
                200,
                json={
                    "count": 1,
                    "approvals": [
                        {
                            "approval_id": _APPROVAL,
                            "signal": "Trash 2 emails",
                            "kind": "destructive",
                            "card": {"title": "Trash 2 emails"},
                        }
                    ],
                },
            )
        if (
            request.method == "POST"
            and request.url.path == f"/governance/approvals/{_APPROVAL}/respond"
        ):
            posted.append(json.loads(request.content))
            if respond_status != 200:
                return httpx.Response(respond_status, json={"detail": "already answered"})
            return httpx.Response(200, json={"resumed": True, "detail": "Trashed them."})
        return httpx.Response(404)

    return httpx.Client(base_url="http://iris-api.test", transport=httpx.MockTransport(handler))


def test_the_gateway_answers_a_tapped_approval_through_the_api() -> None:
    from iris_harness.server.channel_gateway.telegram import ApiApprovalBackend
    from iris_harness.services.channels.approval_commands import handle_approval_command

    posted: list[dict[str, Any]] = []
    backend = ApiApprovalBackend(iris_api_url="http://iris-api.test", client=_approvals_api(posted))

    ask = handle_approval_command(f"/ask {_APPROVAL}", "42", backend, allowed_users=frozenset())
    assert ask is not None and ask.text.startswith("Trash 2 emails?")
    assert posted == []  # asking runs nothing

    done = handle_approval_command(
        f"/approve {_APPROVAL}", "42", backend, allowed_users=frozenset()
    )
    assert posted == [{"status": "approved", "actor": "telegram:42"}]
    assert done is not None and done.text == "Approved: Trash 2 emails.\n\nTrashed them."


def test_an_approval_answered_elsewhere_is_reported_not_raised() -> None:
    from iris_harness.server.channel_gateway.telegram import ApiApprovalBackend
    from iris_harness.services.channels.approval_commands import handle_approval_command

    backend = ApiApprovalBackend(
        iris_api_url="http://iris-api.test", client=_approvals_api([], respond_status=409)
    )
    reply = handle_approval_command(
        f"/approve {_APPROVAL}", "42", backend, allowed_users=frozenset()
    )
    assert reply is not None and reply.text.startswith("Already answered")


def test_the_gateway_poller_answers_approval_commands(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    runtime = build_telegram_runtime_from_env(
        iris_api_url="http://iris-api.test",
        upstream_client=_mock_sync_iris_api([{"event": "done", "response": "ok"}]),
        telegram_poll_client=httpx.Client(transport=httpx.MockTransport(lambda request: None)),
        telegram_send_client=httpx.Client(transport=httpx.MockTransport(lambda request: None)),
    )
    assert runtime is not None
    try:
        assert runtime._poller._command_handler is not None
    finally:
        runtime.stop()


# --- a reminder's Done / Snooze from Telegram, through the API (PR 3b) -----------------

_REMINDER = "3f9c0d2a-0000-4000-8000-0000000000aa"


def _reminders_api(posted: list[tuple[str, dict[str, Any]]]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method != "POST" or not path.startswith("/api/v1/reminders/"):
            return httpx.Response(404)
        body = json.loads(request.content or b"{}")
        posted.append((path, body))
        if "/by-message/telegram/555/99/" in path:
            return httpx.Response(404, json={"detail": "no reminder sent that message"})
        if body.get("for") == "whenever":
            return httpx.Response(422, json={"detail": "I didn't catch that."})
        if path.endswith("/done"):
            return httpx.Response(
                200,
                json={
                    "reminder": {"id": _REMINDER, "status": "done"},
                    "undo": {"kind": "done"},
                    "next": {"id": "n1", "remind_at": "2026-10-05T13:00:00+00:00"},
                },
            )
        return httpx.Response(
            200,
            json={
                "reminder": {"id": _REMINDER, "remind_at": "2026-09-28T14:03:00+00:00"},
                "undo": {"kind": "snooze", "until_before": "2026-09-28T13:00:00+00:00"},
            },
        )

    return httpx.Client(base_url="http://iris-api.test", transport=httpx.MockTransport(handler))


def test_the_gateway_answers_reminder_buttons_and_replies_through_the_api(monkeypatch) -> None:
    from datetime import UTC, datetime
    from zoneinfo import ZoneInfo

    from iris_harness.server.channel_gateway.telegram import ApiReminderBackend
    from iris_harness.services.channels.reminder_commands import ReminderCommands

    posted: list[tuple[str, dict[str, Any]]] = []
    backend = ApiReminderBackend(iris_api_url="http://iris-api.test", client=_reminders_api(posted))
    commands = ReminderCommands(
        backend,
        allowed_users=frozenset(),
        tz=ZoneInfo("America/Chicago"),
        clock=lambda: datetime(2026, 9, 28, 13, 3, tzinfo=UTC),
    )

    done = commands(f"/done {_REMINDER}", "42")
    assert done is not None and done.text == "✅ Marked done. Next: Mon Oct 5 8:00 AM."
    snooze = commands(f"/snooze {_REMINDER} 1h", "42")
    assert snooze is not None and snooze.text == "⏰ Snoozed — I'll remind you at 9:03 AM."
    replied = commands.on_reply("snooze 1h", "42", "555", "70")
    assert replied is not None and replied.text.startswith("⏰ Snoozed")
    assert commands.on_reply("done", "42", "555", "99") is None  # not a reminder: chat
    assert posted == [
        (f"/api/v1/reminders/{_REMINDER}/done", {"source": "telegram"}),
        (f"/api/v1/reminders/{_REMINDER}/snooze", {"for": "1h", "source": "telegram"}),
        (
            "/api/v1/reminders/by-message/telegram/555/70/snooze",
            {"for": "snooze 1h", "source": "telegram"},
        ),
        ("/api/v1/reminders/by-message/telegram/555/99/done", {"source": "telegram"}),
    ]
    bad = commands(f"/snooze {_REMINDER} whenever", "42")
    assert bad is not None and bad.text.startswith("I didn't catch that")


def test_the_gateway_poller_chains_reminders_after_approvals(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    runtime = build_telegram_runtime_from_env(
        iris_api_url="http://iris-api.test",
        upstream_client=_mock_sync_iris_api([{"event": "done", "response": "ok"}]),
        telegram_poll_client=httpx.Client(transport=httpx.MockTransport(lambda request: None)),
        telegram_send_client=httpx.Client(transport=httpx.MockTransport(lambda request: None)),
    )
    assert runtime is not None
    try:
        poller = runtime._poller  # type: ignore[attr-defined]
        assert poller._reply_handler is not None  # replies to a reminder are routed
        assert poller._command_handler("hello", "42") is None  # chat, unchanged
    finally:
        runtime.stop()


def test_the_gateway_says_a_bill_is_already_paid() -> None:
    """Paid through the API on a bill already closed as paid: the API answers 200 with
    ``already_paid``; the gateway's reply names the bill instead of "isn't open any more"
    (PR 4 demo)."""
    from iris_harness.server.channel_gateway.telegram import ApiReminderBackend
    from iris_harness.services.channels.reminder_commands import ReminderCommands

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "reminder": {
                    "id": _REMINDER,
                    "status": "expired",
                    "bill": {"entity": "Example Card"},
                },
                "undo": {"kind": "done"},
                "next": None,
                "already_paid": True,
            },
        )

    client = httpx.Client(base_url="http://iris-api.test", transport=httpx.MockTransport(handler))
    backend = ApiReminderBackend(iris_api_url="http://iris-api.test", client=client)
    reply = ReminderCommands(backend, allowed_users=frozenset())(f"/paid {_REMINDER}", "42")
    assert reply is not None and reply.text == "✅ Example Card is already marked paid."


@pytest.mark.parametrize("audience", ["owner", "other"])
def test_the_bridge_forwards_who_reads_the_answer(audience: str) -> None:
    seen: list[dict[str, Any]] = []
    done = json.dumps({"event": "done", "response": "hi"}) + "\n"

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=done.encode())

    bridge = TelegramIrisChatBridge(
        iris_api_url="http://api.test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert bridge.reply("hello", "telegram:1", audience) == "hi"  # type: ignore[arg-type]
    assert seen[0]["audience"] == audience
