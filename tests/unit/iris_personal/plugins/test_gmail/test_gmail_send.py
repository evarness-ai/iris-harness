"""Gmail send (ADR-0118 amendment): the ``users.messages.send`` payload — a base64url
RFC 2822 plain-text message, threaded when it is a reply — and a read-only grant's error.
A fake Gmail service records the request; nothing is sent."""

from __future__ import annotations

import base64
import email
from email import policy
from pathlib import Path
from typing import Any

import httplib2
import pytest
from googleapiclient.errors import HttpError

from iris_personal.plugins.gmail import gmail_fetch
from iris_personal.plugins.gmail.provider import GmailProvider

ACCT = "gmail:owner@gmail.com"


class _Call:
    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome

    def execute(self) -> Any:
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class _Messages:
    def __init__(self, status: int | None) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self._status = status

    def send(self, *, userId: str, body: dict[str, Any]) -> _Call:
        self.requests.append((userId, body))
        if self._status is not None:
            return _Call(HttpError(httplib2.Response({"status": str(self._status)}), b"{}"))
        return _Call({"id": "18c0ffee00000001", "threadId": body.get("threadId", "new")})


class _Service:
    def __init__(self, status: int | None = None) -> None:
        self.messages_api = _Messages(status)

    def users(self) -> Any:
        service = self

        class _Users:
            def messages(self) -> _Messages:
                return service.messages_api

        return _Users()


def _decode(raw: str) -> email.message.EmailMessage:
    parsed = email.message_from_bytes(base64.urlsafe_b64decode(raw), policy=policy.default)
    assert isinstance(parsed, email.message.EmailMessage)
    return parsed


def test_a_new_message_is_plain_text_rfc_2822(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _Service()
    monkeypatch.setattr(gmail_fetch, "_service_for", lambda account_id: service)

    sent_id = gmail_fetch.send_message(
        ACCT,
        to=["Bob <bob@example.com>"],
        cc=["carol@example.org"],
        subject="Lunch — Friday?",
        body="Friday at 1?\n\nRobin",
    )

    assert sent_id == "18c0ffee00000001"
    ((user_id, request),) = service.messages_api.requests
    assert user_id == "me"
    assert set(request) == {"raw"}  # no threadId on a new message
    assert "+" not in request["raw"] and "/" not in request["raw"]  # base64url, not base64
    msg = _decode(request["raw"])
    assert msg["From"] == "owner@gmail.com"
    assert msg["To"] == "Bob <bob@example.com>"
    assert msg["Cc"] == "carol@example.org"
    assert msg["Subject"] == "Lunch — Friday?"
    assert msg["In-Reply-To"] is None and msg["References"] is None
    assert msg.get_content_type() == "text/plain"
    assert not msg.is_multipart()  # no HTML part, no attachments
    assert msg.get_content().replace("\r\n", "\n").rstrip("\n") == "Friday at 1?\n\nRobin"


def test_a_reply_carries_the_thread_and_its_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _Service()
    monkeypatch.setattr(gmail_fetch, "_service_for", lambda account_id: service)

    gmail_fetch.send_message(
        ACCT,
        to=["alice@example.com"],
        cc=[],
        subject="Re: Dinner plans",
        body="Count me in.",
        in_reply_to="<abc@mail.example.com>",
        references="<r0@x>",
        thread_id="t-1",
    )

    ((_user, request),) = service.messages_api.requests
    assert request["threadId"] == "t-1"
    msg = _decode(request["raw"])
    assert msg["In-Reply-To"] == "<abc@mail.example.com>"
    assert msg["References"] == "<r0@x> <abc@mail.example.com>"
    assert msg["Cc"] is None


def test_a_read_only_grant_says_how_to_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gmail_fetch, "_service_for", lambda account_id: _Service(status=403))
    with pytest.raises(gmail_fetch.GmailScopeError, match="iris auth gmail login"):
        gmail_fetch.send_message(ACCT, to=["b@example.com"], cc=[], subject="s", body="b")


def test_another_gmail_error_is_raised_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gmail_fetch, "_service_for", lambda account_id: _Service(status=400))
    with pytest.raises(HttpError):
        gmail_fetch.send_message(ACCT, to=["b@example.com"], cc=[], subject="s", body="b")


def test_the_provider_delegates_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(account_id: str, **kwargs: Any) -> str:
        seen.update(account_id=account_id, **kwargs)
        return "id-1"

    monkeypatch.setattr(gmail_fetch, "send_message", fake)
    out = GmailProvider().send_message(ACCT, to=("a@example.com",), subject="s", body="b")
    assert out == "id-1"
    assert seen["to"] == ["a@example.com"] and seen["cc"] == []
    assert seen["thread_id"] is None


def test_gmail_modify_covers_messages_send() -> None:
    """The scope question, answered from Google's own discovery document (shipped with
    google-api-python-client): the grant trash already asks for is enough to send."""
    import json

    import googleapiclient

    doc = Path(googleapiclient.__file__).parent / "discovery_cache" / "documents" / "gmail.v1.json"
    methods = json.loads(doc.read_text())["resources"]["users"]["resources"]["messages"]
    scopes = methods["methods"]["send"]["scopes"]
    from iris_personal.plugins.gmail.gmail_oauth import DEFAULT_SCOPES

    assert DEFAULT_SCOPES == ["https://www.googleapis.com/auth/gmail.modify"]
    assert DEFAULT_SCOPES[0] in scopes
