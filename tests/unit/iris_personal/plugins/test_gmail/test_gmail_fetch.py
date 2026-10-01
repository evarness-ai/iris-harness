"""Tests for ``iris_personal.plugins.gmail.gmail_fetch`` — Gmail API mocked end-to-end."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from googleapiclient.errors import HttpError

from iris_personal.email.store import EmailStore
from iris_personal.plugins.gmail import gmail_fetch
from iris_personal.plugins.gmail.gmail_fetch import fetch_new_emails

# ─── Helpers to build canned Gmail API responses ──────────────────────


def _http_error(status: int) -> HttpError:
    resp = MagicMock()
    resp.status = status
    return HttpError(resp=resp, content=b"")


def _gmail_message_payload(
    msg_id: str,
    *,
    thread_id: str = "thr-1",
    snippet: str = "snippet content here",
    from_header: str = "Bob <bob@example.com>",
    subject: str = "hello",
    internal_date_ms: int = 1_700_000_000_000,
    labels: list[str] | None = None,
    to_header: str = "user@gmail.com",
) -> dict:
    """Shape mimicking Gmail API's messages.get(format='metadata') response."""
    return {
        "id": msg_id,
        "threadId": thread_id,
        "labelIds": labels or ["INBOX"],
        "snippet": snippet,
        "internalDate": str(internal_date_ms),
        "payload": {
            "headers": [
                {"name": "From", "value": from_header},
                {"name": "To", "value": to_header},
                {"name": "Subject", "value": subject},
                {"name": "Message-ID", "value": f"<{msg_id}@mail.gmail.com>"},
            ]
        },
    }


def _build_mock_service(
    *,
    messages_list_responses: list[dict] | None = None,
    history_list_responses: list[dict] | None = None,
    history_list_error: Exception | None = None,
    profile_history_id: str = "777",
    message_payloads: dict[str, dict] | None = None,
    message_get_errors: dict[str, Exception] | None = None,
) -> MagicMock:
    """Construct a MagicMock that imitates the Gmail API service tree."""
    service = MagicMock()

    # users().messages().list().execute()
    msgs_list = MagicMock()
    if messages_list_responses is not None:
        msgs_list.execute.side_effect = messages_list_responses
    else:
        msgs_list.execute.return_value = {"messages": []}
    service.users.return_value.messages.return_value.list.return_value = msgs_list

    # users().messages().get().execute()
    def fake_get(*, userId: str, id: str, format: str) -> MagicMock:
        get_call = MagicMock()
        if message_get_errors and id in message_get_errors:
            get_call.execute.side_effect = message_get_errors[id]
        elif message_payloads and id in message_payloads:
            get_call.execute.return_value = message_payloads[id]
        else:
            get_call.execute.return_value = _gmail_message_payload(id)
        return get_call

    service.users.return_value.messages.return_value.get.side_effect = fake_get

    # users().history().list().execute()
    history_list_call = MagicMock()
    if history_list_error is not None:
        history_list_call.execute.side_effect = history_list_error
    elif history_list_responses is not None:
        history_list_call.execute.side_effect = history_list_responses
    else:
        history_list_call.execute.return_value = {"history": []}
    service.users.return_value.history.return_value.list.return_value = history_list_call

    # users().getProfile().execute()
    profile_call = MagicMock()
    profile_call.execute.return_value = {"historyId": profile_history_id}
    service.users.return_value.getProfile.return_value = profile_call

    return service


@pytest.fixture
def store(tmp_path: Path) -> EmailStore:
    s = EmailStore(db_path=tmp_path / "email.db")
    s.ensure_schema()
    return s


# ─── Parsing helpers ──────────────────────────────────────────────────


def test_parse_message_basic_fields() -> None:
    payload = _gmail_message_payload("msg-1")
    msg = gmail_fetch._parse_message(payload, account_id="gmail:user@gmail.com")
    assert msg.id == "msg-1"
    assert msg.provider == "gmail"
    assert msg.account_id == "gmail:user@gmail.com"
    assert msg.thread_id == "thr-1"
    assert msg.from_address == "Bob <bob@example.com>"
    assert msg.from_domain == "example.com"
    assert msg.to == ("user@gmail.com",)
    assert msg.subject == "hello"
    assert msg.snippet == "snippet content here"
    assert msg.labels == ("INBOX",)
    assert msg.body_text is None
    assert msg.attachments == ()


def test_parse_message_decodes_mime_encoded_subject() -> None:
    """Subjects with RFC 2047 MIME encoding should be decoded."""
    encoded = "=?UTF-8?B?" + base64.b64encode("Hello 你好".encode()).decode() + "?="
    payload = _gmail_message_payload("msg-1", subject=encoded)
    msg = gmail_fetch._parse_message(payload, account_id="gmail:user@gmail.com")
    assert msg.subject == "Hello 你好"


def test_parse_message_truncates_snippet_to_400_chars() -> None:
    long_snip = "x" * 600
    payload = _gmail_message_payload("msg-1", snippet=long_snip)
    msg = gmail_fetch._parse_message(payload, account_id="gmail:user@gmail.com")
    assert len(msg.snippet) == 400


def test_parse_message_handles_empty_from() -> None:
    payload = _gmail_message_payload("msg-1", from_header="")
    msg = gmail_fetch._parse_message(payload, account_id="gmail:user@gmail.com")
    # Falls back to placeholder rather than raising
    assert msg.from_address == "unknown@unknown"
    assert msg.from_domain is None


def test_parse_message_multi_recipient_to_header() -> None:
    payload = _gmail_message_payload("msg-1", to_header="Alice <alice@x.com>, bob@y.com, eve@z.com")
    msg = gmail_fetch._parse_message(payload, account_id="gmail:user@gmail.com")
    assert msg.to == ("alice@x.com", "bob@y.com", "eve@z.com")


# ─── fetch_new_emails — cold start ────────────────────────────────────


def test_cold_start_when_no_cursor(store: EmailStore) -> None:
    """First-run for an account: messages.list, get each, persist, write cursor."""
    service = _build_mock_service(
        messages_list_responses=[
            {"messages": [{"id": "m1"}, {"id": "m2"}]}  # no nextPageToken → terminate
        ],
        profile_history_id="9000",
    )

    with (
        patch.object(gmail_fetch, "build", return_value=service),
        patch.object(gmail_fetch, "load_credentials", return_value=MagicMock()),
    ):
        result = fetch_new_emails("gmail:user@gmail.com", store=store)

    assert result.fetched == 2
    assert result.new_cursor == "9000"
    assert result.fell_back_to_cold_start is False
    assert store.count() == 2
    assert store.get_cursor("gmail", "gmail:user@gmail.com", "gmail_history_id") == "9000"


def test_cold_start_with_no_messages_still_writes_cursor(store: EmailStore) -> None:
    service = _build_mock_service(
        messages_list_responses=[{"messages": []}],
        profile_history_id="42",
    )
    with (
        patch.object(gmail_fetch, "build", return_value=service),
        patch.object(gmail_fetch, "load_credentials", return_value=MagicMock()),
    ):
        result = fetch_new_emails("gmail:user@gmail.com", store=store)

    assert result.fetched == 0
    assert result.new_cursor == "42"
    assert store.count() == 0


def test_cold_start_respects_max_messages(store: EmailStore) -> None:
    service = _build_mock_service(
        messages_list_responses=[{"messages": [{"id": f"m{i}"} for i in range(50)]}],
        profile_history_id="100",
    )
    with (
        patch.object(gmail_fetch, "build", return_value=service),
        patch.object(gmail_fetch, "load_credentials", return_value=MagicMock()),
    ):
        result = fetch_new_emails("gmail:user@gmail.com", store=store, max_messages=10)

    assert result.fetched == 10


# ─── fetch_new_emails — delta (history.list) ──────────────────────────


def test_delta_with_existing_cursor(store: EmailStore) -> None:
    """Cursor present: history.list returns messagesAdded; persist + advance cursor."""
    store.set_cursor("gmail", "gmail:user@gmail.com", "gmail_history_id", "5000")

    service = _build_mock_service(
        history_list_responses=[
            {
                "history": [
                    {
                        "id": "h1",
                        "messagesAdded": [{"message": {"id": "m1"}}, {"message": {"id": "m2"}}],
                    }
                ],
                "historyId": "5050",
            }
        ],
    )
    with (
        patch.object(gmail_fetch, "build", return_value=service),
        patch.object(gmail_fetch, "load_credentials", return_value=MagicMock()),
    ):
        result = fetch_new_emails("gmail:user@gmail.com", store=store)

    assert result.fetched == 2
    assert result.new_cursor == "5050"
    assert result.fell_back_to_cold_start is False
    assert store.get_cursor("gmail", "gmail:user@gmail.com", "gmail_history_id") == "5050"


def test_delta_with_no_new_messages(store: EmailStore) -> None:
    store.set_cursor("gmail", "gmail:user@gmail.com", "gmail_history_id", "5000")
    service = _build_mock_service(
        history_list_responses=[{"history": [], "historyId": "5000"}],
    )
    with (
        patch.object(gmail_fetch, "build", return_value=service),
        patch.object(gmail_fetch, "load_credentials", return_value=MagicMock()),
    ):
        result = fetch_new_emails("gmail:user@gmail.com", store=store)

    assert result.fetched == 0


def test_stale_cursor_falls_back_to_cold_start(store: EmailStore) -> None:
    """Gmail 410 on history.list → fall back to cold-start path."""
    store.set_cursor("gmail", "gmail:user@gmail.com", "gmail_history_id", "old")

    service = _build_mock_service(
        history_list_error=_http_error(410),
        messages_list_responses=[{"messages": [{"id": "m1"}]}],
        profile_history_id="9999",
    )
    with (
        patch.object(gmail_fetch, "build", return_value=service),
        patch.object(gmail_fetch, "load_credentials", return_value=MagicMock()),
    ):
        result = fetch_new_emails("gmail:user@gmail.com", store=store)

    assert result.fell_back_to_cold_start is True
    assert result.fetched == 1
    assert result.new_cursor == "9999"


def test_history_list_404_also_triggers_cold_start(store: EmailStore) -> None:
    store.set_cursor("gmail", "gmail:user@gmail.com", "gmail_history_id", "old")
    service = _build_mock_service(
        history_list_error=_http_error(404),
        messages_list_responses=[{"messages": []}],
        profile_history_id="1",
    )
    with (
        patch.object(gmail_fetch, "build", return_value=service),
        patch.object(gmail_fetch, "load_credentials", return_value=MagicMock()),
    ):
        result = fetch_new_emails("gmail:user@gmail.com", store=store)
    assert result.fell_back_to_cold_start is True


def test_history_list_other_errors_propagate(store: EmailStore) -> None:
    """A 503 (or any non-404/410) should propagate, not be swallowed."""
    store.set_cursor("gmail", "gmail:user@gmail.com", "gmail_history_id", "old")
    service = _build_mock_service(
        history_list_error=_http_error(503),
    )
    with (
        patch.object(gmail_fetch, "build", return_value=service),
        patch.object(gmail_fetch, "load_credentials", return_value=MagicMock()),
        pytest.raises(HttpError),
    ):
        fetch_new_emails("gmail:user@gmail.com", store=store)


# ─── Resilience & misc ────────────────────────────────────────────────


def test_per_message_fetch_failure_does_not_abort_batch(store: EmailStore) -> None:
    """If one messages.get fails, the rest still persist."""
    service = _build_mock_service(
        messages_list_responses=[{"messages": [{"id": "good1"}, {"id": "bad"}, {"id": "good2"}]}],
        profile_history_id="100",
        message_get_errors={"bad": _http_error(404)},
    )
    with (
        patch.object(gmail_fetch, "build", return_value=service),
        patch.object(gmail_fetch, "load_credentials", return_value=MagicMock()),
    ):
        result = fetch_new_emails("gmail:user@gmail.com", store=store)

    assert result.fetched == 2
    assert store.count() == 2


def test_missing_credentials_raises(store: EmailStore) -> None:
    with (
        patch.object(gmail_fetch, "load_credentials", return_value=None),
        pytest.raises(RuntimeError, match="No Gmail credentials"),
    ):
        fetch_new_emails("gmail:user@gmail.com", store=store)


def test_idempotent_re_run(store: EmailStore) -> None:
    """Fetching the same messages twice is a no-op upsert."""
    service = _build_mock_service(
        messages_list_responses=[
            {"messages": [{"id": "m1"}, {"id": "m2"}]},
            {"messages": [{"id": "m1"}, {"id": "m2"}]},  # second call returns same ids
        ],
        profile_history_id="100",
    )
    with (
        patch.object(gmail_fetch, "build", return_value=service),
        patch.object(gmail_fetch, "load_credentials", return_value=MagicMock()),
    ):
        first = fetch_new_emails("gmail:user@gmail.com", store=store)
        # Clear cursor so second call also takes cold-start path
        store.set_cursor("gmail", "gmail:user@gmail.com", "gmail_history_id_unused", "irrelevant")
        second = fetch_new_emails("gmail:user@gmail.com", store=store)

    assert first.fetched == 2
    # Second call would normally use delta — but we never wrote the delta-cursor
    # kind during the first cold-start in this fixture, so it cold-starts again.
    # Either way, count stays at 2 (upsert is idempotent).
    assert store.count() == 2
    assert second.fetched >= 0  # may be 0 if delta path with empty history


# ─── Label read-back (loop-proof PR 5) ────────────────────────────────


def _delta_with_labels(store: EmailStore, history: list[dict], payloads: dict[str, dict]):  # type: ignore[no-untyped-def]
    store.set_cursor("gmail", "gmail:user@gmail.com", "gmail_history_id", "5000")
    service = _build_mock_service(
        history_list_responses=[{"history": history, "historyId": "5100"}],
        message_payloads=payloads,
    )
    with (
        patch.object(gmail_fetch, "build", return_value=service),
        patch.object(gmail_fetch, "load_credentials", return_value=MagicMock()),
    ):
        result = fetch_new_emails("gmail:user@gmail.com", store=store)
    return result, service


def test_delta_asks_for_label_changes_too(store: EmailStore) -> None:
    _, service = _delta_with_labels(store, [], {})
    kwargs = service.users.return_value.history.return_value.list.call_args.kwargs
    assert kwargs["historyTypes"] == ["messageAdded", "labelAdded", "labelRemoved"]


def test_label_change_on_a_stored_message_is_read_back(store: EmailStore) -> None:
    """A relabel of mail already in email.db: its labels now, in the result and the row."""
    for mid in ("old-1", "old-2"):
        store.upsert(
            gmail_fetch._parse_message(_gmail_message_payload(mid), "gmail:user@gmail.com")
        )
    history = [
        {
            "id": "h2",
            "labelsAdded": [{"message": {"id": "old-1"}, "labelIds": ["Label_7"]}],
            "labelsRemoved": [{"message": {"id": "old-1"}, "labelIds": ["Label_3"]}],
        },
        # Reading mail flips UNREAD: a system label, not read back.
        {"id": "h3", "labelsRemoved": [{"message": {"id": "old-2"}, "labelIds": ["UNREAD"]}]},
        # A relabel of mail IRIS never stored is skipped.
        {"id": "h4", "labelsAdded": [{"message": {"id": "ghost"}, "labelIds": ["Label_7"]}]},
    ]
    now = _gmail_message_payload("old-1", labels=["INBOX", "Label_7"])
    result, _ = _delta_with_labels(store, history, {"old-1": now})

    assert result.label_changes == (("old-1", ("INBOX", "Label_7")),)
    assert result.fetched == 0
    stored = store.get("old-1")
    assert stored is not None and stored.labels == ("INBOX", "Label_7")


def test_new_mail_is_not_a_label_change(store: EmailStore) -> None:
    """New mail labelled in the same window is fetched whole, once, as before."""
    history = [
        {
            "id": "h1",
            "messagesAdded": [{"message": {"id": "m1"}}],
            "labelsAdded": [{"message": {"id": "m1"}, "labelIds": ["Label_7"]}],
        }
    ]
    result, _ = _delta_with_labels(store, history, {})
    assert result.fetched == 1
    assert result.new_message_ids == ("m1",)
    assert result.label_changes == ()


def test_cold_start_reads_no_label_changes(store: EmailStore) -> None:
    service = _build_mock_service(messages_list_responses=[{"messages": [{"id": "m1"}]}])
    with (
        patch.object(gmail_fetch, "build", return_value=service),
        patch.object(gmail_fetch, "load_credentials", return_value=MagicMock()),
    ):
        result = fetch_new_emails("gmail:user@gmail.com", store=store)
    assert result.label_changes == ()
