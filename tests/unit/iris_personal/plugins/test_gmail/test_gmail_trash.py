"""Gmail trash / untrash (ADR-0118 step 5): the calls, and a read-only grant's error."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httplib2
import pytest
from googleapiclient.errors import HttpError

from iris_personal.email.write_approvals import approve_mailbox_writes
from iris_personal.plugins.gmail import gmail_fetch, gmail_oauth

ACCT = "gmail:owner@gmail.com"


@pytest.fixture(autouse=True)
def _writes_approved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests are about what Gmail does once the owner approved mailbox writes
    (R4); the gate itself is ``test_gmail_write_gate.py``'s."""
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path / "data"))
    approve_mailbox_writes(ACCT, "test")


class _Request:
    def __init__(self, run: Any) -> None:
        self._run = run

    def execute(self) -> Any:
        return self._run()


class _Messages:
    """Gmail as the 2026-09-22 end-to-end run showed it: trash adds TRASH and drops
    INBOX (or SPAM); untrash only drops TRASH."""

    def __init__(self, labels: dict[str, set[str]], errors: dict[str, Any]) -> None:
        self.labels = labels
        self.calls: list[tuple[str, str]] = []
        self._errors = errors

    def _do(self, action: str, id: str, change: Any) -> _Request:
        def run() -> Any:
            self.calls.append((action, id))
            planned = self._errors.get(id)
            # An int fails every call; a list fails the next call with its head
            # (a status, or (status, reason)), None meaning "this call succeeds".
            outcome = planned.pop(0) if isinstance(planned, list) and planned else planned
            if isinstance(planned, list) and outcome is None:
                outcome = None
            if outcome is not None:
                status, reason = outcome if isinstance(outcome, tuple) else (outcome, "")
                body = json.dumps(
                    {
                        "error": {
                            "code": status,
                            "message": "error",
                            "errors": [{"reason": reason}] if reason else [],
                        }
                    }
                ).encode()
                raise HttpError(httplib2.Response({"status": str(status)}), body)
            change(self.labels[id])
            return {"id": id, "labelIds": sorted(self.labels[id])}

        return _Request(run)

    def trash(self, *, userId: str, id: str) -> _Request:
        return self._do(
            "trash", id, lambda ls: (ls.add("TRASH"), ls.difference_update({"INBOX", "SPAM"}))
        )

    def untrash(self, *, userId: str, id: str) -> _Request:
        return self._do("untrash", id, lambda ls: ls.discard("TRASH"))

    def get(self, *, userId: str, id: str, format: str) -> _Request:
        return self._do("get", id, lambda ls: None)

    def batchModify(self, *, userId: str, body: dict[str, Any]) -> _Request:
        def run() -> None:
            self.calls.append(("batchModify", ",".join(body["ids"])))
            for mid in body["ids"]:
                self.labels[mid].update(body["addLabelIds"])

        return _Request(run)


class _Batch:
    def __init__(self, service: _Service, callback: Any) -> None:
        self._service, self._callback, self._items = service, callback, []

    def add(self, request: _Request, request_id: str) -> None:
        self._items.append((request_id, request))

    def execute(self) -> None:
        self._service.batch_sizes.append(len(self._items))
        for request_id, request in self._items:
            try:
                self._callback(request_id, request.execute(), None)
            except HttpError as exc:
                self._callback(request_id, None, exc)


class _Service:
    def __init__(
        self, labels: dict[str, set[str]] | None = None, errors: dict[str, Any] | None = None
    ) -> None:
        self.messages_api = _Messages(labels or {}, errors or {})
        self.batch_sizes: list[int] = []

    def new_batch_http_request(self, callback: Any) -> _Batch:
        return _Batch(self, callback)

    def users(self) -> Any:
        service = self

        class _Users:
            def messages(self) -> _Messages:
                return service.messages_api

        return _Users()


def _inbox(*ids: str) -> dict[str, set[str]]:
    return {mid: {"INBOX", "CATEGORY_PROMOTIONS", "UNREAD"} for mid in ids}


def _use(monkeypatch: pytest.MonkeyPatch, service: _Service) -> None:
    monkeypatch.setattr(gmail_fetch, "_service_for", lambda account_id: service)
    monkeypatch.setattr(gmail_fetch.time, "sleep", lambda seconds: None)


def test_trash_and_untrash_go_through_batch_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _Service(_inbox("a", "b"))
    _use(monkeypatch, service)
    assert gmail_fetch.trash_messages(ACCT, ["a", "b"]) == ["a", "b"]
    assert gmail_fetch.untrash_messages(ACCT, ["a"]) == {"a": ["CATEGORY_PROMOTIONS", "UNREAD"]}
    assert service.messages_api.calls == [("trash", "a"), ("trash", "b"), ("untrash", "a")]
    assert service.batch_sizes == [2, 1]  # one round trip per batch, not per message


def test_requests_go_fifty_to_a_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = [f"m{i}" for i in range(120)]
    service = _Service(_inbox(*ids))
    _use(monkeypatch, service)
    assert gmail_fetch.trash_messages(ACCT, ids) == ids
    assert service.batch_sizes == [10] * 12  # 50 at once drew 429s on the real account


def test_a_message_already_gone_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    _use(monkeypatch, _Service(_inbox("a", "b", "c"), {"b": 404}))
    assert gmail_fetch.trash_messages(ACCT, ["a", "b", "c"]) == ["a", "c"]


def test_rate_limited_items_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """2026-09-22: Gmail answered most of a real 200-message batch with 429 or
    403 userRateLimitExceeded; both are a wait, not a refusal."""
    service = _Service(
        _inbox("a", "b", "c"), {"a": [429, None], "b": [(403, "userRateLimitExceeded"), None]}
    )
    _use(monkeypatch, service)
    assert gmail_fetch.trash_messages(ACCT, ["a", "b", "c"]) == ["a", "b", "c"]
    assert service.batch_sizes == [3, 2]  # the second round is only the limited two


def test_a_rate_limit_that_never_clears_skips_the_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _use(monkeypatch, _Service(_inbox("a", "b"), {"a": 429}))
    assert gmail_fetch.trash_messages(ACCT, ["a", "b"]) == ["b"]


def test_a_read_only_grant_says_how_to_fix_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _use(monkeypatch, _Service(_inbox("a"), {"a": (403, "insufficientPermissions")}))
    with pytest.raises(PermissionError) as err:
        gmail_fetch.trash_messages(ACCT, ["a"])
    assert "iris auth gmail login --user owner@gmail.com" in str(err.value)


def test_restore_puts_back_what_the_trash_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    """2026-09-22: untrash alone left 200 restored promotions archived, and one that
    had been trashed out of Spam must go back to Spam, not the inbox."""
    from iris_personal.plugins.gmail.provider import GmailProvider

    labels = _inbox("promo")
    labels["spammy"] = {"SPAM", "UNREAD"}
    service = _Service(labels)
    _use(monkeypatch, service)
    monkeypatch.setattr(gmail_fetch, "_parse_message", lambda payload, account_id: payload["id"])
    provider = GmailProvider()

    before = provider.current_labels(ACCT, ["promo", "spammy"])
    provider.trash_messages(ACCT, ["promo", "spammy"])
    assert "INBOX" not in labels["promo"] and "SPAM" not in labels["spammy"]
    restored = provider.restore_messages(ACCT, ["promo", "spammy"], labels_before=before)

    assert restored == ["promo", "spammy"]
    assert labels["promo"] == {"INBOX", "CATEGORY_PROMOTIONS", "UNREAD"}
    assert labels["spammy"] == {"SPAM", "UNREAD"}


def test_restore_without_labels_leaves_gmails_untrash_as_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iris_personal.plugins.gmail.provider import GmailProvider

    labels = _inbox("old")
    service = _Service(labels)
    _use(monkeypatch, service)
    monkeypatch.setattr(gmail_fetch, "_parse_message", lambda payload, account_id: payload["id"])
    GmailProvider().trash_messages(ACCT, ["old"])
    assert GmailProvider().restore_messages(ACCT, ["old"]) == ["old"]
    assert "INBOX" not in labels["old"]  # a ledger row from before labels were kept
    assert not any(call[0] == "batchModify" for call in service.messages_api.calls)


def test_new_consents_ask_for_modify_never_full_access() -> None:
    assert gmail_oauth.DEFAULT_SCOPES == ["https://www.googleapis.com/auth/gmail.modify"]
    assert all("mail.google.com" not in s for s in gmail_oauth.DEFAULT_SCOPES)


def test_restore_does_not_trust_the_untrash_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even if untrash reports INBOX, the message is put back in INBOX explicitly
    (2026-09-22: 69 of 200 restored emails stayed archived, with no error)."""
    from iris_personal.plugins.gmail.provider import GmailProvider

    labels = _inbox("p")
    service = _Service(labels)
    _use(monkeypatch, service)
    monkeypatch.setattr(gmail_fetch, "_parse_message", lambda payload, account_id: payload["id"])
    real_untrash = gmail_fetch.untrash_messages

    def _misreporting_untrash(account_id: str, ids: list[str]) -> dict[str, list[str]]:
        return {mid: [*labels_, "INBOX"] for mid, labels_ in real_untrash(account_id, ids).items()}

    provider = GmailProvider()
    before = provider.current_labels(ACCT, ["p"])
    provider.trash_messages(ACCT, ["p"])
    monkeypatch.setattr(gmail_fetch, "untrash_messages", _misreporting_untrash)
    provider.restore_messages(ACCT, ["p"], labels_before=before)

    assert "INBOX" in labels["p"]
