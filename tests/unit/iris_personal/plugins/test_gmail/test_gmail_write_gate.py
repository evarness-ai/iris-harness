"""Gmail's mailbox writes wait for the account's write approval (OSS plan R4).

Every write -- label create, label add/remove, trash, untrash, the restore's label
put-back -- asks the email library's one gate before any Gmail request. A fake service
records every request; without an approval it must record none. No network.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from iris_personal.email.write_approvals import approve_mailbox_writes, revoke_mailbox_writes
from iris_personal.plugins.gmail import gmail_fetch
from iris_personal.plugins.gmail.provider import GmailProvider

ACCT = "gmail:owner@example.com"
APPROVE = f"iris email writes approve --account {ACCT}"


class _Req:
    def __init__(self, log: list[str], name: str, result: Any) -> None:
        self._log, self._name, self._result = log, name, result

    def execute(self) -> Any:
        self._log.append(self._name)
        return self._result


class _Batch:
    def __init__(self, callback: Any) -> None:
        self._callback, self._items = callback, []

    def add(self, request: _Req, request_id: str) -> None:
        self._items.append((request_id, request))

    def execute(self) -> None:
        for request_id, request in self._items:
            self._callback(request_id, request.execute(), None)


class _Service:
    """Every request, read or write, lands in ``log``."""

    def __init__(self, labels: dict[str, str] | None = None) -> None:
        self.log: list[str] = []
        self.labels = dict(labels or {})
        self.built: list[str] = []  # accounts a client was built for

    def new_batch_http_request(self, callback: Any) -> _Batch:
        return _Batch(callback)

    def users(self) -> Any:
        svc = self

        class _Labels:
            def list(self, *, userId: str) -> _Req:
                rows = [{"id": i, "name": n} for n, i in svc.labels.items()]
                return _Req(svc.log, "labels.list", {"labels": rows})

            def create(self, *, userId: str, body: dict[str, Any]) -> _Req:
                new_id = f"Label_{len(svc.labels) + 1}"
                svc.labels[body["name"]] = new_id
                return _Req(svc.log, "labels.create", {"id": new_id})

        class _Messages:
            def trash(self, *, userId: str, id: str) -> _Req:
                return _Req(svc.log, "trash", {"labelIds": ["TRASH"]})

            def untrash(self, *, userId: str, id: str) -> _Req:
                return _Req(svc.log, "untrash", {"labelIds": []})

            def get(self, *, userId: str, id: str, format: str) -> _Req:
                return _Req(svc.log, "get", {"id": id, "labelIds": ["INBOX"]})

            def batchModify(self, *, userId: str, body: dict[str, Any]) -> _Req:
                return _Req(svc.log, "batchModify", None)

            def send(self, *, userId: str, body: dict[str, Any]) -> _Req:
                return _Req(svc.log, "send", {"id": "sent-1"})

        class _Users:
            def labels(self) -> _Labels:
                return _Labels()

            def messages(self) -> _Messages:
                return _Messages()

        return _Users()


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Service]:
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path / "data"))
    gmail_fetch.forget_labels()
    svc = _Service()

    def _service_for(account_id: str) -> _Service:
        svc.built.append(account_id)
        return svc

    monkeypatch.setattr(gmail_fetch, "_service_for", _service_for)
    monkeypatch.setattr(gmail_fetch.time, "sleep", lambda seconds: None)
    yield svc
    gmail_fetch.forget_labels()


def _writes() -> dict[str, Any]:
    """Each Gmail write, as the provider (the only caller) runs it."""
    p = GmailProvider()
    return {
        "change labels": lambda: p.modify_labels(ACCT, ["m1"], ["Label_9"], []),
        "move mail to Trash": lambda: p.trash_messages(ACCT, ["m1"]),
        "restore mail from Trash": lambda: gmail_fetch.untrash_messages(ACCT, ["m1"]),
        "put labels back": lambda: gmail_fetch.add_labels(ACCT, {"m1": {"INBOX"}}),
        "create the label IRIS": lambda: p.ensure_labels(ACCT, ["IRIS/Bill"]),
    }


@pytest.mark.parametrize("what", list(_writes()))
def test_without_approval_a_write_raises_before_any_request(service: _Service, what: str) -> None:
    with pytest.raises(PermissionError) as refused:
        _writes()[what]()
    assert f"did not {what}" in str(refused.value)
    assert APPROVE in str(refused.value)  # the refusal names the one-time command
    writes = [r for r in service.log if r != "labels.list"]
    assert writes == []
    if what != "create the label IRIS":
        # Not even a client: the gate runs before the service is built.
        assert service.built == []


def test_restore_is_refused_before_the_untrash(service: _Service) -> None:
    with pytest.raises(PermissionError):
        GmailProvider().restore_messages(ACCT, ["m1"], labels_before={"m1": ["INBOX"]})
    assert service.log == []


def test_labels_that_already_exist_are_read_without_an_approval(service: _Service) -> None:
    """Listing is a read (the owner's relabels are read back with it); only creating a
    missing label is a write."""
    service.labels = {"IRIS": "Label_1", "IRIS/Bill": "Label_2"}
    assert gmail_fetch.ensure_labels(ACCT, ["IRIS/Bill"]) == {"IRIS/Bill": "Label_2"}
    assert service.log == ["labels.list"]


def test_writes_run_after_approve_and_stop_after_revoke(service: _Service) -> None:
    approve_mailbox_writes(ACCT, "test approval")
    p = GmailProvider()
    assert p.ensure_labels(ACCT, ["IRIS/Bill"]) == {"IRIS/Bill": "Label_2"}
    assert p.modify_labels(ACCT, ["m1"], ["Label_2"], []) == 1
    assert p.trash_messages(ACCT, ["m1"]) == ["m1"]
    gmail_fetch.add_labels(ACCT, {"m1": {"INBOX"}})
    assert gmail_fetch.untrash_messages(ACCT, ["m1"]) == {"m1": []}
    assert service.log.count("labels.create") == 2  # IRIS, then IRIS/Bill
    assert {"batchModify", "trash", "untrash"} <= set(service.log)

    revoke_mailbox_writes(ACCT, actor="test", agent_type="test")
    before = list(service.log)
    for what, write in _writes().items():
        if what.startswith("create"):
            continue  # IRIS/Bill exists now: resolving it creates nothing
        with pytest.raises(PermissionError):
            write()
    assert p.ensure_labels(ACCT, ["IRIS/Bill"]) == {"IRIS/Bill": "Label_2"}  # cached read
    with pytest.raises(PermissionError, match="create the label IRIS/Event"):
        p.ensure_labels(ACCT, ["IRIS/Event"])
    assert [r for r in service.log[len(before) :] if r != "labels.list"] == []


def test_another_accounts_approval_does_not_count(service: _Service) -> None:
    approve_mailbox_writes("gmail:someone-else@example.com", "test")
    approve_mailbox_writes("imap:owner@example.com", "test")  # same address, other provider
    with pytest.raises(PermissionError):
        GmailProvider().modify_labels(ACCT, ["m1"], ["Label_1"], [])
    assert service.log == []


def test_sending_is_not_a_mailbox_write(service: _Service) -> None:
    """Send adds a message; it changes none of the owner's mail, and every send already
    waits for the owner's approval of that message (send_email). It is not gated here."""
    sent = GmailProvider().send_message(ACCT, to=["friend@example.com"], subject="Hi", body="Hi")
    assert sent == "sent-1"
    assert service.log == ["send"]


def test_each_write_leaves_one_ledger_row_with_what_it_changed(
    service: _Service, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R14: every Gmail write records its kind and count (account in the payload, never
    in the reason) for the proof bundle; a refused write records nothing."""
    import json

    from iris_harness.sdk.audit import AuditLog
    from iris_personal.email.write_approvals import WRITE_HOOK

    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "audit.db"))
    with pytest.raises(PermissionError):
        GmailProvider().trash_messages(ACCT, ["m1"])
    approve_mailbox_writes(ACCT, "test approval")
    p = GmailProvider()
    p.ensure_labels(ACCT, ["IRIS/Bill"])  # creates IRIS, then IRIS/Bill
    p.modify_labels(ACCT, ["m1", "m2"], ["Label_2"], [])
    p.trash_messages(ACCT, ["m1"])
    gmail_fetch.untrash_messages(ACCT, ["m1"])
    gmail_fetch.add_labels(ACCT, {"m1": {"INBOX"}})

    rows = [
        r for r in AuditLog(db_path=tmp_path / "audit.db").query() if r.hook_point == WRITE_HOOK
    ]
    payloads = [json.loads(r.payload_json) for r in rows]
    assert [(p["op"], p["count"]) for p in payloads] == [
        ("create_label", 1),
        ("create_label", 1),
        ("label", 2),
        ("trash", 1),
        ("restore", 1),
        ("restore_labels", 1),
    ]
    assert {p["account"] for p in payloads} == {ACCT}
    assert all("owner@example.com" not in r.reason for r in rows)
