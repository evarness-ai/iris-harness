"""Mailbox writes: refused before the write approval (R4), then labels as $-keywords via
the judge's governed ``sync_labels``, the owner's relabel read back, and Trash as a move
there and back -- never an expunge of the owner's mail."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_personal.email.providers import LabellingProvider
from iris_personal.email.store import EmailStore
from iris_personal.email.write_approvals import approve_mailbox_writes
from iris_personal.plugins.email_workflows.judge_config import JudgeConfig
from iris_personal.plugins.email_workflows.judge_labels import sync_labels
from iris_personal.plugins.email_workflows.judgments import JudgmentStore
from iris_personal.plugins.imap.account import ImapAccount
from iris_personal.plugins.imap.keywords import keyword_for
from iris_personal.plugins.imap.provider import ImapProvider

from .conftest import seed
from .fake_imap_server import FakeImapServer, FakeMailbox

WRITES = {"UID STORE", "UID MOVE", "UID COPY", "UID EXPUNGE", "CREATE", "CLOSE"}


def _synced(provider: ImapProvider, account: ImapAccount, store: EmailStore) -> list[str]:
    return list(provider.fetch_new(account.account_id, store=store).new_message_ids)


def test_label_names_map_to_dollar_keywords() -> None:
    assert keyword_for("IRIS/Bill") == "$IRIS_Bill"
    assert keyword_for("IRIS/Needs-Reply") == "$IRIS_Needs-Reply"
    assert keyword_for("$Already") == "$Already"
    assert keyword_for("a (b) c") == "$a__b__c"


def test_the_provider_is_a_labelling_provider(provider: ImapProvider) -> None:
    assert isinstance(provider, LabellingProvider)


def test_every_write_is_refused_before_an_approval_and_nothing_reaches_the_server(
    provider: ImapProvider, account: ImapAccount, mailbox: FakeMailbox, store: EmailStore
) -> None:
    seed(mailbox, 2)
    ids = _synced(provider, account, store)
    before = len(mailbox.commands)
    kw = provider.ensure_labels(account.account_id, ["IRIS/Bill"])["IRIS/Bill"]

    with pytest.raises(PermissionError, match="no approval"):
        provider.modify_labels(account.account_id, ids, [kw], [])
    with pytest.raises(PermissionError, match="no approval"):
        provider.trash_messages(account.account_id, ids)
    with pytest.raises(PermissionError, match="no approval"):
        provider.restore_messages(account.account_id, ids)

    assert mailbox.commands[before:] == []  # not even a connection
    assert all(not {f for f in mailbox.flags_of("INBOX", u) if f.startswith("$")} for u in (1, 2))


def test_judge_sync_labels_writes_one_keyword_per_bucket_after_approval(
    provider: ImapProvider,
    account: ImapAccount,
    mailbox: FakeMailbox,
    store: EmailStore,
    tmp_path: Path,
) -> None:
    seed(mailbox, 3)
    ids = _synced(provider, account, store)
    config = JudgeConfig.load(None)
    first, second = config.keys[0], config.keys[1]
    judgments = JudgmentStore(db_path=tmp_path / "judgments.db")
    judgments.ensure_schema()
    judgments.record(ids[0], account.account_id, bucket=first, confidence=0.9)
    judgments.record(ids[1], account.account_id, bucket=second, confidence=0.9)
    judgments.record(ids[2], account.account_id, bucket="promo", confidence=0.9)

    refused = sync_labels(judgments, config, {account.account_id: provider})
    assert refused.written == 0 and "no approval" in refused.errors[0]

    approve_mailbox_writes(account.account_id, "approval-row-42")
    done = sync_labels(judgments, config, {account.account_id: provider})

    assert done.written == 2 and not done.errors
    kw_first = keyword_for(config.labels[first])
    kw_second = keyword_for(config.labels[second])
    assert kw_first in mailbox.flags_of("INBOX", 1)
    assert kw_second in mailbox.flags_of("INBOX", 2)
    assert not {f for f in mailbox.flags_of("INBOX", 3) if f.startswith("$")}
    for uid in (1, 2, 3):
        assert "\\Seen" not in mailbox.flags_of("INBOX", uid)  # never marks read
    assert "CLOSE" not in mailbox.commands  # CLOSE would expunge \\Deleted mail


def test_the_owner_moving_a_label_comes_back_as_a_label_change(
    provider: ImapProvider, account: ImapAccount, mailbox: FakeMailbox, store: EmailStore
) -> None:
    seed(mailbox, 2)
    ids = _synced(provider, account, store)
    names = provider.ensure_labels(account.account_id, ["IRIS/Bill", "IRIS/FYI"])
    approve_mailbox_writes(account.account_id, "approval-1")
    provider.modify_labels(account.account_id, [ids[0]], [names["IRIS/Bill"]], [])
    echo = provider.fetch_new(account.account_id, store=store)
    assert echo.label_changes == ((ids[0], ("INBOX", "UNREAD", "$IRIS_Bill")),)

    # In their mail client, the owner swaps Bill for FYI on the first mail.
    with mailbox.lock:
        flags = mailbox.folders["INBOX"].messages[1].flags
        flags.discard("$IRIS_Bill")
        flags.add("$IRIS_FYI")
    moved = provider.fetch_new(account.account_id, store=store)

    assert moved.label_changes == ((ids[0], ("INBOX", "UNREAD", "$IRIS_FYI")),)
    stored = store.get(ids[0])
    assert stored is not None and "$IRIS_FYI" in stored.labels and "$IRIS_Bill" not in stored.labels
    assert provider.fetch_new(account.account_id, store=store).label_changes == ()


def test_a_server_without_custom_keywords_is_refused_clearly(
    state: object, tmp_path: Path, store: EmailStore
) -> None:
    from iris_personal.plugins.imap.account import save_account

    from .conftest import PASSWORD, USER

    box = FakeMailbox(users={USER: PASSWORD}, keywords_allowed=False)
    seed(box, 1)
    with FakeImapServer(box) as srv:
        acct = ImapAccount(
            address=USER, host=srv.host, username=USER, password=PASSWORD,
            port=srv.port, security="plain",
        )  # fmt: skip
        save_account(acct)
        prov = ImapProvider(state=state, timeout=5.0)  # type: ignore[arg-type]
        ids = list(prov.fetch_new(acct.account_id, store=store).new_message_ids)
        approve_mailbox_writes(acct.account_id, "approval-1")
        kw = prov.ensure_labels(acct.account_id, ["IRIS/Bill"])["IRIS/Bill"]
        with pytest.raises(PermissionError, match="custom keywords"):
            prov.modify_labels(acct.account_id, ids, [kw], [])
        assert "UID STORE" not in box.commands


def test_system_flags_are_never_written(
    provider: ImapProvider, account: ImapAccount, mailbox: FakeMailbox, store: EmailStore
) -> None:
    seed(mailbox, 1)
    ids = _synced(provider, account, store)
    approve_mailbox_writes(account.account_id, "approval-1")
    assert provider.modify_labels(account.account_id, ids, ["\\Seen", "UNREAD"], ["INBOX"]) == 0
    assert "\\Seen" not in mailbox.flags_of("INBOX", 1)


def test_trash_moves_to_the_trash_folder_and_restore_brings_it_back(
    provider: ImapProvider, account: ImapAccount, mailbox: FakeMailbox, store: EmailStore
) -> None:
    seed(mailbox, 2)
    ids = _synced(provider, account, store)
    approve_mailbox_writes(account.account_id, "approval-1")

    moved = provider.trash_messages(account.account_id, [ids[0]])

    assert moved == [ids[0]]
    assert mailbox.find(b"<msg-0@shop.example>", "INBOX") is None
    assert mailbox.find(b"<msg-0@shop.example>", "Trash") is not None

    restored = provider.restore_messages(account.account_id, [ids[0]])

    assert [m.id for m in restored] == [ids[0]] and restored[0].subject == "Order 0"
    assert mailbox.find(b"<msg-0@shop.example>", "INBOX") is not None
    assert mailbox.find(b"<msg-0@shop.example>", "Trash") is None
    assert "UID EXPUNGE" not in mailbox.commands  # MOVE, not delete + expunge


def test_without_move_the_fallback_expunges_only_the_moved_uids(
    state: object, store: EmailStore
) -> None:
    from iris_personal.plugins.imap.account import save_account

    from .conftest import PASSWORD, USER

    box = FakeMailbox(users={USER: PASSWORD}, capabilities=("UIDPLUS",))
    seed(box, 2)
    box.folders["INBOX"].messages[2].flags.add("\\Deleted")  # the owner's own
    with FakeImapServer(box) as srv:
        acct = ImapAccount(
            address=USER, host=srv.host, username=USER, password=PASSWORD,
            port=srv.port, security="plain",
        )  # fmt: skip
        save_account(acct)
        prov = ImapProvider(state=state, timeout=5.0)  # type: ignore[arg-type]
        ids = list(prov.fetch_new(acct.account_id, store=store).new_message_ids)
        approve_mailbox_writes(acct.account_id, "approval-1")
        assert prov.trash_messages(acct.account_id, [ids[0]]) == [ids[0]]
    assert box.find(b"<msg-0@shop.example>", "Trash") is not None
    assert box.find(b"<msg-1@shop.example>", "INBOX") is not None  # not purged


def test_a_server_with_neither_move_nor_uidplus_is_refused(
    state: object, store: EmailStore
) -> None:
    from iris_personal.plugins.imap.account import save_account

    from .conftest import PASSWORD, USER

    box = FakeMailbox(users={USER: PASSWORD}, capabilities=())
    seed(box, 1)
    with FakeImapServer(box) as srv:
        acct = ImapAccount(
            address=USER, host=srv.host, username=USER, password=PASSWORD,
            port=srv.port, security="plain",
        )  # fmt: skip
        save_account(acct)
        prov = ImapProvider(state=state, timeout=5.0)  # type: ignore[arg-type]
        ids = list(prov.fetch_new(acct.account_id, store=store).new_message_ids)
        approve_mailbox_writes(acct.account_id, "approval-1")
        with pytest.raises(PermissionError, match="neither MOVE nor UIDPLUS"):
            prov.trash_messages(acct.account_id, ids)
    assert box.find(b"<msg-0@shop.example>", "INBOX") is not None


def test_a_stale_uid_is_refound_by_message_id_after_a_reset(
    provider: ImapProvider, account: ImapAccount, mailbox: FakeMailbox, store: EmailStore
) -> None:
    seed(mailbox, 1)
    ids = _synced(provider, account, store)
    mailbox.reset_uidvalidity("INBOX")
    approve_mailbox_writes(account.account_id, "approval-1")
    kw = provider.ensure_labels(account.account_id, ["IRIS/Bill"])["IRIS/Bill"]

    assert provider.modify_labels(account.account_id, ids, [kw], []) == 1
    (uid,) = mailbox.folders["INBOX"].uids()
    assert kw in mailbox.flags_of("INBOX", uid)


def test_each_write_that_reached_the_server_leaves_one_ledger_row(
    provider: ImapProvider,
    account: ImapAccount,
    mailbox: FakeMailbox,
    store: EmailStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R14: label, trash and restore each record what they changed (the account in the
    payload, never in the reason), so the proof bundle observes them; a refused write
    records nothing."""
    import json

    from iris_harness.sdk.audit import AuditLog
    from iris_personal.email.write_approvals import WRITE_HOOK

    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "audit.db"))
    seed(mailbox, 2)
    ids = _synced(provider, account, store)
    kw = provider.ensure_labels(account.account_id, ["IRIS/Bill"])["IRIS/Bill"]
    with pytest.raises(PermissionError):
        provider.modify_labels(account.account_id, ids, [kw], [])
    approve_mailbox_writes(account.account_id, "approval-1")

    assert provider.modify_labels(account.account_id, ids, [kw], []) == 2
    provider.trash_messages(account.account_id, [ids[0]])
    provider.restore_messages(account.account_id, [ids[0]])

    rows = [
        r for r in AuditLog(db_path=tmp_path / "audit.db").query() if r.hook_point == WRITE_HOOK
    ]
    payloads = [json.loads(r.payload_json) for r in rows]
    assert [(p["op"], p["count"]) for p in payloads] == [("label", 2), ("trash", 1), ("restore", 1)]
    assert {p["account"] for p in payloads} == {account.account_id}
    assert all(account.address not in r.reason for r in rows)
