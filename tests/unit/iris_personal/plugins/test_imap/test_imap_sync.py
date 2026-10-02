"""Sync against the fake server: first sync, incremental, capped batches, UIDVALIDITY
reset, and that syncing never marks mail read."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from iris_personal.email.providers import MailProvider
from iris_personal.email.store import EmailStore
from iris_personal.plugins.imap.account import ImapAccount
from iris_personal.plugins.imap.provider import CURSOR_KIND, ImapProvider

from .conftest import build_message, seed
from .fake_imap_server import FakeMailbox


def test_the_provider_satisfies_the_mail_provider_protocol(provider: ImapProvider) -> None:
    assert isinstance(provider, MailProvider)
    assert provider.name == "imap"


def test_fetch_new_reports_progress(
    provider: ImapProvider, account: ImapAccount, mailbox: FakeMailbox, store: EmailStore
) -> None:
    seed(mailbox, 5)
    calls: list[tuple[float, str]] = []

    result = provider.fetch_new(
        account.account_id, store=store, progress=lambda f, m: calls.append((f, m))
    )

    assert result.fetched == 5
    assert calls == [(1.0, "fetched 5/5")]


def test_first_sync_stores_every_message_and_the_uid_mark(
    provider: ImapProvider, account: ImapAccount, mailbox: FakeMailbox, store: EmailStore
) -> None:
    seed(mailbox, 5)

    result = provider.fetch_new(account.account_id, store=store)

    assert result.fetched == 5 and len(result.new_message_ids) == 5
    assert result.fell_back_to_cold_start is False
    assert store.count(account.account_id) == 5
    stored = store.get(result.new_message_ids[0])
    assert stored is not None
    assert stored.provider == "imap" and stored.account_id == "imap:owner@example.test"
    assert stored.subject == "Order 0" and "order 0 shipped" in stored.snippet.lower()
    assert stored.labels == ("INBOX", "UNREAD")
    assert stored.headers_subset["Message-ID"] == "<msg-0@shop.example>"
    cursor = json.loads(store.get_cursor("imap", account.account_id, CURSOR_KIND) or "{}")
    inbox = mailbox.folders["INBOX"]
    assert cursor == {"folder": "INBOX", "uidvalidity": inbox.uidvalidity, "uid": 5}


def test_sync_never_marks_mail_read_and_opens_folders_read_only(
    provider: ImapProvider, account: ImapAccount, mailbox: FakeMailbox, store: EmailStore
) -> None:
    seed(mailbox, 3)
    provider.fetch_new(account.account_id, store=store)
    for uid in mailbox.folders["INBOX"].uids():
        assert "\\Seen" not in mailbox.flags_of("INBOX", uid)
    assert "SELECT" not in mailbox.commands and "EXAMINE" in mailbox.commands
    assert not {"UID STORE", "UID MOVE", "UID COPY", "CLOSE"} & set(mailbox.commands)


def test_incremental_sync_fetches_only_new_uids(
    provider: ImapProvider, account: ImapAccount, mailbox: FakeMailbox, store: EmailStore
) -> None:
    seed(mailbox, 3)
    first = provider.fetch_new(account.account_id, store=store)
    seed(mailbox, 2, start=3)

    second = provider.fetch_new(account.account_id, store=store)
    third = provider.fetch_new(account.account_id, store=store)

    assert len(first.new_message_ids) == 3
    assert len(second.new_message_ids) == 2
    assert not set(second.new_message_ids) & set(first.new_message_ids)
    assert {store.get(i).subject for i in second.new_message_ids} == {"Order 3", "Order 4"}  # type: ignore[union-attr]
    assert third.fetched == 0 and third.new_message_ids == ()
    assert store.count(account.account_id) == 5


def test_a_capped_batch_takes_the_oldest_first_and_leaves_no_gap(
    provider: ImapProvider, account: ImapAccount, mailbox: FakeMailbox, store: EmailStore
) -> None:
    seed(mailbox, 1)
    provider.fetch_new(account.account_id, store=store)
    seed(mailbox, 5, start=1)

    runs = [provider.fetch_new(account.account_id, store=store, max_messages=2) for _ in range(4)]

    subjects = [[store.get(i).subject for i in r.new_message_ids] for r in runs]  # type: ignore[union-attr]
    assert subjects == [["Order 1", "Order 2"], ["Order 3", "Order 4"], ["Order 5"], []]


def test_cold_start_takes_the_newest_within_the_window(
    provider: ImapProvider, account: ImapAccount, mailbox: FakeMailbox, store: EmailStore
) -> None:
    mailbox.add_message(
        build_message(subject="Ancient", message_id="<old@x.example>"),
        internaldate=datetime.now(UTC) - timedelta(days=90),
    )
    seed(mailbox, 4)

    result = provider.fetch_new(account.account_id, store=store, max_messages=3, cold_start_days=30)

    assert [store.get(i).subject for i in result.new_message_ids] == [  # type: ignore[union-attr]
        "Order 1",
        "Order 2",
        "Order 3",
    ]
    # The mark is the top of the mailbox, so neither the skipped nor the ancient mail
    # comes back as "new" later.
    assert provider.fetch_new(account.account_id, store=store).new_message_ids == ()


def test_a_uidvalidity_reset_cold_starts_without_doubling_rows(
    provider: ImapProvider, account: ImapAccount, mailbox: FakeMailbox, store: EmailStore
) -> None:
    seed(mailbox, 4)
    first = provider.fetch_new(account.account_id, store=store)
    mailbox.reset_uidvalidity("INBOX")

    after = provider.fetch_new(account.account_id, store=store)

    assert after.fell_back_to_cold_start is True
    assert set(after.new_message_ids) == set(first.new_message_ids)  # Message-ID ids
    assert store.count(account.account_id) == 4
    cursor = json.loads(store.get_cursor("imap", account.account_id, CURSOR_KIND) or "{}")
    assert cursor["uidvalidity"] == mailbox.folders["INBOX"].uidvalidity
    seed(mailbox, 1, start=10)
    assert len(provider.fetch_new(account.account_id, store=store).new_message_ids) == 1


def test_reset_cursor_makes_the_next_fetch_cold_start(
    provider: ImapProvider, account: ImapAccount, mailbox: FakeMailbox, store: EmailStore
) -> None:
    seed(mailbox, 2)
    provider.fetch_new(account.account_id, store=store)
    provider.reset_cursor(account.account_id, store=store)
    assert store.get_cursor("imap", account.account_id, CURSOR_KIND) is None
    assert len(provider.fetch_new(account.account_id, store=store).new_message_ids) == 2


def test_an_oversized_message_is_synced_from_its_headers(
    state: object, account: ImapAccount, mailbox: FakeMailbox, store: EmailStore
) -> None:
    mailbox.add_message(
        build_message(
            subject="Big scan",
            message_id="<big@x.example>",
            attachments=[("scan.pdf", "application/pdf", b"%PDF" + b"0" * 4096)],
        )
    )
    small = ImapProvider(state=state, max_full_fetch_bytes=1024, timeout=5.0)  # type: ignore[arg-type]

    result = small.fetch_new(account.account_id, store=store)

    stored = store.get(result.new_message_ids[0])
    assert stored is not None and stored.subject == "Big scan"
    assert stored.snippet == "" and stored.attachments == ()


def test_the_email_sweep_reaches_an_imap_account_through_the_registry(
    provider: ImapProvider,
    account: ImapAccount,
    mailbox: FakeMailbox,
    store: EmailStore,
    tmp_path: Path,
) -> None:
    """``iris email`` wiring: an ``imap:`` row plus the registered provider is all the
    provider-agnostic sweep needs, exactly as for Gmail and the demo."""
    from iris_harness.services.heartbeat.models import HeartbeatDefinition, HeartbeatStatus
    from iris_personal.email.accounts import EmailAccountStore
    from iris_personal.email.providers import clear_mail_providers, register_mail_provider
    from iris_personal.email.sweep import EmailSweepHandler

    seed(mailbox, 3)
    accounts = EmailAccountStore(db_path=tmp_path / "iris.db")
    accounts.ensure_schema()
    accounts.add(provider="imap", address=account.address)
    clear_mail_providers()
    register_mail_provider(provider)
    try:
        run = EmailSweepHandler(accounts_store=accounts, email_store=store)(
            HeartbeatDefinition(
                name="email_sweep",
                handler="email_sweep",
                schedule="interval:600",
                enabled=True,
                params={"max_messages": 50},
            )
        )
    finally:
        clear_mail_providers()
    assert run.status is HeartbeatStatus.SUCCESS, run.error
    assert "imap:owner@example.test: +3" in (run.output or "")
    assert store.count(account.account_id) == 3
