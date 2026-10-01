"""__tmpl_title: a mailbox IRIS syncs, written against the mail-provider facade.

A mail provider implements ``MailProvider`` (``iris_personal.email.provider_api``) and
registers itself in ``setup``. The email sweep then syncs every connected account of
its ``name`` into the owner's mail record; the read, attachment and Trash tools reach the
mailbox through it. It registers none of the six ``PluginAPI`` kinds, so the manifest's
``provides`` is empty.

The owner's address is personal data: the provider hands the addresses it signs in to
to the owner-identity guards (``api.register_owner_identity_source``), declared in the
manifest under ``identity: provides``, so they are masked wherever they must not leave.

This provider serves a fixed demo mailbox, so it works offline. Replace ``_messages``
and ``fetch_message_body`` with your service's API; make every write inside
``mailbox_write``, which checks the owner's approval first and records the write in the
governance ledger after (the proof bundle's evidence that it was approved).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from iris_harness.sdk import PluginAPI
from iris_personal.email.provider_api import (
    AttachmentCandidate,
    DownloadedAttachment,
    EmailMessage,
    FetchResult,
    MailSyncStore,
    connect_account,
    default_sync_store,
    mailbox_write,
    register_mail_provider,
)

NAME = "__tmpl_tool"
CURSOR = "offset"

# The demo mailbox: (id, sender, subject, body). Your service's API replaces it.
DEMO_MAILBOX: tuple[tuple[str, str, str, str], ...] = (
    ("demo-1", "ada@example.org", "Seed order", "The seed order ships on Friday."),
    ("demo-2", "grace@example.org", "Loft repairs", "The roofer comes on Tuesday."),
)


class Provider:
    """The ``NAME`` mailbox provider."""

    name = NAME

    def __init__(self) -> None:
        self._addresses: set[str] = set()

    # -- connecting an account ---------------------------------------------------
    def connect(self, address: str) -> str:
        """Connect the owner's ``address`` and return its account id.

        A real provider first checks it can sign in and keeps the credentials in the
        vault; then it records the account, and the sweep starts syncing it.
        """
        account_id = connect_account(NAME, address)
        self._addresses.add(address.strip().lower())
        return account_id

    def owner_identity(self) -> dict[str, list[str]]:
        """The owner's addresses this provider signs in to, for the identity guards."""
        return {"email": sorted(self._addresses)}

    # -- MailProvider --------------------------------------------------------------
    def _messages(self, account_id: str) -> list[EmailMessage]:
        address = account_id.split(":", 1)[-1]
        return [
            EmailMessage(
                id=message_id,
                provider=NAME,
                account_id=account_id,
                from_address=sender,
                from_domain=sender.split("@", 1)[1],
                to=(address,),
                subject=subject,
                received_at=datetime(2026, 9, 1, 9, 0, tzinfo=UTC),
                snippet=body,
            )
            for message_id, sender, subject, body in DEMO_MAILBOX
        ]

    def fetch_new(
        self,
        account_id: str,
        *,
        store: MailSyncStore | None = None,
        max_messages: int = 100,
        cold_start_days: int = 30,
    ) -> FetchResult:
        record = store if store is not None else default_sync_store()
        start = int(record.get_cursor(NAME, account_id, CURSOR) or 0)
        batch = self._messages(account_id)[start : start + max(0, max_messages)]
        written = record.upsert_many(batch)
        end = start + len(batch)
        record.set_cursor(NAME, account_id, CURSOR, str(end))
        return FetchResult(
            account_id=account_id,
            fetched=written,
            new_message_ids=tuple(message.id for message in batch),
            new_cursor=str(end),
            fell_back_to_cold_start=start == 0,
        )

    def reset_cursor(self, account_id: str, *, store: MailSyncStore) -> None:
        store.clear_cursor(NAME, account_id, CURSOR)

    def fetch_message_body(self, account_id: str, message_id: str, *, max_chars: int = 4000) -> str:
        for demo_id, _sender, _subject, body in DEMO_MAILBOX:
            if demo_id == message_id:
                return body[:max_chars]
        raise LookupError(message_id)

    # -- writes: each inside mailbox_write -----------------------------------------
    # mailbox_write checks the owner's approval before the body runs (none: it raises
    # PermissionError and nothing is written), and records what the body ``add``s to
    # the tally in the governance ledger after it -- also when the body fails part way.
    def trash_messages(self, account_id: str, message_ids: Sequence[str]) -> list[str]:
        with mailbox_write(account_id, "move mail to Trash", op="trash") as tally:
            trashed = list(message_ids)  # your service's "move to Trash" call
            tally.add(len(trashed))
        return trashed

    def restore_messages(
        self,
        account_id: str,
        message_ids: Sequence[str],
        *,
        labels_before: Mapping[str, Sequence[str]] | None = None,
    ) -> list[EmailMessage]:
        with mailbox_write(account_id, "restore mail from Trash", op="restore") as tally:
            restored = [m for m in self._messages(account_id) if m.id in message_ids]
            tally.add(len(restored))
        return restored

    def fetch_message_attachments(
        self,
        account_id: str,
        message_id: str,
        *,
        mime_types: tuple[str, ...] | None = None,
        service: Any | None = None,
    ) -> list[DownloadedAttachment]:
        return []

    def list_attachment_candidates(
        self, account_id: str, terms: Sequence[str], *, limit: int = 15
    ) -> list[AttachmentCandidate]:
        return []


def mount(provider: Provider) -> Callable[[PluginAPI], None]:
    """A ``setup`` that mounts ``provider`` (the tests mount one they hold)."""

    def _setup(api: PluginAPI) -> None:
        register_mail_provider(provider)
        api.register_owner_identity_source(provider.owner_identity)

    return _setup


def setup(api: PluginAPI) -> None:
    mount(Provider())(api)
