"""A third-party mail provider, written only against the stable tier.

``test_provider_api.py`` holds this file to the tier (``check_stable_imports``) and
mounts it in a harness: if a provider needed anything the facade does not give, this
file would have to import it, and that test would fail. Not collected (no ``test_``
prefix); imported by the test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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

NAME = "pigeon"
CURSOR = "offset"
ADDRESS = "owner@pigeon.example"

# The whole mailbox: (id, sender, subject, body).
MAILBOX: tuple[tuple[str, str, str, str], ...] = (
    ("p-1", "ada@loft.example", "Seed order", "The seed order ships on Friday."),
    ("p-2", "grace@loft.example", "Loft repairs", "The roofer comes on Tuesday."),
)


class PigeonProvider:
    """Serves ``MAILBOX``; changing the mailbox needs the owner's write approval."""

    name = NAME

    def __init__(self) -> None:
        self.labels: dict[str, tuple[str, ...]] = {}
        self.created: set[str] = set()

    def _message(self, account_id: str, row: tuple[str, str, str, str]) -> EmailMessage:
        mid, sender, subject, body = row
        return EmailMessage(
            id=mid,
            provider=NAME,
            account_id=account_id,
            from_address=sender,
            from_domain=sender.split("@", 1)[1],
            to=(ADDRESS,),
            subject=subject,
            received_at=datetime(2026, 9, 1, 9, 0, tzinfo=UTC),
            snippet=body,
        )

    def fetch_new(
        self,
        account_id: str,
        *,
        store: MailSyncStore | None = None,
        max_messages: int = 100,
        cold_start_days: int = 30,
    ) -> FetchResult:
        s = store if store is not None else default_sync_store()
        start = int(s.get_cursor(NAME, account_id, CURSOR) or 0)
        batch = MAILBOX[start : start + max(0, max_messages)]
        messages = [self._message(account_id, row) for row in batch]
        written = s.upsert_many(messages)
        end = start + len(batch)
        s.set_cursor(NAME, account_id, CURSOR, str(end))
        return FetchResult(
            account_id=account_id,
            fetched=written,
            new_message_ids=tuple(m.id for m in messages),
            new_cursor=str(end),
            fell_back_to_cold_start=start == 0,
        )

    def reset_cursor(self, account_id: str, *, store: MailSyncStore) -> None:
        store.clear_cursor(NAME, account_id, CURSOR)

    def fetch_message_body(self, account_id: str, message_id: str, *, max_chars: int = 4000) -> str:
        for mid, _sender, _subject, body in MAILBOX:
            if mid == message_id:
                return body[:max_chars]
        raise LookupError(message_id)

    def trash_messages(self, account_id: str, message_ids: Sequence[str]) -> list[str]:
        with mailbox_write(account_id, "move mail to Trash", op="trash") as tally:
            trashed = list(message_ids)
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
            restored = [self._message(account_id, row) for row in MAILBOX if row[0] in message_ids]
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

    # LabellingProvider
    def ensure_labels(self, account_id: str, names: Sequence[str]) -> dict[str, str]:
        with mailbox_write(account_id, "create labels", op="create_label") as tally:
            created = [name for name in names if name not in self.created]
            self.created.update(created)
            tally.add(len(created))
        return {name: name for name in names}

    def modify_labels(
        self,
        account_id: str,
        message_ids: Sequence[str],
        add_ids: Sequence[str],
        remove_ids: Sequence[str],
    ) -> int:
        with mailbox_write(account_id, "change labels", op="label") as tally:
            for mid in message_ids:
                now = set(self.labels.get(mid, ())) - set(remove_ids)
                self.labels[mid] = tuple(sorted(now | set(add_ids)))
                tally.add(1)
        return len(message_ids)


def connect() -> str:
    """Connect the owner's pigeon account (a real provider does this once it holds the
    account's credentials) and return its account id."""
    return connect_account(NAME, ADDRESS)


def setup(api: PluginAPI) -> None:
    register_mail_provider(PigeonProvider())
