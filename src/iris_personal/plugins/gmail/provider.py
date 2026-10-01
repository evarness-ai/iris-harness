"""The Gmail ``MailProvider`` — the object the core's mail registry dispatches to.

Every method delegates to the module functions in :mod:`gmail_fetch` and
:mod:`gmail_attachments` *at call time*, by attribute lookup on the module, so a test
that patches ``gmail_fetch.fetch_message_body`` patches what the provider runs.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from iris_personal.email.provider_api import (
    AttachmentCandidate,
    DownloadedAttachment,
    EmailMessage,
    FetchResult,
    MailSyncStore,
)

from . import gmail_attachments, gmail_fetch
from .vendor_categories import load_table

logger = logging.getLogger(__name__)


#: The labels Gmail's trash takes away: where a message lives, not what it is about.
_LOCATION_LABELS = frozenset({"INBOX", "SPAM"})


class GmailProvider:
    """Gmail behind the core's ``MailProvider`` interface: reading, moving mail to
    Trash and back (never permanent deletion), and sending plain text."""

    name = gmail_fetch.GMAIL_PROVIDER

    def fetch_new(
        self,
        account_id: str,
        *,
        store: MailSyncStore | None = None,
        max_messages: int = gmail_fetch.DEFAULT_MAX_MESSAGES,
        cold_start_days: int = gmail_fetch.DEFAULT_COLD_START_DAYS,
    ) -> FetchResult:
        return gmail_fetch.fetch_new_emails(
            account_id, store=store, max_messages=max_messages, cold_start_days=cold_start_days
        )

    def reset_cursor(self, account_id: str, *, store: MailSyncStore) -> None:
        """Delete the history cursor so the next fetch cold-starts -- the one deliberate
        "start from scratch" signal, which is why no fetch path does it on its own."""
        store.clear_cursor(
            gmail_fetch.GMAIL_PROVIDER, account_id, gmail_fetch.GMAIL_HISTORY_CURSOR_KIND
        )

    def fetch_message_body(self, account_id: str, message_id: str, *, max_chars: int = 4000) -> str:
        return gmail_fetch.fetch_message_body(account_id, message_id, max_chars=max_chars)

    def category_labels(self) -> dict[str, str]:
        """Gmail's inbox-tab labels and the topic path each maps to (label → path), from
        the plugin's ``vendor_categories.yaml``. A label mapped to no path is left out.
        The email tools use it to match a tab's mail after triage has refiled it."""
        return {label: path for label, path in load_table().items() if path}

    def current_labels(self, account_id: str, message_ids: Sequence[str]) -> dict[str, list[str]]:
        """Labels right before a trash: Gmail's trash drops INBOX (or SPAM) and its
        untrash does not add it back, so the restore needs to know what was there."""
        return gmail_fetch.message_labels(account_id, list(message_ids))

    def trash_messages(self, account_id: str, message_ids: Sequence[str]) -> list[str]:
        return gmail_fetch.trash_messages(account_id, list(message_ids))

    def restore_messages(
        self,
        account_id: str,
        message_ids: Sequence[str],
        *,
        labels_before: Mapping[str, Sequence[str]] | None = None,
    ) -> list[EmailMessage]:
        """Untrash, then put the message back where it was: Gmail's trash removes its
        location label (INBOX, or SPAM for mail trashed out of Spam) and untrash does
        not add it back. Without ``labels_before`` the mail comes back archived.

        The location comes from ``labels_before`` alone, re-added unconditionally
        (adding a label a message has is a no-op). An earlier version added only what
        the untrash response lacked; on the owner's inbox 69 of 200 restored emails
        stayed archived with no error, which that comparison would explain if the
        response ever lists a label the message does not end up with (2026-09-22).
        """
        after = gmail_fetch.untrash_messages(account_id, list(message_ids))
        if labels_before:
            locations = {mid: set(labels_before.get(mid, ())) & _LOCATION_LABELS for mid in after}
            gmail_fetch.add_labels(account_id, locations)
            logger.info(
                "gmail restore %s: untrashed %d of %d, location put back on %d",
                account_id,
                len(after),
                len(message_ids),
                sum(1 for labels in locations.values() if labels),
            )
        return gmail_fetch.fetch_messages_metadata(account_id, list(after))

    def ensure_labels(self, account_id: str, names: Sequence[str]) -> dict[str, str]:
        """``{name: label id}``, creating missing labels (the email judge's IRIS/*)."""
        return gmail_fetch.ensure_labels(account_id, list(names))

    def modify_labels(
        self,
        account_id: str,
        message_ids: Sequence[str],
        add_ids: Sequence[str],
        remove_ids: Sequence[str],
    ) -> int:
        """Add and remove labels in ``batchModify`` calls. Never archives or marks read:
        what is added and removed is exactly what the caller passes."""
        return gmail_fetch.modify_labels(
            account_id, list(message_ids), list(add_ids), list(remove_ids)
        )

    def send_message(
        self,
        account_id: str,
        *,
        to: Sequence[str],
        cc: Sequence[str] = (),
        subject: str,
        body: str,
        in_reply_to: str | None = None,
        references: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        """Send a plain-text email (``send_email``, ADR-0118 amendment); return its id.
        An optional capability: the core checks for it rather than requiring it of
        every mailbox provider."""
        return gmail_fetch.send_message(
            account_id,
            to=list(to),
            cc=list(cc),
            subject=subject,
            body=body,
            in_reply_to=in_reply_to,
            references=references,
            thread_id=thread_id,
        )

    def fetch_message_attachments(
        self,
        account_id: str,
        message_id: str,
        *,
        mime_types: tuple[str, ...] | None = None,
        service: Any | None = None,
    ) -> list[DownloadedAttachment]:
        return gmail_attachments.fetch_message_attachments(
            account_id, message_id, mime_types=mime_types, service=service
        )

    def list_attachment_candidates(
        self, account_id: str, terms: Sequence[str], *, limit: int = 15
    ) -> list[AttachmentCandidate]:
        """A targeted ``has:attachment <terms>`` search, headers + attachment metadata.

        Raises whatever the Gmail client raises on a revoked token or a network
        failure; the caller turns that into a re-auth nudge.
        """
        gq = ("has:attachment " + " ".join(terms)).strip() if terms else "has:attachment"
        svc = gmail_attachments._build_service(account_id)
        listing = svc.users().messages().list(userId="me", q=gq, maxResults=limit).execute()
        out: list[AttachmentCandidate] = []
        for ref in listing.get("messages", []):
            payload = svc.users().messages().get(userId="me", id=ref["id"], format="full").execute()
            atts = gmail_attachments.attachments_from_payload(payload)
            if not atts:
                continue
            hdrs = {
                h["name"].lower(): h["value"] for h in payload.get("payload", {}).get("headers", [])
            }
            out.append(
                AttachmentCandidate(
                    subject=hdrs.get("subject", ""),
                    from_address=hdrs.get("from", "unknown"),
                    attachments=atts,
                )
            )
        return out


__all__ = ["GmailProvider"]
