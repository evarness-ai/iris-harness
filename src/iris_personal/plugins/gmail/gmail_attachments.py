"""Read-only Gmail attachment fetch — the deferred half of gmail-inbox.

``gmail_fetch.py`` deliberately syncs envelope + snippet only
(``format="metadata"``) and notes that attachment handling waits "until
the finance-statements skill needs it" (its module docstring). This is
that module: it fetches a message in ``format="full"``, walks the MIME
tree for attachment parts, and downloads their bytes via
``users().messages().attachments().get`` — all under the existing
``gmail.readonly`` scope (no new OAuth grant).

Bytes are returned to the caller; this module never decides where they
live. The finance ingest layer owns storage paths (ADR-0006:
``data/finance/statements/<provider>/<YYYY>/<file>``).
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from iris_personal.email.contracts import EmailAttachment
from iris_personal.email.providers import DownloadedAttachment

from .gmail_oauth import load_credentials

logger = logging.getLogger(__name__)


# ``DownloadedAttachment`` is the core interface's type (``email.providers``);
# re-exported here so existing imports of it from this module keep working.


def _address_from_account_id(account_id: str) -> str:
    """'gmail:user@gmail.com' → 'user@gmail.com' (ADR-0016 slug)."""
    if ":" in account_id:
        return account_id.split(":", 1)[1]
    return account_id


def _build_service(account_id: str) -> Any:
    creds = load_credentials(_address_from_account_id(account_id))
    if creds is None:
        raise RuntimeError(
            f"No Gmail credentials for {account_id}. "
            "Run `iris auth gmail login --user <address>` first."
        )
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _walk_parts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Depth-first flatten of a Gmail MIME ``payload`` into its parts.

    Includes the top-level payload itself so single-part messages with
    an attachment body are not missed.
    """
    out: list[dict[str, Any]] = []
    stack = [payload]
    while stack:
        part = stack.pop()
        out.append(part)
        stack.extend(part.get("parts", []) or [])
    return out


def _part_to_attachment(part: dict[str, Any]) -> EmailAttachment | None:
    """An attachment part has a non-empty filename + a body.attachmentId."""
    filename = (part.get("filename") or "").strip()
    body = part.get("body", {}) or {}
    attachment_id = body.get("attachmentId")
    if not filename or not attachment_id:
        return None
    return EmailAttachment(
        filename=filename,
        mime_type=part.get("mimeType") or "application/octet-stream",
        size_bytes=int(body.get("size", 0) or 0),
        attachment_id=attachment_id,
    )


def attachments_from_payload(payload: dict[str, Any]) -> tuple[EmailAttachment, ...]:
    """Pure helper: extract attachment descriptors from a full-format payload.

    Split out so detection/tests can run without any network call.
    """
    msg_payload = payload.get("payload", payload)
    found = (_part_to_attachment(p) for p in _walk_parts(msg_payload))
    return tuple(a for a in found if a is not None)


def list_attachments(
    account_id: str, message_id: str, *, service: Any | None = None
) -> tuple[EmailAttachment, ...]:
    """Attachment descriptors for one message (read-only, no bytes)."""
    svc = service if service is not None else _build_service(account_id)
    payload = svc.users().messages().get(userId="me", id=message_id, format="full").execute()
    return attachments_from_payload(payload)


def download_attachment(
    account_id: str,
    message_id: str,
    attachment: EmailAttachment,
    *,
    service: Any | None = None,
) -> DownloadedAttachment:
    """Fetch and base64url-decode one attachment's bytes (read-only)."""
    svc = service if service is not None else _build_service(account_id)
    resp = (
        svc.users()
        .messages()
        .attachments()
        .get(userId="me", messageId=message_id, id=attachment.attachment_id)
        .execute()
    )
    data = resp.get("data", "")
    content = base64.urlsafe_b64decode(data) if data else b""
    return DownloadedAttachment(meta=attachment, content=content)


def fetch_message_attachments(
    account_id: str,
    message_id: str,
    *,
    mime_types: tuple[str, ...] | None = None,
    service: Any | None = None,
) -> list[DownloadedAttachment]:
    """List + download all (optionally MIME-filtered) attachments on a message.

    One ``format="full"`` GET reused across all attachments on the
    message. Per-attachment download failures are logged and skipped so
    one bad part doesn't abort the batch.
    """
    svc = service if service is not None else _build_service(account_id)
    payload = svc.users().messages().get(userId="me", id=message_id, format="full").execute()
    metas = attachments_from_payload(payload)
    if mime_types is not None:
        metas = tuple(m for m in metas if m.mime_type in mime_types)
    out: list[DownloadedAttachment] = []
    for meta in metas:
        try:
            out.append(download_attachment(account_id, message_id, meta, service=svc))
        except HttpError as exc:  # pragma: no cover - network failure path
            logger.debug("gmail-attachments: skip %s/%s (%s)", message_id, meta.filename, exc)
    return out


__all__ = [
    "DownloadedAttachment",
    "attachments_from_payload",
    "list_attachments",
    "download_attachment",
    "fetch_message_attachments",
]
