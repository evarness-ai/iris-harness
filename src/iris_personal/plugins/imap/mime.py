"""RFC 822 bytes -> the ``EmailMessage`` shape the store expects.

Stdlib ``email`` with the modern policy, which decodes RFC 2047 headers and each part's
charset / transfer encoding. A part whose declared charset is unknown or lies is decoded
as UTF-8 with replacement rather than dropped: a readable body with a few odd characters
beats no body.

Identity: IMAP UIDs are only stable inside one ``(folder, UIDVALIDITY)``, so they make a
poor ``emails.id``. The id is a hash of the account and the ``Message-ID`` header, which
survives a UIDVALIDITY reset and a move to Trash and back, so a re-sync upserts the same
rows instead of doubling them. Mail without a ``Message-ID`` falls back to a hash of its
date, sender and subject.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from email import message_from_bytes, policy
from email.message import EmailMessage as MIMEMessage
from email.message import Message
from email.utils import getaddresses, parsedate_to_datetime

from iris_personal.email.contracts import EmailAttachment, EmailMessage

from .keywords import labels_from_flags

SNIPPET_CHARS = 400
HEADERS_OF_INTEREST = ("Message-ID", "In-Reply-To", "References")
_WS = re.compile(r"\s+")
_ANGLE = re.compile(r"<[^<>\s]+>")


@dataclass(frozen=True)
class ParsedMail:
    message_id: str | None
    thread_root: str | None
    from_raw: str
    to: tuple[str, ...]
    cc: tuple[str, ...]
    subject: str
    date: datetime | None
    headers: dict[str, str]
    body_text: str | None
    body_html: str | None
    attachments: tuple[EmailAttachment, ...]


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:32]


def message_key(account_id: str, message_id: str | None, fallback: str) -> str:
    """The stable ``emails.id`` for one message of one account."""
    return f"imap-{_digest(account_id, message_id or 'fp:' + fallback)}"


def thread_key(account_id: str, root: str) -> str:
    return f"imap-t-{_digest(account_id, root)}"


def _addresses(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(addr for _name, addr in getaddresses([raw]) if "@" in addr)


def _domain(from_raw: str) -> str | None:
    pairs = getaddresses([from_raw]) if from_raw else []
    addr = next((a for _n, a in pairs if "@" in a), "")
    return addr.rsplit("@", 1)[1].strip().lower() or None if addr else None


def _text_of(part: Message) -> str:
    """A text part's content, decoded; a bad charset falls back to UTF-8 w/ replace."""
    if isinstance(part, MIMEMessage):
        try:
            content = part.get_content()
            if isinstance(content, str):
                return content
        except (LookupError, UnicodeDecodeError, KeyError, ValueError):
            pass
    payload = part.get_payload(decode=True)
    if isinstance(payload, bytes):
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except LookupError:
            return payload.decode("utf-8", errors="replace")
    return ""


def _is_attachment(part: Message) -> bool:
    if part.is_multipart():
        return False
    disposition = part.get_content_disposition()
    if disposition == "attachment":
        return True
    # An inline part with a file name that is not the message text (an inline image).
    return bool(part.get_filename()) and part.get_content_maintype() != "text"


def iter_attachments(msg: Message) -> list[tuple[str, Message]]:
    """``(attachment id, part)`` for each attachment; the id is the part's position in
    the MIME walk, so the same message always yields the same ids."""
    out: list[tuple[str, Message]] = []
    for index, part in enumerate(msg.walk()):
        if _is_attachment(part):
            out.append((f"part-{index}", part))
    return out


def _header(msg: Message, name: str) -> str:
    value = msg.get(name)
    return _WS.sub(" ", str(value)).strip() if value is not None else ""


def parse_bytes(raw: bytes) -> Message:
    return message_from_bytes(raw, policy=policy.default)


def parse_mail(raw: bytes) -> ParsedMail:
    msg = parse_bytes(raw)
    message_id = _header(msg, "Message-ID") or None
    references = _ANGLE.findall(_header(msg, "References"))
    in_reply_to = _ANGLE.findall(_header(msg, "In-Reply-To"))
    root = (references or in_reply_to or [message_id])[0]
    try:
        date = parsedate_to_datetime(_header(msg, "Date")) if _header(msg, "Date") else None
        if date is not None and date.tzinfo is None:
            date = date.replace(tzinfo=UTC)
    except (TypeError, ValueError, IndexError):
        date = None

    body_text: str | None = None
    body_html: str | None = None
    attachments: list[EmailAttachment] = []
    attachment_parts = {id(p) for _i, p in iter_attachments(msg)}
    for attachment_id, part in iter_attachments(msg):
        payload = part.get_payload(decode=True)
        attachments.append(
            EmailAttachment(
                filename=part.get_filename() or f"attachment-{attachment_id}",
                mime_type=part.get_content_type() or "application/octet-stream",
                size_bytes=len(payload) if isinstance(payload, bytes) else 0,
                attachment_id=attachment_id,
            )
        )
    for part in msg.walk():
        if part.is_multipart() or id(part) in attachment_parts:
            continue
        ctype = part.get_content_type()
        if ctype == "text/plain" and body_text is None:
            body_text = _text_of(part)
        elif ctype == "text/html" and body_html is None:
            body_html = _text_of(part)

    return ParsedMail(
        message_id=message_id,
        thread_root=root,
        from_raw=_header(msg, "From"),
        to=_addresses(_header(msg, "To")),
        cc=_addresses(_header(msg, "Cc")),
        subject=_header(msg, "Subject"),
        date=date,
        headers={h: _header(msg, h) for h in HEADERS_OF_INTEREST if _header(msg, h)},
        body_text=body_text,
        body_html=body_html,
        attachments=tuple(attachments),
    )


def html_to_text(markup: str) -> str:
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", markup)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</tr>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return html_lib.unescape(text)


def readable_text(mail: ParsedMail) -> str:
    """The plain body, else the HTML body stripped to text."""
    if mail.body_text and mail.body_text.strip():
        return mail.body_text
    return html_to_text(mail.body_html) if mail.body_html else ""


def snippet_of(mail: ParsedMail) -> str:
    return _WS.sub(" ", readable_text(mail)).strip()[:SNIPPET_CHARS]


def to_email_message(
    mail: ParsedMail,
    *,
    account_id: str,
    folder: str,
    flags: tuple[str, ...],
    internaldate: datetime | None,
) -> EmailMessage:
    """The envelope for the store. Bodies ride along (transit only; the store keeps the
    snippet), labels are the folder, read state and keywords (``keywords.py``)."""
    received = internaldate or mail.date or datetime.now(UTC)
    fingerprint = f"{mail.date.isoformat() if mail.date else ''}|{mail.from_raw}|{mail.subject}"
    return EmailMessage(
        id=message_key(account_id, mail.message_id, fingerprint),
        provider="imap",
        account_id=account_id,
        thread_id=thread_key(account_id, mail.thread_root or fingerprint),
        from_address=mail.from_raw if len(mail.from_raw) >= 3 else "unknown@unknown",
        from_domain=_domain(mail.from_raw),
        to=mail.to,
        cc=mail.cc,
        subject=mail.subject,
        received_at=received,
        snippet=snippet_of(mail),
        body_text=mail.body_text,
        body_html=mail.body_html,
        labels=labels_from_flags(folder, flags),
        attachments=mail.attachments,
        headers_subset=dict(mail.headers),
    )


__all__ = [
    "HEADERS_OF_INTEREST",
    "SNIPPET_CHARS",
    "ParsedMail",
    "html_to_text",
    "iter_attachments",
    "message_key",
    "parse_bytes",
    "parse_mail",
    "readable_text",
    "snippet_of",
    "thread_key",
    "to_email_message",
]
