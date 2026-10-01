"""``send_email``: a plain-text email, sent only once the owner approves the exact message.

Sending is not data loss, so the tool is declared ``effect: write`` — but it cannot be
taken back either, so its manifest also declares ``approval: pinned`` (ADR-0118
amendment). Every call then takes the destructive tools' path: the loop checks the
arguments here (``validate``), the card shows the whole message in the owner's words
(``describe``), the approval row pins the exact call, approving runs exactly that call,
and rejecting runs nothing.

``reply_to_id`` makes it a reply: it names a message in the local store, whose account
sends the reply, whose sender receives it when ``to`` is left out, whose subject gets a
"Re:" when ``subject`` is, and whose Message-ID threads it (In-Reply-To, References, and
Gmail's threadId). Plain text only: no HTML, no attachments, no drafts.

Two guards from the owner's first real send (2026-09-22), where the model turned the
"name@…" the owner typed into an address that does not exist and the mail
would have gone from whichever account received mail last:

* Every recipient must appear in the owner's request or already be in their mail
  (as a sender, recipient or cc). Anything else is refused before a card exists, with
  the reason, so the model can correct itself.
* A new message is sent from ``from`` when given (a connected account), else from
  ``IRIS_EMAIL_DEFAULT_SENDER`` (the owner's local setting, never in the repo), else
  from the first connected account in a fixed order. A reply keeps the account that
  received the original.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from email.utils import formataddr, getaddresses, parseaddr
from pathlib import Path
from typing import Any

from iris_harness.sdk.types import ToolDescription, ToolSpec
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore

logger = logging.getLogger(__name__)

# A deliberately plain check: one @, no spaces or brackets, a dot in the domain. The
# provider does the real validation; this keeps a card from ever naming "Bob" or "me".
_ADDRESS = re.compile(r"^[^@\s<>,;\"]+@[^@\s<>,;\"]+\.[^@\s<>,;\"]+$")
#: The owner's default sending account (an address or an account id), set locally.
DEFAULT_SENDER_ENV = "IRIS_EMAIL_DEFAULT_SENDER"
# Arguments a plain-text send refuses rather than silently dropping.
_UNSUPPORTED = ("attachments", "attachment", "html", "body_html", "bcc", "draft")


def _as_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return [str(raw)]
    return [str(v) for v in raw if str(v).strip()]


def _recipients(raw: Any) -> tuple[list[str], list[str]]:
    """``(addresses, rejected)``: each entry parsed as ``Name <a@b>`` or ``a@b``; a
    comma-separated string counts as several. Rejected entries are kept verbatim so the
    refusal can quote them."""
    good: list[str] = []
    bad: list[str] = []
    for entry in _as_list(raw):
        if "\r" in entry or "\n" in entry:
            bad.append(entry.strip())
            continue
        pairs = getaddresses([entry])
        if not pairs or any(not _ADDRESS.match(addr) for _name, addr in pairs):
            bad.append(entry.strip())
            continue
        for name, addr in pairs:
            formatted = formataddr((name, addr)) if name else addr
            if formatted not in good:
                good.append(formatted)
    return good, bad


def _reply_subject(original: str) -> str:
    subject = (original or "").strip()
    return subject if subject.lower().startswith("re:") else f"Re: {subject}".strip()


def _bare(address: str) -> str:
    return parseaddr(address)[1].strip().lower()


def _account_for(wanted: str, accounts: list[str]) -> str | None:
    """The connected account ``wanted`` names (an address or an account id), or None."""
    want = wanted.strip().lower()
    for account_id in accounts:
        if want in (account_id.lower(), account_id.split(":", 1)[-1].lower()):
            return account_id
    return None


def build_send_tools(
    *,
    data_dir: Path,
    provider_for: Callable[[str], Any] | None = None,
    current_query: Callable[[], str] | None = None,
) -> list[ToolSpec]:
    """``send_email`` over the store at ``data_dir``. ``provider_for`` maps an account to
    its mail provider (the registered one by default; tests pass a fake).
    ``current_query`` returns the owner's request for this turn; a recipient must appear
    in it or in the stored mail."""

    def _store() -> EmailStore:
        store = EmailStore(db_path=data_dir / "email.db")
        store.ensure_schema()
        return store

    def _provider(account_id: str) -> Any:
        if provider_for is not None:
            return provider_for(account_id)
        from iris_personal.email.providers import mail_provider_for

        return mail_provider_for(account_id)

    def _plan(args: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        """The message this call would send, or why it cannot be sent. ``validate``,
        ``describe`` and the send all read it, so the card shows what is sent."""
        unsupported = [key for key in _UNSUPPORTED if args.get(key)]
        if unsupported:
            return None, (
                f"send_email sends plain text only; it does not take {', '.join(unsupported)}. "
                "Leave those out, or tell the owner it cannot be done here."
            )
        store = _store()
        original: EmailMessage | None = None
        reply_to_id = str(args.get("reply_to_id") or "").strip()
        if reply_to_id:
            original = store.get(reply_to_id)
            if original is None:
                return None, (
                    f"reply_to_id {reply_to_id!r} is not in the owner's mail. Use only an "
                    "[id …] value that search_inbox or list_by_category printed in this "
                    "conversation."
                )
        to, bad_to = _recipients(args.get("to"))
        cc, bad_cc = _recipients(args.get("cc"))
        if bad_to or bad_cc:
            return None, (
                f"not an email address: {', '.join(bad_to + bad_cc)}. Use the full address "
                "(name@example.com); if you do not know it, ask the owner."
            )
        if not to and original is not None:
            to, _bad = _recipients(original.from_address)
        if not to:
            return None, (
                'send_email needs a recipient: {"to": ["name@example.com"]}, or a '
                "reply_to_id to answer that email's sender."
            )
        asked = (current_query() if current_query is not None else "").lower()
        unknown = [
            a for a in [*to, *cc] if _bare(a) not in asked and not store.address_known(_bare(a))
        ]
        if unknown:
            return None, (
                f"{', '.join(unknown)} is not in the owner's request and has never appeared "
                "in their mail, so it may be misspelt or made up. Use each address exactly "
                "as the owner wrote it, or ask them for it."
            )
        subject = str(args.get("subject") or "").strip()
        if not subject and original is not None:
            subject = _reply_subject(original.subject)
        if not subject:
            return None, "send_email needs a subject."
        if "\r" in subject or "\n" in subject:
            return None, "the subject must be one line."
        body = str(args.get("body") or "")
        if not body.strip():
            return None, "send_email needs the message text in body; it is empty."
        if original is not None:
            account_id = original.account_id
        else:
            accounts = sorted(store.list_accounts())
            if not accounts:
                return None, "no mail account is connected, so nothing can be sent."
            wanted = str(args.get("from") or args.get("sender") or "").strip()
            configured = os.environ.get(DEFAULT_SENDER_ENV, "").strip()
            if wanted:
                chosen = _account_for(wanted, accounts)
                if chosen is None:
                    return None, (
                        f"{wanted} is not a connected account. Send from one of: "
                        f"{', '.join(a.split(':', 1)[-1] for a in accounts)}."
                    )
            elif configured:
                chosen = _account_for(configured, accounts)
                if chosen is None:
                    return None, (
                        f"{DEFAULT_SENDER_ENV} names {configured}, which is not a connected "
                        "account, so nothing can be sent until it is fixed."
                    )
            else:
                chosen = accounts[0]  # a fixed order, never "whichever got mail last"
            account_id = chosen
        headers = dict(original.headers_subset) if original is not None else {}
        return {
            "account_id": account_id,
            "sender": account_id.split(":", 1)[-1],
            "to": to,
            "cc": [c for c in cc if c not in to],
            "subject": subject,
            "body": body,
            "original": original,
            "in_reply_to": headers.get("Message-ID") or None,
            "references": headers.get("References") or None,
            "thread_id": original.thread_id if original is not None else None,
        }, None

    def validate_send(args: dict[str, Any]) -> str | None:
        """Refuse a message the owner could only approve to fail: bad addresses, no
        body, a reply to an email that is not in their mail, attachments or HTML."""
        _, problem = _plan(args)
        return problem

    def describe_send(args: dict[str, Any]) -> ToolDescription:
        plan, problem = _plan(args)
        if plan is None:
            # validate refuses these before a card exists; this only guards the shape.
            return ToolDescription(title="Send an email", lines=(problem or "",))
        to = plan["to"]
        who = to[0] if len(to) == 1 else f"{to[0]} and {len(to) - 1} more"
        lines = [f"From: {plan['sender']}", f"To: {', '.join(to)}"]
        if plan["cc"]:
            lines.append(f"Cc: {', '.join(plan['cc'])}")
        lines.append(f"Subject: {plan['subject']}")
        original = plan["original"]
        if original is not None:
            lines.append(
                f"In reply to: {(original.subject or '(no subject)').strip()} — "
                f"{original.from_address} · {original.received_at:%d %b}"
            )
        lines.append("Message:")
        lines.extend(line for line in plan["body"].splitlines() if line.strip())
        return ToolDescription(title=f"Send email to {who} — {plan['subject']}", lines=tuple(lines))

    def send(args: dict[str, Any]) -> str:
        plan, problem = _plan(args)
        if plan is None:
            return f"Error: {problem} Nothing was sent."
        provider = _provider(plan["account_id"])
        if provider is None or not hasattr(provider, "send_message"):
            return f"Error: no mail provider can send from {plan['account_id']}. Nothing was sent."
        try:
            sent_id = provider.send_message(
                plan["account_id"],
                to=plan["to"],
                cc=plan["cc"],
                subject=plan["subject"],
                body=plan["body"],
                in_reply_to=plan["in_reply_to"],
                references=plan["references"],
                thread_id=plan["thread_id"],
            )
        except PermissionError as exc:
            return f"Not sent: {exc}"
        logger.info("send_email: sent %s from %s", sent_id or "(no id)", plan["account_id"])
        kind = "Reply sent" if plan["original"] is not None else "Sent"
        return f"{kind} to {', '.join(plan['to'])}: {plan['subject']}."

    return [
        ToolSpec(
            name="send_email",
            description=(
                "Send a plain-text email, or reply to one. For when the user asks to send, "
                'write, email or reply to someone. Args: {"to": [str] (full addresses), '
                '"subject": str, "body": str (the whole message), "cc"?: [str], '
                '"reply_to_id"?: str (an [id …] from search_inbox or list_by_category; the '
                "reply goes to that email's sender and thread, so to and subject may be "
                'left out), "from"?: str (only when the user names the account to send '
                "from)}. Copy every address exactly as the user wrote it. No attachments or "
                "formatting. Every call shows the owner the "
                "whole message on an approval card before anything is sent; do not ask "
                "them to confirm first."
            ),
            call=send,
            describe=describe_send,
            validate=validate_send,
        )
    ]


__all__ = ["build_send_tools"]
