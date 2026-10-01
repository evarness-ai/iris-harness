"""Inbox digest — answers "what's my email inbox look like today?".

Deterministic-first, same contract as the Planner (``iris_harness.planner.handler``):
the factual digest is assembled straight from the local ``EmailStore`` (no LLM,
no network — the Gmail sweep heartbeat keeps the store synced), and an optional
LLM narrative is layered *on top* without inventing or dropping anything. The
narrative falls back to the deterministic text whenever the LLM is absent,
errors, or the inbox is empty, so the facts are never at the model's mercy.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore

logger = logging.getLogger(__name__)

# How many accounts/messages to scan and surface. Scan deeper than we show so
# the "today" partition is accurate even on busy mailboxes.
_SCAN_LIMIT = 60
_RECENT_LIMIT = 8


def _display_address(account_id: str) -> str:
    """``"gmail:foo@bar.com"`` -> ``"foo@bar.com"`` (provider prefix stripped)."""
    return account_id.split(":", 1)[1] if ":" in account_id else account_id


def _sender_name(from_address: str) -> str:
    """Pull a friendly sender from ``"Name <a@b.com>"`` or ``"a@b.com"``."""
    addr = from_address.strip()
    if "<" in addr:
        name = addr.split("<", 1)[0].strip().strip('"')
        if name:
            return name
        return addr.split("<", 1)[1].split(">", 1)[0].strip()
    return addr


@dataclass(frozen=True)
class AccountDigest:
    """Per-mailbox slice of the inbox digest."""

    account_id: str
    address: str
    total: int
    today_count: int
    recent: tuple[EmailMessage, ...]


@dataclass(frozen=True)
class InboxDigest:
    """Factual snapshot of every connected mailbox at a point in time."""

    generated_at: datetime
    accounts: tuple[AccountDigest, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not self.accounts or all(a.total == 0 for a in self.accounts)

    @property
    def today_total(self) -> int:
        return sum(a.today_count for a in self.accounts)

    @property
    def grand_total(self) -> int:
        return sum(a.total for a in self.accounts)


def build_inbox_digest(
    store: EmailStore,
    *,
    now: datetime,
    scan_limit: int = _SCAN_LIMIT,
    recent_limit: int = _RECENT_LIMIT,
) -> InboxDigest:
    """Assemble a deterministic inbox digest from the local store.

    ``now`` is timezone-aware; "today" is partitioned in *its* timezone so the
    count matches the user's wall clock regardless of how messages were stored.
    """
    accounts: list[AccountDigest] = []
    for account_id in store.list_accounts():
        recent_msgs = store.list_recent(account_id, limit=scan_limit)
        today_count = sum(1 for m in _on_day(recent_msgs, now))
        accounts.append(
            AccountDigest(
                account_id=account_id,
                address=_display_address(account_id),
                total=store.count(account_id),
                today_count=today_count,
                recent=tuple(recent_msgs[:recent_limit]),
            )
        )
    return InboxDigest(generated_at=now, accounts=tuple(accounts))


def _on_day(messages: list[EmailMessage], now: datetime) -> list[EmailMessage]:
    """Messages received on ``now``'s calendar day, compared in ``now``'s tz."""
    target = now.date()
    same_day: list[EmailMessage] = []
    for m in messages:
        received = m.received_at
        if received.tzinfo is not None and now.tzinfo is not None:
            received = received.astimezone(now.tzinfo)
        if received.date() == target:
            same_day.append(m)
    return same_day


def render_digest_text(digest: InboxDigest) -> str:
    """Render an InboxDigest as plain text (channel-agnostic)."""
    if digest.is_empty:
        return (
            "Your local email store is empty — no messages have been synced yet. "
            "Connect an account with `iris auth gmail login --user <address>`, then "
            "let the sync run, and I'll be able to summarise your inbox."
        )

    day = digest.generated_at.strftime("%Y-%m-%d")
    n_acct = len(digest.accounts)
    header = (
        f"Inbox on {day}: {digest.today_total} new today across "
        f"{n_acct} account{'s' if n_acct != 1 else ''} "
        f"({digest.grand_total} total stored)."
    )
    lines = [header]
    for acct in digest.accounts:
        lines.append("")
        lines.append(f"{acct.address} — {acct.today_count} today, {acct.total} total:")
        if not acct.recent:
            lines.append("- (no messages)")
            continue
        for m in acct.recent:
            when = m.received_at.strftime("%b %d %H:%M")
            sender = _sender_name(m.from_address)
            subject = m.subject.strip() or "(no subject)"
            lines.append(f"- {when} · {sender}: {subject}")
    return "\n".join(lines)


_NARRATE_PROMPT = (
    "You are a personal assistant giving the user a quick read on their inbox. "
    "Below is the FACTUAL digest, already assembled from their local mail store. "
    "Write a brief (2-3 sentence) summary: how busy today is and what looks worth "
    "attention first. Do NOT invent senders, subjects, or counts; do NOT drop or "
    "renumber items. Plain prose, no lists.\n\n"
    "DIGEST:\n{digest}\n\nSUMMARY:"
)


def narrate_digest(digest: InboxDigest, *, llm_call: Callable[[str], str] | None = None) -> str:
    """Deterministic digest text, optionally prefaced by a grounded LLM narrative.

    The factual list is always appended, so the LLM can only *summarise* — never
    replace — the grounded digest. Falls back to plain text on any error.
    """
    text = render_digest_text(digest)
    if llm_call is None or digest.is_empty:
        return text
    try:
        narrative = llm_call(_NARRATE_PROMPT.format(digest=text)).strip()
    except Exception:  # narrative is optional polish
        logger.exception("email digest: narrative LLM call failed; using plain digest")
        return text
    return f"{narrative}\n\n{text}" if narrative else text


__all__ = [
    "AccountDigest",
    "InboxDigest",
    "build_inbox_digest",
    "render_digest_text",
    "narrate_digest",
]
