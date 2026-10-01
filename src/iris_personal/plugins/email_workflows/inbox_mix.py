"""The morning digest's Inbox summary: one line per mailbox, last 24 h by category.

"owner@example.com: 42 in the last 24 h — 12 updates · 9 promo · 6 finance"

Each mailbox's mail from the window is grouped by its triage top-level category — the
first segment under the ADR-0017 ``email/`` root, so ``email/finance/cards`` counts as
``finance``. The group names come from the stored paths; nothing here names one.
Largest groups first; mail triage has not sorted yet closes the line as ``unsorted N``
(only when there is some). Individual emails are Focus's job, not this section's.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows.digest_focus import WINDOW

#: The ADR-0017 root every triage path sits under.
_ROOT = "email/"
#: The closing group for mail triage has not classified yet.
UNSORTED = "unsorted"


def top_group(path: str | None) -> str | None:
    """``email/finance/cards`` → ``finance``; ``None`` (or a bare root) → ``None``."""
    p = (path or "").strip().strip("/")
    if p.startswith(_ROOT):
        p = p[len(_ROOT) :]
    elif p == _ROOT.rstrip("/"):
        p = ""
    head = p.split("/", 1)[0].strip()
    return head or None


def account_label(account_id: str) -> str:
    """``gmail:owner@example.com`` → ``owner@example.com`` (the provider prefix drops)."""
    _, sep, rest = account_id.partition(":")
    return rest if sep and rest else account_id


def mix_line(counts: dict[str | None, int]) -> str:
    """'42 in the last 24 h — 12 updates · 9 promo · unsorted 3' for one mailbox."""
    groups: Counter[str] = Counter()
    unsorted = 0
    for path, n in counts.items():
        group = top_group(path)
        if group is None:
            unsorted += n
        else:
            groups[group] += n
    total = sum(groups.values()) + unsorted
    if total == 0:
        return "nothing in the last 24 h"
    parts = [f"{n} {g}" for g, n in sorted(groups.items(), key=lambda kv: (-kv[1], kv[0]))]
    if unsorted:
        parts.append(f"{UNSORTED} {unsorted}")
    return f"{total} in the last 24 h — {' · '.join(parts)}"


def inbox_mix(
    store: EmailStore, *, now: datetime, account_id: str | None = None
) -> list[dict[str, str]]:
    """One row per mailbox in the store (or just ``account_id``): label, total, line.

    Every known mailbox gets a row — a quiet one says "nothing in the last 24 h" so
    the owner can see both inboxes were looked at.
    """
    accounts = store.list_accounts()
    if account_id is not None:
        accounts = [a for a in accounts if a == account_id]
    since = now - WINDOW
    rows: list[dict[str, str]] = []
    for account in accounts:
        counts = store.path_counts(account, since=since)
        rows.append(
            {
                "account": account_label(account),
                "account_id": account,
                "total": str(sum(counts.values())),
                "summary": mix_line(counts),
            }
        )
    return rows


__all__ = ["UNSORTED", "account_label", "inbox_mix", "mix_line", "top_group"]
