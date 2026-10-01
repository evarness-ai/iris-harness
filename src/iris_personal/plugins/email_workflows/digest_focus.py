"""The morning digest's Focus section (loop-proof PR 2, D17).

Focus = the last 24 hours of mail whose triage category is one the owner chose
(``focus_categories`` in Settings → Digest), minus every sender the owner marked "not
useful": the newest ``focus_per_account`` from each inbox, then newest first across
them, capped at ``focus_limit``. One busy inbox cannot crowd out the others. Deterministic: no judge, no ranking
model — the category is the filter and time is the order.

Each line ends with a 👎 link, ``[👎](iris:not-useful/<quoted sender>)``: the web
rendering turns it into a button that posts to ``/api/digest/not-useful``, the other
renderings strip it. The verdict lands in the surface-suppression ledger under
``email/focus`` keyed by the sender's address, which :func:`focus_messages` consults —
so the sender is gone from tomorrow's Focus and the footer says so.

The category vocabulary is the owner's (Settings / ``config/digest.yaml``); nothing
here names one. A category is matched as a triage path (``finance`` matches
``email/finance`` and everything below it) — the ``email/`` root is the ADR-0017 path
scheme, not a category.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime, timedelta
from urllib.parse import quote

from iris_harness.sdk.learning import SurfaceFeedbackStore
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.feedback_keys import EMAIL_FOCUS_SURFACE, EMAIL_SUBSYSTEM, email_focus_dims
from iris_personal.email.store import CategoryFilter, EmailStore

logger = logging.getLogger(__name__)

#: The ADR-0017 root every triage path sits under.
_ROOT = "email/"
#: How far back Focus looks.
WINDOW = timedelta(hours=24)
#: The scheme stream B's web renderer turns into a button (and strips elsewhere).
NOT_USEFUL_SCHEME = "iris:not-useful/"


def _path(category: str) -> str:
    c = category.strip().strip("/")
    return c if c.startswith(_ROOT) else _ROOT + c


def _display(category: str | None) -> str:
    c = (category or "").strip()
    return c[len(_ROOT) :] if c.startswith(_ROOT) else c


def category_filter(categories: Iterable[str]) -> CategoryFilter:
    """The triage paths the owner's Focus categories name."""
    paths = tuple(dict.fromkeys(_path(c) for c in categories if c and c.strip()))
    return CategoryFilter(paths=paths)


def sender_name(from_address: str) -> str:
    """'GoldenPi <news@goldenpi.com>' → 'GoldenPi'; a bare address stays itself."""
    s = from_address.strip()
    if "<" in s and ">" in s:
        name = s[: s.find("<")].strip().strip('"').strip()
        if name:
            return name
        s = s[s.find("<") + 1 : s.rfind(">")]
    return s.strip()


def not_useful_link(from_address: str) -> str:
    """The 👎 link for one sender, in the exact scheme the renderers know."""
    sender = email_focus_dims(from_address)["sender"]
    return f"[👎]({NOT_USEFUL_SCHEME}{quote(sender, safe='')})"


def focus_messages(
    store: EmailStore,
    suppression: SurfaceFeedbackStore,
    categories: Iterable[str],
    *,
    limit: int,
    now: datetime,
    per_account: int | None = None,
) -> list[EmailMessage]:
    """The Focus emails: the newest ``per_account`` from each inbox (suppressed senders
    removed first), then the newest ``limit`` of those. ``per_account=None`` = no
    per-inbox cap."""
    wanted = category_filter(categories)
    if not wanted.paths or limit <= 0:
        return []
    since = now - WINDOW
    hidden: dict[str, bool] = {}

    def suppressed(message: EmailMessage) -> bool:
        dims = email_focus_dims(message.from_address)
        key = dims["sender"]
        if key not in hidden:
            hidden[key] = suppression.should_suppress(EMAIL_SUBSYSTEM, EMAIL_FOCUS_SURFACE, dims)
        return hidden[key]

    cap = limit if per_account is None or per_account <= 0 else min(per_account, limit)
    found: list[EmailMessage] = []
    for account_id in store.list_accounts():
        # Every row in the window: suppressed senders are filtered after the query,
        # so a capped fetch could come back short.
        count = store.count_by_category(account_id, wanted, since=since)
        if not count:
            continue
        rows = sorted(
            store.list_by_category(account_id, wanted, limit=count, since=since),
            key=lambda m: m.received_at,
            reverse=True,
        )
        found.extend([m for m in rows if not suppressed(m)][:cap])
    found.sort(key=lambda m: m.received_at, reverse=True)
    return found[:limit]


def focus_header(categories: Iterable[str], limit: int, per_account: int | None = None) -> str:
    """'Focus — personal · family · finance, newest 5 per inbox' (or 'newest 10' when
    there is no per-inbox cap below ``limit``)."""
    names = [_display(_path(c)) for c in categories if c and c.strip()]
    newest = (
        f"newest {per_account} per inbox"
        if per_account is not None and 0 < per_account < limit
        else f"newest {limit}"
    )
    return f"Focus — {' · '.join(dict.fromkeys(names))}, {newest}"


def focus_line(message: EmailMessage) -> str:
    """'GoldenPi · Your weekly guide to bond investing · finance [👎](…)'.

    The category shows as its top group only ("finance", not "finance/investing"):
    the heading already names the categories; the line stays short.
    """
    subject = " ".join((message.subject or "(no subject)").split())
    group = _display(message.classified_category).split("/", 1)[0].strip()
    tail = f" · {group}" if group else ""
    return (
        f"{sender_name(message.from_address)} · {subject}{tail} "
        f"{not_useful_link(message.from_address)}"
    )


def render_focus(
    store: EmailStore,
    suppression: SurfaceFeedbackStore,
    categories: Iterable[str],
    *,
    limit: int,
    now: datetime,
    per_account: int | None = None,
) -> str:
    """The whole Focus section as markdown: its heading line, then one bullet per email.

    Never empty: with no categories chosen, or none in the window, the section says so.
    """
    chosen = [c for c in categories if c and c.strip()]
    if not chosen:
        return "## Focus\nNo focus categories chosen — pick them in Settings → Digest."
    header = f"## {focus_header(chosen, limit, per_account)}"
    messages = focus_messages(
        store, suppression, chosen, limit=limit, now=now, per_account=per_account
    )
    if not messages:
        return f"{header}\nNothing in these categories in the last 24 h."
    return header + "\n" + "\n".join(f"- {focus_line(m)}" for m in messages)


__all__ = [
    "NOT_USEFUL_SCHEME",
    "WINDOW",
    "category_filter",
    "focus_header",
    "focus_line",
    "focus_messages",
    "not_useful_link",
    "render_focus",
    "sender_name",
]
