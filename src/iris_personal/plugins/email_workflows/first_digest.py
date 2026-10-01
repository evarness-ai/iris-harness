"""The first digest: what IRIS shows once it has read a mailbox for the first time.

``iris email demo`` and ``iris email setup`` (step 7) both end on it, so it is one
function: the email agent's inbox digest (narrated on the governed tier when a
``narrate`` call is given, deterministic otherwise), the morning digest's Needs reply
section, and the bills and events the judge copied a date out of. Only what the stores
hold: nothing here reaches a mailbox or the network.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# The most emails one dated section reads (the whole demo corpus fits).
_SECTION_MAX = 1000


@dataclass(frozen=True)
class FirstDigest:
    """The digest's sections by name (``inbox``, ``needs_reply``, ``bills``, ``events``)
    and the whole text."""

    sections: dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(self.sections.values())


def build_first_digest(
    data_dir: Path,
    *,
    narrate: Callable[[str], str] | None,
    config_dir: Path | None = None,
) -> FirstDigest:
    """Build the digest from ``data_dir/email.db`` (the judgments beside the mail)."""
    from iris_harness.sdk.digest import expiry_days
    from iris_harness.sdk.time import iris_timezone

    from .agent import _email_inbox_digest
    from .judge_digest import render_needs_reply
    from .judge_view import judged_emails, open_stores
    from .judge_words import SurfaceWords

    inbox, _meta = _email_inbox_digest(narrate, data_dir)
    stores = open_stores(data_dir)
    assert stores is not None  # create=True always opens
    judgments, emails = stores
    tz = iris_timezone()
    needs_reply = render_needs_reply(
        judgments,
        emails,
        SurfaceWords.load(config_dir),
        days=expiry_days("needs_reply_days"),
        now=datetime.now(tz),
        tz=tz,
    )
    bills = render_dated(
        judged_emails(judgments, emails, bucket="bill", limit=_SECTION_MAX),
        title="Bills due",
        date_field="due_date",
        line="{sender}: ${amount} due {date}",
        undated="{count} more bill email(s) need nothing: payments already received.",
    )
    events = render_dated(
        judged_emails(judgments, emails, bucket="event", limit=_SECTION_MAX),
        title="Coming up",
        date_field="event_start",
        line="{date}: {subject} ({sender})",
    )
    return FirstDigest(
        sections={"inbox": inbox, "needs_reply": needs_reply, "bills": bills, "events": events}
    )


def render_dated(
    items: list[Any],
    *,
    title: str,
    date_field: str,
    line: str,
    undated: str = "",
    limit: int = 10,
) -> str:
    """A digest section from judged emails and the date the judge copied out of each
    (``due_date``, ``event_start``), soonest first. Only what the judge recorded."""
    dated = [i for i in items if isinstance(i.judgment.fields.get(date_field), str)]
    dated.sort(key=lambda i: str(i.judgment.fields[date_field]))
    lines = [
        "- "
        + line.format(
            sender=i.sender or "Unknown sender",
            subject=i.subject,
            date=str(i.judgment.fields[date_field]).replace("T", " "),
            amount=i.judgment.fields.get("min_due", "?"),
        )
        for i in dated[:limit]
    ]
    if len(dated) > limit:
        lines.append(f"- +{len(dated) - limit} more")
    rest = len(items) - len(dated)
    if rest and undated:
        lines.append(undated.format(count=rest))
    body = "\n".join(lines) if lines else "Nothing."
    return f"## {title} ({len(dated)})\n{body}"


__all__ = ["FirstDigest", "build_first_digest", "render_dated"]
