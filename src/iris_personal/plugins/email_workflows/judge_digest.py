"""The email judge in the morning digest (loop-proof PR 5).

Three pieces, every word from ``judge.yaml`` (``digest:``):

* **Needs reply (N)** — the Inbox group's section: emails whose effective bucket is
  ``needs_reply``, newest first, at most ``needs_reply_max``, as "Sender: subject". An
  email leaves when the owner answers it (a message in its thread carrying one of
  ``sent_labels``, dated after the email), when the owner moves it to another bucket,
  or when it ages out: D18, every dated item ends — ``needs_reply_days`` local days
  after its judgment (``config/digest.yaml`` ``expiry:``).
* **Judged yesterday** — one line: how many emails the judge read yesterday, by
  effective bucket; the Unsure ones ask to be taught (their Action Center cards) and
  the waiting ones say they are judged next sweep. Nothing at all on a day with neither.
* **learned yesterday** — the footer's source ``email_judgment_corrections``: one
  phrase per correction made that day, "Capital One → Needs reply (you said so in
  chat)", saying where the owner made it.

The push headline is unchanged: the Inbox group's push line is still Focus.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, tzinfo
from pathlib import Path

from iris_personal.email.store import EmailStore

from .judge_cards import days_expired
from .judge_config import JudgeConfig
from .judge_view import JudgedEmail, open_stores, with_emails
from .judge_words import SurfaceWords, bucket_name
from .judgments import PROMO, JudgmentStore

NEEDS_REPLY = "needs_reply"
UNSURE = "unsure"
_SCAN = 1000


def _aware(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def replied(emails: EmailStore, item: JudgedEmail, sent_labels: frozenset[str]) -> bool:
    """Whether the owner answered: a message in the email's thread with a sent label,
    dated after the email."""
    email = item.email
    if email is None or not email.thread_id:
        return False
    for other in emails.list_by_thread(email.account_id, email.thread_id):
        if other.id == email.id or other.received_at <= email.received_at:
            continue
        if sent_labels.intersection(other.labels):
            return True
    return False


def needs_reply_items(
    judgments: JudgmentStore,
    emails: EmailStore,
    *,
    days: int,
    sent_labels: frozenset[str],
    now: datetime,
    tz: tzinfo,
) -> list[JudgedEmail]:
    """The open "needs reply" emails, newest first (not answered, not aged out)."""
    since = now - timedelta(days=days + 2)
    rows = judgments.recent(bucket=NEEDS_REPLY, limit=_SCAN, since=since)
    out: list[JudgedEmail] = []
    for item in with_emails(emails, rows):
        judged = _aware(item.judgment.judged_at)
        if judged is None or days_expired(judged, days, now, tz):
            continue
        if replied(emails, item, sent_labels):
            continue
        out.append(item)
    out.sort(key=lambda i: i.received_at or _aware(i.judgment.judged_at) or now, reverse=True)
    return out


def render_needs_reply(
    judgments: JudgmentStore,
    emails: EmailStore,
    words: SurfaceWords,
    *,
    days: int,
    now: datetime,
    tz: tzinfo,
) -> str:
    """The whole section as markdown: its heading, then one bullet per email."""
    d = words.digest
    items = needs_reply_items(
        judgments, emails, days=days, sent_labels=words.sent_labels, now=now, tz=tz
    )
    if not items:
        return f"## {d['needs_reply_empty_title']}\n{d['needs_reply_empty']}"
    shown = [
        "- " + d["needs_reply_item"].format(sender=i.sender or "Unknown sender", subject=i.subject)
        for i in items[: words.needs_reply_max]
    ]
    more = len(items) - len(shown)
    if more > 0:
        shown.append(f"- +{more} more")
    return f"## {d['needs_reply_title'].format(count=len(items))}\n" + "\n".join(shown)


def judged_line(
    judgments: JudgmentStore,
    config: JudgeConfig,
    words: SurfaceWords,
    *,
    start: datetime,
    end: datetime,
) -> str:
    """ "Judged yesterday: 41 · 3 bill · … · 2 unsure — please teach me (Action Center)
    · 7 waiting — …"; ``""`` when nothing was judged and nothing waits."""
    d = words.digest
    rows = judgments.judged_between(start, end)
    waiting = judgments.count_waiting()
    if not rows and not waiting:
        return ""
    parts = [d["judged_line"].format(total=len(rows))]
    for key in (*config.keys, PROMO):
        if key == UNSURE:
            continue
        n = sum(1 for j in rows if j.effective_bucket == key)
        if n:
            name = bucket_name(config, words, key).lower()
            parts.append(d["judged_part"].format(count=n, name=name))
    unsure = sum(1 for j in rows if j.effective_bucket == UNSURE)
    if unsure:
        parts.append(d["unsure_part"].format(count=unsure))
    if waiting:
        parts.append(d["waiting_part"].format(count=waiting))
    return " · ".join(parts)


def learned_phrases(
    judgments: JudgmentStore,
    emails: EmailStore,
    config: JudgeConfig,
    words: SurfaceWords,
    start: datetime,
    end: datetime,
) -> list[str]:
    """One footer phrase per email the owner corrected in ``[start, end)``."""
    out: list[str] = []
    for item in with_emails(emails, judgments.corrected_between(start, end)):
        j = item.judgment
        if not j.owner_bucket:
            continue
        how = words.learned_how.get(j.owner_source or "", "")
        phrase = words.digest["learned"].format(
            sender=item.sender or item.subject,
            bucket=bucket_name(config, words, j.owner_bucket),
            how=how,
        )
        out.append(phrase.replace(" ()", ""))
    return out


def learned_source(
    data_dir: Path, config_dir: Path | None = None
) -> Callable[[datetime, datetime], list[str]]:
    """The digest footer's learned source for this data dir (``register_learned_source``)."""

    def _source(start: datetime, end: datetime) -> list[str]:
        stores = open_stores(data_dir, create=False)
        if stores is None:
            return []
        judgments, emails = stores
        config = JudgeConfig.load(config_dir)
        return learned_phrases(judgments, emails, config, SurfaceWords.load(config_dir), start, end)

    return _source


__all__ = [
    "judged_line",
    "learned_phrases",
    "learned_source",
    "needs_reply_items",
    "render_needs_reply",
    "replied",
]
