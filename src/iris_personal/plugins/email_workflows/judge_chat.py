"""Correcting the email judge in chat, deterministically (loop-proof PR 5).

"the Capital One email isn't FYI, it needs a reply", "that dental email is an event",
"the X email is a bill", "X is promo" — the owner telling IRIS which bucket an email
belongs in. It must move THAT email through :func:`.judge_corrections.apply_correction`
(``source="chat"``) and never depend on a model choosing a tool, so it is an intercept,
in the house pattern for grounded actions (ADR-0097, ADR-0102): parse, find the email
among the judged ones, act, report.

The grammar, every word of it from ``judge.yaml`` (``rebucket.chat``):

* ``<which email> <link> <bucket>`` — the words before the link ("is", "needs",
  "should be", ...) name the email; the bucket word after it (judge.yaml's
  ``bucket_words``) is where it goes. A bucket in a negated clause ("isn't FYI") is
  what it is NOT: it only narrows which email is meant.
* The email is named by words of its sender (name, address, domain) and subject, among
  the rows judged in the last ``window_days``. Every naming word must match.
* The turn names "the X **email**" (``email_words``) — except "X is promo", the one
  form the owner uses without it.

Outcomes: one email → corrected, and the reply says what moved; already that bucket →
"nothing to change"; several → up to three to pick from; a bucket only denied ("isn't FYI") → which one?;
no email found → falls through to normal routing, unless the turn said "email" and a
bucket, when it says it found none. A question ("is this a bill?") is never a
correction. "what did you judge today?" is answered with the day's counts.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, time, timedelta, tzinfo
from typing import Any

from .judge_config import JudgeConfig, labels_enabled
from .judge_corrections import Emit, apply_correction, valid_buckets
from .judge_view import JudgedEmail, judged_emails
from .judge_words import SurfaceWords, bucket_name
from .judgments import PROMO, JudgmentStore

_WORD = re.compile(r"[a-z0-9]+")
_CLAUSE_BREAK = re.compile(r"[,;:.!]|\bbut\b|\bit\b|\bit's\b|\bits\b|\bthey\b|\s[-—–]\s")
#: A target longer than this many naming words is a sentence, not an email.
_MAX_TARGET_WORDS = 6
_CHOICES = 3
_SCAN = 500


@dataclass(frozen=True)
class RebucketTurn:
    """What the turn did: ``kind`` is changed | already | ambiguous | which_bucket |
    not_found | counts."""

    kind: str
    reply: str
    message_id: str = ""


def _phrase_pattern(phrases: Iterable[str]) -> re.Pattern[str]:
    ordered = sorted({p for p in phrases if p}, key=len, reverse=True)
    alternation = "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in ordered)
    return re.compile(rf"(?<![a-z0-9])(?:{alternation})(?![a-z0-9])")


def _normal(text: str) -> str:
    return " ".join(text.lower().replace("’", "'").split())


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _words_of(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower())}


def _email_words(item: JudgedEmail) -> set[str]:
    email = item.email
    if email is None:
        return set()
    return _words_of(f"{email.from_address} {email.from_domain or ''} {email.subject}")


@dataclass(frozen=True)
class _Parsed:
    target: tuple[str, ...]
    bucket: str | None
    not_bucket: str | None
    said_email: bool


class _Grammar:
    def __init__(self, config: JudgeConfig, words: SurfaceWords) -> None:
        chat = words.chat
        self.links = _phrase_pattern(chat.links)
        self.negation = _phrase_pattern(chat.negations)
        self.email_word = _phrase_pattern(chat.email_words)
        self.question = _phrase_pattern(chat.question_starts)
        self.judged_today = _phrase_pattern(chat.judged_today)
        self.filler = {_stem(w) for w in chat.filler_words} | {
            _stem(w) for p in chat.email_words for w in p.split()
        }
        allowed = set(valid_buckets(config))
        self.bucket_of: dict[str, str] = {}
        for key, phrases in config.bucket_words.items():
            if key in allowed:
                for phrase in phrases:
                    self.bucket_of.setdefault(_normal(phrase), key)
        self.buckets = _phrase_pattern(self.bucket_of)

    def is_question(self, text: str) -> bool:
        if text.rstrip().endswith("?"):
            return True
        first = self.question.match(text)
        return first is not None

    def _clause_buckets(self, rest: str) -> tuple[str | None, str | None]:
        """(the bucket said, the bucket denied) in the words after the link."""
        said: str | None = None
        denied: str | None = None
        start = 0
        breaks = [m.start() for m in _CLAUSE_BREAK.finditer(rest)] + [len(rest)]
        for end in breaks:
            clause = rest[start:end]
            start = end
            found = [self.bucket_of[_normal(m.group(0))] for m in self.buckets.finditer(clause)]
            if not found:
                continue
            if self.negation.search(clause):
                denied = denied or found[0]
            else:
                said = found[-1]
        return said, denied

    def parse(self, text: str) -> _Parsed | None:
        link = self.links.search(text)
        if link is None or link.start() == 0:
            return None
        head, rest = text[: link.start()], text[link.start() :]
        if self.email_word.match(head):
            return None  # "email the dentist that ..." is an instruction to write one
        target = tuple(w for w in (_stem(x) for x in _WORD.findall(head)) if w not in self.filler)
        if not target or len(target) > _MAX_TARGET_WORDS:
            return None
        said, denied = self._clause_buckets(rest)
        return _Parsed(
            target=target,
            bucket=said,
            not_bucket=denied,
            said_email=self.email_word.search(head) is not None,
        )


def _day_text(value: datetime | None, tz: tzinfo) -> str:
    if value is None:
        return ""
    local = value.astimezone(tz)
    return f"{local:%a} {local:%b} {local.day}"


def _counts_reply(
    store: JudgmentStore, config: JudgeConfig, words: SurfaceWords, now: datetime, tz: tzinfo
) -> RebucketTurn:
    start = datetime.combine(now.astimezone(tz).date(), time.min, tzinfo=tz)
    rows = store.judged_between(start, start + timedelta(days=1))
    replies = words.chat.replies
    parts = []
    for key in valid_buckets(config):
        n = sum(1 for j in rows if j.effective_bucket == key)
        if n:
            parts.append(f"{n} {bucket_name(config, words, key)}")
    if not parts:
        return RebucketTurn(kind="counts", reply=replies["counts_none"])
    return RebucketTurn(kind="counts", reply=replies["counts"].format(counts=" · ".join(parts)))


def handle_rebucket_turn(
    message: str,
    store: JudgmentStore,
    emails: Any,
    *,
    config: JudgeConfig,
    words: SurfaceWords,
    now: datetime,
    tz: tzinfo,
    emit: Emit | None = None,
) -> RebucketTurn | None:
    """Correct the email a chat turn names; ``None`` when it is not such a turn."""
    text = _normal(str(message or ""))
    if not text:
        return None
    grammar = _Grammar(config, words)
    if grammar.judged_today.search(text):
        return _counts_reply(store, config, words, now, tz)
    if grammar.is_question(text):
        return None
    parsed = grammar.parse(text.rstrip(" .!"))
    if parsed is None or (parsed.bucket is None and parsed.not_bucket is None):
        return None
    # Without "email" only "X is promo" is this intercept's: "my rent is a bill" is not.
    if not parsed.said_email and parsed.bucket != PROMO:
        return None
    replies = words.chat.replies
    since = now - timedelta(days=words.chat.window_days)
    candidates = judged_emails(store, emails, limit=_SCAN, since=since)
    wanted = set(parsed.target)
    found = [c for c in candidates if wanted <= _email_words(c)]
    if parsed.not_bucket and len(found) > 1:
        narrowed = [c for c in found if c.judgment.effective_bucket == parsed.not_bucket]
        found = narrowed or found
    if parsed.bucket and len(found) > 1:
        # "the X email is a bill" means one that is not a bill yet.
        narrowed = [c for c in found if c.judgment.effective_bucket != parsed.bucket]
        found = narrowed or found
    if not found:
        if parsed.said_email and parsed.bucket is not None:
            return RebucketTurn(
                kind="not_found",
                reply=replies["not_found"].format(
                    words=" ".join(parsed.target), days=words.chat.window_days
                ),
            )
        return None
    if len(found) > 1:
        choices = "; ".join(
            f'"{c.subject}" ({c.sender or "unknown sender"}, {_day_text(c.received_at, tz)})'
            for c in found[:_CHOICES]
        )
        return RebucketTurn(kind="ambiguous", reply=replies["ambiguous"].format(choices=choices))
    (item,) = found
    if parsed.bucket is None:
        names = [bucket_name(config, words, k) for k in valid_buckets(config) if k != "unsure"]
        listed = ", ".join(names[:-1]) + f" or {names[-1]}" if len(names) > 1 else names[0]
        return RebucketTurn(
            kind="which_bucket",
            reply=replies["which_bucket"].format(
                subject=item.subject, sender=item.sender, buckets=listed
            ),
            message_id=item.message_id,
        )
    correction = apply_correction(
        store, config, item.message_id, parsed.bucket, source="chat", emit=emit
    )
    if correction is None:  # the row left between the read and the write
        return None
    name = bucket_name(config, words, parsed.bucket)
    if not correction.changed:
        return RebucketTurn(
            kind="already",
            reply=replies["already"].format(subject=item.subject, bucket=name),
            message_id=item.message_id,
        )
    fields = {
        "subject": item.subject,
        "sender": item.sender or "them",
        "bucket": name,
        "previous": bucket_name(config, words, correction.previous) or "unjudged",
    }
    if parsed.bucket == PROMO:
        reply = replies["changed_promo"].format(**fields)
    elif labels_enabled():
        reply = replies["changed"].format(label=config.bucket(parsed.bucket).label, **fields)
    else:
        reply = replies["changed_no_labels"].format(**fields)
    return RebucketTurn(kind="changed", reply=reply, message_id=item.message_id)


__all__ = ["RebucketTurn", "handle_rebucket_turn"]
