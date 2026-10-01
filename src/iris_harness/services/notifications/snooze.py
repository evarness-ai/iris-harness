"""Snooze and Done in the owner's words (loop-proof D14, PR 3b).

Every surface snoozes a reminder with one of three fixed choices — ``10m``, ``1h``,
``tomorrow_9am`` (the Telegram buttons, the push actions, the sheet) — or with a typed
reply to the reminder ("snooze 1h", "an hour", "tomorrow at 6pm", "at 18:00"). Both
end here: :func:`parse_snooze` turns either into the aware UTC instant the reminder
fires again, or ``None`` when it does not understand.

**No English word lives in this file.** The fixed choices are a contract (button
payloads), not vocabulary; every word a person types — units, "tomorrow", "at",
am/pm, the leading "snooze for", the words that mean done — comes from the
``snooze:`` section of ``config/notifications.yaml``. The code is grammar only: a
number and a unit, a day and a clock time, a bare clock time.

Rules:

* ``<n> <unit>`` / ``<one-word> <unit>`` (``10m``, ``10 min``, ``an hour``,
  ``2 hours``) — that long from ``now``;
* ``<tomorrow> [<at>] [<time>]`` — tomorrow, local, at the time or at
  ``tomorrow_default`` (09:00);
* ``[<at>] <time>`` — today at that local time, or tomorrow when it has passed. An
  hour written without am/pm or minutes past 12 (``at 6``) is the sooner of 6:00 and
  18:00 still ahead.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import Any

import yaml

from iris_harness.foundation.paths import config_dir as resolve_config_dir
from iris_harness.foundation.process_state import track_globals

logger = logging.getLogger(__name__)

#: The fixed choices every surface offers (button payloads — a contract, not words).
SNOOZE_CHOICES = ("10m", "1h", "tomorrow_9am")

_CHOICE_DELTAS = {"10m": timedelta(minutes=10), "1h": timedelta(hours=1)}
_CHOICE_TOMORROW = {"tomorrow_9am": time(9, 0)}

_DEFAULT_TOMORROW = time(9, 0)


# ── vocabulary ───────────────────────────────────────────────────────────────


def _phrase(text: str) -> str:
    return r"\s*".join(re.escape(part) for part in str(text).split())


def _alternation(phrases: Iterable[str]) -> str:
    """Longest first, so "snooze for" wins over "snooze"; never matches nothing."""
    ordered = sorted({str(p).strip().lower() for p in phrases if str(p).strip()}, key=len)
    ordered.reverse()
    return "|".join(_phrase(p) for p in ordered) or r"(?!x)x"


@dataclass(frozen=True)
class SnoozeVocabulary:
    """Compiled patterns built from the ``snooze:`` words."""

    tomorrow_default: time
    lead: re.Pattern[str]
    duration: re.Pattern[str]
    tomorrow: re.Pattern[str]
    clock: re.Pattern[str]
    done: frozenset[str]
    #: Replies that close a bill's reminder as paid (PR 4) — a Done, worded "paid".
    paid: frozenset[str] = frozenset()
    #: Replies that answer a bill's "Did you pay?" with Not yet (PR 4).
    not_yet: frozenset[str] = frozenset()

    @classmethod
    def from_mapping(cls, section: dict[str, Any]) -> SnoozeVocabulary:
        words = section.get("words") or {}

        def alt(key: str) -> str:
            raw = words.get(key) or []
            return _alternation(raw if isinstance(raw, list) else [raw])

        when = _time_of(section.get("tomorrow_default")) or _DEFAULT_TOMORROW
        clock_time = (
            r"(?P<hour>\d{1,2})(?:[:.](?P<minute>[0-5]\d))?\s*"
            rf"(?:(?P<am>{alt('am')})|(?P<pm>{alt('pm')}))?"
        )
        done_raw = section.get("done") or []

        def phrases(key: str) -> frozenset[str]:
            raw = section.get(key) or []
            return frozenset(_norm(str(w)) for w in raw if str(w).strip())

        return cls(
            tomorrow_default=when,
            lead=re.compile(rf"^(?:(?:{alt('lead')})\b\s*)+", re.IGNORECASE),
            duration=re.compile(
                rf"^(?:(?P<count>\d{{1,4}})|(?P<one>{alt('one')}))\s*"
                rf"(?:(?P<minutes>{alt('minutes')})|(?P<hours>{alt('hours')})"
                rf"|(?P<days>{alt('days')}))$",
                re.IGNORECASE,
            ),
            tomorrow=re.compile(
                rf"^(?:{alt('tomorrow')})(?:\s*(?:(?:{alt('at')})\s*)?(?P<time>.+))?$",
                re.IGNORECASE,
            ),
            clock=re.compile(rf"^(?:(?P<at>{alt('at')})\s*)?{clock_time}$", re.IGNORECASE),
            done=frozenset(_norm(str(w)) for w in done_raw if str(w).strip()),
            paid=phrases("paid"),
            not_yet=phrases("not_yet"),
        )


def _time_of(raw: Any) -> time | None:
    match = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", str(raw or "").strip())
    return time(int(match.group(1)), int(match.group(2))) if match else None


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _config_path(config_dir: Path | None) -> Path:
    base = config_dir or resolve_config_dir()
    path = base / "notifications.yaml"
    if path.exists() or config_dir is not None:
        return path
    from iris_harness.foundation.paths import default_config_dir

    return default_config_dir() / "notifications.yaml"


_cache: dict[tuple[str, float], SnoozeVocabulary] = {}


def load_vocabulary(config_dir: Path | None = None) -> SnoozeVocabulary:
    """The ``snooze:`` words from ``notifications.yaml`` (re-read when the file
    changes). A missing or broken file gives an empty vocabulary: the fixed choices
    still work, typed words do not."""
    path = _config_path(config_dir)
    try:
        stamp = path.stat().st_mtime
    except OSError:
        stamp = -1.0
    key = (str(path), stamp)
    if key in _cache:
        return _cache[key]
    section: dict[str, Any] = {}
    try:
        if stamp >= 0:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            found = raw.get("snooze") if isinstance(raw, dict) else None
            section = found if isinstance(found, dict) else {}
    except Exception:  # a bad file must not break Done / Snooze buttons
        logger.warning("failed to read snooze words from %s", path, exc_info=True)
    vocab = SnoozeVocabulary.from_mapping(section)
    _cache.clear()
    _cache[key] = vocab
    return vocab


# ── parsing ──────────────────────────────────────────────────────────────────


def _aware(now: datetime) -> datetime:
    return now if now.tzinfo is not None else now.replace(tzinfo=UTC)


def _at_local(day: date, when: time, tz: tzinfo) -> datetime:
    return datetime.combine(day, when, tzinfo=tz).astimezone(UTC)


def _clock(match: re.Match[str]) -> tuple[time, bool] | None:
    """(time, ambiguous) from a clock match; ambiguous = could be am or pm."""
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    am, pm = match.group("am"), match.group("pm")
    if am or pm:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if pm else 0)
        return time(hour, minute), False
    if hour > 23:
        return None
    return time(hour, minute), 1 <= hour <= 11


def parse_snooze(
    text: str, now: datetime, tz: tzinfo, *, vocabulary: SnoozeVocabulary | None = None
) -> datetime | None:
    """When a snooze of ``text`` fires again, as an aware UTC datetime; ``None`` when
    ``text`` is not understood (or does not land in the future)."""
    now = _aware(now)
    cleaned = _norm(text or "")
    if not cleaned:
        return None
    if cleaned in _CHOICE_DELTAS:
        return now.astimezone(UTC) + _CHOICE_DELTAS[cleaned]
    local = now.astimezone(tz)
    if cleaned in _CHOICE_TOMORROW:
        return _at_local(local.date() + timedelta(days=1), _CHOICE_TOMORROW[cleaned], tz)

    vocab = vocabulary or load_vocabulary()
    cleaned = cleaned.replace("_", " ")
    cleaned = vocab.lead.sub("", cleaned).strip()
    if not cleaned:
        return None

    duration = vocab.duration.match(cleaned)
    if duration is not None:
        count = int(duration.group("count")) if duration.group("count") else 1
        if count <= 0:
            return None
        if duration.group("minutes"):
            delta = timedelta(minutes=count)
        elif duration.group("hours"):
            delta = timedelta(hours=count)
        else:
            delta = timedelta(days=count)
        return now.astimezone(UTC) + delta

    tomorrow = vocab.tomorrow.match(cleaned)
    if tomorrow is not None:
        day = local.date() + timedelta(days=1)
        spoken = (tomorrow.group("time") or "").strip()
        if not spoken:
            return _at_local(day, vocab.tomorrow_default, tz)
        clock = vocab.clock.match(spoken)
        parsed = _clock(clock) if clock is not None else None
        return _at_local(day, parsed[0], tz) if parsed is not None else None

    clock = vocab.clock.match(cleaned)
    if clock is None:
        return None
    # A bare number is a clock time only when it says so ("at 6", "6pm", "6:30").
    if not (clock.group("at") or clock.group("minute") or clock.group("am") or clock.group("pm")):
        return None
    parsed = _clock(clock)
    if parsed is None:
        return None
    when, ambiguous = parsed
    candidates = [when]
    if ambiguous:
        candidates.append(time(when.hour + 12, when.minute))
    today, tomorrow_day = local.date(), local.date() + timedelta(days=1)
    for day in (today, tomorrow_day):
        for candidate in candidates:
            at = _at_local(day, candidate, tz)
            if at > now:
                return at
    return None


def _said(text: str, words: frozenset[str]) -> bool:
    cleaned = _norm(text or "")
    return cleaned.rstrip(".!") in words or cleaned in words


def is_done(text: str, *, vocabulary: SnoozeVocabulary | None = None) -> bool:
    """Does a reply mean "done"? (Exact phrase from the ``done:`` or ``paid:`` words.)"""
    vocab = vocabulary or load_vocabulary()
    return _said(text, vocab.done) or _said(text, vocab.paid)


def is_paid(text: str, *, vocabulary: SnoozeVocabulary | None = None) -> bool:
    """Does a reply say "paid" (a Done worded for a bill)? (``paid:`` words.)"""
    vocab = vocabulary or load_vocabulary()
    return _said(text, vocab.paid)


def is_not_yet(text: str, *, vocabulary: SnoozeVocabulary | None = None) -> bool:
    """Does a reply answer a bill's question with Not yet? (``not_yet:`` words.)"""
    vocab = vocabulary or load_vocabulary()
    return _said(text, vocab.not_yet)


@dataclass(frozen=True)
class ReplyAction:
    """What a typed reply to a reminder asks for."""

    action: str  # "done" | "snooze" | "not_yet"
    until: datetime | None = None
    text: str = ""
    paid: bool = False  # a Done said as "paid" (a bill's reminder)


def parse_reply(
    text: str, now: datetime, tz: tzinfo, *, vocabulary: SnoozeVocabulary | None = None
) -> ReplyAction | None:
    """``done`` or a snooze (with its instant); ``None`` for anything else — that reply
    is ordinary chat."""
    vocab = vocabulary or load_vocabulary()
    if is_done(text, vocabulary=vocab):
        return ReplyAction("done", text=text, paid=is_paid(text, vocabulary=vocab))
    if is_not_yet(text, vocabulary=vocab):
        return ReplyAction("not_yet", text=text)
    until = parse_snooze(text, now, tz, vocabulary=vocab)
    if until is not None:
        return ReplyAction("snooze", until=until, text=text)
    return None


__all__ = [
    "SNOOZE_CHOICES",
    "ReplyAction",
    "SnoozeVocabulary",
    "is_done",
    "is_not_yet",
    "is_paid",
    "load_vocabulary",
    "parse_reply",
    "parse_snooze",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_cache")
