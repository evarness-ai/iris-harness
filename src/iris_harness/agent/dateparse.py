"""Shared natural-language relative-date parsing.

One home for "tomorrow / next friday / June 25 / 2026-07-01" handling so the
planner (`parse_target_day`) and the runtime reminder/meeting intercepts
(`_extract_reminder_date`) never drift. Recognises (in order): day after
tomorrow, tomorrow, today/tonight, weekday names, an ISO date, a month-name
date. Returns ``None`` when nothing matches — callers decide the default.

The compiled regexes are exported so callers can also *strip* these tokens
from a title (e.g. so "dentist appointment friday" → "dentist appointment").
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

ISO_DATE_RE = re.compile(r"\b(?P<date>\d{4}-\d{2}-\d{2})\b")
MONTH_DATE_RE = re.compile(
    r"\b(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+(?P<day>\d{1,2})(?:,?\s+(?P<year>\d{4}))?\b",
    re.IGNORECASE,
)
WEEKDAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE
)
#: Any one date phrase ``parse_relative_date`` reads, as a single span — so a caller
#: can cut the phrase out of a sentence (a reminder's "until Dec 31") and resolve
#: exactly that text. Same precedence as the parser: relative days first.
DATE_PHRASE_RE = re.compile(
    r"\b(?:day\s+after\s+tomorrow|tomorrow|today|tonight)\b"
    rf"|(?:\bnext\s+)?{WEEKDAY_RE.pattern}"
    rf"|{ISO_DATE_RE.pattern}"
    rf"|{MONTH_DATE_RE.pattern}",
    re.IGNORECASE,
)
#: The weekday names, Monday first — index = ``date.weekday()``.
WEEKDAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def parse_relative_date(
    text: str, *, now: datetime, weekday_today_means_next: bool = False
) -> date | None:
    """Resolve a relative/absolute date phrase in ``text`` to a ``date``.

    ``weekday_today_means_next``: when a bare weekday names *today*, return that
    day next week instead of today. Use this for *scheduling* ("schedule it
    friday" on a Friday → next Friday); leave False for *viewing* ("plan my
    friday" → today). Explicit "next <weekday>" always pushes a week.
    """
    lowered = text.lower()
    today = now.date()

    if re.search(r"\bday after tomorrow\b", lowered):
        return today + timedelta(days=2)
    if re.search(r"\btomorrow\b", lowered):
        return today + timedelta(days=1)
    if re.search(r"\b(?:today|tonight)\b", lowered):
        return today

    weekday = WEEKDAY_RE.search(lowered)
    if weekday:
        idx = _WEEKDAYS[weekday.group(1).lower()]
        delta = (idx - today.weekday()) % 7
        if re.search(r"\bnext\b", lowered):
            delta += 7
        elif delta == 0 and weekday_today_means_next:
            delta = 7
        return today + timedelta(days=delta)

    iso = ISO_DATE_RE.search(text)
    if iso:
        try:
            return date.fromisoformat(iso.group("date"))
        except ValueError:
            pass

    month_match = MONTH_DATE_RE.search(text)
    if month_match:
        month = _MONTHS[month_match.group("month").lower()]
        day = int(month_match.group("day"))
        year = int(month_match.group("year") or today.year)
        candidate = date(year, month, day)
        if month_match.group("year") is None and candidate < today:
            candidate = date(year + 1, month, day)
        return candidate

    return None


__all__ = [
    "DATE_PHRASE_RE",
    "ISO_DATE_RE",
    "MONTH_DATE_RE",
    "WEEKDAY_NAMES",
    "WEEKDAY_RE",
    "parse_relative_date",
]


def resolved_dates_line(message: str, *, now: datetime) -> str:
    """One prompt line binding each date phrase in ``message`` to its date, or ``""``.

    ``"Friday" = Friday, October 2, 2026; "next Wednesday" = Wednesday, October 7, 2026``.
    Small models do calendar arithmetic badly: in the 2026-09-27 eval qwen3.5:4b said
    "Friday, October 3" and "next Wednesday is October 1st" even with the week's dates
    listed in the prompt; with this line it stated the right date 17 of 18 times.
    The same parser (and the same scheduling rule, ``weekday_today_means_next``) that
    the reminder tool uses, so what the model says and what the tool writes agree.
    """
    naive = now.replace(tzinfo=None)
    bound: list[str] = []
    for match in DATE_PHRASE_RE.finditer(message or ""):
        phrase = match.group(0)
        day = parse_relative_date(phrase, now=naive, weekday_today_means_next=True)
        if day is None:
            continue
        entry = f'"{phrase}" = {day:%A, %B} {day.day}, {day.year}'
        if entry not in bound:
            bound.append(entry)
    if not bound:
        return ""
    return (
        "Dates in the user's message (already worked out; use these exactly): "
        + "; ".join(bound)
        + "\n"
    )
