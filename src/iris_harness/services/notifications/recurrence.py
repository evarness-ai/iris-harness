"""Reminder recurrence — the next occurrence of a repeating reminder (D14).

A repeating reminder keeps its *local wall time* in the owner's zone (``IRIS_TZ``):
"every Monday 8am" stays 8:00 in Chicago across a daylight-saving change, so its UTC
instant moves by an hour twice a year. Rules:

- ``daily``    — the next day.
- ``weekdays`` — the next Monday–Friday.
- ``weekly``   — seven days on.
- ``monthly``  — the same day next month, clamped to the month's last day (the 31st
  lands on Feb 28/29, Apr 30 …) and back to the 31st when the month has one.
- ``yearly``   — the same date next year; Feb 29 lands on Feb 28 in a common year.

``anchor`` is the series' first local date: monthly and yearly clamp from it, so a
series that started on the 31st does not drift to the 28th for good after February.
Every instant returned is UTC.
"""

from __future__ import annotations

import calendar
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from typing import Literal, get_args

RecurrenceRule = Literal["daily", "weekdays", "weekly", "monthly", "yearly"]
RULES: tuple[str, ...] = get_args(RecurrenceRule)

_WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def validate_rule(rule: str | None) -> str | None:
    """``rule`` normalised (lower-case), or ``None``; raises on an unknown rule."""
    if rule is None:
        return None
    cleaned = rule.strip().lower()
    if not cleaned:
        return None
    if cleaned not in RULES:
        raise ValueError(f"unknown recurrence {rule!r}; expected one of {', '.join(RULES)}")
    return cleaned


def _clamped(year: int, month: int, day: int) -> date:
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def _next_date(rule: str, current: date, anchor: date) -> date:
    if rule == "daily":
        return current + timedelta(days=1)
    if rule == "weekdays":
        nxt = current + timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += timedelta(days=1)
        return nxt
    if rule == "weekly":
        return current + timedelta(days=7)
    if rule == "monthly":
        year, month = (
            (current.year + 1, 1) if current.month == 12 else (current.year, current.month + 1)
        )
        return _clamped(year, month, anchor.day)
    if rule == "yearly":
        return _clamped(current.year + 1, anchor.month, anchor.day)
    raise ValueError(f"unknown recurrence {rule!r}")


def _local(dt: datetime, tz: tzinfo) -> datetime:
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    return aware.astimezone(tz)


def next_occurrence(
    rule: str,
    after: datetime,
    tz: tzinfo,
    *,
    until: datetime | None = None,
    anchor: date | None = None,
) -> datetime | None:
    """The occurrence following ``after`` (UTC), or ``None`` past ``until``.

    ``after`` is the current occurrence; its local wall time in ``tz`` is kept.
    """
    checked = validate_rule(rule)
    if checked is None:
        return None
    local = _local(after, tz)
    wall = time(local.hour, local.minute, local.second)
    nxt_date = _next_date(checked, local.date(), anchor or local.date())
    # zoneinfo resolves a wall time that does not exist (the spring-forward hour) to
    # the instant an hour on, and an ambiguous one (fall-back) to its first reading.
    candidate = datetime.combine(nxt_date, wall, tzinfo=tz).astimezone(UTC)
    if until is not None and candidate > _local(until, UTC):
        return None
    return candidate


def next_occurrence_after(
    rule: str,
    current: datetime,
    tz: tzinfo,
    *,
    not_before: datetime,
    until: datetime | None = None,
    anchor: date | None = None,
) -> datetime | None:
    """Step the series from ``current`` to its first occurrence after ``not_before``.

    A reminder that was stuck (the host was off for days) must not fire a burst of
    catch-up occurrences: the series resumes at its next future slot.
    """
    candidate: datetime | None = current
    for _ in range(3700):  # ten years of daily steps is plenty
        assert candidate is not None
        candidate = next_occurrence(rule, candidate, tz, until=until, anchor=anchor)
        if candidate is None or candidate > not_before:
            return candidate
    return None


def describe(rule: str | None, at: datetime, tz: tzinfo) -> str:
    """How the rule reads to the owner: "every Monday", "every month on the 31st"."""
    checked = validate_rule(rule)
    if checked is None:
        return ""
    local = _local(at, tz)
    if checked == "daily":
        return "every day"
    if checked == "weekdays":
        return "every weekday"
    if checked == "weekly":
        return f"every {_WEEKDAY_NAMES[local.weekday()]}"
    if checked == "monthly":
        return f"every month on the {_ordinal(local.day)}"
    return f"every year on {local:%b} {local.day}"


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


__all__ = [
    "RULES",
    "RecurrenceRule",
    "describe",
    "next_occurrence",
    "next_occurrence_after",
    "validate_rule",
]
