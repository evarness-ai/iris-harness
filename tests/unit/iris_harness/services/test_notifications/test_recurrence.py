"""Reminder recurrence (loop-proof D14): local wall time kept across DST, weekdays skip
the weekend, monthly clamps and recovers, yearly Feb 29 lands on Feb 28."""

from __future__ import annotations

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pytest

from iris_harness.services.notifications.recurrence import (
    describe,
    next_occurrence,
    next_occurrence_after,
    validate_rule,
)

CT = ZoneInfo("America/Chicago")


def _ct(*args: int) -> datetime:
    return datetime(*args, tzinfo=CT).astimezone(UTC)


def test_daily_is_the_next_day_same_local_time() -> None:
    nxt = next_occurrence("daily", _ct(2026, 9, 28, 8, 0), CT)
    assert nxt == _ct(2026, 9, 29, 8, 0)
    assert nxt is not None and nxt.tzinfo == UTC


def test_weekly_keeps_8am_across_the_fall_back() -> None:
    # 2026-11-01 is the US fall-back: 8:00 CDT is 13:00Z, 8:00 CST is 14:00Z.
    before = _ct(2026, 10, 26, 8, 0)
    nxt = next_occurrence("weekly", before, CT)
    assert before.hour == 13
    assert nxt is not None and nxt.hour == 14
    assert nxt.astimezone(CT).hour == 8 and nxt.astimezone(CT).weekday() == 0


def test_daily_keeps_local_time_across_spring_forward() -> None:
    nxt = next_occurrence("daily", _ct(2026, 3, 7, 9, 30), CT)  # DST starts Mar 8
    assert nxt is not None and nxt.astimezone(CT).hour == 9
    assert nxt.astimezone(CT).minute == 30


def test_weekdays_skip_the_weekend() -> None:
    friday = _ct(2026, 9, 25, 7, 0)
    nxt = next_occurrence("weekdays", friday, CT)
    assert nxt == _ct(2026, 9, 28, 7, 0)  # Monday
    assert next_occurrence("weekdays", _ct(2026, 9, 26, 7, 0), CT) == nxt  # from Saturday


def test_monthly_clamps_the_31st_and_recovers() -> None:
    jan31 = _ct(2027, 1, 31, 9, 0)
    feb = next_occurrence("monthly", jan31, CT, anchor=date(2027, 1, 31))
    assert feb == _ct(2027, 2, 28, 9, 0)
    mar = next_occurrence("monthly", feb, CT, anchor=date(2027, 1, 31))  # type: ignore[arg-type]
    assert mar == _ct(2027, 3, 31, 9, 0)
    apr = next_occurrence("monthly", mar, CT, anchor=date(2027, 1, 31))  # type: ignore[arg-type]
    assert apr == _ct(2027, 4, 30, 9, 0)


def test_monthly_rolls_over_the_year() -> None:
    assert next_occurrence("monthly", _ct(2026, 12, 15, 9, 0), CT) == _ct(2027, 1, 15, 9, 0)


def test_yearly_feb_29_lands_on_feb_28_then_back() -> None:
    leap = _ct(2028, 2, 29, 10, 0)
    nxt = next_occurrence("yearly", leap, CT, anchor=date(2028, 2, 29))
    assert nxt == _ct(2029, 2, 28, 10, 0)
    later = nxt
    for _ in range(3):
        later = next_occurrence("yearly", later, CT, anchor=date(2028, 2, 29))  # type: ignore[arg-type]
    assert later == _ct(2032, 2, 29, 10, 0)


def test_until_ends_the_series() -> None:
    start = _ct(2026, 9, 28, 8, 0)
    assert next_occurrence("daily", start, CT, until=_ct(2026, 9, 29, 8, 0)) is not None
    assert next_occurrence("daily", start, CT, until=_ct(2026, 9, 29, 7, 59)) is None


def test_a_stuck_series_resumes_in_the_future() -> None:
    start = _ct(2026, 9, 1, 8, 0)
    nxt = next_occurrence_after("daily", start, CT, not_before=_ct(2026, 9, 25, 12, 0))
    assert nxt == _ct(2026, 9, 26, 8, 0)


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("daily", "every day"),
        ("weekdays", "every weekday"),
        ("weekly", "every Monday"),
        ("monthly", "every month on the 28th"),
        ("yearly", "every year on Sep 28"),
    ],
)
def test_describe(rule: str, expected: str) -> None:
    assert describe(rule, _ct(2026, 9, 28, 8, 0), CT) == expected


def test_unknown_rule_is_refused() -> None:
    assert validate_rule(" Weekly ") == "weekly"
    assert validate_rule(None) is None
    with pytest.raises(ValueError, match="unknown recurrence"):
        validate_rule("fortnightly")
