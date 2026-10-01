"""Shared relative-date parser used by the planner + reminder/meeting intercepts."""

from __future__ import annotations

from datetime import UTC, date, datetime

from iris_harness.agent.dateparse import parse_relative_date

NOW = datetime(2026, 6, 23, 9, 0, tzinfo=UTC)  # a Tuesday


def test_today_tomorrow_day_after() -> None:
    assert parse_relative_date("plan today", now=NOW) == date(2026, 6, 23)
    assert parse_relative_date("tonight please", now=NOW) == date(2026, 6, 23)
    assert parse_relative_date("do it tomorrow", now=NOW) == date(2026, 6, 24)
    assert parse_relative_date("the day after tomorrow", now=NOW) == date(2026, 6, 25)


def test_iso_and_month_dates() -> None:
    assert parse_relative_date("on 2026-07-01", now=NOW) == date(2026, 7, 1)
    assert parse_relative_date("July 4", now=NOW) == date(2026, 7, 4)
    # a month/day already past this year rolls to next year
    assert parse_relative_date("Jan 5", now=NOW) == date(2027, 1, 5)


def test_weekday_viewing_vs_scheduling() -> None:
    # bare future weekday is the same either way
    assert parse_relative_date("friday", now=NOW) == date(2026, 6, 26)
    # naming TODAY's weekday: viewing → today; scheduling → next week
    assert parse_relative_date("tuesday", now=NOW) == date(2026, 6, 23)
    assert parse_relative_date("tuesday", now=NOW, weekday_today_means_next=True) == date(
        2026, 6, 30
    )
    # "next <weekday>" always pushes a week
    assert parse_relative_date("next monday", now=NOW) == date(2026, 7, 6)


def test_none_when_no_date_phrase() -> None:
    assert parse_relative_date("plan my day", now=NOW) is None
    assert parse_relative_date("what do I have", now=NOW) is None
