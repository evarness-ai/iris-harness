"""The prompt carries the user's day words already resolved (PR B, 2026-09-27 eval).

qwen3.5:4b said "Friday, October 3" / "next Wednesday is October 1st" when doing the
arithmetic itself, even with the week's dates listed; given the resolved dates it was
right 17 of 18 times. Same parser and scheduling rule as the reminder tool.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from iris_harness.agent.dateparse import resolved_dates_line

SUNDAY = datetime(2026, 9, 27, 15, 0, tzinfo=ZoneInfo("America/Chicago"))


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Add a task to renew my passport by Friday", '"Friday" = Friday, October 2, 2026'),
        ("What date is next Wednesday?", '"next Wednesday" = Wednesday, October 7, 2026'),
        ("remind me tomorrow", '"tomorrow" = Monday, September 28, 2026'),
    ],
)
def test_day_words_are_resolved_like_the_reminder_tool(message: str, expected: str) -> None:
    line = resolved_dates_line(message, now=SUNDAY)
    assert expected in line
    assert line.startswith("Dates in the user's message") and line.endswith("\n")


def test_a_weekday_that_is_today_means_next_week_when_scheduling() -> None:
    friday = datetime(2026, 10, 2, 9, 0, tzinfo=ZoneInfo("America/Chicago"))
    assert '"Friday" = Friday, October 9, 2026' in resolved_dates_line("call on Friday", now=friday)


def test_no_date_words_no_line() -> None:
    assert resolved_dates_line("plan my day", now=SUNDAY) == ""
    assert resolved_dates_line("", now=SUNDAY) == ""


def test_a_repeated_phrase_is_listed_once() -> None:
    line = resolved_dates_line("Friday, or maybe Friday", now=SUNDAY)
    assert line.count('"Friday"') == 1


def test_both_prompts_carry_it_only_when_the_message_has_a_date() -> None:
    from iris_harness.agent.agentic_core import _clock_line
    from iris_harness.runtime.handlers.general_invoke import _general_system_prompt

    assert "Dates in the user's message" in _clock_line("book it for Friday")
    assert "Dates in the user's message" not in _clock_line("plan my day")
    assert "Dates in the user's message" in _general_system_prompt("I am IRIS.", message="Friday?")
    assert "Dates in the user's message" in _general_system_prompt("", message="Friday?")
    assert "Dates in the user's message" not in _general_system_prompt("I am IRIS.", message="hi")
