"""Schedules in words: validate, describe, and read what a person typed."""

from __future__ import annotations

import pytest

from iris_harness.services.heartbeat.schedule_text import (
    MIN_INTERVAL_SECONDS,
    describe_schedule,
    parse_schedule_words,
    validate_schedule,
)


@pytest.mark.parametrize(
    ("raw", "normal"),
    [
        ("interval:600", "interval:600"),
        (" interval: 60 ", "interval:60"),
        ("0 8 * * *", "0 8 * * *"),
        ("0  8 *  * 1-5", "0 8 * * 1-5"),
    ],
)
def test_valid_schedules_come_back_normalised(raw: str, normal: str) -> None:
    assert validate_schedule(raw) == normal


@pytest.mark.parametrize(
    "bad",
    [
        f"interval:{MIN_INTERVAL_SECONDS - 1}",
        "interval:ten",
        "interval:",
        "0 25 * * *",
        "0 8 * *",
        "every day",
        "",
    ],
)
def test_invalid_schedules_are_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_schedule(bad)


@pytest.mark.parametrize(
    ("schedule", "words"),
    [
        ("interval:30", "every 30 s"),
        ("interval:60", "every minute"),
        ("interval:600", "every 10 min"),
        ("interval:3600", "every hour"),
        ("interval:21600", "every 6 hours"),
        ("interval:86400", "every day"),
        ("interval:172800", "every 2 days"),
        ("0 8 * * *", "daily at 08:00"),
        ("45 3 * * *", "daily at 03:45"),
        ("0 8 * * 1-5", "cron 0 8 * * 1-5"),
    ],
)
def test_describe_uses_plain_words(schedule: str, words: str) -> None:
    assert describe_schedule(schedule) == words


@pytest.mark.parametrize(
    ("typed", "schedule"),
    [
        (["every", "10m"], "interval:600"),
        (["every", "10", "min"], "interval:600"),
        (["every", "hour"], "interval:3600"),
        (["every", "2h"], "interval:7200"),
        (["every", "30s"], "interval:30"),
        (["daily", "08:30"], "30 8 * * *"),
        (["daily", "7:05"], "5 7 * * *"),
        (["cron", "0", "8", "*", "*", "1-5"], "0 8 * * 1-5"),
    ],
)
def test_typed_words_become_a_schedule(typed: list[str], schedule: str) -> None:
    assert parse_schedule_words(typed) == schedule


@pytest.mark.parametrize(
    "typed",
    [[], ["every"], ["every", "10"], ["every", "5s"], ["daily", "25:00"], ["weekly", "mon"]],
)
def test_unreadable_words_are_refused(typed: list[str]) -> None:
    with pytest.raises(ValueError):
        parse_schedule_words(typed)


def test_every_described_schedule_parses_back_to_itself() -> None:
    """The words the app shows are words the REPL accepts."""
    for schedule in ("interval:600", "interval:3600", "interval:86400", "30 8 * * *"):
        words = describe_schedule(schedule).replace("daily at", "daily").split()
        assert parse_schedule_words(words) == schedule
