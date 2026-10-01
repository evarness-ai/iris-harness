"""The snooze grammar (loop-proof D14, PR 3b): fixed choices and the owner's words.

``now`` is Mon Sep 28 2026, 8:03 AM in Chicago (13:03Z). Every word comes from the
shipped ``config/notifications.yaml``; one test swaps in its own vocabulary to prove
the words are read from the file, not from the code.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from iris_harness.foundation.paths import repo_root
from iris_harness.services.notifications.snooze import (
    SNOOZE_CHOICES,
    is_done,
    load_vocabulary,
    parse_reply,
    parse_snooze,
)

CT = ZoneInfo("America/Chicago")
NOW = datetime(2026, 9, 28, 13, 3, tzinfo=UTC)  # Mon 8:03 AM CDT
SHIPPED = repo_root() / "config"


def _local(y: int, mo: int, d: int, h: int, mi: int = 0) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=CT).astimezone(UTC)


@pytest.fixture(autouse=True)
def _shipped_words(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_CONFIG_DIR", str(SHIPPED))


def test_the_three_choices() -> None:
    assert SNOOZE_CHOICES == ("10m", "1h", "tomorrow_9am")
    assert parse_snooze("10m", NOW, CT) == NOW + timedelta(minutes=10)
    assert parse_snooze("1h", NOW, CT) == NOW + timedelta(hours=1)
    assert parse_snooze("tomorrow_9am", NOW, CT) == _local(2026, 9, 29, 9)


def test_choices_work_without_any_vocabulary(tmp_path: Path) -> None:
    empty = load_vocabulary(tmp_path)  # no notifications.yaml there
    assert parse_snooze("1h", NOW, CT, vocabulary=empty) == NOW + timedelta(hours=1)
    assert parse_snooze("tomorrow_9am", NOW, CT, vocabulary=empty) == _local(2026, 9, 29, 9)
    assert parse_snooze("an hour", NOW, CT, vocabulary=empty) is None


@pytest.mark.parametrize(
    ("text", "delta"),
    [
        ("10 min", timedelta(minutes=10)),
        ("10 minutes", timedelta(minutes=10)),
        ("an hour", timedelta(hours=1)),
        ("2 hours", timedelta(hours=2)),
        ("snooze 1h", timedelta(hours=1)),
        ("Snooze for 30 mins", timedelta(minutes=30)),
        ("remind me in 2 hrs", timedelta(hours=2)),
        ("in a day", timedelta(days=1)),
    ],
)
def test_durations(text: str, delta: timedelta) -> None:
    assert parse_snooze(text, NOW, CT) == NOW + delta


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tomorrow", _local(2026, 9, 29, 9)),
        ("tomorrow 9am", _local(2026, 9, 29, 9)),
        ("tomorrow at 6pm", _local(2026, 9, 29, 18)),
        ("snooze until tomorrow 7:30 am", _local(2026, 9, 29, 7, 30)),
        ("tomorrow 18:00", _local(2026, 9, 29, 18)),
        ("6pm", _local(2026, 9, 28, 18)),
        ("at 18:00", _local(2026, 9, 28, 18)),
        ("at 6", _local(2026, 9, 28, 18)),  # 6:00 has passed: the 18:00 still ahead
        ("at 9", _local(2026, 9, 28, 9)),
        ("7am", _local(2026, 9, 29, 7)),  # already past today: tomorrow
        ("until 12pm", _local(2026, 9, 28, 12)),
        ("12am", _local(2026, 9, 29, 0)),
    ],
)
def test_days_and_clock_times(text: str, expected: datetime) -> None:
    assert parse_snooze(text, NOW, CT) == expected


@pytest.mark.parametrize(
    "text", ["", "later", "whenever", "10", "at 25:00", "13pm", "0 min", "next year", "hello"]
)
def test_not_understood(text: str) -> None:
    assert parse_snooze(text, NOW, CT) is None


def test_tomorrow_default_and_words_come_from_the_file(tmp_path: Path) -> None:
    (tmp_path / "notifications.yaml").write_text(
        "snooze:\n"
        "  tomorrow_default: '07:15'\n"
        "  words:\n"
        "    lead: [aplazar]\n"
        "    one: [una]\n"
        "    hours: [hora, horas]\n"
        "    tomorrow: [manana]\n"
        "    at: [a las]\n"
        "  done: [hecho]\n",
        encoding="utf-8",
    )
    words = load_vocabulary(tmp_path)
    assert parse_snooze("manana", NOW, CT, vocabulary=words) == _local(2026, 9, 29, 7, 15)
    assert parse_snooze("aplazar una hora", NOW, CT, vocabulary=words) == NOW + timedelta(hours=1)
    assert parse_snooze("an hour", NOW, CT, vocabulary=words) is None
    assert is_done("Hecho", vocabulary=words) and not is_done("done", vocabulary=words)


def test_a_naive_now_is_utc() -> None:
    assert parse_snooze("1h", NOW.replace(tzinfo=None), CT) == NOW + timedelta(hours=1)


def test_done_and_reply_parsing() -> None:
    assert is_done("done") and is_done("Done!") and is_done("  mark   done ") and is_done("✅")
    assert not is_done("not done yet")
    assert parse_reply("done", NOW, CT).action == "done"  # type: ignore[union-attr]
    snooze = parse_reply("snooze 1h", NOW, CT)
    assert snooze is not None and snooze.action == "snooze"
    assert snooze.until == NOW + timedelta(hours=1)
    assert parse_reply("what's on today?", NOW, CT) is None


@pytest.mark.parametrize("text", ["paid", "Paid!", "I paid it", "already paid"])
def test_paid_is_a_done_said_as_paid(text: str) -> None:
    """PR 4: "paid" in reply to a bill's reminder is its Done, answered "Marked paid"."""
    reply = parse_reply(text, NOW, CT)
    assert reply is not None and reply.action == "done" and reply.paid is True
    assert is_done(text)


@pytest.mark.parametrize("text", ["not yet", "Not paid", "haven't paid"])
def test_not_yet_is_its_own_answer(text: str) -> None:
    reply = parse_reply(text, NOW, CT)
    assert reply is not None and reply.action == "not_yet"


def test_done_is_not_paid() -> None:
    reply = parse_reply("done", NOW, CT)
    assert reply is not None and reply.action == "done" and reply.paid is False
