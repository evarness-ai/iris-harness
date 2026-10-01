"""Heartbeat schedules in words: check one, describe one, read one a person typed.

A schedule is ``interval:<seconds>`` or a five-field crontab (``min hour day month
weekday``), read in the scheduler's timezone. The API, the REPL and the web app all
describe and accept schedules through this module, so "every 10 min" means the same
thing on every surface.
"""

from __future__ import annotations

import re

from apscheduler.triggers.cron import CronTrigger

INTERVAL_PREFIX = "interval:"

# The shortest interval the app may set. The fastest shipped heartbeat (pressure_tick)
# runs every 30 s; anything tighter is a typo, not a schedule.
MIN_INTERVAL_SECONDS = 30

_DAILY = re.compile(r"^(\d{1,2}) (\d{1,2}) \* \* \*$")
_CLOCK = re.compile(r"^(\d{1,2}):(\d{2})$")
_AMOUNT = re.compile(r"^(\d+)\s*([a-z]*)$")
_UNITS = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "m": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "h": 3600,
    "hr": 3600,
    "hrs": 3600,
    "hour": 3600,
    "hours": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
}


def validate_schedule(schedule: str) -> str:
    """Return ``schedule`` stripped, or raise ``ValueError`` saying what is wrong."""
    text = schedule.strip()
    if text.startswith(INTERVAL_PREFIX):
        raw = text[len(INTERVAL_PREFIX) :].strip()
        if not raw.isdigit():
            raise ValueError(f"interval must be whole seconds, got {raw!r}")
        seconds = int(raw)
        if seconds < MIN_INTERVAL_SECONDS:
            raise ValueError(f"shortest interval is {MIN_INTERVAL_SECONDS} seconds")
        return f"{INTERVAL_PREFIX}{seconds}"
    if len(text.split()) != 5:
        raise ValueError(
            "schedule must be 'interval:<seconds>' or a 5-field cron "
            f"(min hour day month weekday), got {text!r}"
        )
    try:
        CronTrigger.from_crontab(text)
    except ValueError as exc:
        raise ValueError(f"invalid cron {text!r}: {exc}") from exc
    return " ".join(text.split())


def describe_schedule(schedule: str) -> str:
    """Plain words for a schedule: ``every 10 min``, ``daily at 08:00``, ``cron …``."""
    text = schedule.strip()
    if text.startswith(INTERVAL_PREFIX):
        raw = text[len(INTERVAL_PREFIX) :].strip()
        if not raw.isdigit():
            return text
        seconds = int(raw)
        for size, one, many in ((86400, "day", "days"), (3600, "hour", "hours")):
            if seconds % size == 0:
                count = seconds // size
                return f"every {one}" if count == 1 else f"every {count} {many}"
        if seconds % 60 == 0:
            minutes = seconds // 60
            return "every minute" if minutes == 1 else f"every {minutes} min"
        return f"every {seconds} s"
    daily = _DAILY.match(" ".join(text.split()))
    if daily:
        minute, hour = int(daily.group(1)), int(daily.group(2))
        if minute < 60 and hour < 24:
            return f"daily at {hour:02d}:{minute:02d}"
    return f"cron {text}"


def parse_schedule_words(words: list[str]) -> str:
    """Read a schedule typed as words and return it validated.

    ``every 10m`` / ``every 10 min`` / ``every hour``, ``daily 08:30``, or
    ``cron 0 8 * * 1-5``. Raises ``ValueError`` with the accepted forms.
    """
    usage = "use 'every <n><s|m|h|d>', 'daily HH:MM' or 'cron <5 fields>'"
    if not words:
        raise ValueError(usage)
    kind, rest = words[0].lower(), words[1:]
    if kind == "every":
        amount = " ".join(rest).strip().lower()
        if amount in _UNITS:  # "every hour", "every day"
            amount = f"1 {amount}"
        match = _AMOUNT.match(amount)
        if not match or not match.group(2) or match.group(2) not in _UNITS:
            raise ValueError(f"cannot read {amount!r}: {usage}")
        return validate_schedule(f"{INTERVAL_PREFIX}{int(match.group(1)) * _UNITS[match.group(2)]}")
    if kind == "daily":
        clock = _CLOCK.match(" ".join(rest).strip())
        if not clock or int(clock.group(1)) > 23 or int(clock.group(2)) > 59:
            raise ValueError(f"daily needs a 24-hour time like 08:30: {usage}")
        return validate_schedule(f"{int(clock.group(2))} {int(clock.group(1))} * * *")
    if kind == "cron":
        return validate_schedule(" ".join(rest))
    raise ValueError(usage)
