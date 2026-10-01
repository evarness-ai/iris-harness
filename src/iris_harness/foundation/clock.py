"""One home for "now" and for reading stored timestamps (code-standards batch).

The code base grew ~45 private copies of the same few lines (``_utc_now``, ``_now``,
``_parse_iso`` ...). They drift: some returned naive datetimes, some raised on a bad
string, some kept a non-UTC offset. New code uses these four instead, and each copy
moves here once it is shown to mean exactly the same thing.

Every datetime this module returns is timezone-aware and in UTC. A **naive** datetime
or ISO string handed in is read as UTC, which is how every store here writes them.
The owner's local zone is a different question: see ``iris_harness.sdk.time``.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, date, datetime, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """The current instant, timezone-aware, in UTC."""
    return datetime.now(UTC)


def utc_now_iso() -> str:
    """:func:`utc_now` as an ISO-8601 string (``...+00:00``), the stores' format."""
    return datetime.now(UTC).isoformat()


def as_utc(dt: datetime) -> datetime:
    """``dt`` in UTC. A naive ``dt`` is read as UTC; an aware one is converted."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def parse_iso(value: object, *, default: datetime | None = None) -> datetime | None:
    """Parse an ISO-8601 timestamp into an aware UTC datetime; never raises.

    A trailing ``Z`` is accepted and a naive value is read as UTC. ``None``, an empty
    or non-string value, or text that is not ISO-8601 gives ``default``.
    """
    if not isinstance(value, str) or not value.strip():
        return default
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        return as_utc(datetime.fromisoformat(text))
    except ValueError:
        return default


def iris_timezone() -> ZoneInfo:
    """The owner's zone from ``IRIS_TZ``; UTC when it is unset or not a zone name."""
    name = (os.environ.get("IRIS_TZ") or "").strip()
    if name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            logger.warning("IRIS_TZ=%r is not a time zone name; using UTC", name)
    return ZoneInfo("UTC")


def local_zone() -> tzinfo:
    """The zone "now" and "today" are read in: ``IRIS_TZ`` when set, else this machine's.

    Not :func:`iris_timezone`'s UTC fallback: a run with no ``IRIS_TZ`` has always
    meant "the machine's own clock" for these, and turning it into UTC would move a
    local run's "today" by hours. When ``IRIS_TZ`` is set -- the VM, a Docker install --
    it wins over the machine's zone, which a container often leaves at UTC.
    """
    if (os.environ.get("IRIS_TZ") or "").strip():
        return iris_timezone()
    zone = datetime.now().astimezone().tzinfo
    return zone if zone is not None else UTC


def local_now() -> datetime:
    """The owner's wall clock, timezone-aware (see :func:`local_zone`)."""
    return datetime.now(local_zone())


def local_today() -> date:
    """The owner's calendar date (see :func:`local_zone`)."""
    return local_now().date()


__all__ = [
    "as_utc",
    "iris_timezone",
    "local_now",
    "local_today",
    "local_zone",
    "parse_iso",
    "utc_now",
    "utc_now_iso",
]
