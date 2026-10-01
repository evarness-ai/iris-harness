"""The digest footer's ``learned yesterday`` line (loop-proof D17, graph §9).

Rule: every owner signal has exactly one store and one effect, and the next digest
names what it learned. This module is the registry the footer reads: each signal
store contributes one **learned source** — a function that, given the previous local
day as ``[start, end)``, returns short human phrases ("news@goldenpi.com hidden from
Focus"). The renderer joins whatever the sources return; a day with nothing reads
``learned yesterday: nothing`` and the line is never omitted (V36).

Adding a signal (reminder snooze pattern in PR 3, judgment corrections in PR 5, …)
is one :func:`register_learned_source` call from the code that owns the store — the
renderer never changes. A source that fails is skipped and logged, never fatal: a
footer must not take the digest down.

Core registers the two signal stores it owns today:

* ``surface_feedback`` — 👎 / 👍 verdicts in ``learning.db`` (the Focus 👎 first);
* ``settings`` — edits in ``settings.db`` (ADR-0120), e.g. a digest topic added.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from iris_harness.foundation.process_state import track_globals
from iris_harness.services.learning.suppression import (
    EMAIL_FOCUS_SURFACE,
    EMAIL_FOLLOWUP_SURFACE,
    EMAIL_SEARCH_SUBSYSTEM,
    EMAIL_SEARCH_SURFACE,
    NOT_USEFUL,
    SurfaceFeedbackStore,
)

logger = logging.getLogger(__name__)

#: A learned source: ``(start, end)`` of the window (tz-aware) → phrases, in order.
LearnedSource = Callable[[datetime, datetime], list[str]]

PREFIX = "learned yesterday: "
NOTHING = "nothing"
SEPARATOR = " · "

_SOURCES: dict[str, LearnedSource] = {}


def register_learned_source(name: str, source: LearnedSource) -> None:
    """Add (or replace) the learned source ``name``. Order of first registration is kept."""
    _SOURCES[name] = source


def unregister_learned_source(name: str) -> None:
    """Remove ``name`` if present (tests, a plugin unloading)."""
    _SOURCES.pop(name, None)


def learned_sources() -> dict[str, LearnedSource]:
    """A copy of the registry, in registration order."""
    return dict(_SOURCES)


def previous_local_day(now: datetime, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """``[start, end)`` of the local calendar day before ``now``, as tz-aware datetimes."""
    today = now.astimezone(tz).date()
    end = datetime.combine(today, time.min, tzinfo=tz)
    start = datetime.combine(today - timedelta(days=1), time.min, tzinfo=tz)
    return start, end


def learned_between(start: datetime, end: datetime) -> list[str]:
    """Every registered source's phrases for ``[start, end)``, de-duplicated, in order."""
    phrases: list[str] = []
    for name, source in learned_sources().items():
        try:
            found = source(start, end)
        except Exception:  # one broken store must not blank the footer
            logger.warning("learned source %r failed; skipped", name, exc_info=True)
            continue
        for phrase in found:
            text = " ".join(str(phrase).split())
            if text and text not in phrases:
                phrases.append(text)
    return phrases


def learned_yesterday_line(now: datetime | None = None, tz: ZoneInfo | None = None) -> str:
    """The footer line for the local day before ``now``. Never empty."""
    if tz is None:
        from iris_harness.services.digest.settings import iris_timezone

        tz = iris_timezone()
    start, end = previous_local_day(now or datetime.now(UTC), tz)
    phrases = learned_between(start, end)
    return PREFIX + (SEPARATOR.join(phrases) if phrases else NOTHING)


# ── core sources ──────────────────────────────────────────────────────────────

# How a verdict on a surface reads in the footer. A surface not listed falls back to a
# generic phrase built from its own name, so a new surface is reported, never dropped.
_HIDDEN_FROM: dict[tuple[str, str], str] = {
    (EMAIL_SEARCH_SUBSYSTEM, EMAIL_FOCUS_SURFACE): "hidden from Focus",
    (EMAIL_SEARCH_SUBSYSTEM, EMAIL_FOLLOWUP_SURFACE): "no longer tracked as a followup",
    (EMAIL_SEARCH_SUBSYSTEM, EMAIL_SEARCH_SURFACE): "hidden from email search",
}
_SHOWN_AGAIN: dict[tuple[str, str], str] = {
    (EMAIL_SEARCH_SUBSYSTEM, EMAIL_FOCUS_SURFACE): "back in Focus",
}


def _subject_of(dims: dict[str, str]) -> str:
    for key in ("sender", "from_domain", "from", "name", "label"):
        if dims.get(key):
            return dims[key]
    return next(iter(dims.values()), "an item")


def surface_feedback_source(start: datetime, end: datetime) -> list[str]:
    """👎 / 👍 the owner gave on surfaced items that day, one phrase per item.

    Several verdicts on one item that day collapse to the last one.
    """
    store = SurfaceFeedbackStore()
    store.ensure_schema()
    last: dict[tuple[str, str, str], tuple[str, str]] = {}
    for entry in store.feedback_between(start, end):
        subject = _subject_of(entry.dims)
        key = (entry.subsystem, entry.surface_kind, subject)
        surface = (entry.subsystem.lower(), entry.surface_kind.lower())
        if entry.verdict == NOT_USEFUL:
            effect = _HIDDEN_FROM.get(surface, f"{entry.surface_kind.replace('_', ' ')} hidden")
        else:
            effect = _SHOWN_AGAIN.get(surface, f"{entry.surface_kind.replace('_', ' ')} shown")
        last.pop(key, None)  # re-insert so the phrase sits where its last verdict did
        last[key] = (subject, effect)
    return [f"{subject} {effect}" for subject, effect in last.values()]


def settings_source(start: datetime, end: datetime) -> list[str]:
    """Settings the owner changed that day (ADR-0120 history), one phrase per section."""
    from iris_harness.foundation.settings.store import SettingsStore

    changed: dict[str, list[str]] = {}
    for change in reversed(SettingsStore().history(limit=500)):
        at = change.at if change.at.tzinfo else change.at.replace(tzinfo=UTC)
        if not start <= at < end:
            continue
        keys = changed.setdefault(change.section, [])
        if change.key not in keys:
            keys.append(change.key)
    return [
        f"{section} settings changed ({', '.join(k.replace('_', ' ') for k in keys)})"
        for section, keys in changed.items()
    ]


register_learned_source("surface_feedback", surface_feedback_source)
register_learned_source("settings", settings_source)


__all__ = [
    "NOTHING",
    "PREFIX",
    "LearnedSource",
    "learned_between",
    "learned_sources",
    "learned_yesterday_line",
    "previous_local_day",
    "register_learned_source",
    "settings_source",
    "surface_feedback_source",
    "unregister_learned_source",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_SOURCES")
