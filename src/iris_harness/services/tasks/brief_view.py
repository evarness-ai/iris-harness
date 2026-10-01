"""How a title reads on one digest line (the Today group's tasks and plan).

:func:`short_title` — repeated `` - `` / ``: `` separated segments collapsed
(calendar feeds often repeat a course name and stage: "Swim Lessons - Swim Lessons:
Stage 3 - ... Stage 3"), then capped at a word boundary with "…". Pure and generic:
no vocabulary, no store; the full title stays in the store. Which items show at all
is ``iris_harness.services.digest.expiry``'s business.
"""

from __future__ import annotations

import re

TITLE_CAP = 80

# Segment separators, only at bracket depth 0 ("(ages 5 - 12 years)" is one segment).
_SEPARATORS = (" - ", " – ", " — ", ": ", " | ")
_OPEN = "([{"
_CLOSE = ")]}"


def _split_segments(title: str) -> list[tuple[str, str]]:
    """``[(separator_before, segment), ...]``; the first separator is ""."""
    out: list[tuple[str, str]] = []
    depth = 0
    sep = ""
    start = 0
    i = 0
    while i < len(title):
        ch = title[i]
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth = max(0, depth - 1)
        elif depth == 0:
            hit = next((s for s in _SEPARATORS if title.startswith(s, i)), None)
            if hit is not None:
                out.append((sep, title[start:i]))
                sep = hit
                i += len(hit)
                start = i
                continue
        i += 1
    out.append((sep, title[start:]))
    return out


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _contains_words(haystack: str, needle: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def collapse_repeats(title: str) -> str:
    """Drop the segments of ``title`` that only repeat an earlier one.

    A segment is dropped when it equals an earlier kept segment, or (4+ characters)
    appears as whole words inside one; a segment that *starts* with an earlier one
    keeps only what follows it ("Water Stamina Tue 5:00pm" after "Water Stamina"
    becomes "Tue 5:00pm"). Case- and spacing-insensitive; never empties a title.
    """
    kept: list[tuple[str, str]] = []
    seen: list[str] = []
    for sep, raw in _split_segments(title):
        segment = raw.strip()
        norm = _norm(segment)
        if not norm:
            continue
        if any(norm == s or (len(norm) >= 4 and _contains_words(s, norm)) for s in seen):
            continue
        for prior in sorted(seen, key=len, reverse=True):
            if norm.startswith(prior + " "):
                segment = " ".join(segment.split()[len(prior.split()) :])
                norm = _norm(segment)
                break
        seen.append(norm)
        kept.append((sep if kept else "", segment))
    text = "".join(sep + seg for sep, seg in kept)
    return text or title.strip()


def short_title(title: str, cap: int = TITLE_CAP) -> str:
    """``title`` for one digest line: repeats collapsed, at most ``cap`` characters."""
    text = " ".join(collapse_repeats(title).split())
    if len(text) <= cap:
        return text
    cut = text[: cap - 1]
    space = cut.rfind(" ")
    if space >= cap // 2:
        cut = cut[:space]
    return cut.rstrip(" -–—:|,;") + "…"


__all__ = ["TITLE_CAP", "collapse_repeats", "short_title"]
