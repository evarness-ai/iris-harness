"""Lessons: captured from evidence, reviewed by the owner, approved into behaviors.

What was there before: `learning_signals` rows written on every successful turn, all
of the shape "For 'finance' requests like this, the finance agent using finance
answered it cleanly." Four rows, no information. They were fetched on every turn and
then dropped unless an opt-in flag was set, which it never was.

A lesson now needs evidence — the user corrected the answer, a tool failed and another
path worked, or the evaluator halted a run — and it says what to do next time, not
that something went fine. Approving one writes a behavior file, so it reaches the
prompt through the mechanism that already exists instead of a second injection path.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Evidence kinds. Anything else is not a lesson — it is telemetry.
SOURCE_CORRECTION = "user_correction"
SOURCE_TOOL_RECOVERY = "tool_recovery"
SOURCE_EVALUATOR_HALT = "evaluator_halt"

_MAX_LESSON_CHARS = 400


def _keywords(text: str, *, limit: int = 8) -> tuple[str, ...]:
    """Content words from the trigger, for the behavior's `match_keywords`."""
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "for",
        "with",
        "about",
        "from",
        "into",
        "onto",
        "to",
        "of",
        "in",
        "on",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "its",
        "my",
        "me",
        "i",
        "you",
        "your",
        "we",
        "our",
        "us",
        "do",
        "does",
        "did",
        "not",
        "no",
        "yes",
        "please",
    }
    words = [w for w in re.findall(r"[a-z0-9']{3,}", (text or "").lower()) if w not in stop]
    seen: list[str] = []
    for word in words:
        if word not in seen:
            seen.append(word)
    return tuple(seen[:limit])


class LessonCurator:
    """Propose lessons from evidence; approve one into a behavior."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def propose(self, *, trigger: str, lesson: str, source: str, evidence: str = "") -> int | None:
        """Queue a lesson for review. Returns its id, or None when it is not one.

        Rejected here: an empty lesson, and anything that only says things went well —
        "answered it cleanly" is not something to do differently next time.
        """
        trigger = " ".join((trigger or "").split())
        lesson = " ".join((lesson or "").split())[:_MAX_LESSON_CHARS]
        if not trigger or not lesson:
            return None
        if _reads_as_success(lesson):
            logger.debug("lesson rejected — it only reports success: %r", lesson)
            return None
        proposal_id: int | None = self._store.add_lesson_proposal(
            trigger=trigger, lesson=lesson, source=source, evidence=evidence
        )
        return proposal_id

    def approve(self, proposal_id: int, *, match_intents: tuple[str, ...] = ()) -> str | None:
        """Approve a lesson: it becomes a behavior file, matchable on the next turn."""
        proposal = self._store.fetch_lesson_proposal(proposal_id)
        if proposal is None or proposal.status != "pending":
            return None
        from iris_harness.memory.identity import write_behavior

        name = _behavior_name(proposal.trigger)
        body = (
            f"# Behavior — {name}\n\n"
            f"When: {proposal.trigger}\n\n"
            f"Do: {proposal.lesson}\n\n"
            f"_Learned from: {proposal.source}._\n"
        )
        write_behavior(
            name,
            body,
            match_keywords=_keywords(proposal.trigger),
            match_intents=match_intents,
            description=proposal.lesson[:120],
            source=f"learned:{proposal.source}",
        )
        self._store.resolve_lesson_proposal(proposal_id, "approved")
        return name

    def reject(self, proposal_id: int) -> bool:
        return bool(self._store.resolve_lesson_proposal(proposal_id, "rejected"))


_SUCCESS_RE = re.compile(
    r"\b(answered it cleanly|worked fine|went well|no issues|successfully|as expected)\b",
    re.IGNORECASE,
)


def _reads_as_success(lesson: str) -> bool:
    return bool(_SUCCESS_RE.search(lesson))


def _behavior_name(trigger: str) -> str:
    words = _keywords(trigger, limit=4)
    return "-".join(words) if words else "learned-lesson"


__all__ = [
    "SOURCE_CORRECTION",
    "SOURCE_EVALUATOR_HALT",
    "SOURCE_TOOL_RECOVERY",
    "LessonCurator",
]
