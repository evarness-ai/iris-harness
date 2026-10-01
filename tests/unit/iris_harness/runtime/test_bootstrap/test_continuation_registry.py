"""ADR-0106 M5.C2 — the continuation registry's policy over the store.

The store's own tests pin the row rules. These pin the two things the registry
adds: asking again supersedes rather than raising, and a message is only read as
an answer to the question that was actually asked.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_harness.memory.state.continuations import (
    DEFAULT_CONTINUATION_TTL,
    ContinuationStore,
)
from iris_harness.runtime.continuations import ContinuationRegistry, reads_as_answer


def _registry(tmp_path: Path) -> ContinuationRegistry:
    return ContinuationRegistry(store=ContinuationStore(db_path=tmp_path / "checkpoints.db"))


def test_ask_records_the_owner_and_question(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    reg.ask(
        "s1",
        "planner",
        question="Would you like to proceed with this plan?",
        intent="planner",
    )

    pending = reg.pending("s1")
    assert pending is not None
    assert pending.owner == "planner"
    assert pending.intent == "planner"
    assert pending.status == "pending"


def test_asking_again_supersedes_rather_than_raising(tmp_path: Path) -> None:
    """The store refuses a second pending row on purpose; re-asking is normal, so
    the registry supersedes for the caller instead of making it their job."""
    reg = _registry(tmp_path)
    first = reg.ask("s1", "planner", question="first?")
    second = reg.ask("s1", "research", question="second?")

    pending = reg.pending("s1")
    assert pending is not None
    assert pending.continuation_id == second.continuation_id
    assert pending.owner == "research"

    history = reg.history("s1")
    assert [c.status for c in history] == ["superseded", "pending"]
    assert history[0].continuation_id == first.continuation_id


def test_pending_is_never_another_sessions(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    reg.ask("other", "planner", question="proceed?")

    assert reg.pending("mine") is None


def test_answered_closes_it_and_frees_the_slot(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    opened = reg.ask("s1", "planner")

    reg.answered(opened.continuation_id)

    assert reg.pending("s1") is None
    assert reg.history("s1")[0].status == "answered"


def test_drop_withdraws_without_an_answer(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    reg.ask("s1", "organize_confirmation", question="approve the plan?")

    assert reg.drop("s1") == 1
    assert reg.pending("s1") is None
    # Marked, not deleted — the user was really asked this.
    assert reg.history("s1")[0].status == "superseded"


def test_pending_expires_on_read(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    reg.ask("s1", "planner")

    later = datetime.now(UTC) + DEFAULT_CONTINUATION_TTL + timedelta(hours=1)
    assert reg.pending("s1", now=later) is None


# ── reads_as_answer ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("yes go head with the plan", "approve"),  # the incident turn
        ("yes", "approve"),
        ("go ahead", "approve"),
        ("sounds good", "approve"),
        ("approve", "approve"),
        ("no", "reject"),
        ("cancel", "reject"),
        ("don't", "reject"),
        ("never mind", "reject"),
    ],
)
def test_reads_approval_answers(tmp_path: Path, message: str, expected: str) -> None:
    reg = _registry(tmp_path)
    continuation = reg.ask("s1", "planner", kind="approval")

    assert reads_as_answer(continuation, message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "what's my email digest?",
        "the plan looks yes-ish to me",  # affirmation not at the start
        "tell me more about Stardog",
        "",
    ],
)
def test_does_not_read_a_non_answer_as_one(tmp_path: Path, message: str) -> None:
    reg = _registry(tmp_path)
    continuation = reg.ask("s1", "planner", kind="approval")

    assert reads_as_answer(continuation, message) is None


def test_question_continuations_are_not_classified_here(tmp_path: Path) -> None:
    """A free-text question needs the chain to judge answer-vs-changed-subject.
    That is M5.C3's call, so this function declines rather than guessing."""
    reg = _registry(tmp_path)
    continuation = reg.ask("s1", "research", kind="question", question="which vendors?")

    assert reads_as_answer(continuation, "yes") is None
    assert reads_as_answer(continuation, "Neo4j and Stardog") is None
