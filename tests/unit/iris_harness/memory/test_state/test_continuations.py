"""ADR-0106 M5.C1 — the continuation storage layer.

A continuation records that an owner asked this session something and is waiting.
These tests pin the two properties the organize-plan incident turned on: the TTL is
enforced **on read**, and a pending continuation is never resolved across sessions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_harness.memory.state.continuations import (
    DEFAULT_CONTINUATION_TTL,
    ContinuationConflictError,
    ContinuationStore,
)


def _store(tmp_path: Path) -> ContinuationStore:
    return ContinuationStore(db_path=tmp_path / "checkpoints.db")


def test_open_then_pending_for_roundtrip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    opened = store.open(
        session_id="s1",
        owner="planner",
        kind="approval",
        question="Would you like to proceed with this plan?",
        intent="planner",
    )

    assert opened.status == "pending"
    assert opened.is_resumable_run is False  # Tier A — no run behind it

    pending = store.pending_for("s1")
    assert pending is not None
    assert pending.continuation_id == opened.continuation_id
    assert pending.owner == "planner"


def test_pending_is_scoped_to_its_own_session(tmp_path: Path) -> None:
    """The incident in one assertion: another session's pending question is not
    this session's to answer, even when it is the only one open anywhere."""
    store = _store(tmp_path)
    store.open(session_id="other", owner="planner", question="proceed?")

    assert store.pending_for("mine") is None
    assert store.pending_for("other") is not None


def test_empty_session_id_matches_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.open(session_id="s1", owner="planner")

    assert store.pending_for("") is None
    with pytest.raises(ValueError, match="must belong to a session"):
        store.open(session_id="", owner="planner")


def test_one_pending_per_session_is_a_constraint(tmp_path: Path) -> None:
    """ADR-0106 decision 5 is enforced by a partial unique index, so a second
    pending continuation cannot exist even if a caller forgets to supersede."""
    store = _store(tmp_path)
    store.open(session_id="s1", owner="planner", question="first?")

    with pytest.raises(ContinuationConflictError, match="already has a pending"):
        store.open(session_id="s1", owner="research", question="second?")


def test_supersede_frees_the_slot_and_keeps_the_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.open(session_id="s1", owner="planner", question="first?")

    assert store.supersede("s1") == 1
    second = store.open(session_id="s1", owner="research", question="second?")

    assert store.pending_for("s1").continuation_id == second.continuation_id
    # Superseded, not deleted — the question the user was asked stays on the record.
    old = store.get(first.continuation_id)
    assert old is not None
    assert old.status == "superseded"
    assert old.decided_at is not None


def test_ttl_is_enforced_on_read_not_only_on_sweep(tmp_path: Path) -> None:
    """PR #412's lesson: a store whose TTL is honoured only where some caller
    remembers to expire first will hand a dead row to the path that forgot."""
    store = _store(tmp_path)
    store.open(session_id="s1", owner="planner", question="proceed?")

    later = datetime.now(UTC) + DEFAULT_CONTINUATION_TTL + timedelta(hours=1)

    # No explicit expire_stale call — the read must do it.
    assert store.pending_for("s1", now=later) is None
    assert store.history_for("s1")[0].status == "expired"


def test_answered_continuation_is_no_longer_pending(tmp_path: Path) -> None:
    store = _store(tmp_path)
    opened = store.open(session_id="s1", owner="planner")

    store.set_status(opened.continuation_id, "answered")

    assert store.pending_for("s1") is None
    assert store.get(opened.continuation_id).status == "answered"
    # The slot is free again.
    store.open(session_id="s1", owner="research")


def test_unknown_kind_and_status_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="unknown continuation kind"):
        store.open(session_id="s1", owner="planner", kind="whatever")

    opened = store.open(session_id="s1", owner="planner")
    with pytest.raises(ValueError, match="unknown continuation status"):
        store.set_status(opened.continuation_id, "done")


def test_tier_b_continuation_points_at_a_checkpoint(tmp_path: Path) -> None:
    store = _store(tmp_path)
    opened = store.open(
        session_id="s1",
        owner="research",
        kind="question",
        question="which vendor shortlist?",
        run_id="run-abc",
        step_id=3,
    )

    assert opened.is_resumable_run is True
    assert (opened.run_id, opened.step_id) == ("run-abc", 3)


def test_history_is_oldest_first(tmp_path: Path) -> None:
    store = _store(tmp_path)
    base = datetime.now(UTC)
    first = store.open(session_id="s1", owner="planner", now=base)
    store.supersede("s1", now=base + timedelta(minutes=1))
    second = store.open(session_id="s1", owner="research", now=base + timedelta(minutes=2))

    history = store.history_for("s1")
    assert [c.continuation_id for c in history] == [first.continuation_id, second.continuation_id]
