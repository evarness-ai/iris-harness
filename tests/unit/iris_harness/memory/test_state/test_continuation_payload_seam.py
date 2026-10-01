"""The payload seam — a continuation can carry what happens if you say yes.

`_pending_confirmations` was an in-memory dict keyed by session, holding the action to
run on approval. `Continuation` was durable and held ownership. The two never met,
which ADR-0106 decision 6 flagged as the same asymmetry that caused the incident,
pointing the other way: the *answer* was durable and the thing it would execute died
with the process. A restart turned a pending "approve" into a message that resolved
nothing and routed on as though the question had never been asked — and because the
confirmation held no ownership, any of the other 21 confirmation-resolving intercepts
could claim that "approve" first.

The seam is two nullable columns: `executor_kind` (which registered executor runs it)
and `payload_json` (opaque to governance, which owns ownership and durability and never
what an action means).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.memory.state.continuations import ContinuationStore


@pytest.fixture()
def store(tmp_path: Path) -> ContinuationStore:
    return ContinuationStore(db_path=tmp_path / "checkpoints.db")


def test_a_question_can_carry_an_action(store: ContinuationStore) -> None:
    store.open(
        session_id="s1",
        owner="confirmation",
        question="Create 'Standup' and invite 2 attendees?",
        executor_kind="calendar_event",
        payload={"summary": "Standup", "attendee_count": 2},
    )

    pending = store.pending_for("s1")

    assert pending is not None
    assert pending.is_executable is True
    assert pending.executor_kind == "calendar_event"
    assert pending.payload == {"summary": "Standup", "attendee_count": 2}


def test_an_ordinary_question_carries_none(store: ContinuationStore) -> None:
    """Most continuations only route a turn; answering them executes nothing."""
    store.open(session_id="s1", owner="planner", question="proceed?")

    pending = store.pending_for("s1")

    assert pending is not None
    assert pending.is_executable is False
    assert pending.executor_kind is None
    assert pending.payload is None


def test_the_payload_survives_a_new_store_over_the_same_file(tmp_path: Path) -> None:
    """The whole point. In-memory, this is where a restart lost the action."""
    db = tmp_path / "checkpoints.db"
    ContinuationStore(db_path=db).open(
        session_id="s1",
        owner="confirmation",
        executor_kind="calendar_event",
        payload={"summary": "Standup"},
    )

    pending = ContinuationStore(db_path=db).pending_for("s1")

    assert pending is not None
    assert pending.payload == {"summary": "Standup"}


# ── the store refuses what it cannot keep ─────────────────────────────────────


def test_an_unserialisable_payload_is_refused_at_write_time(store: ContinuationStore) -> None:
    """Checked when the question is opened, not when the answer arrives — by then the
    user has already been asked and a failure is far too late to be useful."""
    with pytest.raises(ValueError, match="JSON-serialisable"):
        store.open(
            session_id="s1",
            owner="confirmation",
            executor_kind="calendar_event",
            payload={"start": object()},
        )


def test_a_payload_without_an_executor_is_refused(store: ContinuationStore) -> None:
    """An action nothing is registered to run is a question that cannot be answered."""
    with pytest.raises(ValueError, match="executor_kind"):
        store.open(session_id="s1", owner="confirmation", payload={"summary": "Standup"})


def test_an_oversized_payload_is_refused(store: ContinuationStore) -> None:
    """A governance table is not where an unbounded blob should end up — the checkpoint
    payload is capped for the same reason."""
    with pytest.raises(ValueError, match="cap"):
        store.open(
            session_id="s1",
            owner="confirmation",
            executor_kind="calendar_event",
            payload={"blob": "x" * 70_000},
        )


# ── and reads never fail on a bad row ─────────────────────────────────────────


def test_undecodable_stored_json_costs_the_action_not_the_conversation(
    tmp_path: Path,
) -> None:
    """A row hand-edited, truncated, or written by a future version: the question still
    reads back, it simply has nothing to execute. Raising here would break every turn in
    the session rather than one approval."""
    import sqlite3

    db = tmp_path / "checkpoints.db"
    store = ContinuationStore(db_path=db)
    store.open(
        session_id="s1",
        owner="confirmation",
        executor_kind="calendar_event",
        payload={"summary": "Standup"},
    )
    conn = sqlite3.connect(db)
    conn.execute("UPDATE continuations SET payload_json = '{not json'")
    conn.commit()
    conn.close()

    pending = ContinuationStore(db_path=db).pending_for("s1")

    assert pending is not None
    assert pending.payload is None
    assert pending.executor_kind == "calendar_event"


def test_a_non_object_payload_reads_back_as_none(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "checkpoints.db"
    ContinuationStore(db_path=db).open(
        session_id="s1", owner="confirmation", executor_kind="k", payload={"a": 1}
    )
    conn = sqlite3.connect(db)
    conn.execute("UPDATE continuations SET payload_json = '[1, 2, 3]'")
    conn.commit()
    conn.close()

    assert ContinuationStore(db_path=db).pending_for("s1").payload is None  # type: ignore[union-attr]


# ── the migration ─────────────────────────────────────────────────────────────


def test_a_pre_seam_database_gains_the_columns(tmp_path: Path) -> None:
    """CREATE TABLE IF NOT EXISTS would leave an existing table untouched, so the
    columns are added explicitly — the same in-place migration ADR-0106 C1 used to put
    `session_id` on `checkpoints`."""
    import sqlite3

    db = tmp_path / "checkpoints.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE continuations (
            continuation_id TEXT PRIMARY KEY,
            session_id      TEXT NOT NULL,
            owner           TEXT NOT NULL,
            kind            TEXT NOT NULL,
            question        TEXT NOT NULL DEFAULT '',
            intent          TEXT NOT NULL DEFAULT '',
            run_id          TEXT,
            step_id         INTEGER,
            status          TEXT NOT NULL DEFAULT 'pending',
            created_at      TEXT NOT NULL,
            decided_at      TEXT,
            expires_at      TEXT NOT NULL
        );
        INSERT INTO continuations(continuation_id, session_id, owner, kind, created_at,
                                  expires_at)
        VALUES ('old1', 's1', 'planner', 'approval', '2026-09-01T00:00:00+00:00',
                '2999-01-01T00:00:00+00:00');
        """)
    conn.commit()
    conn.close()

    pending = ContinuationStore(db_path=db).pending_for("s1")

    # The pre-existing row still reads, and simply answers "nothing to execute".
    assert pending is not None
    assert pending.continuation_id == "old1"
    assert pending.executor_kind is None
    assert pending.payload is None
