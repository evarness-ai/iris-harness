from __future__ import annotations

import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_harness.memory.state import (
    CheckpointNotFoundError,
    CheckpointStore,
    CheckpointTooLargeError,
)
from iris_harness.memory.state.store import MAX_PAYLOAD_BYTES


def _store(tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(db_path=tmp_path / "checkpoints.db")


def test_db_chmod_0o600_on_create(tmp_path: Path) -> None:
    db_path = tmp_path / "checkpoints.db"
    CheckpointStore(db_path=db_path)
    mode = stat.S_IMODE(os.stat(db_path).st_mode)
    assert mode == 0o600


def test_write_then_get_roundtrip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cp = store.write(
        run_id="r1",
        step_id=3,
        agent_type="chat",
        payload={"query": "hi", "steps": []},
        signal="halt",
    )
    assert cp.run_id == "r1"
    assert cp.step_id == 3
    assert cp.signal == "halt"
    assert cp.pinned is False
    assert cp.payload == {"query": "hi", "steps": []}

    fetched = store.get(run_id="r1", step_id=3)
    assert fetched.payload_json == cp.payload_json


def test_write_overwrites_same_run_step(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(run_id="r", step_id=1, agent_type="chat", payload={"v": 1})
    store.write(run_id="r", step_id=1, agent_type="chat", payload={"v": 2})
    assert store.get(run_id="r", step_id=1).payload == {"v": 2}


def test_get_latest_returns_highest_step(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(run_id="r", step_id=1, agent_type="chat", payload={})
    store.write(run_id="r", step_id=5, agent_type="chat", payload={})
    store.write(run_id="r", step_id=3, agent_type="chat", payload={})
    assert store.get_latest("r").step_id == 5


def test_get_missing_raises(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(CheckpointNotFoundError):
        store.get(run_id="nope", step_id=0)
    with pytest.raises(CheckpointNotFoundError):
        store.get_latest("nope")


def test_pin_protects_from_sweep(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
    )  # default TTL fine
    past = datetime.now(UTC) - timedelta(days=8)
    store.write(
        run_id="pinned",
        step_id=1,
        agent_type="chat",
        payload={},
        ts=past,
    )
    store.write(
        run_id="loose",
        step_id=1,
        agent_type="chat",
        payload={},
        ts=past,
    )
    store.pin("pinned")

    deleted = store.sweep_expired()
    assert deleted == 1
    assert store.get(run_id="pinned", step_id=1).pinned is True
    with pytest.raises(CheckpointNotFoundError):
        store.get(run_id="loose", step_id=1)


def test_list_filters_by_agent_type_and_expired(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(run_id="r1", step_id=1, agent_type="chat", payload={})
    store.write(run_id="r2", step_id=1, agent_type="coding", payload={})
    past = datetime.now(UTC) - timedelta(days=8)
    store.write(run_id="r3", step_id=1, agent_type="chat", payload={}, ts=past)

    fresh = store.list()
    assert {c.run_id for c in fresh} == {"r1", "r2"}

    fresh_chat = store.list(agent_type="chat")
    assert {c.run_id for c in fresh_chat} == {"r1"}

    all_chat = store.list(agent_type="chat", include_expired=True)
    assert {c.run_id for c in all_chat} == {"r1", "r3"}


def test_remove(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(run_id="r", step_id=1, agent_type="chat", payload={})
    store.write(run_id="r", step_id=2, agent_type="chat", payload={})
    deleted = store.remove("r")
    assert deleted == 2
    assert store.list() == ()


def test_oversized_payload_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    big_value = "x" * (MAX_PAYLOAD_BYTES + 100)
    with pytest.raises(CheckpointTooLargeError):
        store.write(run_id="r", step_id=1, agent_type="chat", payload={"v": big_value})


def test_unpin(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(run_id="r", step_id=1, agent_type="chat", payload={})
    store.pin("r")
    assert store.get(run_id="r", step_id=1).pinned is True
    store.unpin("r")
    assert store.get(run_id="r", step_id=1).pinned is False


# ── ADR-0106 M5.C1: session keying ────────────────────────────────────────────


def test_write_records_session_id_and_by_session_reads_it(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(run_id="r1", step_id=1, agent_type="chat", payload={}, session_id="s1")
    store.write(run_id="r2", step_id=1, agent_type="chat", payload={}, session_id="s2")

    mine = store.by_session("s1")
    assert [cp.run_id for cp in mine] == ["r1"]
    assert mine[0].session_id == "s1"


def test_by_session_never_leaks_across_sessions_or_sessionless_runs(tmp_path: Path) -> None:
    """A run with no session must not be collected by a caller that has none."""
    store = _store(tmp_path)
    store.write(run_id="cli", step_id=1, agent_type="chat", payload={})  # no session

    assert store.by_session("s1") == ()
    assert store.by_session("") == ()


def test_by_session_enforces_ttl_on_read(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(run_id="r1", step_id=1, agent_type="chat", payload={}, session_id="s1")

    later = datetime.now(UTC) + timedelta(days=8)
    assert store.by_session("s1", now=later) == ()
    assert len(store.by_session("s1", now=later, include_expired=True)) == 1


def test_by_session_returns_pinned_rows_past_ttl(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(run_id="r1", step_id=1, agent_type="chat", payload={}, session_id="s1")
    store.pin("r1")

    later = datetime.now(UTC) + timedelta(days=8)
    assert len(store.by_session("s1", now=later)) == 1


def test_session_id_column_is_added_to_a_pre_adr_database(tmp_path: Path) -> None:
    """A checkpoints.db written before ADR-0106 has no session_id. Opening it must
    migrate in place and leave the existing rows readable, answering "no session"."""
    import sqlite3

    db_path = tmp_path / "checkpoints.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE checkpoints (
            run_id        TEXT NOT NULL,
            step_id       INTEGER NOT NULL,
            ts            TEXT NOT NULL,
            agent_type    TEXT NOT NULL,
            payload_json  TEXT NOT NULL,
            signal        TEXT,
            pinned        INTEGER NOT NULL DEFAULT 0,
            expires_at    TEXT NOT NULL,
            PRIMARY KEY (run_id, step_id)
        );
        """)
    conn.execute(
        "INSERT INTO checkpoints VALUES ('old', 1, ?, 'chat', '{}', NULL, 0, ?)",
        (datetime.now(UTC).isoformat(), (datetime.now(UTC) + timedelta(days=7)).isoformat()),
    )
    conn.commit()
    conn.close()

    store = CheckpointStore(db_path=db_path)

    legacy = store.get(run_id="old", step_id=1)
    assert legacy.session_id is None
    assert store.by_session("anything") == ()
    # And the migrated database takes new session-keyed rows.
    store.write(run_id="new", step_id=1, agent_type="chat", payload={}, session_id="s1")
    assert [cp.run_id for cp in store.by_session("s1")] == ["new"]
