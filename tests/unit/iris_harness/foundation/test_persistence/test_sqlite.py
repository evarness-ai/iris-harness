"""Tests for the hardened SQLite connection helper (Task 1: memory-integrity foundation)."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris_harness.foundation.persistence import connect, sqlite_conn, with_locked_retry
from iris_harness.memory.store import MemoryStore, UserFact


def test_connect_helper_sets_wal_and_busy_timeout(tmp_path: Path) -> None:
    conn = connect(tmp_path / "x.db", row_factory=sqlite3.Row)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert conn.row_factory is sqlite3.Row
    finally:
        conn.close()


def test_app_stores_are_wal_after_write(tmp_path: Path) -> None:
    """The hot app stores route _connect() through connect(), so they land in WAL."""
    from iris_harness.services.learning.store import LearningMetricsStore
    from iris_harness.services.tasks.store import TaskStore

    tasks_db = tmp_path / "tasks.db"
    ts = TaskStore(db_path=tasks_db)
    ts.ensure_schema()
    ts.create(title="buy milk")
    assert sqlite3.connect(tasks_db).execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

    learn_db = tmp_path / "learning.db"
    ls = LearningMetricsStore(db_path=learn_db)
    ls.ensure_schema()
    assert sqlite3.connect(learn_db).execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def _fact(key: str, value: str) -> UserFact:
    now = datetime.now(UTC)
    return UserFact(
        key=key,
        value=value,
        confidence=0.9,
        source="test",
        first_seen=now,
        last_confirmed=now,
        times_confirmed=1,
    )


def test_wal_mode_enabled_after_write(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path=db_path)
    store.upsert_user_fact(_fact("name", "Robin"))

    # Open a fresh connection and confirm the DB file is in WAL mode (persisted on file).
    with sqlite_conn(db_path) as conn:
        (mode,) = conn.execute("PRAGMA journal_mode").fetchone()
    assert mode.lower() == "wal"


# `credit_card` (a multi-valued key) comes from the test vocabulary fragment.
@pytest.mark.usefixtures("test_vocabulary")
def test_concurrent_writers_no_lock_errors(tmp_path: Path) -> None:
    """Many threads writing the same memory.db must not raise 'database is locked'."""
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path=db_path)
    store.ensure_schema()

    errors: list[Exception] = []
    n_threads = 12
    writes_per_thread = 15
    barrier = threading.Barrier(n_threads)

    def writer(tid: int) -> None:
        barrier.wait()  # maximize contention
        try:
            for i in range(writes_per_thread):
                # distinct values of a property with no limit: every write must land
                store.upsert_user_fact(_fact("credit_card", f"card {tid}-{i}"))
        except Exception as exc:  # noqa: BLE001 - capture for assertion
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"unexpected errors under concurrency: {errors}"
    # All rows landed.
    assert len(store.fetch_all_user_facts()) == n_threads * writes_per_thread


def test_sqlite_conn_commits_and_closes(tmp_path: Path) -> None:
    db_path = tmp_path / "t.db"
    with sqlite_conn(db_path) as conn:
        conn.execute("CREATE TABLE t(x INTEGER)")
        conn.execute("INSERT INTO t(x) VALUES(1)")
        # no explicit commit — the context manager commits on clean exit

    # Reopen: the row must have been committed, and the prior connection closed cleanly.
    with sqlite_conn(db_path) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM t").fetchone()
    assert count == 1


def test_sqlite_conn_rolls_back_on_exception(tmp_path: Path) -> None:
    db_path = tmp_path / "t.db"
    with sqlite_conn(db_path) as conn:
        conn.execute("CREATE TABLE t(x INTEGER)")
    try:
        with sqlite_conn(db_path) as conn:
            conn.execute("INSERT INTO t(x) VALUES(1)")
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    with sqlite_conn(db_path) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM t").fetchone()
    assert count == 0  # the insert was rolled back


def test_with_locked_retry_retries_then_succeeds() -> None:
    calls = {"n": 0}

    @with_locked_retry(attempts=4, base_delay=0.001)
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_with_locked_retry_reraises_non_lock_error() -> None:
    @with_locked_retry(attempts=3, base_delay=0.001)
    def boom() -> None:
        raise sqlite3.OperationalError("no such table: foo")

    try:
        boom()
    except sqlite3.OperationalError as exc:
        assert "no such table" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected OperationalError to propagate")
