"""The approvals queue gains one nullable ``call_id`` column (#134, stage 1).

A DB created by the current release has no such column. Opening it adds the column in place
(``PRAGMA table_info`` guard, additive ``ALTER TABLE``), keeps every old row readable with
``call_id`` None, is a no-op the second time, and tolerates losing the race to a second
process that adds it first (``duplicate column name``).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from iris_harness.kernel.governance.approvals.queue import ApprovalQueue
from iris_harness.kernel.governance.approvals.store import ApprovalStore

# The table exactly as the current release (before #134) creates it.
_RELEASE_SCHEMA = """
CREATE TABLE approval_queue (
    approval_id       TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL,
    checkpoint_id     TEXT,
    signal            TEXT NOT NULL,
    context_summary   TEXT NOT NULL,
    requested_at      TEXT NOT NULL,
    channel           TEXT NOT NULL DEFAULT 'cli',
    status            TEXT NOT NULL DEFAULT 'pending',
    responded_at      TEXT,
    response_actor    TEXT,
    timeout_at        TEXT NOT NULL,
    policy_on_timeout TEXT NOT NULL DEFAULT 'fail_closed',
    session_id        TEXT,
    items_json        TEXT,
    card_json         TEXT,
    caller            TEXT,
    executed_at       TEXT
);
CREATE INDEX idx_approval_pending ON approval_queue(status, timeout_at);
"""


def _release_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(_RELEASE_SCHEMA)
    conn.execute(
        "INSERT INTO approval_queue (approval_id, run_id, signal, context_summary, requested_at,"
        " timeout_at) VALUES ('old-1', 'run-old', 's', 'c', '2026-01-01T00:00:00+00:00',"
        " '2099-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()


def _columns(path: Path) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [r[1] for r in conn.execute("PRAGMA table_info(approval_queue)")]
    finally:
        conn.close()


def test_a_release_db_gains_the_column_and_keeps_its_rows(tmp_path: Path) -> None:
    db = tmp_path / "approvals.db"
    _release_db(db)
    assert "call_id" not in _columns(db)

    store = ApprovalStore(db_path=db)
    assert _columns(db).count("call_id") == 1
    old = store.get("old-1")
    assert old is not None and old.call_id is None and old.run_id == "run-old"
    new = store.get(
        store.enqueue("run-new", None, "sig", "ctx", call_id="01JABCDEFGHJKMNPQRSTVWXYZ0")
    )
    assert new is not None and new.call_id == "01JABCDEFGHJKMNPQRSTVWXYZ0"
    plain = store.get(store.enqueue("run-new2", None, "sig", "ctx"))
    assert plain is not None and plain.call_id is None


def test_the_migration_runs_twice_without_harm(tmp_path: Path) -> None:
    db = tmp_path / "approvals.db"
    _release_db(db)
    ApprovalStore(db_path=db)
    first = _columns(db)
    store = ApprovalStore(db_path=db)  # the second open: nothing to do
    assert _columns(db) == first and first.count("call_id") == 1
    assert store.get("old-1") is not None


def test_a_fresh_db_has_the_column_from_the_schema(tmp_path: Path) -> None:
    db = tmp_path / "approvals.db"
    ApprovalStore(db_path=db)
    assert _columns(db).count("call_id") == 1


def test_losing_the_alter_race_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another process added the column between our ``PRAGMA table_info`` and our ALTER."""
    db = tmp_path / "approvals.db"
    _release_db(db)
    other = sqlite3.connect(db)
    other.execute("ALTER TABLE approval_queue ADD COLUMN call_id TEXT")
    other.commit()
    other.close()

    real_connect = ApprovalStore._connect

    class _Stale:
        """Hides ``call_id`` from the table_info read, as a stale view of the table would."""

        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

        def execute(self, sql: str, *args: Any) -> Any:
            cur = self._conn.execute(sql, *args)
            if sql.startswith("PRAGMA table_info(approval_queue)"):
                return [r for r in cur.fetchall() if r[1] != "call_id"]
            return cur

        def __getattr__(self, name: str) -> Any:
            return getattr(self._conn, name)

    @contextmanager
    def racy(self: ApprovalStore) -> Iterator[Any]:
        with real_connect(self) as conn:
            yield _Stale(conn)

    monkeypatch.setattr(ApprovalStore, "_connect", racy)
    store = ApprovalStore(db_path=db)  # the ALTER raises "duplicate column name": tolerated
    monkeypatch.undo()
    assert _columns(db).count("call_id") == 1
    assert store.get("old-1") is not None


def test_another_operational_error_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "approvals.db"
    _release_db(db)
    real_connect = ApprovalStore._connect

    class _Locked:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

        def execute(self, sql: str, *args: Any) -> Any:
            if "ADD COLUMN call_id" in sql:
                raise sqlite3.OperationalError("database is locked")
            return self._conn.execute(sql, *args)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._conn, name)

    @contextmanager
    def locked(self: ApprovalStore) -> Iterator[Any]:
        with real_connect(self) as conn:
            yield _Locked(conn)

    monkeypatch.setattr(ApprovalStore, "_connect", locked)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        ApprovalStore(db_path=db)


def test_the_queue_audit_rows_name_the_held_call(tmp_path: Path) -> None:
    import json

    from iris_harness.kernel.governance.audit import AuditLog

    audit = AuditLog(db_path=tmp_path / "audit.db")
    queue = ApprovalQueue(store=ApprovalStore(db_path=tmp_path / "a.db"), audit_log=audit)
    held = "01JABCDEFGHJKMNPQRSTVWXYZ0"
    approval_id = queue.enqueue("run-1", None, "sig", "ctx", call_id=held)
    queue.respond(approval_id, status="approved", actor="owner")
    other = queue.enqueue("run-2", None, "sig", "ctx")  # no tool call raised this one
    rows = [json.loads(r.payload_json) for r in audit.query()]
    assert [p.get("call_id") for p in rows] == [held, held, None]
    assert queue.get(other) is not None and queue.get(other).call_id is None  # type: ignore[union-attr]
