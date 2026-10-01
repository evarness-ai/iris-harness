"""CheckpointStore — SQLite store for resume checkpoints.

Schema (design §10.2):

    run_id          TEXT NOT NULL
    step_id         INTEGER NOT NULL
    ts              TEXT NOT NULL                         -- ISO-8601
    agent_type      TEXT NOT NULL                         -- chat / coding / ...
    payload_json    TEXT NOT NULL                         -- ChatCheckpointPayload, etc.
    signal          TEXT                                  -- halt|require_approval|manual
    pinned          INTEGER NOT NULL DEFAULT 0
    expires_at      TEXT NOT NULL                         -- ts + TTL

A row is uniquely keyed by ``(run_id, step_id)``: a halt at step 3
overwrites any earlier checkpoint for the same run+step (re-running
the same step shouldn't accumulate). Multiple checkpoints per run
across different step_ids are fine — the typical resume path is
``get_latest(run_id)``.

Row size capped at 1 MiB (raises ``CheckpointTooLargeError``); the
chat shape is well below that, the coding agent's side-effect ledger
is sized separately in Phase 4.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from iris_harness.foundation.paths import governance_data_dir
from iris_harness.foundation.persistence.sqlite import connect

logger = logging.getLogger(__name__)

# Override slot (``None``: resolved from ``IRIS_HOME`` on every use, never frozen at
# import -- a process that relocates the home after importing IRIS writes into the
# new one). Set it to point the default somewhere else outright (tests).
DEFAULT_CHECKPOINT_DB_PATH: Path | None = None


def default_checkpoint_db_path() -> Path:
    """``DEFAULT_CHECKPOINT_DB_PATH`` when set, else ``<governance data dir>/checkpoints.db``."""
    return (
        DEFAULT_CHECKPOINT_DB_PATH
        if DEFAULT_CHECKPOINT_DB_PATH is not None
        else governance_data_dir() / "checkpoints.db"
    )


DEFAULT_TTL = timedelta(days=7)
MAX_PAYLOAD_BYTES = 1 * 1024 * 1024  # 1 MiB


class CheckpointNotFoundError(LookupError):
    """No checkpoint matches the requested run_id / step_id."""


class CheckpointTooLargeError(ValueError):
    """Encoded checkpoint payload exceeds ``MAX_PAYLOAD_BYTES``."""


@dataclass(frozen=True)
class Checkpoint:
    """One materialized row for callers (CLI, AgenticCore resume)."""

    run_id: str
    step_id: int
    ts: str
    agent_type: str
    payload_json: str
    signal: str | None
    pinned: bool
    expires_at: str
    # ADR-0106: which conversation this run belonged to. Nullable — rows written
    # before the column existed, and runs with no session (CLI, heartbeats), have
    # none. A checkpoint is still keyed by (run_id, step_id); this only answers
    # "what was this session doing?".
    session_id: str | None = None

    @property
    def payload(self) -> dict[str, Any]:
        try:
            data = json.loads(self.payload_json)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


class CheckpointStore:
    """Append-with-overwrite SQLite store for resume checkpoints."""

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        ttl: timedelta = DEFAULT_TTL,
    ) -> None:
        self.db_path = db_path or default_checkpoint_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl
        self._init_schema()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        *,
        run_id: str,
        step_id: int,
        agent_type: str,
        payload: dict[str, Any],
        signal: str | None = None,
        ts: datetime | None = None,
        session_id: str | None = None,
    ) -> Checkpoint:
        payload_json = json.dumps(payload, default=str, sort_keys=True)
        if len(payload_json.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise CheckpointTooLargeError(
                f"checkpoint payload exceeds {MAX_PAYLOAD_BYTES} bytes "
                f"(run_id={run_id} step_id={step_id})"
            )

        now = ts or datetime.now(UTC)
        expires_at = now + self._ttl
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints(
                    run_id, step_id, ts, agent_type, payload_json,
                    signal, pinned, expires_at, session_id
                )
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(run_id, step_id) DO UPDATE SET
                    ts=excluded.ts,
                    agent_type=excluded.agent_type,
                    payload_json=excluded.payload_json,
                    signal=excluded.signal,
                    expires_at=excluded.expires_at,
                    session_id=excluded.session_id
                """,
                (
                    run_id,
                    step_id,
                    now.isoformat(),
                    agent_type,
                    payload_json,
                    signal,
                    expires_at.isoformat(),
                    session_id,
                ),
            )
            conn.commit()
        return self.get(run_id=run_id, step_id=step_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, *, run_id: str, step_id: int) -> Checkpoint:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE run_id = ? AND step_id = ?",
                (run_id, step_id),
            ).fetchone()
        if row is None:
            raise CheckpointNotFoundError(f"no checkpoint for run={run_id!r} step={step_id}")
        return _row_to_checkpoint(row)

    def get_latest(self, run_id: str) -> Checkpoint:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE run_id = ?
                ORDER BY step_id DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            raise CheckpointNotFoundError(f"no checkpoints for run={run_id!r}")
        return _row_to_checkpoint(row)

    def by_session(
        self,
        session_id: str,
        *,
        include_expired: bool = False,
        now: datetime | None = None,
    ) -> tuple[Checkpoint, ...]:
        """Checkpoints written by ``session_id``, newest first (ADR-0106).

        Answers "what was this conversation doing?" — the question the intercept
        chain could not ask while checkpoints were keyed by ``run_id`` alone.

        TTL is enforced **on read**: an expired, unpinned row is not returned even
        if ``sweep_expired`` has not run yet. The organize-plan incident (PR #412)
        came from the opposite arrangement, where a store's TTL was only honoured
        wherever some caller remembered to expire first, so a months-dead row still
        read as live on the path that forgot.

        An empty session id matches nothing: a caller with no session must not
        collect the rows of runs that had none.
        """
        if not session_id:
            return ()
        clauses = ["session_id = ?"]
        params: list[Any] = [session_id]
        if not include_expired:
            clauses.append("(pinned = 1 OR expires_at >= ?)")
            params.append((now or datetime.now(UTC)).isoformat())
        sql = f"SELECT * FROM checkpoints WHERE {' AND '.join(clauses)} ORDER BY ts DESC"  # noqa: S608
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return tuple(_row_to_checkpoint(row) for row in rows)

    def list(
        self, *, agent_type: str | None = None, include_expired: bool = False
    ) -> tuple[Checkpoint, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if agent_type is not None:
            clauses.append("agent_type = ?")
            params.append(agent_type)
        if not include_expired:
            clauses.append("(pinned = 1 OR expires_at >= ?)")
            params.append(datetime.now(UTC).isoformat())
        sql = "SELECT * FROM checkpoints"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts DESC"
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return tuple(_row_to_checkpoint(row) for row in rows)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def pin(self, run_id: str) -> int:
        """Pin every checkpoint for ``run_id``. Returns rows updated."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE checkpoints SET pinned = 1 WHERE run_id = ?",
                (run_id,),
            )
            conn.commit()
        return int(cur.rowcount or 0)

    def unpin(self, run_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE checkpoints SET pinned = 0 WHERE run_id = ?",
                (run_id,),
            )
            conn.commit()
        return int(cur.rowcount or 0)

    def remove(self, run_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM checkpoints WHERE run_id = ?",
                (run_id,),
            )
            conn.commit()
        return int(cur.rowcount or 0)

    def sweep_expired(self, *, now: datetime | None = None) -> int:
        """Delete unpinned, expired rows. Returns rows deleted."""
        cutoff = (now or datetime.now(UTC)).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM checkpoints WHERE pinned = 0 AND expires_at < ?",
                (cutoff,),
            )
            conn.commit()
        return int(cur.rowcount or 0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = connect(self.db_path)  # WAL + busy_timeout; the caller still commits
        try:
            try:
                os.chmod(self.db_path, 0o600)
            except OSError as exc:  # pragma: no cover - non-POSIX or perms issue
                logger.warning("could not chmod %s to 0o600: %s", self.db_path, exc)
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    run_id        TEXT NOT NULL,
                    step_id       INTEGER NOT NULL,
                    ts            TEXT NOT NULL,
                    agent_type    TEXT NOT NULL,
                    payload_json  TEXT NOT NULL,
                    signal        TEXT,
                    pinned        INTEGER NOT NULL DEFAULT 0,
                    expires_at    TEXT NOT NULL,
                    session_id    TEXT,
                    PRIMARY KEY (run_id, step_id)
                );
                CREATE INDEX IF NOT EXISTS idx_checkpoints_expires
                    ON checkpoints(expires_at);
                """)
            # A database written before ADR-0106 has no session_id, and CREATE TABLE
            # IF NOT EXISTS leaves it that way. Add the column before anything that
            # references it — indexing first fails on exactly the legacy databases
            # this migration exists for. Nullable, so pre-existing rows stay valid
            # and simply answer "no session" to by_session().
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(checkpoints)")}
            if "session_id" not in columns:
                conn.execute("ALTER TABLE checkpoints ADD COLUMN session_id TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON checkpoints(session_id)"
            )
            conn.commit()


def _row_to_checkpoint(row: sqlite3.Row) -> Checkpoint:
    session_id = row["session_id"] if "session_id" in row.keys() else None
    return Checkpoint(
        run_id=str(row["run_id"]),
        step_id=int(row["step_id"]),
        ts=str(row["ts"]),
        agent_type=str(row["agent_type"]),
        payload_json=str(row["payload_json"]),
        signal=row["signal"] if row["signal"] is None else str(row["signal"]),
        pinned=bool(row["pinned"]),
        expires_at=str(row["expires_at"]),
        session_id=None if session_id is None else str(session_id),
    )
