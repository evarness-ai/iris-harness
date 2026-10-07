"""SideEffectLedger — SQLite-backed ledger of irreversible tool side-effects.

Story 12.gov-4.10 / design §10.3.

Each row records one side effect. A plain write is recorded by the PostToolUse hook, after
the call. A high-risk call (destructive, or a write the owner approves per call) is
written *before* it runs by the PreToolUse hook as a ``pending`` row, and the PostToolUse
hook finalises that same row (``finalize``), so a process that dies mid-call leaves
evidence of the attempt. At resume time, ``pending(run_id)`` returns rows whose probes haven't
confirmed completion yet; the resume flow then decides to skip,
re-execute, or enqueue an approval.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_harness.foundation.ids import new_ulid
from iris_harness.foundation.paths import governance_data_dir
from iris_harness.foundation.persistence.sqlite import add_columns_if_missing

logger = logging.getLogger(__name__)

#: Identity columns added to a ledger a released version created (issue #134, stage 3).
IDENTITY_COLUMNS: dict[str, str] = {
    "call_id": "TEXT",
    "parent_call_id": "TEXT",
    "attempt": "INTEGER",
    "replay_of": "TEXT",
    "record_id": "TEXT",
}

# Override slot (``None``: resolved from ``IRIS_HOME`` on every use, never frozen at
# import -- a process that relocates the home after importing IRIS writes into the
# new one). Set it to point the default somewhere else outright (tests).
DEFAULT_LEDGER_DB_PATH: Path | None = None


def default_ledger_db_path() -> Path:
    """``DEFAULT_LEDGER_DB_PATH`` when set, else ``<governance data dir>/side_effects.db``."""
    return (
        DEFAULT_LEDGER_DB_PATH
        if DEFAULT_LEDGER_DB_PATH is not None
        else governance_data_dir() / "side_effects.db"
    )


_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

ProbeStatus = str  # "pending" | "completed" | "not_completed" | "ambiguous" | "error"


class SideEffectKeyExists(Exception):
    """An exclusive ``record`` found its key already holding a row (nothing was written).

    Carries the key only: it is a call's id, never an argument or a result.
    """

    def __init__(self, side_effect_id: str) -> None:
        super().__init__(side_effect_id)
        self.side_effect_id = side_effect_id


@dataclass(frozen=True)
class SideEffectRow:
    side_effect_id: str
    run_id: str
    step_id: int
    tool: str
    verification_probe: str
    probe_metadata: dict[str, Any]
    status: ProbeStatus
    completed_at: str | None
    error: str | None

    @property
    def probe_subject(self) -> str:
        """What the probe checks: the effect's own id (a commit SHA, a PR URL) when the
        recorder knew it (``probe_metadata["subject"]``), else the row's key."""
        subject = self.probe_metadata.get("subject")
        return subject if isinstance(subject, str) and subject else self.side_effect_id


class SideEffectLedger:
    """Open-or-create the side-effect ledger; enforces append-only semantics."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or default_ledger_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if not self.db_path.exists():
            fd = os.open(str(self.db_path), os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
        else:
            try:
                os.chmod(str(self.db_path), 0o600)
            except OSError as exc:  # pragma: no cover
                logger.warning("could not chmod %s to 0o600: %s", self.db_path, exc)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        schema = _SCHEMA_PATH.read_text(encoding="utf-8")
        # The base connection, never a subclass's: a lazily opening ``_connect`` must not be
        # re-entered from the very step that makes it ready.
        with SideEffectLedger._connect(self) as conn:
            conn.executescript(schema)
            conn.commit()
        # A ledger a released version created lacks the identity columns: add them on a
        # connection of their own, under ``BEGIN IMMEDIATE`` (see ``add_columns_if_missing``).
        add_columns_if_missing(self.db_path, "side_effect_ledger", IDENTITY_COLUMNS)

    @staticmethod
    def _row(r: sqlite3.Row) -> SideEffectRow:
        try:
            meta = json.loads(r["probe_metadata"] or "{}")
        except json.JSONDecodeError:
            meta = {}
        return SideEffectRow(
            side_effect_id=r["side_effect_id"],
            run_id=r["run_id"],
            step_id=r["step_id"],
            tool=r["tool"],
            verification_probe=r["verification_probe"],
            probe_metadata=meta if isinstance(meta, dict) else {},
            status=r["status"],
            completed_at=r["completed_at"],
            error=r["error"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        run_id: str,
        step_id: int,
        tool: str,
        verification_probe: str,
        probe_metadata: dict[str, Any] | None = None,
        side_effect_id: str | None = None,
        exclusive: bool = False,
        call_id: str | None = None,
        parent_call_id: str | None = None,
        attempt: int | None = None,
        replay_of: str | None = None,
    ) -> str:
        """Insert a new pending side-effect row; returns the side_effect_id.

        ``side_effect_id`` is the caller's key for the effect (the ledger hook's
        ``<run_id>:<step_id>:<tool_call_id>``). Without one, a fresh UUID is minted.

        A key that already has a row is, by default, left as it is and the call returns as
        if it had recorded: recording one effect twice is a no-op, but the caller cannot
        tell that nothing was written. A caller that must know passes ``exclusive=True``:
        a taken key raises ``SideEffectKeyExists`` and the existing row is untouched. The
        ledger hooks do (a row they believe they wrote must be theirs).

        ``call_id`` / ``parent_call_id`` / ``attempt`` / ``replay_of`` are the call's identity
        (#134): ids and a count the hooks read from the harness's own record of the call. The
        store mints the row's ``record_id`` and appends a ``pending`` event.
        """
        side_effect_id = side_effect_id or str(uuid.uuid4())
        meta_json = json.dumps(probe_metadata or {}, default=str, sort_keys=True)
        verb = "INSERT" if exclusive else "INSERT OR IGNORE"
        with self._connect() as conn:
            try:
                cur = conn.execute(
                    f"""
                    {verb} INTO side_effect_ledger
                        (side_effect_id, run_id, step_id, tool,
                         verification_probe, probe_metadata, status,
                         call_id, parent_call_id, attempt, replay_of, record_id)
                    VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                    """,
                    (
                        side_effect_id,
                        run_id,
                        step_id,
                        tool,
                        verification_probe,
                        meta_json,
                        call_id,
                        parent_call_id,
                        attempt,
                        replay_of,
                        new_ulid(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise SideEffectKeyExists(side_effect_id) from exc
            if cur.rowcount:  # INSERT OR IGNORE of a taken key wrote nothing: no event either
                _append_event(conn, side_effect_id, "pending", call_id)
            conn.commit()
        logger.debug(
            "side_effect recorded %s run=%s step=%d tool=%s probe=%s",
            side_effect_id,
            run_id,
            step_id,
            tool,
            verification_probe,
        )
        return side_effect_id

    def get(self, side_effect_id: str) -> SideEffectRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM side_effect_ledger WHERE side_effect_id = ?",
                (side_effect_id,),
            ).fetchone()
        return self._row(row) if row else None

    def pending(self, run_id: str) -> list[SideEffectRow]:
        """Return rows for run_id whose status is not 'completed'."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM side_effect_ledger
                WHERE run_id = ? AND status != 'completed'
                ORDER BY step_id, rowid
                """,
                (run_id,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def list_by_run(self, run_id: str) -> list[SideEffectRow]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM side_effect_ledger WHERE run_id = ? ORDER BY step_id, rowid",
                (run_id,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def set_status(
        self,
        side_effect_id: str,
        *,
        status: ProbeStatus,
        error: str | None = None,
    ) -> None:
        """Update a row's status (and optionally error)."""
        completed_at = datetime.now(UTC).isoformat() if status == "completed" else None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE side_effect_ledger
                   SET status = ?, completed_at = ?, error = ?
                 WHERE side_effect_id = ?
                """,
                (status, completed_at, error, side_effect_id),
            )
            _append_event(conn, side_effect_id, status, _call_id_of(conn, side_effect_id))
            conn.commit()

    def finalize(
        self,
        side_effect_id: str,
        *,
        status: ProbeStatus,
        error: str | None = None,
        verification_probe: str | None = None,
        probe_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Settle a row written before its call ran; ``False`` when there is no such row.

        ``status`` is the outcome (``completed`` / ``error``); ``error`` only ever names
        the exception class, never its message. ``verification_probe`` and
        ``probe_metadata`` replace the pre-call values when the call's result named the
        effect (a commit SHA, a PR URL): the metadata is merged over the row's own, so the
        pre-call keys survive.
        """
        completed_at = datetime.now(UTC).isoformat() if status == "completed" else None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT probe_metadata, verification_probe FROM side_effect_ledger "
                "WHERE side_effect_id = ?",
                (side_effect_id,),
            ).fetchone()
            if row is None:
                return False
            try:
                merged = json.loads(row["probe_metadata"] or "{}")
            except json.JSONDecodeError:
                merged = {}
            if not isinstance(merged, dict):
                merged = {}
            merged.update(probe_metadata or {})
            conn.execute(
                """
                UPDATE side_effect_ledger
                   SET status = ?, completed_at = ?, error = ?,
                       verification_probe = ?, probe_metadata = ?
                 WHERE side_effect_id = ?
                """,
                (
                    status,
                    completed_at,
                    error,
                    row["verification_probe"] if verification_probe is None else verification_probe,
                    json.dumps(merged, default=str, sort_keys=True),
                    side_effect_id,
                ),
            )
            _append_event(conn, side_effect_id, status, _call_id_of(conn, side_effect_id))
            conn.commit()
        return True


def _call_id_of(conn: sqlite3.Connection, side_effect_id: str) -> str | None:
    row = conn.execute(
        "SELECT call_id FROM side_effect_ledger WHERE side_effect_id = ?", (side_effect_id,)
    ).fetchone()
    return row[0] if row else None


def _append_event(
    conn: sqlite3.Connection, side_effect_id: str, status: str, call_id: str | None
) -> None:
    """Append one transition to ``side_effect_events`` (identifiers and a status only)."""
    conn.execute(
        "INSERT INTO side_effect_events(event_id, side_effect_id, status, ts, call_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (new_ulid(), side_effect_id, status, datetime.now(UTC).isoformat(), call_id),
    )


class DeferredSideEffectLedger(SideEffectLedger):
    """A ``SideEffectLedger`` that creates its database on first use, not at construction.

    The default ledger scope is the high-risk class only: most processes never run such a
    call, and a kernel is built many times per process. Opening eagerly would create the
    file and its schema (a SQLite commit) for all of them. Here the first read or write
    does; a failure to open surfaces then, from that call, so a high-risk call whose ledger
    will not open is denied (``PreToolUseLedgerHook``) and the next call tries again.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or default_ledger_db_path()
        self._opened = False
        self._open_lock = threading.Lock()

    def _open(self) -> None:
        """Create the directory and schema once; ``_opened`` is set only when both exist.

        Callers are serialised: the kernel runs tool calls on several threads, and one that
        arrived while another was still creating the schema would query a table that is not
        there yet. A failure leaves ``_opened`` unset, so the next call tries again.
        """
        if self._opened and self.db_path.exists():
            return
        with self._open_lock:
            # A file removed since the schema was made (a cleared data dir, a test's temp
            # dir) is a database that needs its schema again: this handle is shared.
            if self._opened and self.db_path.exists():
                return
            recreated = self._opened
            self._opened = False
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()
            self._opened = True
            if recreated:
                logger.warning(
                    "governance: the side-effect ledger database %s was removed while the "
                    "process was running and has been created again, empty. Any pending "
                    "write-ahead rows it held are gone, so resuming a halted run (iris run "
                    "resume) will not see them.",
                    self.db_path,
                )

    def open(self) -> None:
        """Create the database and its schema now (an eager caller; raises if it cannot)."""
        self._open()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._open()
        with super()._connect() as conn:
            yield conn


_SHARED: dict[Path, DeferredSideEffectLedger] = {}
_SHARED_LOCK = threading.Lock()


def shared_side_effect_ledger(db_path: Path | None = None) -> DeferredSideEffectLedger:
    """The process's one ledger for the database at ``db_path`` (default path when ``None``).

    Kernels are built at many sites of one process (the runtime, the ReAct handler, the
    API server, each model client). A ledger per kernel meant a schema script and a commit
    per kernel on its first use, and an open-lock (one instance only) that did not cover the
    others. The ledger holds no connection -- each operation opens and closes its own -- so
    what is shared is the one thing worth sharing: the database being known to exist with
    its schema, created once under one lock.

    Keyed by the resolved path, so a process that moves ``IRIS_HOME`` (or a test that gives
    each case its own directory) gets a ledger for the new file, never the old one. Deferred:
    nothing is created until ``open()`` or the first operation.
    """
    path = Path(os.path.abspath((db_path or default_ledger_db_path()).expanduser()))
    with _SHARED_LOCK:
        ledger = _SHARED.get(path)
        if ledger is None:
            ledger = _SHARED[path] = DeferredSideEffectLedger(path)
        return ledger
