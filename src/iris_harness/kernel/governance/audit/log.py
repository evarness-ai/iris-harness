"""AuditLog — append-only SQLite store for every hook decision.

Schema from design §13.2. One row per ``kernel.fire()`` per
registered hook (allow / transform / deny / require_approval / hook
exception). WAL mode for concurrent reads while the agent writes.

The DB file is created ``0o600`` and the perm is reasserted on every
open — same defense-in-depth pattern as the vault store. Phase 3
ships the hot tier (30-day retention); the compaction sweep + Parquet
archive land in Phase 5.

This module is intentionally dependency-light: no Pydantic at the
write path (audit must not fail because a payload is exotic — we
JSON-encode with ``default=str``), and no async — writes are sync
inside the kernel's hook loop.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from iris_harness.foundation.ids import new_ulid
from iris_harness.foundation.persistence.sqlite import add_columns_if_missing
from iris_harness.kernel.governance.audit import sequence, spool

logger = logging.getLogger(__name__)

#: The identity columns of a row (issue #134, stage 3), all nullable: a row written before a
#: database was migrated has NULL in every one, which is how it is told apart as written in the
#: "pre-identity era". ``record_id`` is minted by the store; the rest are the identifiers the
#: kernel stamped into the payload (a closed set of ids and one count, never text).
IDENTITY_COLUMNS: dict[str, str] = {
    "record_id": "TEXT",
    "session_id": "TEXT",
    "turn_id": "TEXT",
    "call_id": "TEXT",
    "parent_call_id": "TEXT",
    "attempt": "INTEGER",
    "replay_of": "TEXT",
    "resumed_from_run": "TEXT",
}
_IDENTITY_INDEXES: tuple[str, ...] = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_audit_record_id ON audit_log(record_id) "
    "WHERE record_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_audit_call_id ON audit_log(call_id) "
    "WHERE call_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_audit_session_ts ON audit_log(session_id, ts) "
    "WHERE session_id IS NOT NULL",
)
#: ``audit_meta`` key for the boundary of the pre-identity era (see ``_record_boundary``).
IDENTITY_META_KEY = "identity"
IDENTITY_SCHEMA_VERSION = 2

#: The completeness columns (issue #134, stage 4): which writer wrote the row and its place in
#: that writer's sequence, and ``kind`` for the rows the store writes about itself (``gap``,
#: ``writer``, ``compaction``; NULL for the row of a hook firing). Nullable like the identity
#: columns: a row written before a database was migrated has NULL in all three.
SEQUENCE_COLUMNS: dict[str, str] = {
    "writer_id": "TEXT",
    "writer_seq": "INTEGER",
    "kind": "TEXT",
}
_SEQUENCE_INDEXES: tuple[str, ...] = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_audit_writer_seq ON audit_log(writer_id, writer_seq) "
    "WHERE writer_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_audit_kind ON audit_log(kind) WHERE kind IS NOT NULL",
)
#: ``audit_meta`` key for the boundary of the pre-sequence era (see ``_record_sequence_boundary``).
SEQUENCE_META_KEY = "sequence"
SEQUENCE_SCHEMA_VERSION = 1

#: Where a row went: the database, or (when it would not take it) the local spool.
Durability = Literal["db", "spool"]


# Override slot (``None``: resolved on every use, never frozen at import -- a process
# that relocates the home after importing IRIS writes into the new one's ledger).
DEFAULT_AUDIT_DB_PATH: Path | None = None


def _default_audit_db_path() -> Path:
    """Resolve the audit DB path (the ``DEFAULT_AUDIT_DB_PATH`` override, when set).

    Production (nothing set): ``~/.local/share/iris/audit.db``. ``IRIS_GOVERNANCE_AUDIT_DB_PATH``
    overrides it explicitly. ``IRIS_HOME`` (set to a throwaway temp dir by the test
    conftest before any import) relocates it under that home — so the suite never
    appends to the developer's real governance ledger via a bare ``AuditLog()``
    (which several call sites use), the same leak class as the session log.
    """
    if DEFAULT_AUDIT_DB_PATH is not None:
        return DEFAULT_AUDIT_DB_PATH
    from iris_harness.foundation.paths import audit_db_path

    return audit_db_path()


@dataclass(frozen=True)
class AuditRow:
    """One audit_log row, materialized for callers (CLI / tests)."""

    id: int
    ts: str
    run_id: str
    step_id: int | None
    agent_type: str
    hook_point: str
    plugin: str
    decision: str
    classification: str | None
    tier: str | None
    cost_usd: float | None
    severity: str
    reason: str
    payload_json: str
    # Identity (issue #134, stage 3). None on a row written before the database was migrated
    # (the "pre-identity era"); ``session_id`` falls back to the payload's copy, because old
    # rows are never rewritten.
    record_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    call_id: str | None = None
    parent_call_id: str | None = None
    attempt: int | None = None
    replay_of: str | None = None
    resumed_from_run: str | None = None
    # Completeness (issue #134, stage 4). None on a row written before the sequence existed.
    writer_id: str | None = None
    writer_seq: int | None = None
    kind: str | None = None


class AuditLog:
    """Append-only audit store with simple filtered queries."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _default_audit_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        sequence.set_close_hook(_close_writer)
        self.drain_spool()

    def record(
        self,
        *,
        run_id: str,
        step_id: int | None,
        agent_type: str,
        hook_point: str,
        plugin: str,
        decision: str,
        severity: str,
        reason: str,
        classification: str | None = None,
        tier: str | None = None,
        cost_usd: float | None = None,
        payload: dict[str, Any] | None = None,
        ts: datetime | None = None,
    ) -> int:
        """Append one row. Returns the assigned ``id`` (``0`` when the spool took the row).

        Audit writes must not fail because of an exotic payload. JSON
        encoding uses ``default=str`` and falls back to ``{}`` on a
        rare encoding error (logged at WARN).

        When the database will not take the row it goes to the local spool instead (see
        ``audit.spool``) and the call returns ``0``; it raises only when the spool cannot
        keep the row either.
        """
        _, row_id = self._write(
            dict(  # noqa: C408
                run_id=run_id,
                step_id=step_id,
                agent_type=agent_type,
                hook_point=hook_point,
                plugin=plugin,
                decision=decision,
                severity=severity,
                reason=reason,
                classification=classification,
                tier=tier,
                cost_usd=cost_usd,
                payload=payload,
                ts=ts,
            )
        )
        return row_id

    def _write(self, row: dict[str, Any], *, kind: str | None = None) -> tuple[Durability, int]:
        """Number the row, write it (with the writer's start and any gap rows), or spool it."""
        writer, seq = sequence.allocate(self.db_path)
        row["ts"] = row.get("ts") or datetime.now(UTC)
        record_id = new_ulid()
        claimed: list[sequence.GapRange] = []
        extra: list[int] = []
        try:
            with self._connect() as conn:
                row_id = _insert(
                    conn,
                    **row,
                    record_id=record_id,
                    writer_id=writer.writer_id,
                    writer_seq=seq,
                    kind=kind,
                )
                claimed = self._write_preamble(conn, writer, extra)
                conn.commit()
        except Exception as exc:  # noqa: BLE001 - any database failure spools
            sequence.restore_gaps(self.db_path, writer, claimed)
            for lost_seq in extra:  # a gap row's own number: a hole too, but not a lost event
                sequence.note_failure(self.db_path, writer, lost_seq, exc, spooled=False, own=True)
            return self._spool_or_raise(row, record_id, writer, seq, kind, exc), 0
        sequence.mark_started(self.db_path, writer)
        self._after_write()
        return "db", row_id

    def _write_preamble(
        self, conn: sqlite3.Connection, writer: sequence.Writer, allocated: list[int]
    ) -> list[sequence.GapRange]:
        """In the row's transaction: the writer's ``writer.start`` and the gaps it owes.

        ``allocated`` collects the sequence numbers taken for gap rows, so a write that then
        fails can report them as holes. A gap that cannot be written is handed back.
        """
        if not writer.started:
            _insert_meta(
                conn,
                writer,
                seq=sequence.START_SEQ,
                kind=sequence.KIND_WRITER,
                hook_point=sequence.WRITER_START,
                reason="this process began writing to the ledger",
                payload={"pid": os.getpid()},
            )
        gaps = sequence.claim_gaps(self.db_path, writer)
        try:
            for gap in gaps:
                _, gap_seq = sequence.allocate(self.db_path)
                allocated.append(gap_seq)
                _insert_meta(
                    conn,
                    writer,
                    seq=gap_seq,
                    kind=sequence.KIND_GAP,
                    hook_point="audit.gap",
                    reason="audit rows of this writer did not reach the ledger when written",
                    payload=gap.as_payload(),
                    severity="error" if gap.lost else "warning",
                )
        except BaseException:
            sequence.restore_gaps(self.db_path, writer, gaps)
            raise
        return gaps

    def _spool_or_raise(
        self,
        row: Mapping[str, Any],
        record_id: str,
        writer: sequence.Writer,
        seq: int,
        kind: str | None,
        exc: BaseException,
    ) -> Durability:
        """The database failed: keep the row in the spool, or re-raise when it cannot."""
        try:
            spool.append(
                spool.spool_path_for(self.db_path),
                _spool_record(row, record_id, writer.writer_id, seq, kind),
            )
        except Exception as spool_exc:
            sequence.note_failure(self.db_path, writer, seq, exc, spooled=False)
            logger.error(
                "audit_log: row %s/%d lost: the database (%s) and the spool (%s) both failed",
                writer.writer_id,
                seq,
                exc.__class__.__name__,
                spool_exc.__class__.__name__,
            )
            raise exc from spool_exc
        sequence.note_failure(self.db_path, writer, seq, exc, spooled=True)
        logger.warning(
            "audit_log: row %s/%d kept in the spool: %s",
            writer.writer_id,
            seq,
            exc.__class__.__name__,
        )
        return "spool"

    def _after_write(self) -> None:
        if spool.spool_path_for(self.db_path).exists():
            self.drain_spool()

    def drain_spool(self) -> int:
        """Put the rows waiting in the spool into the database. Never raises; returns how many."""
        path = spool.spool_path_for(self.db_path)
        if not path.exists():
            return 0
        try:
            return spool.drain(path, self._apply_spooled)
        except Exception as exc:  # noqa: BLE001 - a broken spool must not break a write
            logger.warning("audit_log: spool drain failed: %s", exc.__class__.__name__)
            return 0

    def _apply_spooled(self, records: list[dict[str, Any]]) -> bool:
        """Insert spooled rows idempotently. ``False`` leaves the spool as it was."""
        try:
            with self._connect() as conn:
                for rec in records:
                    _insert_spooled(conn, rec)
                conn.commit()
        except sqlite3.Error:
            return False
        return True

    def spool_state(self) -> spool.SpoolState:
        return spool.state(spool.spool_path_for(self.db_path))

    def record_many(self, rows: Sequence[Mapping[str, Any]]) -> list[int | None]:
        """Append several rows in one connection, each succeeding or failing on its own.

        ``rows`` are the keyword arguments of :meth:`record`. A row that cannot be written
        (an exotic value SQLite refuses) is rolled back alone and reported as ``None`` in
        the result, in order; the others are kept. That is per-row failure semantics: a
        failed row is neither hidden nor does it take the rest with it. (Atomic-per-firing is
        a separate decision.) Each row is numbered like a single write, and a failed row
        leaves its hole.
        """
        ids: list[int | None] = []
        writer = sequence.writer_for(self.db_path)
        claimed: list[sequence.GapRange] = []
        extra: list[int] = []
        try:
            with self._connect() as conn:
                claimed = self._write_preamble(conn, writer, extra)
                for n, row in enumerate(rows):
                    _, seq = sequence.allocate(self.db_path)
                    fields = dict(row)
                    fields["ts"] = fields.get("ts") or datetime.now(UTC)
                    record_id = new_ulid()
                    conn.execute("SAVEPOINT audit_row")
                    try:
                        ids.append(
                            _insert(
                                conn,
                                **fields,
                                record_id=record_id,
                                writer_id=writer.writer_id,
                                writer_seq=seq,
                            )
                        )
                    except (sqlite3.Error, TypeError, ValueError) as exc:
                        conn.execute("ROLLBACK TO audit_row")
                        logger.warning("audit_log: row %d of a batch failed to write: %s", n, exc)
                        ids.append(None)
                        try:
                            self._spool_or_raise(fields, record_id, writer, seq, None, exc)
                        except Exception:
                            logger.debug("batch row %d not spooled", n, exc_info=True)
                    finally:
                        conn.execute("RELEASE audit_row")
                conn.commit()
        except Exception as exc:
            sequence.restore_gaps(self.db_path, writer, claimed)
            for lost_seq in extra:
                sequence.note_failure(self.db_path, writer, lost_seq, exc, spooled=False, own=True)
            raise
        sequence.mark_started(self.db_path, writer)
        self._after_write()
        return ids

    def query(
        self,
        *,
        run_id: str | None = None,
        decision: str | None = None,
        severity: str | None = None,
        plugin: str | None = None,
        caller: str | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        limit: int | None = None,
        include_store_rows: bool = False,
    ) -> tuple[AuditRow, ...]:
        """Return rows matching the filter, ordered by ``ts`` ascending.

        The store's own rows (``writer.start`` / ``writer.close``, ``gap``, ``compaction``)
        are left out unless ``include_store_rows``: they describe the ledger, not a decision.

        ``caller`` matches the payload's ``caller`` exactly, or, when it ends in ``:``,
        by that namespace (``mcp:`` is every MCP client, ``plugin:`` every plugin's code).
        """
        clauses: list[str] = [] if include_store_rows else ["kind IS NULL"]
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if decision is not None:
            clauses.append("decision = ?")
            params.append(decision)
        if severity is not None:
            clauses.append("severity = ?")
            params.append(severity)
        if plugin is not None:
            clauses.append("plugin = ?")
            params.append(plugin)
        if caller is not None:
            if caller.endswith(":"):
                clauses.append("substr(json_extract(payload_json, '$.caller'), 1, ?) = ?")
                params.extend([len(caller), caller])
            else:
                clauses.append("json_extract(payload_json, '$.caller') = ?")
                params.append(caller)
        if since is not None:
            clauses.append("ts >= ?")
            params.append(_iso(since))
        if until is not None:
            clauses.append("ts <= ?")
            params.append(_iso(until))

        sql = "SELECT * FROM audit_log"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts ASC, id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return tuple(_row_to_audit(row) for row in rows)

    def callers(self) -> tuple[str, ...]:
        """Every distinct ``caller`` the ledger's payloads name, sorted."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT json_extract(payload_json, '$.caller') FROM audit_log "
                "WHERE json_extract(payload_json, '$.caller') IS NOT NULL"
            ).fetchall()
        return tuple(sorted(str(value) for (value,) in rows if isinstance(value, str) and value))

    def count(self, *, include_store_rows: bool = False) -> int:
        sql = "SELECT COUNT(*) FROM audit_log"
        if not include_store_rows:
            sql += " WHERE kind IS NULL"
        with self._connect() as conn:
            (n,) = conn.execute(sql).fetchone()
        return int(n)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            try:
                os.chmod(self.db_path, 0o600)
            except OSError as exc:  # pragma: no cover - non-POSIX or perms issue
                logger.warning("could not chmod %s to 0o600: %s", self.db_path, exc)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN")
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts              TEXT NOT NULL,
                    run_id          TEXT NOT NULL,
                    step_id         INTEGER,
                    agent_type      TEXT NOT NULL,
                    hook_point      TEXT NOT NULL,
                    plugin          TEXT NOT NULL,
                    decision        TEXT NOT NULL,
                    classification  TEXT,
                    tier            TEXT,
                    cost_usd        REAL,
                    severity        TEXT NOT NULL,
                    reason          TEXT NOT NULL,
                    payload_json    TEXT NOT NULL,
                    record_id        TEXT,
                    session_id       TEXT,
                    turn_id          TEXT,
                    call_id          TEXT,
                    parent_call_id   TEXT,
                    attempt          INTEGER,
                    replay_of        TEXT,
                    resumed_from_run TEXT,
                    writer_id        TEXT,
                    writer_seq       INTEGER,
                    kind             TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_audit_run_ts ON audit_log(run_id, ts);
                CREATE INDEX IF NOT EXISTS idx_audit_decision_ts ON audit_log(decision, ts);
                CREATE TABLE IF NOT EXISTS audit_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """)
            conn.commit()
        # A database a released version created has the table without the identity columns:
        # add them on a connection of their own (never inside ``_connect``'s transaction, see
        # ``add_columns_if_missing``). Only the process that adds them records the boundary.
        add_columns_if_missing(
            self.db_path,
            "audit_log",
            IDENTITY_COLUMNS,
            indexes=_IDENTITY_INDEXES,
            on_added=_record_boundary,
        )
        add_columns_if_missing(
            self.db_path,
            "audit_log",
            SEQUENCE_COLUMNS,
            indexes=_SEQUENCE_INDEXES,
            on_added=_record_sequence_boundary,
        )


def _record_boundary(conn: sqlite3.Connection, added: list[str]) -> None:
    """Write the pre-identity boundary, in the transaction that added the columns.

    Every row up to the largest ``id`` now present was written before identity existed;
    rows after it carry a ``record_id`` (a row without one is pre-identity, whatever its id:
    an older process may still be writing). Old rows are not touched (never backfilled).
    A fresh database has no such era and records nothing.
    """
    if "record_id" not in added:
        return
    (last_id,) = conn.execute("SELECT COALESCE(MAX(id), 0) FROM audit_log").fetchone()
    boundary = {
        "schema": IDENTITY_SCHEMA_VERSION,
        "first_identity_row_id": int(last_id) + 1,
        "migrated_at": datetime.now(UTC).isoformat(),
    }
    conn.execute(
        "CREATE TABLE IF NOT EXISTS audit_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT OR IGNORE INTO audit_meta(key, value) VALUES (?, ?)",
        (IDENTITY_META_KEY, json.dumps(boundary, sort_keys=True)),
    )


def _identity_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The identity columns' values from a payload the kernel stamped (ids and one count).

    A value of the wrong type is dropped, never coerced: the columns hold identifiers and
    nothing else, so no text from an argument can land in one.
    """
    out: dict[str, Any] = {}
    for name, declaration in IDENTITY_COLUMNS.items():
        if name == "record_id":
            continue
        value = payload.get(name)
        if declaration == "INTEGER":
            ok = isinstance(value, int) and not isinstance(value, bool)
        else:
            ok = isinstance(value, str) and bool(value)
        out[name] = value if ok else None
    return out


def _insert(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    step_id: int | None,
    agent_type: str,
    hook_point: str,
    plugin: str,
    decision: str,
    severity: str,
    reason: str,
    classification: str | None = None,
    tier: str | None = None,
    cost_usd: float | None = None,
    payload: Mapping[str, Any] | None = None,
    ts: datetime | str | None = None,
    record_id: str | None = None,
    writer_id: str | None = None,
    writer_seq: int | None = None,
    kind: str | None = None,
    or_ignore: bool = False,
) -> int:
    payload_dict = dict(payload or {})
    ids = _identity_values(payload_dict)
    stamp = ts.isoformat() if isinstance(ts, datetime) else (ts or datetime.now(UTC).isoformat())
    cur = conn.execute(
        f"""
        INSERT {"OR IGNORE " if or_ignore else ""}INTO audit_log(
            ts, run_id, step_id, agent_type, hook_point, plugin,
            decision, classification, tier, cost_usd, severity,
            reason, payload_json,
            record_id, session_id, turn_id, call_id, parent_call_id,
            attempt, replay_of, resumed_from_run,
            writer_id, writer_seq, kind
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,  # noqa: S608 - the only interpolation is the fixed OR IGNORE keyword
        (
            stamp,
            run_id,
            step_id,
            agent_type,
            hook_point,
            plugin,
            decision,
            classification,
            tier,
            cost_usd,
            severity,
            reason,
            _safe_json(payload_dict),
            record_id or new_ulid(),
            ids["session_id"],
            ids["turn_id"],
            ids["call_id"],
            ids["parent_call_id"],
            ids["attempt"],
            ids["replay_of"],
            ids["resumed_from_run"],
            writer_id,
            writer_seq,
            kind,
        ),
    )
    return int(cur.lastrowid or 0) if cur.rowcount else 0


def _insert_meta(
    conn: sqlite3.Connection,
    writer: sequence.Writer,
    *,
    seq: int,
    kind: str,
    hook_point: str,
    reason: str,
    payload: Mapping[str, Any],
    severity: str = "info",
) -> None:
    """A row the store writes about itself (the writer's start or close, a gap).

    ``OR IGNORE``: two threads of one process may both write ``writer.start``; the unique
    ``(writer_id, writer_seq)`` index keeps one.
    """
    _insert(
        conn,
        run_id=f"audit:{writer.writer_id}",
        step_id=None,
        agent_type="audit",
        hook_point=hook_point,
        plugin="audit",
        decision="allow",
        severity=severity,
        reason=reason,
        payload={"writer_id": writer.writer_id, **payload},
        record_id=new_ulid(),
        writer_id=writer.writer_id,
        writer_seq=seq,
        kind=kind,
        or_ignore=True,
    )


def _spool_record(
    row: Mapping[str, Any], record_id: str, writer_id: str, seq: int, kind: str | None
) -> dict[str, Any]:
    """The closed set of fields a spool line carries (see ``audit.spool``)."""
    ts = row.get("ts")
    return {
        "record_id": record_id,
        "writer_id": writer_id,
        "writer_seq": seq,
        "kind": kind,
        "ts": ts.isoformat() if isinstance(ts, datetime) else str(ts),
        "run_id": row["run_id"],
        "step_id": row.get("step_id"),
        "agent_type": row["agent_type"],
        "hook_point": row["hook_point"],
        "plugin": row["plugin"],
        "decision": row["decision"],
        "classification": row.get("classification"),
        "tier": row.get("tier"),
        "cost_usd": row.get("cost_usd"),
        "severity": row["severity"],
        "reason": row["reason"],
        "payload": json.loads(_safe_json(dict(row.get("payload") or {}))),
    }


def _insert_spooled(conn: sqlite3.Connection, rec: Mapping[str, Any]) -> None:
    """Replay one validated spool line. A row the ledger already holds is left untouched."""
    _insert(
        conn,
        run_id=rec["run_id"],
        step_id=rec["step_id"],
        agent_type=rec["agent_type"],
        hook_point=rec["hook_point"],
        plugin=rec["plugin"],
        decision=rec["decision"],
        severity=rec["severity"],
        reason=rec["reason"],
        classification=rec["classification"],
        tier=rec["tier"],
        cost_usd=rec["cost_usd"],
        payload=rec["payload"],
        ts=rec["ts"],
        record_id=rec["record_id"],
        writer_id=rec["writer_id"],
        writer_seq=rec["writer_seq"],
        kind=rec["kind"],
        or_ignore=True,
    )


def _record_sequence_boundary(conn: sqlite3.Connection, added: list[str]) -> None:
    """Write the pre-sequence boundary, in the transaction that added the columns.

    Rows up to the largest ``id`` now present were written before rows were numbered; a row
    without a ``writer_seq`` after it came from a writer that does not number (an older
    release still running). Old rows are not touched.
    """
    if "writer_seq" not in added:
        return
    (last_id,) = conn.execute("SELECT COALESCE(MAX(id), 0) FROM audit_log").fetchone()
    boundary = {
        "schema": SEQUENCE_SCHEMA_VERSION,
        "first_sequenced_row_id": int(last_id) + 1,
        "migrated_at": datetime.now(UTC).isoformat(),
    }
    conn.execute(
        "CREATE TABLE IF NOT EXISTS audit_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT OR IGNORE INTO audit_meta(key, value) VALUES (?, ?)",
        (SEQUENCE_META_KEY, json.dumps(boundary, sort_keys=True)),
    )


def _close_writer(db_key: str, writer: sequence.Writer) -> None:
    """``atexit``: write ``writer.close`` (with the last number used) if the ledger is there."""
    if not os.path.exists(db_key):
        return
    log = AuditLog(db_path=Path(db_key))
    last = writer.next_seq - 1
    log._write(
        dict(  # noqa: C408
            run_id=f"audit:{writer.writer_id}",
            step_id=None,
            agent_type="audit",
            hook_point=sequence.WRITER_CLOSE,
            plugin="audit",
            decision="allow",
            severity="info",
            reason="this process finished writing to the ledger",
            payload={"writer_id": writer.writer_id, "last_seq": last, "pid": os.getpid()},
        ),
        kind=sequence.KIND_WRITER,
    )


def _row_to_audit(row: sqlite3.Row) -> AuditRow:
    return AuditRow(
        id=int(row["id"]),
        ts=str(row["ts"]),
        run_id=str(row["run_id"]),
        step_id=row["step_id"] if row["step_id"] is None else int(row["step_id"]),
        agent_type=str(row["agent_type"]),
        hook_point=str(row["hook_point"]),
        plugin=str(row["plugin"]),
        decision=str(row["decision"]),
        classification=(
            row["classification"] if row["classification"] is None else str(row["classification"])
        ),
        tier=row["tier"] if row["tier"] is None else str(row["tier"]),
        cost_usd=row["cost_usd"] if row["cost_usd"] is None else float(row["cost_usd"]),
        severity=str(row["severity"]),
        reason=str(row["reason"]),
        payload_json=str(row["payload_json"]),
        **_identity_of(row),
        **{name: (row[name] if name in set(row.keys()) else None) for name in SEQUENCE_COLUMNS},
    )


def _identity_of(row: sqlite3.Row) -> dict[str, Any]:
    """The identity columns of ``row``; absent (a connection to an old file) reads as None.

    ``session_id`` falls back to the payload's own copy: rows from before the column existed
    are not backfilled (#134 D6), and they still say which session they belong to.
    """
    keys = set(row.keys())
    out: dict[str, Any] = {name: (row[name] if name in keys else None) for name in IDENTITY_COLUMNS}
    if out["session_id"] is None:
        try:
            payload = json.loads(row["payload_json"])
        except ValueError:
            payload = None
        copy = payload.get("session_id") if isinstance(payload, dict) else None
        out["session_id"] = copy if isinstance(copy, str) and copy else None
    return out


def _iso(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _safe_json(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(payload, default=str, sort_keys=True)
    except Exception as exc:  # noqa: BLE001 - audit must never crash the kernel
        logger.warning("audit_log: failed to JSON-encode payload (%s); writing {}", exc)
        return "{}"
