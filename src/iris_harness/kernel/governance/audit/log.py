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
from typing import Any

from iris_harness.foundation.ids import new_ulid
from iris_harness.foundation.persistence.sqlite import add_columns_if_missing

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


class AuditLog:
    """Append-only audit store with simple filtered queries."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _default_audit_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

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
        """Append one row. Returns the assigned ``id``.

        Audit writes must not fail because of an exotic payload. JSON
        encoding uses ``default=str`` and falls back to ``{}`` on a
        rare encoding error (logged at WARN).
        """
        with self._connect() as conn:
            row_id = _insert(
                conn,
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
            conn.commit()
            return row_id

    def record_many(self, rows: Sequence[Mapping[str, Any]]) -> list[int | None]:
        """Append several rows in one connection, each succeeding or failing on its own.

        ``rows`` are the keyword arguments of :meth:`record`. A row that cannot be written
        (an exotic value SQLite refuses) is rolled back alone and reported as ``None`` in
        the result, in order; the others are kept. That is per-row failure semantics: a
        failed row is neither hidden nor does it take the rest with it. (Atomic-per-firing is
        a separate decision, #134 stage 4.)
        """
        ids: list[int | None] = []
        with self._connect() as conn:
            for n, row in enumerate(rows):
                conn.execute("SAVEPOINT audit_row")
                try:
                    ids.append(_insert(conn, **row))
                except (sqlite3.Error, TypeError, ValueError) as exc:
                    conn.execute("ROLLBACK TO audit_row")
                    logger.warning("audit_log: row %d of a batch failed to write: %s", n, exc)
                    ids.append(None)
                finally:
                    conn.execute("RELEASE audit_row")
            conn.commit()
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
    ) -> tuple[AuditRow, ...]:
        """Return rows matching the filter, ordered by ``ts`` ascending.

        ``caller`` matches the payload's ``caller`` exactly, or, when it ends in ``:``,
        by that namespace (``mcp:`` is every MCP client, ``plugin:`` every plugin's code).
        """
        clauses: list[str] = []
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

    def count(self) -> int:
        with self._connect() as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()
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
                    resumed_from_run TEXT
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
    ts: datetime | None = None,
) -> int:
    payload_dict = dict(payload or {})
    ids = _identity_values(payload_dict)
    cur = conn.execute(
        """
        INSERT INTO audit_log(
            ts, run_id, step_id, agent_type, hook_point, plugin,
            decision, classification, tier, cost_usd, severity,
            reason, payload_json,
            record_id, session_id, turn_id, call_id, parent_call_id,
            attempt, replay_of, resumed_from_run
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (ts or datetime.now(UTC)).isoformat(),
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
            new_ulid(),
            ids["session_id"],
            ids["turn_id"],
            ids["call_id"],
            ids["parent_call_id"],
            ids["attempt"],
            ids["replay_of"],
            ids["resumed_from_run"],
        ),
    )
    return int(cur.lastrowid or 0)


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
