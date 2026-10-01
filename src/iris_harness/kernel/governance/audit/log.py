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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
        payload_json = _safe_json(payload or {})
        ts_iso = (ts or datetime.now(UTC)).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO audit_log(
                    ts, run_id, step_id, agent_type, hook_point, plugin,
                    decision, classification, tier, cost_usd, severity,
                    reason, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_iso,
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
                    payload_json,
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

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
                    payload_json    TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_run_ts ON audit_log(run_id, ts);
                CREATE INDEX IF NOT EXISTS idx_audit_decision_ts ON audit_log(decision, ts);
                """)
            conn.commit()


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
    )


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
