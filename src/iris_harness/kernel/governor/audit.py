"""Append-only SQLite audit logging for IRIS governor decisions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from iris_harness.foundation.persistence.sqlite import sqlite_conn

from .exceptions import GovernorAuditError
from .models import GovernorAuditEntry, GovernorGuardDecision

DEFAULT_AUDIT_DB_PATH = Path("data/audit.db")


class GovernorAuditLogger:
    """Write immutable governor decisions to a local SQLite database.

    The database is created on first use, not on construction: the API builds a
    governor while it starts even with MCP off, and building one must not write a
    file (it used to leave ``data/audit.db`` in whatever directory imported the app).
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self._initialized = False

    @classmethod
    def from_repo_root(
        cls,
        repo_root: Path,
        *,
        db_path: Path | None = None,
    ) -> GovernorAuditLogger:
        """Construct an audit logger using the default IRIS audit location."""
        return cls((db_path or (repo_root / DEFAULT_AUDIT_DB_PATH)).resolve())

    def record_decision(self, decision: GovernorGuardDecision) -> None:
        """Append one guard decision to the audit store."""
        self._ensure_initialized()
        try:
            with sqlite_conn(self.db_path) as connection:
                connection.execute(
                    """
                    INSERT INTO governor_guard_audit (
                        route,
                        action,
                        allowed,
                        reason,
                        matched_policy,
                        requires_approval,
                        retry_after_seconds,
                        metadata_json,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        decision.route,
                        decision.action,
                        1 if decision.allowed else 0,
                        decision.reason,
                        decision.matched_policy,
                        1 if decision.requires_approval else 0,
                        decision.retry_after_seconds,
                        json.dumps(decision.metadata, sort_keys=True),
                        decision.created_at.isoformat(),
                    ),
                )
        except sqlite3.Error as exc:  # pragma: no cover - exercised through caller failure paths
            raise GovernorAuditError(f"unable to record governor decision: {exc}") from exc

    def list_entries(self, *, limit: int = 100) -> tuple[GovernorAuditEntry, ...]:
        """Return the oldest-to-newest audit records up to the requested limit."""
        self._ensure_initialized()
        with sqlite_conn(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, route, action, allowed, reason, matched_policy,
                       requires_approval, retry_after_seconds, metadata_json, created_at
                FROM governor_guard_audit
                ORDER BY id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(
            GovernorAuditEntry(
                event_id=int(row["id"]),
                route=str(row["route"]),
                action=str(row["action"]),
                allowed=bool(row["allowed"]),
                reason=str(row["reason"]),
                matched_policy=(
                    str(row["matched_policy"]) if row["matched_policy"] is not None else None
                ),
                requires_approval=bool(row["requires_approval"]),
                retry_after_seconds=(
                    int(row["retry_after_seconds"])
                    if row["retry_after_seconds"] is not None
                    else None
                ),
                metadata=json.loads(str(row["metadata_json"]) or "{}"),
                created_at=str(row["created_at"]),
            )
            for row in rows
        )

    def _ensure_initialized(self) -> None:
        # Every statement is IF NOT EXISTS, so two threads racing here is harmless.
        if not self._initialized:
            self._initialize()
            self._initialized = True

    def _initialize(self) -> None:
        """Create the append-only schema if it does not already exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite_conn(self.db_path) as connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS governor_guard_audit (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        route TEXT NOT NULL,
                        action TEXT NOT NULL,
                        allowed INTEGER NOT NULL,
                        reason TEXT NOT NULL,
                        matched_policy TEXT,
                        requires_approval INTEGER NOT NULL,
                        retry_after_seconds INTEGER,
                        metadata_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """)
                connection.execute("""
                    CREATE TRIGGER IF NOT EXISTS governor_guard_audit_no_update
                    BEFORE UPDATE ON governor_guard_audit
                    BEGIN
                        SELECT RAISE(ABORT, 'governor_guard_audit is append-only');
                    END
                    """)
                connection.execute("""
                    CREATE TRIGGER IF NOT EXISTS governor_guard_audit_no_delete
                    BEFORE DELETE ON governor_guard_audit
                    BEGIN
                        SELECT RAISE(ABORT, 'governor_guard_audit is append-only');
                    END
                    """)
        except sqlite3.Error as exc:  # pragma: no cover - exercised through caller failure paths
            raise GovernorAuditError(f"unable to initialize governor audit db: {exc}") from exc
