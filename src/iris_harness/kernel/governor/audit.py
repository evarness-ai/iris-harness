"""Append-only SQLite audit logging for IRIS governor decisions."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Mapping
from pathlib import Path

from iris_harness.foundation.ids import new_ulid
from iris_harness.foundation.persistence.sqlite import add_columns_if_missing, sqlite_conn

from .exceptions import GovernorAuditError
from .models import GovernorAuditEntry, GovernorGuardDecision

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_DB_PATH = Path("data/audit.db")

#: Identity columns (issue #134, stage 3): which run, call and session a guard decision was
#: made for, and the row's own id. Nullable: a decision made before the column existed, or
#: for a request that is not a tool call, has none. Identifiers only, never text.
IDENTITY_COLUMNS: dict[str, str] = {
    "run_id": "TEXT",
    "call_id": "TEXT",
    "session_id": "TEXT",
    "record_id": "TEXT",
}


class GovernorAuditLogger:
    """Write immutable governor decisions to a local SQLite database.

    The database is created on first use, not on construction: the API builds a
    governor while it starts even with MCP off, and building one must not write a
    file (it used to leave ``data/audit.db`` in whatever directory imported the app).
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self._initialized = False
        self._has_identity = False

    @classmethod
    def from_repo_root(
        cls,
        repo_root: Path,
        *,
        db_path: Path | None = None,
    ) -> GovernorAuditLogger:
        """Construct an audit logger using the default IRIS audit location."""
        return cls((db_path or (repo_root / DEFAULT_AUDIT_DB_PATH)).resolve())

    def record_decision(
        self,
        decision: GovernorGuardDecision,
        *,
        identity: Mapping[str, str | None] | None = None,
    ) -> None:
        """Append one guard decision to the audit store.

        ``identity`` names the run, call and session the decision was made for (``run_id``,
        ``call_id``, ``session_id``; any may be absent). Only those three keys are read, and
        only a non-empty string is kept.
        """
        self._ensure_initialized()
        ids = {
            key: value
            for key in ("run_id", "call_id", "session_id")
            if isinstance(value := (identity or {}).get(key), str) and value
        }
        columns = [
            "route",
            "action",
            "allowed",
            "reason",
            "matched_policy",
            "requires_approval",
            "retry_after_seconds",
            "metadata_json",
            "created_at",
        ]
        values: list[object] = [
            decision.route,
            decision.action,
            1 if decision.allowed else 0,
            decision.reason,
            decision.matched_policy,
            1 if decision.requires_approval else 0,
            decision.retry_after_seconds,
            json.dumps(decision.metadata, sort_keys=True),
            decision.created_at.isoformat(),
        ]
        if (
            self._has_identity
        ):  # the columns exist; a store that could not migrate writes the old shape
            columns += ["run_id", "call_id", "session_id", "record_id"]
            values += [ids.get("run_id"), ids.get("call_id"), ids.get("session_id"), new_ulid()]
        try:
            with sqlite_conn(self.db_path) as connection:
                connection.execute(
                    f"INSERT INTO governor_guard_audit ({', '.join(columns)}) "  # noqa: S608
                    f"VALUES ({', '.join('?' for _ in columns)})",
                    values,
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
        try:
            add_columns_if_missing(self.db_path, "governor_guard_audit", IDENTITY_COLUMNS)
            self._has_identity = True
        except sqlite3.Error:
            # The decisions must still be written: the old schema keeps working, without the
            # identity columns, until the next start tries the migration again.
            logger.error(
                "governor audit: could not add the identity columns to %s; "
                "decisions are written without them",
                self.db_path,
                exc_info=True,
            )
