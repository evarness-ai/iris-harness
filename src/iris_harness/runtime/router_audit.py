"""Append-only audit of intent-classifier decisions per chat turn.

Writes one row per ``IrisRuntime.chat`` / ``chat_stream`` turn into the
``router_decisions`` table inside ``data/audit.db``. The store lets
operators ask real questions like:

    sqlite3 data/audit.db "
      SELECT intent, source, COUNT(*) AS hits
      FROM router_decisions
      GROUP BY intent, source
      ORDER BY hits DESC;
    "

Privacy: messages are never stored verbatim. Only a SHA-256 hex prefix
+ message length are recorded so duplicates can be counted without
exposing PII / secret content. Classification of the message itself
(public / internal / personal / secret) is enforced upstream by the
governance kernel — this audit log is internal-only.

Schema is append-only via SQLite triggers that mirror the governor
audit pattern in ``src/iris_harness/kernel/governor/audit.py``.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from iris_harness.agent.intent_router import IntentResult
from iris_harness.foundation.ids import new_ulid
from iris_harness.foundation.observability.session_log import current_turn_id
from iris_harness.foundation.persistence.sqlite import add_columns_if_missing, sqlite_conn

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_DB_PATH = Path("data/audit.db")
TABLE_NAME = "router_decisions"

#: Identity columns (issue #134, stage 3): the row's own id and the chat turn it classified.
#: Nullable: a decision made before the column existed has neither. The table is append-only
#: (triggers), so old rows are never rewritten.
IDENTITY_COLUMNS: dict[str, str] = {"record_id": "TEXT", "turn_id": "TEXT"}


class RouterAuditLogger:
    """Persist intent-classifier decisions to ``data/audit.db``."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self._has_identity = False
        self._initialize()

    @classmethod
    def from_repo_root(cls, repo_root: Path | None = None) -> RouterAuditLogger:
        """Construct using the default ``<repo_root>/data/audit.db`` path."""
        base = (repo_root or Path.cwd()).resolve()
        return cls(base / DEFAULT_AUDIT_DB_PATH)

    def record(
        self,
        *,
        session_id: str,
        message: str,
        result: IntentResult,
        router_model: str | None = None,
        channel: str | None = None,
    ) -> None:
        """Append one classifier decision to the audit store.

        Swallows any sqlite errors after logging — audit failures must
        never break a user-facing chat turn.
        """

        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]
        now = datetime.now(UTC).isoformat()
        try:
            columns = (
                "session_id, message_hash, message_length, intent, agent_type, "
                "confidence, source, is_multi_step, router_model, channel, created_at"
            )
            values: list[object] = [
                session_id,
                digest,
                len(message),
                result.intent,
                result.agent_type,
                float(result.confidence),
                result.source or "",
                1 if result.is_multi_step else 0,
                router_model,
                channel,
                now,
            ]
            if self._has_identity:  # a store that could not migrate keeps writing the old shape
                columns += ", record_id, turn_id"
                values += [new_ulid(), current_turn_id()]
            with sqlite_conn(self.db_path) as connection:
                connection.execute(
                    # TABLE_NAME is a module constant, not user input; values are bound below
                    f"INSERT INTO {TABLE_NAME} ("  # noqa: S608
                    f"{columns}) VALUES ({', '.join('?' for _ in values)})",
                    values,
                )
        except sqlite3.Error as exc:
            # Never break the chat turn over an audit write failure.
            logger.warning("router audit write failed: %s", exc)

    def _initialize(self) -> None:
        """Create the append-only schema if it does not already exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite_conn(self.db_path) as connection:
                connection.execute(f"""
                    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        message_hash TEXT NOT NULL,
                        message_length INTEGER NOT NULL,
                        intent TEXT NOT NULL,
                        agent_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        source TEXT NOT NULL,
                        is_multi_step INTEGER NOT NULL,
                        router_model TEXT,
                        channel TEXT,
                        created_at TEXT NOT NULL
                    )
                    """)
                connection.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_intent
                    ON {TABLE_NAME}(intent)
                    """)
                connection.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_source
                    ON {TABLE_NAME}(source)
                    """)
                connection.execute(f"""
                    CREATE TRIGGER IF NOT EXISTS {TABLE_NAME}_no_update
                    BEFORE UPDATE ON {TABLE_NAME}
                    BEGIN
                        SELECT RAISE(ABORT, '{TABLE_NAME} is append-only');
                    END
                    """)
                connection.execute(f"""
                    CREATE TRIGGER IF NOT EXISTS {TABLE_NAME}_no_delete
                    BEFORE DELETE ON {TABLE_NAME}
                    BEGIN
                        SELECT RAISE(ABORT, '{TABLE_NAME} is append-only');
                    END
                    """)
        except sqlite3.Error as exc:
            logger.warning("could not initialize router audit db: %s", exc)
            return
        try:
            add_columns_if_missing(self.db_path, TABLE_NAME, IDENTITY_COLUMNS)
            self._has_identity = True
        except sqlite3.Error:
            # Decisions must still be written: the old schema keeps working, without the
            # identity columns, and the next start tries the migration again.
            logger.error(
                "router audit: could not add the identity columns to %s; "
                "decisions are written without them",
                self.db_path,
                exc_info=True,
            )
