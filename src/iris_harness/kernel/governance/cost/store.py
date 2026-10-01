"""CostStore — SQLite ledger of per-LLM-call cost (story 12.gov-3.7).

Append-only. One row per allowed ``PreLLMCall`` (the ``CostLimiter``
plugin writes an estimated-cost row when it allows the call; an
optional reconciliation path can update the row once the upstream
returns actual token counts — not implemented in v1).

Schema (design §4.1 routes table — `cost_ceiling_usd`):

    ts                  TEXT NOT NULL    -- ISO-8601
    run_id              TEXT NOT NULL
    agent_type          TEXT NOT NULL
    tier                TEXT NOT NULL
    prompt_tokens       INTEGER NOT NULL
    completion_tokens   INTEGER NOT NULL
    cost_usd            REAL NOT NULL
    user_id             TEXT NOT NULL

WAL mode for concurrent reads; ``0o600`` perm reasserted on every open
— same defense-in-depth pattern as the audit log + vault store.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from iris_harness.foundation.paths import governance_data_dir

logger = logging.getLogger(__name__)

# Override slot (``None``: resolved from ``IRIS_HOME`` on every use, never frozen at
# import -- a process that relocates the home after importing IRIS writes into the
# new one). Set it to point the default somewhere else outright (tests).
DEFAULT_COST_LEDGER_DB_PATH: Path | None = None


def default_cost_ledger_db_path() -> Path:
    """``DEFAULT_COST_LEDGER_DB_PATH`` when set, else ``<governance data dir>/cost-ledger.db``."""
    return (
        DEFAULT_COST_LEDGER_DB_PATH
        if DEFAULT_COST_LEDGER_DB_PATH is not None
        else governance_data_dir() / "cost-ledger.db"
    )


@dataclass(frozen=True)
class CostEntry:
    """One materialized row from the ledger."""

    id: int
    ts: str
    run_id: str
    agent_type: str
    tier: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    user_id: str


class CostStore:
    """Append-only SQLite cost ledger."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or default_cost_ledger_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def record(
        self,
        *,
        run_id: str,
        agent_type: str,
        tier: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        user_id: str,
        ts: datetime | None = None,
    ) -> int:
        """Append one row. Returns the assigned ``id``."""
        ts_iso = (ts or datetime.now(UTC)).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO cost_ledger(
                    ts, run_id, agent_type, tier, prompt_tokens,
                    completion_tokens, cost_usd, user_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_iso,
                    run_id,
                    agent_type,
                    tier,
                    int(prompt_tokens),
                    int(completion_tokens),
                    float(cost_usd),
                    user_id,
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def sum_today(self, *, user_id: str, now: datetime | None = None) -> float:
        """Sum ``cost_usd`` for ``user_id`` since midnight (UTC) today."""
        today = (now or datetime.now(UTC)).date()
        return self.sum_since(user_id=user_id, since=today)

    def sum_since(self, *, user_id: str, since: date | datetime | str) -> float:
        """Sum ``cost_usd`` for ``user_id`` with ``ts >= since``."""
        cutoff = _iso_at_midnight(since)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(cost_usd), 0.0)
                FROM cost_ledger
                WHERE user_id = ? AND ts >= ?
                """,
                (user_id, cutoff),
            ).fetchone()
        return float(row[0])

    def sum_by_tier_since(self, *, user_id: str, since: date | datetime | str) -> dict[str, float]:
        """Per-tier spend breakdown for ``user_id`` since ``since`` (date-floored)."""
        cutoff = _iso_at_midnight(since)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT tier, COALESCE(SUM(cost_usd), 0.0)
                FROM cost_ledger
                WHERE user_id = ? AND ts >= ?
                GROUP BY tier
                ORDER BY tier
                """,
                (user_id, cutoff),
            ).fetchall()
        return {str(tier): float(spend) for tier, spend in rows}

    def list_since(
        self,
        *,
        user_id: str,
        since: date | datetime | str,
        limit: int | None = None,
    ) -> tuple[CostEntry, ...]:
        cutoff = _iso_at_midnight(since)
        sql = (
            "SELECT * FROM cost_ledger " "WHERE user_id = ? AND ts >= ? " "ORDER BY ts ASC, id ASC"
        )
        params: list[Any] = [user_id, cutoff]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return tuple(_row_to_entry(row) for row in rows)

    def count(self) -> int:
        with self._connect() as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM cost_ledger").fetchone()
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
                CREATE TABLE IF NOT EXISTS cost_ledger (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts                TEXT    NOT NULL,
                    run_id            TEXT    NOT NULL,
                    agent_type        TEXT    NOT NULL,
                    tier              TEXT    NOT NULL,
                    prompt_tokens     INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    cost_usd          REAL    NOT NULL,
                    user_id           TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cost_user_ts
                    ON cost_ledger(user_id, ts);
                CREATE INDEX IF NOT EXISTS idx_cost_tier_ts
                    ON cost_ledger(tier, ts);
                """)
            conn.commit()


def _row_to_entry(row: sqlite3.Row) -> CostEntry:
    return CostEntry(
        id=int(row["id"]),
        ts=str(row["ts"]),
        run_id=str(row["run_id"]),
        agent_type=str(row["agent_type"]),
        tier=str(row["tier"]),
        prompt_tokens=int(row["prompt_tokens"]),
        completion_tokens=int(row["completion_tokens"]),
        cost_usd=float(row["cost_usd"]),
        user_id=str(row["user_id"]),
    )


def _iso_at_midnight(value: date | datetime | str) -> str:
    """Floor a date/datetime/string to UTC midnight and return ISO-8601.

    Strings are accepted as-is (caller's responsibility to format
    correctly). Dates and naive datetimes are interpreted in UTC.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    # date (or date-like) — floor to midnight UTC.
    return datetime(value.year, value.month, value.day, tzinfo=UTC).isoformat()
