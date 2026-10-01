"""Health incidents — the record the health watch keeps (ADR-0116).

ADR-0069 §3 made alerts a projection that is never stored. That was enough for a
banner, but not for a watch that repairs and notifies: it has to remember that it
already told the owner (or a restart would re-send every alert), what it tried, and
how the problem ended. An incident is that memory — one row per stretch of time a
check stayed red, keyed by ``HealthCheck.key``.

States: ``repairing`` (the watch is still trying fixes) → ``needs_user`` (fixes ran
out; the owner was told) → ``resolved``. A resolution is ``self_healed`` (green
before the owner was needed) or ``user_fixed`` (green after the owner was told). A
check that stops reporting counts as green: a heartbeat diagnostic only exists while
its heartbeat is failing.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from iris_harness.foundation.clock import utc_now
from iris_harness.foundation.persistence.sqlite import sqlite_conn

REPAIRING = "repairing"
NEEDS_USER = "needs_user"
RESOLVED = "resolved"

SELF_HEALED = "self_healed"
USER_FIXED = "user_fixed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS health_incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    target TEXT NOT NULL,
    subject TEXT,
    kind TEXT NOT NULL,
    state TEXT NOT NULL,
    detail TEXT NOT NULL,
    action TEXT,
    opened_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    repairs TEXT NOT NULL DEFAULT '[]',
    notified_at TEXT,
    notify_count INTEGER NOT NULL DEFAULT 0,
    resolved_at TEXT,
    resolution TEXT
);
CREATE INDEX IF NOT EXISTS idx_health_incidents_key ON health_incidents(key, resolved_at);
"""


@dataclass
class Incident:
    """One stretch of a check being red, and what the watch did about it."""

    id: int
    key: str
    target: str
    subject: str | None
    kind: str
    state: str
    detail: str
    action: str | None
    opened_at: str
    updated_at: str
    attempts: int = 0
    repairs: list[dict[str, Any]] = field(default_factory=list)
    notified_at: str | None = None
    notify_count: int = 0
    resolved_at: str | None = None
    resolution: str | None = None

    @property
    def is_open(self) -> bool:
        return self.resolved_at is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "key": self.key,
            "target": self.target,
            "subject": self.subject,
            "kind": self.kind,
            "state": self.state,
            "detail": self.detail,
            "action": self.action,
            "opened_at": self.opened_at,
            "updated_at": self.updated_at,
            "attempts": self.attempts,
            "repairs": list(self.repairs),
            "notified_at": self.notified_at,
            "notify_count": self.notify_count,
            "resolved_at": self.resolved_at,
            "resolution": self.resolution,
        }


def _row(row: sqlite3.Row) -> Incident:
    data = dict(row)
    data["repairs"] = json.loads(data.get("repairs") or "[]")
    return Incident(**data)


class IncidentStore:
    """SQLite-backed incidents (``<data_dir>/health.db``)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite_conn(self.db_path) as conn:
            conn.executescript(_SCHEMA)

    def _query(self, sql: str, args: tuple[Any, ...] = ()) -> list[Incident]:
        with sqlite_conn(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [_row(r) for r in conn.execute(sql, args).fetchall()]

    def open_incidents(self) -> list[Incident]:
        return self._query("SELECT * FROM health_incidents WHERE resolved_at IS NULL ORDER BY id")

    def get(self, incident_id: int) -> Incident | None:
        found = self._query("SELECT * FROM health_incidents WHERE id = ?", (incident_id,))
        return found[0] if found else None

    def recent(self, *, limit: int = 50, open_only: bool = False) -> list[Incident]:
        where = "WHERE resolved_at IS NULL " if open_only else ""
        return self._query(
            f"SELECT * FROM health_incidents {where}ORDER BY id DESC LIMIT ?",  # noqa: S608
            (max(1, limit),),
        )

    def count_since(self, key: str, *, days: int, now: datetime | None = None) -> int:
        """How many incidents ``key`` has opened in the last ``days`` (this one included)."""
        since = ((now or utc_now()) - timedelta(days=days)).isoformat()
        with sqlite_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM health_incidents WHERE key = ? AND opened_at >= ?",
                (key, since),
            ).fetchone()
        return int(row[0]) if row else 0

    def open(
        self,
        *,
        key: str,
        target: str,
        subject: str | None,
        kind: str,
        detail: str,
        action: str | None,
        now: datetime | None = None,
    ) -> Incident:
        stamp = (now or utc_now()).isoformat()
        with sqlite_conn(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO health_incidents (key, target, subject, kind, state, detail, action,"
                " opened_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (key, target, subject, kind, REPAIRING, detail, action, stamp, stamp),
            )
            new_id = int(cur.lastrowid or 0)
        found = self.get(new_id)
        assert found is not None
        return found

    def save(self, incident: Incident, *, now: datetime | None = None) -> None:
        """Write back every mutable field of ``incident``."""
        incident.updated_at = (now or utc_now()).isoformat()
        with sqlite_conn(self.db_path) as conn:
            conn.execute(
                "UPDATE health_incidents SET state = ?, detail = ?, action = ?, updated_at = ?,"
                " attempts = ?, repairs = ?, notified_at = ?, notify_count = ?,"
                " resolved_at = ?, resolution = ? WHERE id = ?",
                (
                    incident.state,
                    incident.detail,
                    incident.action,
                    incident.updated_at,
                    incident.attempts,
                    json.dumps(incident.repairs),
                    incident.notified_at,
                    incident.notify_count,
                    incident.resolved_at,
                    incident.resolution,
                    incident.id,
                ),
            )


__all__ = [
    "NEEDS_USER",
    "REPAIRING",
    "RESOLVED",
    "SELF_HEALED",
    "USER_FIXED",
    "Incident",
    "IncidentStore",
]
