"""SQLite-backed mission persistence with crash recovery."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from iris_harness.foundation.persistence import connect

from .models import Mission, MissionStatus, MissionStep, StepStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS missions (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    handler     TEXT NOT NULL,
    status      TEXT NOT NULL,
    cursor      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    steps       TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_missions_status ON missions(status);
"""


class MissionStore:
    """Persists missions to SQLite for checkpointing and crash recovery."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = connect(self.db_path, row_factory=sqlite3.Row)
        return conn

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_steps(steps: list[MissionStep]) -> str:
        return json.dumps(
            [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "output": s.output,
                    "error": s.error,
                    "started_at": s.started_at.isoformat() if s.started_at else None,
                    "finished_at": s.finished_at.isoformat() if s.finished_at else None,
                    "payload": s.payload,
                }
                for s in steps
            ]
        )

    @staticmethod
    def _deserialize_steps(blob: str) -> list[MissionStep]:
        raw = json.loads(blob or "[]")
        steps: list[MissionStep] = []
        for entry in raw:
            steps.append(
                MissionStep(
                    name=entry["name"],
                    status=StepStatus(entry["status"]),
                    output=entry.get("output", ""),
                    error=entry.get("error", ""),
                    started_at=(
                        datetime.fromisoformat(entry["started_at"])
                        if entry.get("started_at")
                        else None
                    ),
                    finished_at=(
                        datetime.fromisoformat(entry["finished_at"])
                        if entry.get("finished_at")
                        else None
                    ),
                    payload=entry.get("payload", {}),
                )
            )
        return steps

    @classmethod
    def _row_to_mission(cls, row: sqlite3.Row) -> Mission:
        return Mission(
            id=row["id"],
            name=row["name"],
            handler=row["handler"],
            status=MissionStatus(row["status"]),
            cursor=row["cursor"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata"] or "{}"),
            steps=cls._deserialize_steps(row["steps"]),
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, mission: Mission) -> None:
        mission.updated_at = datetime.now(UTC)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO missions (id, name, handler, status, cursor, created_at, updated_at, metadata, steps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    handler=excluded.handler,
                    status=excluded.status,
                    cursor=excluded.cursor,
                    updated_at=excluded.updated_at,
                    metadata=excluded.metadata,
                    steps=excluded.steps
                """,
                (
                    mission.id,
                    mission.name,
                    mission.handler,
                    mission.status.value,
                    mission.cursor,
                    mission.created_at.isoformat(),
                    mission.updated_at.isoformat(),
                    json.dumps(mission.metadata),
                    self._serialize_steps(mission.steps),
                ),
            )

    def load(self, mission_id: str) -> Mission | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM missions WHERE id = ?", (mission_id,)).fetchone()
        return self._row_to_mission(row) if row else None

    def list_active(self) -> list[Mission]:
        """Return missions that should be resumed after a crash (PENDING/RUNNING/PAUSED)."""
        active = (
            MissionStatus.PENDING.value,
            MissionStatus.RUNNING.value,
            MissionStatus.PAUSED.value,
        )
        placeholders = ",".join("?" * len(active))
        query = f"SELECT * FROM missions WHERE status IN ({placeholders}) ORDER BY created_at"  # noqa: S608 — placeholders only, values bound
        with self._connect() as conn:
            rows = conn.execute(query, active).fetchall()
        return [self._row_to_mission(r) for r in rows]

    def list_all(self) -> list[Mission]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM missions ORDER BY created_at").fetchall()
        return [self._row_to_mission(r) for r in rows]

    def delete(self, mission_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM missions WHERE id = ?", (mission_id,))
        return cursor.rowcount > 0

    def save_many(self, missions: Iterable[Mission]) -> None:
        for m in missions:
            self.save(m)
