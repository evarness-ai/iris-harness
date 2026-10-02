"""SQLite-backed store for background Activities (``data/activities.db``).

Mirrors ``iris_harness.services.tasks.store.TaskStore``'s shape: hardened WAL connection,
one row per record, typed bus events on every status transition. Kept
deliberately small — an Activity is an append-and-transition record, not
a richly-queried entity.

Concurrency: the ``ActivityRunner`` writes from a background worker thread
while the request path (``GET /activities``) reads from another connection.
WAL + busy_timeout (via ``persistence.connect``) make that safe.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from iris_harness.foundation.clock import utc_now
from iris_harness.foundation.eventbus import EventBus
from iris_harness.foundation.persistence import connect, data_path

from .events import (
    ACTIVITY_COMPLETED,
    ACTIVITY_FAILED,
    ACTIVITY_PROGRESS,
    ACTIVITY_STARTED,
    ActivityCompletedPayload,
    ActivityFailedPayload,
    ActivityProgressPayload,
    ActivityStartedPayload,
)
from .models import Activity, ActivityStatus

logger = logging.getLogger(__name__)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness check. A pid a platform can't answer about (a
    permissions error, an exotic OS) is treated as alive — reaping is only ever
    safe to do when we're SURE the owner is gone."""
    try:
        return bool(psutil.pid_exists(pid))
    except Exception:  # noqa: BLE001 — unsure beats wrongly reaping a live job
        return True


def _row_to_activity(row: sqlite3.Row) -> Activity:
    return Activity(
        id=row["id"],
        kind=row["kind"],
        title=row["title"],
        status=row["status"],
        progress=row["progress"],
        progress_message=row["progress_message"] or "",
        origin=row["origin"] or "",
        result_summary=row["result_summary"] or "",
        error=row["error"] or "",
        undo_ref=row["undo_ref"],
        owner_pid=row["owner_pid"],
        metadata=json.loads(row["metadata"] or "{}"),
        started_at=_parse_dt(row["started_at"]),
        finished_at=_parse_dt(row["finished_at"]),
        created_at=_parse_dt(row["created_at"]) or utc_now(),
        updated_at=_parse_dt(row["updated_at"]) or utc_now(),
    )


@dataclass
class ActivityStore:
    """SQLite-backed Activity store.

    Pass ``bus=None`` (default) for silent operation (tests, the read-only
    API path); pass a bus to fire ``activity.*`` events on transitions.
    """

    db_path: Path = field(default_factory=lambda: data_path("activities.db"))
    bus: EventBus | None = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            _add_column_if_missing(conn, "activities", "owner_pid", "INTEGER")

    def _connect(self) -> sqlite3.Connection:
        return connect(self.db_path, row_factory=sqlite3.Row)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, activity_id: str) -> Activity | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM activities WHERE id = ?", (activity_id,)).fetchone()
        return _row_to_activity(row) if row else None

    def list(
        self,
        *,
        status: ActivityStatus | None = None,
        origin: str | None = None,
        limit: int = 100,
    ) -> list[Activity]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if origin is not None:
            clauses.append("origin = ?")
            params.append(origin)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        # `clauses` holds only hardcoded fragments; values bind through `params`.
        sql = f"SELECT * FROM activities {where} ORDER BY created_at DESC LIMIT ?"  # noqa: S608
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_activity(r) for r in rows]

    # ------------------------------------------------------------------
    # Writes / transitions
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        kind: str,
        title: str,
        origin: str = "",
        metadata: dict[str, Any] | None = None,
        status: ActivityStatus = "queued",
    ) -> Activity:
        activity = Activity(
            id=str(uuid.uuid4()),
            kind=kind,
            title=title,
            status=status,
            origin=origin,
            owner_pid=os.getpid(),
            metadata=metadata or {},
        )
        with self._connect() as conn:
            conn.execute(_INSERT_SQL, _to_row(activity))
        return activity

    def _update(self, activity_id: str, **fields: Any) -> Activity:
        current = self.get(activity_id)
        if current is None:
            raise KeyError(f"activity not found: {activity_id}")
        merged = current.model_dump()
        merged.update(fields)
        merged["updated_at"] = utc_now()
        updated = Activity(**merged)
        with self._connect() as conn:
            conn.execute(_UPDATE_SQL, _to_row(updated))
        return updated

    def mark_running(self, activity_id: str) -> Activity:
        updated = self._update(activity_id, status="running", started_at=utc_now())
        self._emit(
            ACTIVITY_STARTED,
            ActivityStartedPayload(
                activity_id=updated.id,
                kind=updated.kind,
                title=updated.title,
                origin=updated.origin,
                started_at=updated.started_at or utc_now(),
            ),
        )
        return updated

    def mark_progress(self, activity_id: str, progress: float, message: str = "") -> Activity:
        frac = max(0.0, min(1.0, float(progress)))
        updated = self._update(activity_id, progress=frac, progress_message=message)
        self._emit(
            ACTIVITY_PROGRESS,
            ActivityProgressPayload(activity_id=updated.id, progress=frac, message=message),
        )
        return updated

    def mark_completed(
        self,
        activity_id: str,
        *,
        result_summary: str = "",
        undo_ref: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Activity:
        current = self.get(activity_id)
        merged_meta = dict(current.metadata) if current else {}
        if metadata:
            merged_meta.update(metadata)
        now = utc_now()
        updated = self._update(
            activity_id,
            status="completed",
            progress=1.0,
            result_summary=result_summary,
            undo_ref=undo_ref,
            metadata=merged_meta,
            finished_at=now,
        )
        self._emit(
            ACTIVITY_COMPLETED,
            ActivityCompletedPayload(
                activity_id=updated.id,
                kind=updated.kind,
                title=updated.title,
                origin=updated.origin,
                result_summary=updated.result_summary,
                undo_ref=updated.undo_ref,
                metadata=updated.metadata,
                finished_at=now,
            ),
        )
        return updated

    def mark_failed(self, activity_id: str, error: str) -> Activity:
        now = utc_now()
        updated = self._update(activity_id, status="failed", error=error, finished_at=now)
        self._emit(
            ACTIVITY_FAILED,
            ActivityFailedPayload(
                activity_id=updated.id,
                kind=updated.kind,
                title=updated.title,
                origin=updated.origin,
                error=error,
                finished_at=now,
            ),
        )
        return updated

    def mark_cancelled(self, activity_id: str) -> Activity:
        return self._update(activity_id, status="cancelled", finished_at=utc_now())

    def reconcile_orphaned(self) -> int:
        """Reap rows left ``queued``/``running`` by a process that is no longer
        alive (a crash or a kill, not a clean shutdown — those finish their rows).

        ``activities.db`` can be shared by more than one live process (an API
        server and a CLI command, say), so a row is only ever reaped by checking
        ITS OWN ``owner_pid`` for life, never by "I'm starting up, so anything
        not mine must be dead" — that would kill a sibling process's real job.
        Call once, right after a process's own startup; never from a per-request
        read path (a fresh ``ActivityStore`` there would re-run this on every
        call and could race a job this same process just started).
        """
        reaped = 0
        for row in self.list(status="queued", limit=10_000) + self.list(
            status="running", limit=10_000
        ):
            if row.owner_pid is not None and _pid_alive(row.owner_pid):
                continue
            try:
                self.mark_failed(row.id, "interrupted: the process running it did not exit cleanly")
            except KeyError:
                continue
            reaped += 1
        if reaped:
            logger.warning("activities: reaped %d orphaned row(s) on startup", reaped)
        return reaped

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _emit(self, topic: str, payload: Any) -> None:
        if self.bus is None:
            return
        self.bus.emit_sync(topic, payload)


# ---------------------------------------------------------------------------
# Row helpers + schema
# ---------------------------------------------------------------------------


def _to_row(activity: Activity) -> dict[str, Any]:
    return {
        "id": activity.id,
        "kind": activity.kind,
        "title": activity.title,
        "status": activity.status,
        "progress": activity.progress,
        "progress_message": activity.progress_message,
        "origin": activity.origin,
        "result_summary": activity.result_summary,
        "error": activity.error,
        "undo_ref": activity.undo_ref,
        "owner_pid": activity.owner_pid,
        "metadata": json.dumps(activity.metadata),
        "started_at": _iso(activity.started_at),
        "finished_at": _iso(activity.finished_at),
        "created_at": _iso(activity.created_at),
        "updated_at": _iso(activity.updated_at),
    }


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS activities (
    id                TEXT    PRIMARY KEY,
    kind              TEXT    NOT NULL,
    title             TEXT    NOT NULL,
    status            TEXT    NOT NULL DEFAULT 'queued',
    progress          REAL    NOT NULL DEFAULT 0.0,
    progress_message  TEXT    NOT NULL DEFAULT '',
    origin            TEXT    NOT NULL DEFAULT '',
    result_summary    TEXT    NOT NULL DEFAULT '',
    error             TEXT    NOT NULL DEFAULT '',
    undo_ref          TEXT,
    owner_pid         INTEGER,
    metadata          TEXT    NOT NULL DEFAULT '{}',
    started_at        TEXT,
    finished_at       TEXT,
    created_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_activities_status ON activities(status, created_at);
CREATE INDEX IF NOT EXISTS idx_activities_origin ON activities(origin, created_at);
"""


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    """Idempotent additive migration for an existing table."""
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


_INSERT_SQL = """
INSERT INTO activities (
    id, kind, title, status, progress, progress_message, origin,
    result_summary, error, undo_ref, owner_pid, metadata,
    started_at, finished_at, created_at, updated_at
) VALUES (
    :id, :kind, :title, :status, :progress, :progress_message, :origin,
    :result_summary, :error, :undo_ref, :owner_pid, :metadata,
    :started_at, :finished_at, :created_at, :updated_at
)
"""


_UPDATE_SQL = """
UPDATE activities SET
    kind = :kind,
    title = :title,
    status = :status,
    progress = :progress,
    progress_message = :progress_message,
    origin = :origin,
    result_summary = :result_summary,
    error = :error,
    undo_ref = :undo_ref,
    owner_pid = :owner_pid,
    metadata = :metadata,
    started_at = :started_at,
    finished_at = :finished_at,
    updated_at = :updated_at
WHERE id = :id
"""
