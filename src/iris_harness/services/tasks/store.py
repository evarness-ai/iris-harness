"""SQLite-backed CRUD for the user-facing Task and Goal subsystem.

See ADR-0005 (vocabulary + soft migration) and ADR-0014
(implementation shape). Schema lives in this module's
``ensure_schema()`` for now; if/when migrations matter we'll split it
out.

Bus integration: each mutation optionally emits a typed event payload
on ``iris_harness.services.tasks.events`` topics when a bus is wired. See that module
for the topic constants and payload dataclasses.
"""

from __future__ import annotations

import builtins
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from iris_harness.foundation.clock import utc_now
from iris_harness.foundation.eventbus import EventBus
from iris_harness.foundation.persistence import connect, data_path
from iris_harness.foundation.persistence.sqlite import ensure_columns

from .events import (
    GOAL_ACHIEVED,
    GOAL_CREATED,
    GOAL_UPDATED,
    TASK_COMPLETED,
    TASK_CREATED,
    TASK_DROPPED,
    TASK_UPDATED,
    TASK_WAIT_RESOLVED,
    GoalAchievedPayload,
    GoalCreatedPayload,
    GoalUpdatedPayload,
    TaskCompletedPayload,
    TaskCreatedPayload,
    TaskDroppedPayload,
    TaskUpdatedPayload,
    TaskWaitResolvedPayload,
)
from .models import Goal, GoalStatus, SourceKind, Task, TaskAction, TaskStatus, WaitFor


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _row_to_task(row: sqlite3.Row) -> Task:
    wait_for_raw = row["wait_for"]
    wait_for: WaitFor | None = None
    if wait_for_raw:
        data = json.loads(wait_for_raw)
        wait_for = WaitFor(**data)
    # ``action`` is an additive column (ADR-0073); guard for pre-migration rows.
    action: TaskAction | None = None
    action_raw = row["action"] if "action" in row.keys() else None
    if action_raw:
        action = TaskAction(**json.loads(action_raw))
    return Task(
        id=row["id"],
        title=row["title"],
        description=row["description"] or "",
        status=row["status"],
        priority=row["priority"],
        source_kind=row["source_kind"],
        source_id=row["source_id"],
        parent_task_id=row["parent_task_id"],
        dedup_key=row["dedup_key"],
        due_at=_parse_dt(row["due_at"]),
        wait_for=wait_for,
        wait_for_resolved_at=_parse_dt(row["wait_for_resolved_at"]),
        parent_goal_id=row["parent_goal_id"],
        related_wikilinks=tuple(json.loads(row["related_wikilinks"] or "[]")),
        calendar_event_id=row["calendar_event_id"],
        calendar_visible=bool(row["calendar_visible"]),
        action=action,
        created_at=_parse_dt(row["created_at"]) or utc_now(),
        updated_at=_parse_dt(row["updated_at"]) or utc_now(),
        completed_at=_parse_dt(row["completed_at"]),
        closed_reason=row["closed_reason"] if "closed_reason" in row.keys() else None,
    )


def _row_to_goal(row: sqlite3.Row) -> Goal:
    return Goal(
        id=row["id"],
        title=row["title"],
        description=row["description"] or "",
        status=row["status"],
        target_date=_parse_dt(row["target_date"]),
        success_criteria=row["success_criteria"] or "",
        created_at=_parse_dt(row["created_at"]) or utc_now(),
        updated_at=_parse_dt(row["updated_at"]) or utc_now(),
        completed_at=_parse_dt(row["completed_at"]),
    )


@dataclass
class TaskStore:
    """SQLite-backed Task and Goal store.

    Pass ``bus=None`` (the default) to operate silently — useful for
    tests and one-off scripts. Pass a runtime bus to fire events on
    every mutation.
    """

    db_path: Path = field(default_factory=lambda: data_path("tasks.db"))
    bus: EventBus | None = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
        # Additive columns, on a connection of their own under BEGIN IMMEDIATE (#201): a read of
        # ``table_info`` then an ``ALTER`` raised "duplicate column name" for the process that
        # lost a race to open an older tasks.db.
        ensure_columns(
            self.db_path,
            "tasks",
            {"action": "TEXT", "closed_reason": "TEXT"},  # ADR-0073; expiry
        )

    def _connect(self) -> sqlite3.Connection:
        conn = connect(self.db_path, row_factory=sqlite3.Row)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def create(self, **fields: Any) -> Task:
        """Insert a new Task. Generates ``id`` if not provided."""
        fields.setdefault("id", str(uuid.uuid4()))
        task = Task(**fields)
        with self._connect() as conn:
            conn.execute(_INSERT_TASK_SQL, _task_to_row(task))
        self._emit(
            TASK_CREATED,
            TaskCreatedPayload(
                task_id=task.id,
                title=task.title,
                status=task.status,
                source_kind=task.source_kind,
                source_id=task.source_id,
                dedup_key=task.dedup_key,
                parent_task_id=task.parent_task_id,
                parent_goal_id=task.parent_goal_id,
                due_at=task.due_at,
                created_at=task.created_at,
            ),
        )
        return task

    def upsert(self, *, dedup_key: str, **fields: Any) -> Task:
        """First-write wins on collision; subsequent calls return the
        existing Task unchanged. See ADR-0005 (dedup_key mechanism)."""
        if not dedup_key:
            raise ValueError("upsert() requires a non-empty dedup_key")
        existing = self._get_by_dedup_key(dedup_key)
        if existing is not None:
            return existing
        fields["dedup_key"] = dedup_key
        return self.create(**fields)

    def get(self, task_id: str) -> Task | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _row_to_task(row) if row else None

    def _get_by_dedup_key(self, dedup_key: str) -> Task | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE dedup_key = ? AND status != 'dropped'",
                (dedup_key,),
            ).fetchone()
        return _row_to_task(row) if row else None

    def get_by_dedup_key(self, dedup_key: str) -> Task | None:
        """Public lookup by ``dedup_key``. Returns the live (non-dropped)
        task with this key, or None. Matches the upsert visibility rule
        (ADR-0014 #3): dropped tasks don't surface.
        """
        return self._get_by_dedup_key(dedup_key)

    def list(
        self,
        *,
        status: TaskStatus | None = None,
        source_kind: SourceKind | None = None,
        parent_task_id: str | None = None,
        parent_goal_id: str | None = None,
        due_before: datetime | None = None,
        has_action: bool | None = None,
        limit: int = 100,
    ) -> list[Task]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if source_kind is not None:
            clauses.append("source_kind = ?")
            params.append(source_kind)
        if parent_task_id is not None:
            clauses.append("parent_task_id = ?")
            params.append(parent_task_id)
        if parent_goal_id is not None:
            clauses.append("parent_goal_id = ?")
            params.append(parent_goal_id)
        if due_before is not None:
            clauses.append("due_at IS NOT NULL AND due_at <= ?")
            params.append(_iso(due_before))
        if has_action is not None:
            # The Action Center discriminator (ADR-0073): a pending action is a
            # task carrying an ``action``; a plain user todo has none.
            clauses.append("action IS NOT NULL" if has_action else "action IS NULL")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        # Why S608 is safe: `clauses` is built from hardcoded SQL fragments
        # only ("status = ?", etc.); all user-supplied values flow through
        # `params` and are bound by sqlite3's parameterization.
        sql = (
            f"SELECT * FROM tasks {where} ORDER BY priority DESC, due_at ASC LIMIT ?"  # noqa: S608
        )
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_task(r) for r in rows]

    def update(self, task_id: str, **fields: Any) -> Task:
        """Update mutable fields. Forbids changing id, created_at,
        dedup_key (mutation of identity is a bug)."""
        forbidden = {"id", "created_at", "dedup_key"}
        bad = forbidden & set(fields)
        if bad:
            raise ValueError(f"update() cannot change immutable fields: {sorted(bad)}")
        current = self.get(task_id)
        if current is None:
            raise KeyError(f"task not found: {task_id}")
        merged = current.model_dump()
        merged.update(fields)
        merged["updated_at"] = utc_now()
        if merged["status"] in ("open", "doing") and "closed_reason" not in fields:
            merged["closed_reason"] = None  # reopened: the close's reason no longer holds
        updated = Task(**merged)
        with self._connect() as conn:
            conn.execute(_UPDATE_TASK_SQL, _task_to_row(updated))
        self._emit(
            TASK_UPDATED,
            TaskUpdatedPayload(
                task_id=updated.id,
                status=updated.status,
                updated_at=updated.updated_at,
                changed_fields=tuple(sorted(fields.keys())),
            ),
        )
        return updated

    def complete(self, task_id: str) -> Task:
        now = utc_now()
        updated = self.update(task_id, status="done", completed_at=now)
        self._emit(
            TASK_COMPLETED,
            TaskCompletedPayload(
                task_id=updated.id,
                title=updated.title,
                completed_at=now,
                parent_goal_id=updated.parent_goal_id,
            ),
        )
        return updated

    def drop(self, task_id: str) -> Task:
        updated = self.update(task_id, status="dropped")
        self._emit(
            TASK_DROPPED,
            TaskDroppedPayload(
                task_id=updated.id,
                title=updated.title,
                dropped_at=updated.updated_at,
                parent_goal_id=updated.parent_goal_id,
            ),
        )
        return updated

    def expire(self, task_id: str, reason: str) -> Task:
        """Close a task that aged out (status ``expired``, ``closed_reason`` = why).

        The system's close, not the owner's: nothing is deleted, and unlike ``drop``
        the task keeps its ``dedup_key``, so a producer re-scanning the same source
        (tomorrow's meetings, a mail thread) never raises it again.
        """
        return self.update(task_id, status="expired", closed_reason=reason)

    def resolve_wait(self, task_id: str, by_event: str) -> Task:
        """Mark a follow-up task as resolved (the awaited event arrived).
        Does NOT auto-complete the task — caller decides what to do
        with the now-resolved followup."""
        current = self.get(task_id)
        if current is None:
            raise KeyError(f"task not found: {task_id}")
        if current.wait_for is None:
            raise ValueError(f"task {task_id} has no wait_for to resolve")
        resolved_at = utc_now()
        updated = self.update(task_id, wait_for_resolved_at=resolved_at)
        self._emit(
            TASK_WAIT_RESOLVED,
            TaskWaitResolvedPayload(task_id=updated.id, by_event=by_event, resolved_at=resolved_at),
        )
        return updated

    # ------------------------------------------------------------------
    # Goals
    # ------------------------------------------------------------------

    def create_goal(self, **fields: Any) -> Goal:
        fields.setdefault("id", str(uuid.uuid4()))
        goal = Goal(**fields)
        with self._connect() as conn:
            conn.execute(_INSERT_GOAL_SQL, _goal_to_row(goal))
        self._emit(
            GOAL_CREATED,
            GoalCreatedPayload(
                goal_id=goal.id,
                title=goal.title,
                status=goal.status,
                target_date=goal.target_date,
                created_at=goal.created_at,
            ),
        )
        return goal

    def get_goal(self, goal_id: str) -> Goal | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM goals WHERE id = ?", (goal_id,)).fetchone()
        return _row_to_goal(row) if row else None

    def list_goals(
        self, *, status: GoalStatus | None = None, limit: int = 100
    ) -> builtins.list[Goal]:
        sql = "SELECT * FROM goals"
        params: list[Any] = []
        if status is not None:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_goal(r) for r in rows]

    def update_goal(self, goal_id: str, **fields: Any) -> Goal:
        forbidden = {"id", "created_at"}
        bad = forbidden & set(fields)
        if bad:
            raise ValueError(f"update_goal() cannot change immutable fields: {sorted(bad)}")
        current = self.get_goal(goal_id)
        if current is None:
            raise KeyError(f"goal not found: {goal_id}")
        merged = current.model_dump()
        merged.update(fields)
        merged["updated_at"] = utc_now()
        updated = Goal(**merged)
        with self._connect() as conn:
            conn.execute(_UPDATE_GOAL_SQL, _goal_to_row(updated))
        if updated.status == "achieved" and updated.completed_at is not None:
            self._emit(
                GOAL_ACHIEVED,
                GoalAchievedPayload(
                    goal_id=updated.id, title=updated.title, completed_at=updated.completed_at
                ),
            )
        else:
            self._emit(
                GOAL_UPDATED,
                GoalUpdatedPayload(
                    goal_id=updated.id,
                    status=updated.status,
                    updated_at=updated.updated_at,
                    changed_fields=tuple(sorted(fields.keys())),
                ),
            )
        return updated

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _emit(self, topic: str, payload: Any) -> None:
        if self.bus is None:
            return
        self.bus.emit_sync(topic, payload)


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------


def _task_to_row(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "description": task.description,
        "status": task.status,
        "priority": task.priority,
        "source_kind": task.source_kind,
        "source_id": task.source_id,
        "parent_task_id": task.parent_task_id,
        "dedup_key": task.dedup_key,
        "due_at": _iso(task.due_at),
        "wait_for": json.dumps(task.wait_for.model_dump(mode="json")) if task.wait_for else None,
        "wait_for_resolved_at": _iso(task.wait_for_resolved_at),
        "parent_goal_id": task.parent_goal_id,
        "related_wikilinks": json.dumps(list(task.related_wikilinks)),
        "calendar_event_id": task.calendar_event_id,
        "calendar_visible": int(task.calendar_visible),
        "action": json.dumps(task.action.model_dump(mode="json")) if task.action else None,
        "created_at": _iso(task.created_at),
        "updated_at": _iso(task.updated_at),
        "completed_at": _iso(task.completed_at),
        "closed_reason": task.closed_reason,
    }


def _goal_to_row(goal: Goal) -> dict[str, Any]:
    return {
        "id": goal.id,
        "title": goal.title,
        "description": goal.description,
        "status": goal.status,
        "target_date": _iso(goal.target_date),
        "success_criteria": goal.success_criteria,
        "created_at": _iso(goal.created_at),
        "updated_at": _iso(goal.updated_at),
        "completed_at": _iso(goal.completed_at),
    }


# ---------------------------------------------------------------------------
# Schema and SQL constants
# ---------------------------------------------------------------------------


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS goals (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'active',
    target_date     TEXT,
    success_criteria TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    completed_at    TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    id                  TEXT    PRIMARY KEY,
    title               TEXT    NOT NULL,
    description         TEXT    NOT NULL DEFAULT '',
    status              TEXT    NOT NULL DEFAULT 'open',
    priority            INTEGER NOT NULL DEFAULT 0,
    source_kind         TEXT,
    source_id           TEXT,
    parent_task_id      TEXT    REFERENCES tasks(id),
    dedup_key           TEXT,
    due_at              TEXT,
    wait_for            TEXT,
    wait_for_resolved_at TEXT,
    parent_goal_id      TEXT    REFERENCES goals(id),
    related_wikilinks   TEXT    NOT NULL DEFAULT '[]',
    calendar_event_id   TEXT,
    calendar_visible    INTEGER NOT NULL DEFAULT 0,
    action              TEXT,
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL,
    completed_at        TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_dedup
    ON tasks(dedup_key) WHERE dedup_key IS NOT NULL AND status != 'dropped';

CREATE INDEX IF NOT EXISTS idx_tasks_status_due ON tasks(status, due_at);
CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_task_id);
CREATE INDEX IF NOT EXISTS idx_tasks_goal ON tasks(parent_goal_id);
"""


_INSERT_TASK_SQL = """
INSERT INTO tasks (
    id, title, description, status, priority,
    source_kind, source_id, parent_task_id, dedup_key,
    due_at, wait_for, wait_for_resolved_at,
    parent_goal_id, related_wikilinks,
    calendar_event_id, calendar_visible, action,
    created_at, updated_at, completed_at, closed_reason
) VALUES (
    :id, :title, :description, :status, :priority,
    :source_kind, :source_id, :parent_task_id, :dedup_key,
    :due_at, :wait_for, :wait_for_resolved_at,
    :parent_goal_id, :related_wikilinks,
    :calendar_event_id, :calendar_visible, :action,
    :created_at, :updated_at, :completed_at, :closed_reason
)
"""


_UPDATE_TASK_SQL = """
UPDATE tasks SET
    title = :title,
    description = :description,
    status = :status,
    priority = :priority,
    source_kind = :source_kind,
    source_id = :source_id,
    parent_task_id = :parent_task_id,
    due_at = :due_at,
    wait_for = :wait_for,
    wait_for_resolved_at = :wait_for_resolved_at,
    parent_goal_id = :parent_goal_id,
    related_wikilinks = :related_wikilinks,
    calendar_event_id = :calendar_event_id,
    calendar_visible = :calendar_visible,
    action = :action,
    updated_at = :updated_at,
    completed_at = :completed_at,
    closed_reason = :closed_reason
WHERE id = :id
"""


_INSERT_GOAL_SQL = """
INSERT INTO goals (
    id, title, description, status, target_date, success_criteria,
    created_at, updated_at, completed_at
) VALUES (
    :id, :title, :description, :status, :target_date, :success_criteria,
    :created_at, :updated_at, :completed_at
)
"""


_UPDATE_GOAL_SQL = """
UPDATE goals SET
    title = :title,
    description = :description,
    status = :status,
    target_date = :target_date,
    success_criteria = :success_criteria,
    updated_at = :updated_at,
    completed_at = :completed_at
WHERE id = :id
"""
