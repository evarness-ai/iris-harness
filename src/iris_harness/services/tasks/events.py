"""Event topic constants and typed payloads for the tasks subsystem.

Producer: ``iris_harness.services.tasks.store.TaskStore`` (mutations emit on the
runtime ``EventBus`` when one is wired).

See ADR-0013 for the topic-naming convention (subsystem topics live
with their producer, not in ``iris_harness.foundation.eventbus.topics``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .models import GoalStatus, SourceKind, TaskStatus

# ---------------------------------------------------------------------------
# Topic name constants
# ---------------------------------------------------------------------------

TASK_CREATED = "task.created"
TASK_UPDATED = "task.updated"
TASK_COMPLETED = "task.completed"
TASK_DROPPED = "task.dropped"
TASK_WAIT_RESOLVED = "task.wait_resolved"

GOAL_CREATED = "goal.created"
GOAL_UPDATED = "goal.updated"
GOAL_ACHIEVED = "goal.achieved"


# ---------------------------------------------------------------------------
# Typed payloads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskCreatedPayload:
    task_id: str
    title: str
    status: TaskStatus
    source_kind: SourceKind | None
    source_id: str | None
    dedup_key: str | None
    parent_task_id: str | None
    parent_goal_id: str | None
    due_at: datetime | None
    created_at: datetime


@dataclass(frozen=True)
class TaskUpdatedPayload:
    task_id: str
    status: TaskStatus
    updated_at: datetime
    changed_fields: tuple[str, ...]  # names of fields included in update()


@dataclass(frozen=True)
class TaskCompletedPayload:
    task_id: str
    title: str
    completed_at: datetime
    parent_goal_id: str | None


@dataclass(frozen=True)
class TaskDroppedPayload:
    task_id: str
    title: str
    dropped_at: datetime
    parent_goal_id: str | None


@dataclass(frozen=True)
class TaskWaitResolvedPayload:
    task_id: str
    by_event: str  # producer-supplied identifier of what resolved the wait
    resolved_at: datetime


@dataclass(frozen=True)
class GoalCreatedPayload:
    goal_id: str
    title: str
    status: GoalStatus
    target_date: datetime | None
    created_at: datetime


@dataclass(frozen=True)
class GoalUpdatedPayload:
    goal_id: str
    status: GoalStatus
    updated_at: datetime
    changed_fields: tuple[str, ...]


@dataclass(frozen=True)
class GoalAchievedPayload:
    goal_id: str
    title: str
    completed_at: datetime
