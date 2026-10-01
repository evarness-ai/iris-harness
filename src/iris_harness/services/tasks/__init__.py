"""User-facing tasks and goals subsystem.

Introduced in Phase 0 of the personal-assistant upgrade (see
ADR-0005 + ADR-0014). The legacy ``src/iris_harness/reminders.py`` is left
untouched — soft migration; status fields on the legacy Reminder
model are deprecated organically as callers migrate.

Public surface:
  - Task, Goal Pydantic models
  - TaskStore (SQLite-backed CRUD + dedup upsert)
  - Task / goal event constants and payloads (lands in B-ii commit)
"""

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
from .models import (
    ActionCard,
    ActionChoice,
    ActionEvidence,
    ActionFact,
    ActionKind,
    ActionOptions,
    ActionOptionValue,
    Goal,
    GoalStatus,
    SourceKind,
    Task,
    TaskAction,
    TaskStatus,
    WaitFor,
)
from .pending_actions import (
    DesiredAction,
    PendingAction,
    PendingActionProvider,
    PendingActionsSummary,
    pending_action_from_task,
    reconcile,
)
from .store import TaskStore

__all__ = [
    "ActionKind",
    "DesiredAction",
    "GOAL_ACHIEVED",
    "GOAL_CREATED",
    "GOAL_UPDATED",
    "Goal",
    "GoalAchievedPayload",
    "GoalCreatedPayload",
    "GoalStatus",
    "GoalUpdatedPayload",
    "PendingAction",
    "PendingActionProvider",
    "PendingActionsSummary",
    "SourceKind",
    "TASK_COMPLETED",
    "TASK_CREATED",
    "TASK_DROPPED",
    "TASK_UPDATED",
    "TASK_WAIT_RESOLVED",
    "Task",
    "ActionCard",
    "ActionChoice",
    "ActionEvidence",
    "ActionFact",
    "ActionOptionValue",
    "ActionOptions",
    "TaskAction",
    "TaskCompletedPayload",
    "TaskCreatedPayload",
    "TaskDroppedPayload",
    "TaskStatus",
    "TaskStore",
    "TaskUpdatedPayload",
    "TaskWaitResolvedPayload",
    "WaitFor",
    "pending_action_from_task",
    "reconcile",
]
