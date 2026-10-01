"""Async Activity spine — durable background job records + a runner.

See ``docs/architecture/async-activities-and-notifications.md``. An
Activity is one row per non-chat system operation (FileManager jobs,
heartbeat ticks, routine runs) that runs in the background instead of
blocking a chat turn.
"""

from __future__ import annotations

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
from .models import Activity, ActivityKind, ActivityOutcome, ActivityStatus
from .runner import ActivityRunner, ProgressFn, WorkFn
from .store import ActivityStore

__all__ = [
    "ACTIVITY_COMPLETED",
    "ACTIVITY_FAILED",
    "ACTIVITY_PROGRESS",
    "ACTIVITY_STARTED",
    "Activity",
    "ActivityCompletedPayload",
    "ActivityFailedPayload",
    "ActivityKind",
    "ActivityOutcome",
    "ActivityProgressPayload",
    "ActivityRunner",
    "ActivityStartedPayload",
    "ActivityStatus",
    "ActivityStore",
    "ProgressFn",
    "WorkFn",
]
