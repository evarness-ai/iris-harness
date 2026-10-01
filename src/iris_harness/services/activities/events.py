"""Event topics + typed payloads for the Activity spine.

Producer: ``iris_harness.services.activities.store.ActivityStore`` (status transitions
emit on the bus the store is wired to). Mirrors ``iris_harness.services.tasks.events``
and ``iris_harness.services.notifications.events`` — subsystem topics live with their
producer (ADR-0013).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Topic name constants
# ---------------------------------------------------------------------------

ACTIVITY_STARTED = "activity.started"
ACTIVITY_PROGRESS = "activity.progress"
ACTIVITY_COMPLETED = "activity.completed"
ACTIVITY_FAILED = "activity.failed"


# ---------------------------------------------------------------------------
# Typed payloads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivityStartedPayload:
    activity_id: str
    kind: str
    title: str
    origin: str
    started_at: datetime


@dataclass(frozen=True)
class ActivityProgressPayload:
    activity_id: str
    progress: float
    message: str


@dataclass(frozen=True)
class ActivityCompletedPayload:
    activity_id: str
    kind: str
    title: str
    origin: str
    result_summary: str
    undo_ref: str | None
    metadata: dict[str, Any]
    finished_at: datetime


@dataclass(frozen=True)
class ActivityFailedPayload:
    activity_id: str
    kind: str
    title: str
    origin: str
    error: str
    finished_at: datetime
