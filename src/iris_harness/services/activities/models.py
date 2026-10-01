"""Domain models for the async Activity spine.

An *Activity* is one durable record of a non-chat system operation —
a FileManager categorize/cleanup/organize job, a heartbeat tick, a
routine run, a RAG ingest — that runs in the background rather than
blocking a chat turn. See ``docs/architecture/async-activities-and-notifications.md``.

The shape mirrors ``iris_harness.services.tasks.models.Task`` intentionally so the store
CRUD + event pattern can be reused; Activities are distinct from
user-facing Tasks/Goals (they live in their own ``activities.db`` and
only cross into the Action Center via the ADR-0073 provider seam when a
completed Activity needs a user decision).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from iris_harness.foundation.clock import utc_now

# queued  -> submitted, not yet picked up by a worker
# running -> a worker is executing the job
# completed / failed / cancelled -> terminal
ActivityStatus = Literal["queued", "running", "completed", "failed", "cancelled"]

# ``kind`` is a free string with a ``<subsystem>.<op>`` convention so new
# producers don't need a code change. Known values today:
#   filemanager.categorize | filemanager.cleanup | filemanager.organize
#   heartbeat.<name> | routine.<id> | rag.ingest
ActivityKind = str


class Activity(BaseModel):
    """A durable record of a background system operation."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: ActivityKind
    title: str
    status: ActivityStatus = "queued"
    # 0.0..1.0 fraction complete + a short human message ("analyzed 40/200 images").
    progress: float = 0.0
    progress_message: str = ""
    # Where the job was launched from: "chat:<session_id>" | "heartbeat" |
    # "routine" | "api". The completion notifier parses this to route the
    # in-chat notice back to the originating session.
    origin: str = ""
    result_summary: str = ""
    error: str = ""
    # A reversible-job handle (e.g. the OrganizePlan id) so the feed can offer undo.
    undo_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def is_terminal(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")


@dataclass
class ActivityOutcome:
    """What a background work function returns to the runner.

    Kept as a plain dataclass (not the Pydantic row) so work functions
    stay ignorant of persistence; the runner maps this onto the row on
    completion.
    """

    result_summary: str = ""
    undo_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
