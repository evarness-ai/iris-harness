"""Domain models for the mission engine."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class MissionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class MissionStep:
    """A single step within a mission. Steps are executed sequentially."""

    name: str
    status: StepStatus = StepStatus.PENDING
    output: str = ""
    error: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    payload: dict[str, object] = field(default_factory=dict)


@dataclass
class Mission:
    """A multi-step long-running task with crash-recovery checkpointing."""

    name: str
    handler: str
    steps: list[MissionStep] = field(default_factory=list)
    status: MissionStatus = MissionStatus.PENDING
    cursor: int = 0  # index of the next step to execute
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_done(self) -> bool:
        return self.status in {
            MissionStatus.COMPLETED,
            MissionStatus.FAILED,
            MissionStatus.CANCELLED,
        }

    def current_step(self) -> MissionStep | None:
        if 0 <= self.cursor < len(self.steps):
            return self.steps[self.cursor]
        return None
