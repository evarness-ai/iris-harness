"""Mission engine — executes missions step-by-step with checkpointing."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from .models import Mission, MissionStatus, MissionStep, StepStatus
from .store import MissionStore

logger = logging.getLogger(__name__)


class MissionHandler(Protocol):
    """Executes a single mission step. Mutates the step in-place and returns the new status."""

    def __call__(self, mission: Mission, step: MissionStep) -> StepStatus: ...


HandlerLookup = Callable[[str], MissionHandler | None]


class MissionEngine:
    """Coordinates mission execution with checkpointing after every step."""

    def __init__(
        self,
        store: MissionStore,
        *,
        handlers: dict[str, MissionHandler] | None = None,
    ) -> None:
        self.store = store
        self._handlers: dict[str, MissionHandler] = dict(handlers or {})

    # ------------------------------------------------------------------
    # Handler registry
    # ------------------------------------------------------------------

    def register_handler(self, name: str, handler: MissionHandler) -> None:
        self._handlers[name] = handler

    # ------------------------------------------------------------------
    # Mission lifecycle
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        handler: str,
        steps: list[MissionStep] | list[str],
        *,
        metadata: dict[str, object] | None = None,
    ) -> Mission:
        normalized = [s if isinstance(s, MissionStep) else MissionStep(name=s) for s in steps]
        mission = Mission(
            name=name,
            handler=handler,
            steps=normalized,
            metadata=dict(metadata or {}),
        )
        self.store.save(mission)
        return mission

    def run(self, mission: Mission) -> Mission:
        """Execute a mission until completion, failure, or a step pauses it.

        Each step is checkpointed to the store before and after execution so a
        crash anywhere in the loop leaves the mission resumable via ``resume_pending``.
        """
        handler = self._handlers.get(mission.handler)
        if handler is None:
            mission.status = MissionStatus.FAILED
            errors = mission.metadata.setdefault("errors", [])
            if isinstance(errors, list):
                errors.append(f"no handler registered for {mission.handler!r}")
            self.store.save(mission)
            return mission

        if mission.status in {
            MissionStatus.COMPLETED,
            MissionStatus.FAILED,
            MissionStatus.CANCELLED,
        }:
            return mission

        mission.status = MissionStatus.RUNNING
        self.store.save(mission)

        while mission.cursor < len(mission.steps):
            step = mission.steps[mission.cursor]
            if step.status in {StepStatus.COMPLETED, StepStatus.SKIPPED}:
                mission.cursor += 1
                self.store.save(mission)
                continue

            step.status = StepStatus.RUNNING
            step.started_at = datetime.now(UTC)
            self.store.save(mission)  # checkpoint BEFORE running

            try:
                new_status = handler(mission, step)
            except Exception as exc:  # noqa: BLE001 — handlers must not crash the engine
                step.status = StepStatus.FAILED
                step.error = f"{type(exc).__name__}: {exc}"
                step.finished_at = datetime.now(UTC)
                mission.status = MissionStatus.FAILED
                self.store.save(mission)
                return mission

            step.status = new_status
            step.finished_at = datetime.now(UTC)

            if new_status is StepStatus.FAILED:
                mission.status = MissionStatus.FAILED
                self.store.save(mission)
                return mission

            mission.cursor += 1
            self.store.save(mission)  # checkpoint AFTER running

        mission.status = MissionStatus.COMPLETED
        self.store.save(mission)
        return mission

    def resume_pending(self) -> list[Mission]:
        """Resume any missions that were left in an active state by a crash."""
        resumed: list[Mission] = []
        for mission in self.store.list_active():
            logger.info(
                "resuming mission %s (%s) from cursor %d", mission.id, mission.name, mission.cursor
            )
            resumed.append(self.run(mission))
        return resumed

    def cancel(self, mission_id: str) -> Mission | None:
        mission = self.store.load(mission_id)
        if mission is None:
            return None
        if not mission.is_done:
            mission.status = MissionStatus.CANCELLED
            self.store.save(mission)
        return mission
