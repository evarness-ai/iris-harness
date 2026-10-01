"""Mission proposals — the propose-not-act half of auto-created missions (HITL).

Two triggers propose a mission and neither runs one: the turn pipeline's record stage
(a successful multi-step turn becomes a mission proposal) and the ``mission_proposal_tick``
heartbeat (recurring episodic patterns do). A proposal is a PENDING mission in the mission
store plus an Action Center approval task; the user approves it into a run. Opt-in via
``IRIS_MISSION_AUTOCREATE``, off by default.

Carved out of ``IrisRuntime`` at OSS plan M5.7 track C slice 13 as
``MissionProposals(host)``, held as ``runtime.mission_proposals``. Stateless.
:class:`MissionProposalsHost` declares the two runtime members read; the host is read
**at call time**, not captured. ``mission_autocreate_enabled`` and ``propose_mission`` are
public because the record stage reaches them through ``TurnHost.mission_proposals``;
``mission_proposal_heartbeat`` because the runtime registers it.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from iris_harness.services.heartbeat import HeartbeatDefinition, HeartbeatRun, HeartbeatStatus

if TYPE_CHECKING:
    from pathlib import Path

    from iris_harness.services.missions.engine import MissionEngine
    from iris_harness.services.missions.models import Mission

logger = logging.getLogger(__name__)


class MissionProposalsHost(Protocol):
    """The two runtime members mission proposals reach."""

    data_dir: Path
    mission_engine: MissionEngine


class MissionProposals:
    """Proposes missions for one runtime; never runs one. See the module docstring."""

    def __init__(self, host: MissionProposalsHost) -> None:
        self._host = host

    def mission_autocreate_enabled(self) -> bool:
        """Opt-in auto-creation of missions (propose-not-act). Off by default."""
        return os.getenv("IRIS_MISSION_AUTOCREATE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def propose_mission(self, mission: Mission) -> bool:
        """Persist a PENDING auto-created mission + surface an Action-Center approval
        task (propose-not-act, HITL). De-duplicated by ``metadata['dedup_id']`` so the
        same goal/pattern isn't re-proposed while still un-finished. Returns True when a
        new proposal was created. Best-effort — never raises into the caller."""
        try:
            dedup_id = str(mission.metadata.get("dedup_id") or mission.id)
            for existing in self._host.mission_engine.store.list_active():
                if not existing.is_done and str(existing.metadata.get("dedup_id")) == dedup_id:
                    return False  # already proposed / running
            self._host.mission_engine.store.save(mission)
            self._record_mission_approval_task(mission)
            logger.info(
                "auto-proposed mission %s (%s, source=%s, steps=%d)",
                mission.id,
                mission.name,
                mission.metadata.get("source"),
                len(mission.steps),
            )
            return True
        except Exception:  # proposal is best-effort
            logger.exception("failed to propose mission")
            return False

    def _record_mission_approval_task(self, mission: Mission) -> None:
        """Surface a PENDING mission as an Action-Center approval task. The action is a
        copy_command (`iris mission run <id>`) — channel-agnostic, runs nothing until
        the user approves. Mirrors the calendar-approval pattern (ADR-0076)."""
        try:
            from iris_harness.services.tasks import TaskStore
            from iris_harness.services.tasks.models import TaskAction

            store = TaskStore(db_path=self._host.data_dir / "tasks.db")
            store.ensure_schema()
            store.upsert(
                dedup_key=f"mission-approval:{mission.id}",
                title=f"Approve mission: {mission.name}",
                description=(
                    f"{len(mission.steps)} step(s) · source="
                    f"{mission.metadata.get('source', '?')}. Approve to run it, or reject "
                    f"to dismiss. Steps: {'; '.join(s.name for s in mission.steps)[:300]}"
                ),
                source_kind="mission-proposal",
                source_id=mission.id,
                action=TaskAction(
                    kind="copy_command",
                    label="Approve & run",
                    command=f"iris mission run {mission.id}",
                    safe=False,
                ),
            )
        except Exception:  # visibility is best-effort
            logger.exception("failed to record mission approval task")

    def mission_proposal_heartbeat(self, definition: HeartbeatDefinition) -> HeartbeatRun:
        """Proactive trigger: mine recurring episodic patterns into PENDING mission
        proposals (opt-in ``IRIS_MISSION_AUTOCREATE``). Never runs a mission."""
        proposed = 0
        if self.mission_autocreate_enabled():
            try:
                from iris_harness.memory.identity import list_episodic_patterns
                from iris_harness.services.missions.proposer import (
                    build_episodic_mission,
                )

                for pattern in list_episodic_patterns():
                    text = (getattr(pattern, "text", "") or "").strip()
                    if len(text) < 12:
                        continue
                    if self.propose_mission(build_episodic_mission(text)):
                        proposed += 1
            except Exception as exc:  # must not crash the scheduler
                logger.exception("mission_proposal_tick failed")
                return HeartbeatRun(
                    name=definition.name,
                    status=HeartbeatStatus.FAILED,
                    finished_at=datetime.now(UTC),
                    error=f"{type(exc).__name__}: {exc}",
                )
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            finished_at=datetime.now(UTC),
            output=json.dumps({"proposed": proposed}, sort_keys=True),
        )
