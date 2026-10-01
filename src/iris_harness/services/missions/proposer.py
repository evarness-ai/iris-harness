"""Propose (never auto-run) missions from real signals — the auto-creation seam.

Closes the "missions are never auto-created" edge (ARCHITECTURE §2). Two sources:

1. **Multi-step chat goals** — a turn the IntentRouter flags ``is_multi_step``.
2. **Recurring episodic patterns** — bullets mined from ``~/.iris/memory/episodic.md``.

These builders are **pure**: they construct ``PENDING`` :class:`Mission` objects with
metadata marking the source + ``pending_approval``. They never persist or run anything
— the runtime saves them and surfaces an Action-Center approval task (propose-not-act,
HITL). All of it is gated off by default behind ``IRIS_MISSION_AUTOCREATE`` at the call
sites. The bound handler is :data:`MISSION_HANDLER` (a generic agent-query runner).
"""

from __future__ import annotations

import hashlib

from iris_harness.services.missions.models import Mission, MissionStatus, MissionStep

MISSION_HANDLER = "agent_query"


def proposal_dedup_id(source: str, key: str) -> str:
    """Stable id for a proposal so the same goal/pattern doesn't re-propose."""
    norm = " ".join((key or "").lower().split())
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
    return f"mission-{source}-{digest}"


def _title(text: str, limit: int = 60) -> str:
    t = " ".join((text or "").split())
    return (t[: limit - 1] + "…") if len(t) > limit else (t or "mission")


def build_multi_step_mission(query: str, *, intent: str = "") -> Mission:
    """A PENDING mission proposed from a multi-step chat goal.

    The whole goal is stored as a **single step**; the model-driven decomposition into
    sub-tasks happens at *run time* via the TaskPlanner when the ``agent_query`` step
    executes (``runtime.chat`` runs the full agentic pipeline). We deliberately do NOT
    pre-split here — a naive connector split mangles goals like "add 3 and 4" and the
    planner already does this well.
    """
    q = " ".join((query or "").split())
    return Mission(
        name=_title(q),
        handler=MISSION_HANDLER,
        steps=[MissionStep(name=q[:120], payload={"query": q})],
        status=MissionStatus.PENDING,
        metadata={
            "source": "multi_step_chat",
            "origin_query": query,
            "intent": intent,
            "pending_approval": True,
            "dedup_id": proposal_dedup_id("chat", query),
        },
    )


def build_episodic_mission(pattern: str) -> Mission:
    """A PENDING mission proposed from a recurring episodic pattern."""
    return Mission(
        name=_title(pattern),
        handler=MISSION_HANDLER,
        steps=[MissionStep(name=pattern[:120], payload={"query": pattern})],
        status=MissionStatus.PENDING,
        metadata={
            "source": "episodic",
            "pattern": pattern,
            "pending_approval": True,
            "dedup_id": proposal_dedup_id("episodic", pattern),
        },
    )


__all__ = [
    "MISSION_HANDLER",
    "proposal_dedup_id",
    "build_multi_step_mission",
    "build_episodic_mission",
]
