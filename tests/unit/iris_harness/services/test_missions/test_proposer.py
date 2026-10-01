"""Mission auto-creation proposers + generic handler (propose-not-act)."""

from __future__ import annotations

from iris_harness.services.missions.engine import MissionEngine
from iris_harness.services.missions.handlers import make_agent_query_handler
from iris_harness.services.missions.models import MissionStatus, StepStatus
from iris_harness.services.missions.proposer import (
    MISSION_HANDLER,
    build_episodic_mission,
    build_multi_step_mission,
    proposal_dedup_id,
)
from iris_harness.services.missions.store import MissionStore


def test_build_multi_step_mission_is_single_step_proposal() -> None:
    # No pre-splitting: the whole goal is one step; the TaskPlanner decomposes at
    # run time (a naive connector split would mangle "add 3 and 4").
    m = build_multi_step_mission(
        "add 3 and 4 together and then tell me a fun fact", intent="system"
    )
    assert m.status is MissionStatus.PENDING
    assert m.handler == MISSION_HANDLER
    assert len(m.steps) == 1
    assert m.steps[0].payload["query"] == "add 3 and 4 together and then tell me a fun fact"
    assert m.metadata["source"] == "multi_step_chat"
    assert m.metadata["pending_approval"] is True


def test_build_episodic_mission() -> None:
    m = build_episodic_mission("Every Monday reconcile finances across 3 accounts")
    assert m.status is MissionStatus.PENDING
    assert m.metadata["source"] == "episodic"
    assert m.metadata["dedup_id"] == proposal_dedup_id(
        "episodic", "Every Monday reconcile finances across 3 accounts"
    )


def test_dedup_id_stable_and_case_insensitive() -> None:
    a = proposal_dedup_id("chat", "Draft the Report")
    b = proposal_dedup_id("chat", "draft   the report")
    assert a == b  # normalized + lowercased


def test_agent_query_handler_runs_each_step(tmp_path) -> None:  # type: ignore[no-untyped-def]
    calls: list[str] = []

    def run_query(q: str) -> str:
        calls.append(q)
        return f"answer to {q}"

    store = MissionStore(db_path=tmp_path / "missions.db")
    engine = MissionEngine(
        store=store, handlers={MISSION_HANDLER: make_agent_query_handler(run_query)}
    )
    # A single-step mission runs its whole goal through the agent (one run_query call).
    mission = build_multi_step_mission("do A and do B")
    store.save(mission)

    done = engine.run(mission)
    assert done.status is MissionStatus.COMPLETED
    assert calls == ["do A and do B"]
    assert all(s.status is StepStatus.COMPLETED for s in done.steps)
    assert done.steps[0].output == "answer to do A and do B"


def test_agent_query_handler_step_error_fails_cleanly(tmp_path) -> None:  # type: ignore[no-untyped-def]
    def boom(_q: str) -> str:
        raise RuntimeError("model down")

    store = MissionStore(db_path=tmp_path / "missions.db")
    engine = MissionEngine(store=store, handlers={MISSION_HANDLER: make_agent_query_handler(boom)})
    mission = build_episodic_mission("weekly review of open items")
    store.save(mission)

    done = engine.run(mission)
    assert done.status is MissionStatus.FAILED
    assert done.steps[0].status is StepStatus.FAILED
    assert "model down" in done.steps[0].error
