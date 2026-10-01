"""Mission auto-creation wiring (propose-not-act, HITL) at the runtime level."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from iris_harness.runtime import build_runtime
from iris_harness.services.heartbeat.models import HeartbeatStatus
from iris_harness.services.missions.proposer import build_episodic_mission
from iris_harness.services.tasks import TaskStore

_PATTERN = "Every Monday reconcile finances across 3 accounts"


def _runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    config_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    return build_runtime(config_dir=config_dir, data_dir=data_dir, use_background_scheduler=False)


def test_propose_mission_saves_pending_and_action_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rt = _runtime(monkeypatch, tmp_path)

    created = rt.mission_proposals.propose_mission(build_episodic_mission(_PATTERN))
    assert created is True

    # PENDING mission persisted
    actives = rt.mission_engine.store.list_active()
    assert any(m.metadata.get("source") == "episodic" for m in actives)

    # Action-Center approval task surfaced, with a runnable approve command
    ts = TaskStore(db_path=tmp_path / "data" / "tasks.db")
    tasks = [t for t in ts.list(has_action=True, limit=50) if t.source_kind == "mission-proposal"]
    assert len(tasks) == 1
    assert tasks[0].action is not None
    assert tasks[0].action.command.startswith("iris mission run ")

    # de-dup: an identical pattern does not re-propose
    assert rt.mission_proposals.propose_mission(build_episodic_mission(_PATTERN)) is False
    assert len(rt.mission_engine.store.list_active()) == 1


def test_mission_proposal_heartbeat_inert_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("IRIS_MISSION_AUTOCREATE", raising=False)
    rt = _runtime(monkeypatch, tmp_path)

    run = rt.mission_proposals.mission_proposal_heartbeat(
        SimpleNamespace(name="mission_proposal_tick")
    )
    assert run.status is HeartbeatStatus.SUCCESS
    assert json.loads(run.output)["proposed"] == 0  # disabled → nothing proposed
    assert rt.mission_engine.store.list_active() == []


# ── the carve (OSS plan M5.7 track C, slice 13) ───────────────────────────────


def test_the_record_stage_proposes_a_multi_step_turn_through_the_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The record stage reaches mission proposals through ``TurnHost.mission_proposals``.
    The unit tests above drive the collaborator directly, so without this nothing fails
    if the stage stops proposing or reads the gate from somewhere else."""
    from iris_harness.agent.intent_router import IntentResult
    from iris_harness.runtime.turn.stages import record
    from iris_harness.runtime.turn.state import TurnRequest, TurnState
    from iris_harness.runtime.types import ChatResult

    monkeypatch.setenv("IRIS_MISSION_AUTOCREATE", "1")
    rt = _runtime(monkeypatch, tmp_path)
    state = TurnState(request=TurnRequest(message=_PATTERN, session_id="mission-stage"))
    state.intent_result = IntentResult(
        intent="finance", agent_type="finance", confidence=0.9, is_multi_step=True
    )
    state.result = ChatResult(
        response="Done.",
        intent="finance",
        agent_type="finance",
        has_errors=False,
        sources=[],
        error_summary="",
        metadata={},
    )

    list(record.run(rt, state))

    assert [m.metadata.get("source") for m in rt.mission_engine.store.list_active()] == [
        "multi_step_chat"
    ]


def test_the_scheduled_tick_is_the_collaborators_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from iris_harness.runtime.mission_proposals import MissionProposals

    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    rt = _runtime(monkeypatch, tmp_path)
    rt.startup()
    try:
        handler = rt.heartbeats._handlers["mission_proposal_tick"]
        assert handler.__self__ is rt.mission_proposals
        assert handler.__func__ is MissionProposals.mission_proposal_heartbeat
    finally:
        rt.shutdown()


def test_mission_proposals_host_declares_exactly_what_the_module_reaches() -> None:
    import ast

    tree = ast.parse(Path("src/iris_harness/runtime/mission_proposals.py").read_text())
    reached = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "_host"
    }
    host = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MissionProposalsHost"
    )
    declared = {n.target.id for n in host.body if isinstance(n, ast.AnnAssign)}
    assert reached == declared == {"data_dir", "mission_engine"}
