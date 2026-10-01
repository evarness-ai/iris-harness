"""Tests for MissionStore SQLite roundtrip."""

from __future__ import annotations

from pathlib import Path

from iris_harness.services.missions import (
    Mission,
    MissionStatus,
    MissionStep,
    MissionStore,
    StepStatus,
)


def _make_store(tmp_path: Path) -> MissionStore:
    return MissionStore(tmp_path / "missions.db")


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    mission = Mission(
        name="onboard",
        handler="onboard_handler",
        steps=[
            MissionStep(name="welcome", payload={"channel": "telegram"}),
            MissionStep(name="profile"),
        ],
        metadata={"user_id": "u1"},
    )
    store.save(mission)
    loaded = store.load(mission.id)
    assert loaded is not None
    assert loaded.name == "onboard"
    assert loaded.handler == "onboard_handler"
    assert loaded.metadata == {"user_id": "u1"}
    assert [s.name for s in loaded.steps] == ["welcome", "profile"]
    assert loaded.steps[0].payload == {"channel": "telegram"}


def test_save_updates_existing_mission(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    mission = Mission(name="m", handler="h", steps=[MissionStep(name="s1")])
    store.save(mission)

    mission.cursor = 1
    mission.steps[0].status = StepStatus.COMPLETED
    mission.status = MissionStatus.COMPLETED
    store.save(mission)

    loaded = store.load(mission.id)
    assert loaded is not None
    assert loaded.cursor == 1
    assert loaded.status is MissionStatus.COMPLETED
    assert loaded.steps[0].status is StepStatus.COMPLETED


def test_list_active_filters_terminal_states(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    pending = Mission(name="p", handler="h", status=MissionStatus.PENDING)
    running = Mission(name="r", handler="h", status=MissionStatus.RUNNING)
    paused = Mission(name="pp", handler="h", status=MissionStatus.PAUSED)
    completed = Mission(name="c", handler="h", status=MissionStatus.COMPLETED)
    failed = Mission(name="f", handler="h", status=MissionStatus.FAILED)
    cancelled = Mission(name="x", handler="h", status=MissionStatus.CANCELLED)
    for m in (pending, running, paused, completed, failed, cancelled):
        store.save(m)

    names = sorted(m.name for m in store.list_active())
    assert names == ["p", "pp", "r"]


def test_delete_removes_mission(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    mission = Mission(name="m", handler="h")
    store.save(mission)
    assert store.delete(mission.id) is True
    assert store.load(mission.id) is None
    assert store.delete(mission.id) is False
