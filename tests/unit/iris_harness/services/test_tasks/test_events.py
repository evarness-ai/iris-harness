"""Tests for tasks-subsystem event emission (ADR-0005 + ADR-0014).

Verifies the TaskStore emits the right typed payload on each mutation
when a bus is wired, and stays silent when bus is None.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from iris_harness.foundation.eventbus import EventBus
from iris_harness.services.tasks import (
    GOAL_ACHIEVED,
    GOAL_CREATED,
    GOAL_UPDATED,
    TASK_COMPLETED,
    TASK_CREATED,
    TASK_DROPPED,
    TASK_UPDATED,
    TASK_WAIT_RESOLVED,
    GoalAchievedPayload,
    GoalCreatedPayload,
    GoalUpdatedPayload,
    TaskCompletedPayload,
    TaskCreatedPayload,
    TaskDroppedPayload,
    TaskStore,
    TaskUpdatedPayload,
    TaskWaitResolvedPayload,
    WaitFor,
)


def _now() -> datetime:
    return datetime.now(UTC)


@pytest.fixture
def bus_and_store(tmp_path: Path) -> tuple[EventBus, TaskStore, dict[str, list[Any]]]:
    """A bus, a wired store, and a captured-payloads dict per topic."""
    bus = EventBus()
    store = TaskStore(db_path=tmp_path / "tasks.db", bus=bus)
    store.ensure_schema()
    captured: dict[str, list[Any]] = {
        TASK_CREATED: [],
        TASK_UPDATED: [],
        TASK_COMPLETED: [],
        TASK_DROPPED: [],
        TASK_WAIT_RESOLVED: [],
        GOAL_CREATED: [],
        GOAL_UPDATED: [],
        GOAL_ACHIEVED: [],
    }
    for topic, sink in captured.items():
        bus.on(topic, sink.append)
    return bus, store, captured


# ─── Task events ──────────────────────────────────────────────────────


def test_create_emits_task_created(bus_and_store) -> None:
    _, store, captured = bus_and_store
    t = store.create(title="Reply to Bob", source_kind="manual", dedup_key="k1")

    assert len(captured[TASK_CREATED]) == 1
    payload = captured[TASK_CREATED][0]
    assert isinstance(payload, TaskCreatedPayload)
    assert payload.task_id == t.id
    assert payload.title == "Reply to Bob"
    assert payload.status == "open"
    assert payload.source_kind == "manual"
    assert payload.dedup_key == "k1"


def test_upsert_collision_does_not_re_emit_task_created(bus_and_store) -> None:
    """First write fires task.created; collision returns existing without re-emitting."""
    _, store, captured = bus_and_store
    store.upsert(dedup_key="k1", title="First")
    store.upsert(dedup_key="k1", title="Second")

    assert len(captured[TASK_CREATED]) == 1


def test_update_emits_task_updated_with_changed_fields(bus_and_store) -> None:
    _, store, captured = bus_and_store
    t = store.create(title="x")
    store.update(t.id, title="y", priority=5)

    assert len(captured[TASK_UPDATED]) == 1
    payload = captured[TASK_UPDATED][0]
    assert isinstance(payload, TaskUpdatedPayload)
    assert payload.task_id == t.id
    assert payload.status == "open"
    assert set(payload.changed_fields) == {"title", "priority"}


def test_complete_emits_task_completed(bus_and_store) -> None:
    _, store, captured = bus_and_store
    t = store.create(title="x", parent_goal_id=None)
    store.complete(t.id)

    # `complete` calls `update` internally, so we expect both events.
    assert len(captured[TASK_UPDATED]) == 1
    assert len(captured[TASK_COMPLETED]) == 1

    payload = captured[TASK_COMPLETED][0]
    assert isinstance(payload, TaskCompletedPayload)
    assert payload.task_id == t.id
    assert payload.completed_at is not None


def test_drop_emits_task_dropped(bus_and_store) -> None:
    _, store, captured = bus_and_store
    t = store.create(title="x")
    store.drop(t.id)

    assert len(captured[TASK_DROPPED]) == 1
    payload = captured[TASK_DROPPED][0]
    assert isinstance(payload, TaskDroppedPayload)
    assert payload.task_id == t.id


def test_resolve_wait_emits_task_wait_resolved(bus_and_store) -> None:
    _, store, captured = bus_and_store
    wf = WaitFor(kind="reply_from", payload={"address": "bob@example.com"})
    t = store.create(title="Followup", wait_for=wf)
    store.resolve_wait(t.id, by_event="email/abc")

    assert len(captured[TASK_WAIT_RESOLVED]) == 1
    payload = captured[TASK_WAIT_RESOLVED][0]
    assert isinstance(payload, TaskWaitResolvedPayload)
    assert payload.task_id == t.id
    assert payload.by_event == "email/abc"
    assert payload.resolved_at is not None


# ─── Goal events ──────────────────────────────────────────────────────


def test_create_goal_emits_goal_created(bus_and_store) -> None:
    _, store, captured = bus_and_store
    g = store.create_goal(title="Save $5K")

    assert len(captured[GOAL_CREATED]) == 1
    payload = captured[GOAL_CREATED][0]
    assert isinstance(payload, GoalCreatedPayload)
    assert payload.goal_id == g.id
    assert payload.title == "Save $5K"
    assert payload.status == "active"


def test_update_goal_pauses_emits_goal_updated(bus_and_store) -> None:
    _, store, captured = bus_and_store
    g = store.create_goal(title="x")
    store.update_goal(g.id, status="paused")

    assert len(captured[GOAL_UPDATED]) == 1
    assert len(captured[GOAL_ACHIEVED]) == 0
    payload = captured[GOAL_UPDATED][0]
    assert isinstance(payload, GoalUpdatedPayload)
    assert payload.status == "paused"
    assert set(payload.changed_fields) == {"status"}


def test_update_goal_to_achieved_emits_goal_achieved_not_updated(bus_and_store) -> None:
    """Achieved is a terminal state with its own topic; goal.updated should NOT fire."""
    _, store, captured = bus_and_store
    g = store.create_goal(title="x")
    store.update_goal(g.id, status="achieved", completed_at=_now())

    assert len(captured[GOAL_ACHIEVED]) == 1
    assert len(captured[GOAL_UPDATED]) == 0
    payload = captured[GOAL_ACHIEVED][0]
    assert isinstance(payload, GoalAchievedPayload)
    assert payload.goal_id == g.id
    assert payload.completed_at is not None


# ─── Silent mode ──────────────────────────────────────────────────────


def test_no_bus_means_no_emission(tmp_path: Path) -> None:
    """A store without a bus must not raise on any mutation."""
    store = TaskStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    t = store.create(title="x")
    store.update(t.id, title="y")
    store.complete(t.id)
    g = store.create_goal(title="g")
    store.update_goal(g.id, status="paused")
    # No assertions beyond "did not raise" — silence is the contract.
