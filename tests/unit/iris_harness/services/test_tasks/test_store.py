"""Tests for the SQLite-backed TaskStore (ADR-0005)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_harness.services.tasks import TaskStore, WaitFor


def _now() -> datetime:
    return datetime.now(UTC)


@pytest.fixture
def store(tmp_path: Path) -> TaskStore:
    s = TaskStore(db_path=tmp_path / "tasks.db")
    s.ensure_schema()
    return s


# ─── Tasks ────────────────────────────────────────────────────────────


def test_create_assigns_id_when_omitted(store: TaskStore) -> None:
    t = store.create(title="Reply to Bob")
    assert t.id
    assert store.get(t.id) is not None


def test_create_respects_provided_id(store: TaskStore) -> None:
    t = store.create(id="custom-id", title="x")
    assert t.id == "custom-id"
    assert store.get("custom-id") is not None


def test_get_missing_returns_none(store: TaskStore) -> None:
    assert store.get("nonexistent") is None


def test_upsert_first_write_wins(store: TaskStore) -> None:
    first = store.upsert(dedup_key="k1", title="Original")
    second = store.upsert(dedup_key="k1", title="Different")
    assert first.id == second.id
    assert second.title == "Original"  # not overwritten


def test_upsert_after_drop_creates_new(store: TaskStore) -> None:
    """A dropped task does NOT block re-upserting with the same key —
    the UNIQUE index excludes status='dropped'."""
    a = store.upsert(dedup_key="k1", title="First")
    store.drop(a.id)
    b = store.upsert(dedup_key="k1", title="Second")
    assert b.id != a.id
    assert b.title == "Second"


def test_upsert_requires_nonempty_key(store: TaskStore) -> None:
    with pytest.raises(ValueError):
        store.upsert(dedup_key="", title="x")


def test_update_changes_fields_and_bumps_updated_at(store: TaskStore) -> None:
    t = store.create(title="x")
    before = t.updated_at
    updated = store.update(t.id, title="y", priority=5)
    assert updated.title == "y"
    assert updated.priority == 5
    assert updated.updated_at >= before


def test_update_rejects_immutable_fields(store: TaskStore) -> None:
    t = store.create(title="x")
    with pytest.raises(ValueError):
        store.update(t.id, id="new-id")
    with pytest.raises(ValueError):
        store.update(t.id, dedup_key="k1")


def test_update_missing_raises(store: TaskStore) -> None:
    with pytest.raises(KeyError):
        store.update("nonexistent", title="x")


def test_complete_sets_status_and_timestamp(store: TaskStore) -> None:
    t = store.create(title="x")
    done = store.complete(t.id)
    assert done.status == "done"
    assert done.completed_at is not None


def test_drop_sets_status_no_timestamp(store: TaskStore) -> None:
    t = store.create(title="x")
    dropped = store.drop(t.id)
    assert dropped.status == "dropped"
    assert dropped.completed_at is None  # 'dropped' uses status, not timestamp


def test_resolve_wait_sets_resolution_timestamp(store: TaskStore) -> None:
    wf = WaitFor(kind="reply_from", payload={"address": "bob@example.com"})
    t = store.create(title="Followup", wait_for=wf)

    resolved = store.resolve_wait(t.id, by_event="email/abc123")
    assert resolved.wait_for_resolved_at is not None
    assert resolved.status == "open"  # NOT auto-completed; caller decides


def test_resolve_wait_without_condition_raises(store: TaskStore) -> None:
    t = store.create(title="x")
    with pytest.raises(ValueError):
        store.resolve_wait(t.id, by_event="anything")


def test_list_filters_by_status(store: TaskStore) -> None:
    a = store.create(title="A")
    store.create(title="B")
    store.complete(a.id)

    opens = store.list(status="open")
    dones = store.list(status="done")
    assert {t.title for t in opens} == {"B"}
    assert {t.title for t in dones} == {"A"}


def test_list_filters_by_parent_task_id(store: TaskStore) -> None:
    parent = store.create(title="parent")
    child = store.create(title="child", parent_task_id=parent.id)
    store.create(title="orphan")

    children = store.list(parent_task_id=parent.id)
    assert [t.id for t in children] == [child.id]


def test_list_filters_by_due_before(store: TaskStore) -> None:
    soon = store.create(title="soon", due_at=_now() + timedelta(hours=1))
    store.create(title="later", due_at=_now() + timedelta(days=30))

    cutoff = _now() + timedelta(hours=2)
    matches = store.list(due_before=cutoff)
    assert [t.id for t in matches] == [soon.id]


def test_related_wikilinks_roundtrip(store: TaskStore) -> None:
    t = store.create(
        title="Discuss bills with Bob",
        related_wikilinks=("bob-landlord", "chase-bank"),
    )
    fetched = store.get(t.id)
    assert fetched is not None
    assert fetched.related_wikilinks == ("bob-landlord", "chase-bank")


def test_wait_for_roundtrip_through_db(store: TaskStore) -> None:
    wf = WaitFor(kind="event", payload={"topic": "calendar.changed", "ref": "evt-1"})
    t = store.create(title="Wait for cal event", wait_for=wf)

    fetched = store.get(t.id)
    assert fetched is not None
    assert fetched.wait_for == wf


# ─── Goals ────────────────────────────────────────────────────────────


def test_goal_create_and_get(store: TaskStore) -> None:
    g = store.create_goal(title="Save $5K", success_criteria="Balance >= 5000")
    fetched = store.get_goal(g.id)
    assert fetched is not None
    assert fetched.title == "Save $5K"


def test_goal_achieve_via_update_sets_completed_at(store: TaskStore) -> None:
    g = store.create_goal(title="x")
    achieved = store.update_goal(g.id, status="achieved", completed_at=_now())
    assert achieved.status == "achieved"
    assert achieved.completed_at is not None


def test_list_goals_filters_by_status(store: TaskStore) -> None:
    a = store.create_goal(title="A")
    store.create_goal(title="B")
    store.update_goal(a.id, status="paused")

    actives = store.list_goals(status="active")
    pauseds = store.list_goals(status="paused")
    assert {g.title for g in actives} == {"B"}
    assert {g.title for g in pauseds} == {"A"}


def test_goal_update_rejects_immutable_fields(store: TaskStore) -> None:
    g = store.create_goal(title="x")
    with pytest.raises(ValueError):
        store.update_goal(g.id, id="new-id")


# ─── Schema idempotency ──────────────────────────────────────────────


def test_ensure_schema_is_idempotent(tmp_path: Path) -> None:
    s = TaskStore(db_path=tmp_path / "tasks.db")
    s.ensure_schema()
    s.ensure_schema()  # second call must not raise


def test_task_carries_into_db_correctly(store: TaskStore) -> None:
    """Round-trip every column we care about."""
    t = store.create(
        title="Pay Chase",
        description="$1,200 due",
        priority=5,
        source_kind="finance-bills",
        source_id="bill/chase/2026-06",
        dedup_key="bill:chase:2026-06",
        due_at=_now() + timedelta(days=7),
        related_wikilinks=("chase-bank",),
        calendar_visible=True,
    )
    fetched = store.get(t.id)
    assert fetched is not None
    assert fetched.title == "Pay Chase"
    assert fetched.priority == 5
    assert fetched.source_kind == "finance-bills"
    assert fetched.dedup_key == "bill:chase:2026-06"
    assert fetched.calendar_visible is True
    assert fetched.related_wikilinks == ("chase-bank",)


def test_bus_optional_silent_without_one(store: TaskStore) -> None:
    """Bus parameter is optional — store works silently when None."""
    assert store.bus is None
    t = store.create(title="x")
    assert t.id  # no error


# ─── Pending actions (ADR-0073) ───────────────────────────────────────


def test_action_round_trips_through_store(store: TaskStore) -> None:
    from iris_harness.services.tasks import TaskAction

    action = TaskAction(
        kind="copy_command",
        label="Set the statement password",
        command="poetry run iris finance secret set-password wingtip_credit_card_in",
    )
    t = store.create(
        title="Wingtip card statement needs a password",
        source_kind="finance-statements",
        action=action,
    )
    fetched = store.get(t.id)
    assert fetched is not None
    assert fetched.action == action
    assert fetched.action is not None and fetched.action.safe is False


def test_has_action_filter_separates_pending_actions_from_todos(store: TaskStore) -> None:
    from iris_harness.services.tasks import TaskAction

    store.create(title="buy milk")  # plain user todo, no action
    store.create(
        title="re-extract Northwind statement",
        source_kind="finance-statements",
        action=TaskAction(kind="re_extract", label="Re-extract", target_id="stmt1", safe=True),
    )
    actions = store.list(has_action=True)
    todos = store.list(has_action=False)
    assert [a.title for a in actions] == ["re-extract Northwind statement"]
    assert [t.title for t in todos] == ["buy milk"]


def test_source_kind_filter(store: TaskStore) -> None:
    store.create(title="buy milk", source_kind="manual")
    store.create(title="bill due", source_kind="finance-bills")
    fin = store.list(source_kind="finance-bills")
    assert [t.title for t in fin] == ["bill due"]


def test_action_survives_update(store: TaskStore) -> None:
    from iris_harness.services.tasks import TaskAction

    t = store.create(
        title="register unknown institution",
        source_kind="finance-statements",
        action=TaskAction(kind="register", label="Register", target_id="stmt9", safe=True),
    )
    # An unrelated field update must not drop the action payload.
    updated = store.update(t.id, priority=5)
    assert updated.action is not None
    assert updated.action.target_id == "stmt9"
