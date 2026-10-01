"""ActivityStore CRUD + status transitions + bus events."""

from __future__ import annotations

from pathlib import Path

from iris_harness.foundation.eventbus import EventBus
from iris_harness.services.activities import (
    ACTIVITY_COMPLETED,
    ACTIVITY_FAILED,
    ACTIVITY_STARTED,
    ActivityCompletedPayload,
    ActivityStore,
)


def _store(tmp_path: Path, bus: EventBus | None = None) -> ActivityStore:
    store = ActivityStore(db_path=tmp_path / "activities.db", bus=bus)
    store.ensure_schema()
    return store


def test_create_defaults_to_queued(tmp_path: Path) -> None:
    store = _store(tmp_path)
    a = store.create(kind="filemanager.cleanup", title="Clean Downloads", origin="chat:s1")
    assert a.status == "queued"
    assert a.progress == 0.0
    fetched = store.get(a.id)
    assert fetched is not None
    assert fetched.title == "Clean Downloads"
    assert fetched.origin == "chat:s1"


def test_lifecycle_running_to_completed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    a = store.create(kind="filemanager.categorize", title="Categorize")
    store.mark_running(a.id)
    store.mark_progress(a.id, 0.5, "analyzed 100/200 images")
    done = store.mark_completed(
        a.id, result_summary="8 groups", undo_ref="plan-1", metadata={"categories": 8}
    )
    assert done.status == "completed"
    assert done.progress == 1.0
    assert done.result_summary == "8 groups"
    assert done.undo_ref == "plan-1"
    assert done.metadata["categories"] == 8
    assert done.started_at is not None
    assert done.finished_at is not None


def test_mark_failed_records_error(tmp_path: Path) -> None:
    store = _store(tmp_path)
    a = store.create(kind="filemanager.organize", title="Organize")
    store.mark_running(a.id)
    failed = store.mark_failed(a.id, "boom")
    assert failed.status == "failed"
    assert failed.error == "boom"
    assert failed.finished_at is not None


def test_progress_is_clamped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    a = store.create(kind="x", title="t")
    assert store.mark_progress(a.id, 5.0, "over").progress == 1.0
    assert store.mark_progress(a.id, -1.0, "under").progress == 0.0


def test_list_filters_by_status_and_origin(tmp_path: Path) -> None:
    store = _store(tmp_path)
    a = store.create(kind="k", title="a", origin="chat:s1")
    store.create(kind="k", title="b", origin="chat:s2")
    store.mark_running(a.id)
    store.mark_completed(a.id, result_summary="ok")
    assert {x.title for x in store.list(status="completed")} == {"a"}
    assert {x.title for x in store.list(origin="chat:s2")} == {"b"}
    assert len(store.list()) == 2


def test_bus_events_fire_on_transitions(tmp_path: Path) -> None:
    bus = EventBus()
    topics: list[str] = []
    completed: list[ActivityCompletedPayload] = []
    bus.on(ACTIVITY_STARTED, lambda _p: topics.append("started"))
    bus.on(ACTIVITY_FAILED, lambda _p: topics.append("failed"))
    bus.on(ACTIVITY_COMPLETED, lambda p: (topics.append("completed"), completed.append(p)))
    store = _store(tmp_path, bus=bus)

    a = store.create(kind="filemanager.cleanup", title="Clean", origin="chat:s1")
    store.mark_running(a.id)
    store.mark_completed(a.id, result_summary="3 groups", undo_ref="plan-9")

    assert topics == ["started", "completed"]
    assert completed[0].origin == "chat:s1"
    assert completed[0].result_summary == "3 groups"
    assert completed[0].undo_ref == "plan-9"
