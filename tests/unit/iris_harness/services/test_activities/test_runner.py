"""ActivityRunner drives work through the store + notifies via the bus."""

from __future__ import annotations

import threading
from pathlib import Path

from iris_harness.foundation.eventbus import EventBus
from iris_harness.services.activities import (
    ACTIVITY_COMPLETED,
    ActivityCompletedPayload,
    ActivityOutcome,
    ActivityRunner,
    ActivityStore,
)


def _runner(tmp_path: Path, bus: EventBus | None = None) -> ActivityRunner:
    store = ActivityStore(db_path=tmp_path / "activities.db", bus=bus)
    store.ensure_schema()
    return ActivityRunner(store=store, bus=bus, max_workers=1)


def test_run_now_executes_and_completes(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    seen: list[tuple[float, str]] = []

    def work(progress):  # type: ignore[no-untyped-def]
        progress(0.5, "halfway")
        return ActivityOutcome(result_summary="done", undo_ref="u1", metadata={"n": 2})

    aid = runner.run_now(kind="test.job", title="Job", work=work, origin="chat:s1")
    a = runner.store.get(aid)
    assert a is not None
    assert a.status == "completed"
    assert a.result_summary == "done"
    assert a.undo_ref == "u1"
    assert a.metadata["n"] == 2
    assert a.progress == 1.0
    assert seen == []  # progress recorded on the row, not this list


def test_work_exception_marks_failed(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    def work(progress):  # type: ignore[no-untyped-def]
        raise RuntimeError("kaboom")

    aid = runner.run_now(kind="test.job", title="Job", work=work)
    a = runner.store.get(aid)
    assert a is not None
    assert a.status == "failed"
    assert "kaboom" in a.error


def test_submit_runs_in_background_and_notifies(tmp_path: Path) -> None:
    bus = EventBus()
    done = threading.Event()
    payloads: list[ActivityCompletedPayload] = []

    def on_completed(p: ActivityCompletedPayload) -> None:
        payloads.append(p)
        done.set()

    bus.on(ACTIVITY_COMPLETED, on_completed)
    runner = _runner(tmp_path, bus=bus)

    def work(progress):  # type: ignore[no-untyped-def]
        progress(0.9, "almost")
        return ActivityOutcome(result_summary="bg done")

    aid = runner.submit(kind="test.bg", title="BG", work=work, origin="chat:s2")
    assert done.wait(timeout=5.0), "completion event never fired"
    runner.shutdown(wait=True)

    assert payloads[0].activity_id == aid
    assert payloads[0].origin == "chat:s2"
    assert payloads[0].result_summary == "bg done"
    assert runner.store.get(aid).status == "completed"  # type: ignore[union-attr]
