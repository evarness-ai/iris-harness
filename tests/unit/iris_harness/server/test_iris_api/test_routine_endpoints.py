"""Tests for routine control endpoints."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.server.iris_api.main import create_app
from iris_harness.services.heartbeat import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatScheduler,
    HeartbeatStatus,
)
from iris_harness.services.routines import RoutineStore
from iris_harness.services.routines.models import RoutineExecutionRecord, RoutineExecutionStatus


def _routine_runtime(tmp_path):
    store = RoutineStore(tmp_path / "routines.db")
    scheduler = HeartbeatScheduler()
    scheduler.register_handler(
        "routine_tick",
        lambda definition: HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            output="routine_tick due=1 executed=1 success=1 failed=0 skipped=0",
        ),
    )
    scheduler.register(
        HeartbeatDefinition(
            name="routine_tick",
            handler="routine_tick",
            schedule="interval:60",
        )
    )
    return SimpleNamespace(routine_store=store, heartbeats=scheduler)


def test_routine_api_create_approve_due_tick_and_delete(tmp_path) -> None:
    runtime = _routine_runtime(tmp_path)
    client = TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    )

    created = client.post(
        "/routines",
        json={
            "title": "Morning brief",
            "schedule": "interval:60",
            "template": "morning_briefing",
        },
    )
    assert created.status_code == 200
    routine = created.json()["routine"]
    routine_id = routine["id"]
    assert routine["approval_status"] == "draft"
    assert routine["goal"] == "Morning brief"

    listed = client.get("/routines")
    assert listed.status_code == 200
    assert listed.json()["count"] == 1
    assert listed.json()["routines"][0]["id"] == routine_id

    approved = client.patch(
        f"/routines/{routine_id}",
        json={"approval_status": "approved"},
    )
    assert approved.status_code == 200
    assert approved.json()["routine"]["approval_status"] == "approved"

    approved_list = client.get("/routines", params={"status": "approved"})
    assert approved_list.status_code == 200
    assert approved_list.json()["count"] == 1

    due = client.get("/routines/due")
    assert due.status_code == 200
    assert due.json()["count"] == 1
    assert due.json()["routines"][0]["id"] == routine_id

    tick = client.post("/routines/tick")
    assert tick.status_code == 200
    assert tick.json()["status"] == "success"
    assert "routine_tick" in tick.json()["output"]

    deleted = client.delete(f"/routines/{routine_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted_routine"]["id"] == routine_id
    assert client.get("/routines").json()["count"] == 0


def test_routine_api_clear_deletes_all_routines(tmp_path) -> None:
    runtime = _routine_runtime(tmp_path)
    client = TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    )
    first = client.post(
        "/routines",
        json={
            "title": "Morning brief",
            "schedule": "daily:08:00",
            "template": "morning_briefing",
        },
    ).json()["routine"]
    second = client.post(
        "/routines",
        json={
            "title": "Daily repo brief",
            "schedule": "daily:09:00",
            "template": "daily_repo_brief",
        },
    ).json()["routine"]

    response = client.delete("/routines")

    assert response.status_code == 200
    payload = response.json()
    assert payload["deleted_count"] == 2
    assert payload["count"] == 2
    assert [item["id"] for item in payload["deleted_routines"]] == [first["id"], second["id"]]
    assert client.get("/routines").json()["count"] == 0


def test_routine_api_returns_404_for_missing_routine(tmp_path) -> None:
    runtime = _routine_runtime(tmp_path)
    client = TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    )

    response = client.patch("/routines/missing", json={"approval_status": "paused"})

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def _run_preview_runtime(tmp_path, *, preview_raises: bool = False):
    """Runtime fake exposing run_routine / preview_routine for endpoint wiring."""
    store = RoutineStore(tmp_path / "routines.db")

    def run_routine(routine_id: str):
        if routine_id == "missing":
            return None
        return RoutineExecutionRecord(
            routine_id=routine_id,
            title="Morning brief",
            template="morning-briefing",
            status=RoutineExecutionStatus.SUCCESS,
            detail="delivered to telegram",
            heartbeat_status="success",
        )

    def preview_routine(routine_id: str):
        if routine_id == "missing":
            return None
        if preview_raises:
            raise ValueError("not a previewable brief")
        return "## Stocks\n- AAA"

    return SimpleNamespace(
        routine_store=store,
        run_routine=run_routine,
        preview_routine=preview_routine,
    )


def test_routine_run_endpoint_executes_single_routine(tmp_path) -> None:
    runtime = _run_preview_runtime(tmp_path)
    created = RoutineStore(tmp_path / "routines.db")  # noqa: F841 — ensure db exists
    client = TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    )

    resp = client.post("/routines/abc123/run")
    assert resp.status_code == 200
    body = resp.json()
    assert body["routine_id"] == "abc123"
    assert body["status"] == "success"
    assert body["detail"] == "delivered to telegram"

    missing = client.post("/routines/missing/run")
    assert missing.status_code == 404


def test_routine_preview_endpoint_returns_body_without_404(tmp_path) -> None:
    # seed one real routine so /preview's load() succeeds
    store = RoutineStore(tmp_path / "routines.db")
    from iris_harness.services.routines import create_routine_spec

    spec = create_routine_spec(
        title="Morning brief",
        goal="brief",
        schedule="daily:09:00",
        template="morning-briefing",
    )
    store.save(spec)
    runtime = _run_preview_runtime(tmp_path)
    runtime.routine_store = store
    client = TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    )

    resp = client.post(f"/routines/{spec.id}/preview")
    assert resp.status_code == 200
    assert "## Stocks" in resp.json()["body"]

    missing = client.post("/routines/missing/preview")
    assert missing.status_code == 404


def test_routine_preview_endpoint_422_for_non_brief(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    from iris_harness.services.routines import create_routine_spec

    spec = create_routine_spec(
        title="Tool routine",
        goal="x",
        schedule="daily:09:00",
        template="web-fetch",
    )
    store.save(spec)
    runtime = _run_preview_runtime(tmp_path, preview_raises=True)
    runtime.routine_store = store
    client = TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    )

    resp = client.post(f"/routines/{spec.id}/preview")
    assert resp.status_code == 422
    assert "previewable" in resp.json()["detail"]
