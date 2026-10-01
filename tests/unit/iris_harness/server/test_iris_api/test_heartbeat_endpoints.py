"""Tests for heartbeat introspection endpoints."""

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


def _heartbeat_runtime() -> SimpleNamespace:
    scheduler = HeartbeatScheduler()
    scheduler.register_handler(
        "demo_tick",
        lambda definition: HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            output="demo_tick ok",
        ),
    )
    scheduler.register(
        HeartbeatDefinition(
            name="demo_tick",
            handler="demo_tick",
            schedule="interval:60",
            description="Demo heartbeat for tests.",
        )
    )
    return SimpleNamespace(heartbeats=scheduler)


def test_heartbeat_list_returns_registered_definitions() -> None:
    runtime = _heartbeat_runtime()
    client = TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    )

    response = client.get("/heartbeat")

    assert response.status_code == 200
    payload = response.json()
    assert payload["heartbeats"] == [
        {
            "name": "demo_tick",
            "schedule": "interval:60",
            "schedule_text": "every minute",
            "enabled": True,
            "description": "Demo heartbeat for tests.",
            "default_schedule": "interval:60",
            "default_enabled": True,
            "overridden": False,
            "runnable": True,
            "unavailable_reason": None,
            "platforms": [],
            "next_run_at": None,
            # Loop-proof D13: no run store on this scheduler, so no kept-run fields.
            "watched": False,
            "last_run": None,
            "last_success_at": None,
            "job": None,
        }
    ]
    assert payload["diagnostic_count"] == 0
    assert payload["diagnostics"] == []


def test_heartbeat_runs_endpoint_returns_run_history() -> None:
    runtime = _heartbeat_runtime()
    client = TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    )

    # Fire twice to populate run history.
    assert client.post("/heartbeat/trigger/demo_tick").status_code == 200
    assert client.post("/heartbeat/trigger/demo_tick").status_code == 200

    response = client.get("/heartbeat/runs")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["total"] == 2
    statuses = [run["status"] for run in payload["runs"]]
    assert statuses == ["success", "success"]
    assert all(run["name"] == "demo_tick" for run in payload["runs"])
    assert all(run["output"] == "demo_tick ok" for run in payload["runs"])


def test_heartbeat_runs_endpoint_filters_by_name_and_limit() -> None:
    runtime = _heartbeat_runtime()
    runtime.heartbeats.register_handler(
        "other_tick",
        lambda definition: HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            output="other_tick ok",
        ),
    )
    runtime.heartbeats.register(
        HeartbeatDefinition(
            name="other_tick",
            handler="other_tick",
            schedule="interval:60",
        )
    )
    client = TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    )

    for _ in range(3):
        client.post("/heartbeat/trigger/demo_tick")
    client.post("/heartbeat/trigger/other_tick")

    filtered = client.get("/heartbeat/runs", params={"name": "demo_tick"}).json()
    assert filtered["count"] == 3
    assert filtered["total"] == 3
    assert all(run["name"] == "demo_tick" for run in filtered["runs"])

    limited = client.get("/heartbeat/runs", params={"limit": 2}).json()
    assert limited["count"] == 2
    assert limited["total"] == 4
    assert [run["name"] for run in limited["runs"]] == ["demo_tick", "other_tick"]


def test_heartbeat_runs_endpoint_returns_empty_when_no_runs() -> None:
    runtime = _heartbeat_runtime()
    client = TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    )

    response = client.get("/heartbeat/runs")

    assert response.status_code == 200
    assert response.json() == {"count": 0, "total": 0, "runs": []}


def test_heartbeat_list_says_which_job_ran_and_which_missed(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Loop-proof D13: kept runs feed the Heartbeats list — last run, last success, and
    a missed slot — and survive a restart (a new scheduler over the same store)."""
    from datetime import UTC, datetime, timedelta

    from iris_harness.services.heartbeat.run_store import HeartbeatRunStore
    from iris_harness.services.heartbeat.slots import clock

    monkeypatch.setenv("IRIS_TZ", "UTC")
    now = datetime.now(UTC)
    store = HeartbeatRunStore(db_path=tmp_path / "heartbeat_runs.db")
    store.note_job("email_sweep", now=now - timedelta(days=2))
    store.note_job("email_judge", now=now - timedelta(days=2))
    two_hours_ago = (now - timedelta(hours=2)).replace(second=0, microsecond=0)
    hour_ago = (now - timedelta(hours=1)).replace(second=0, microsecond=0)
    store.record(
        HeartbeatRun(
            name="email_sweep",
            status=HeartbeatStatus.SUCCESS,
            started_at=two_hours_ago,
            output="23 new",
        )
    )
    store.record(
        HeartbeatRun(
            name="email_judge",
            status=HeartbeatStatus.SUCCESS,
            started_at=hour_ago,
            output="judged 23 · waiting 0",
            result={"judged": 23, "waiting": 0, "unreachable": False},
        )
    )
    # A restart: a new scheduler over the kept runs.
    scheduler = HeartbeatScheduler(run_store=store)
    scheduler.register_handler(
        "h", lambda d: HeartbeatRun(name=d.name, status=HeartbeatStatus.SUCCESS)
    )
    scheduler.register(
        HeartbeatDefinition(
            name="email_sweep", handler="h", schedule=f"{two_hours_ago.minute} * * * *"
        )
    )
    scheduler.register(
        HeartbeatDefinition(name="email_judge", handler="h", schedule=f"{hour_ago.minute} * * * *")
    )
    client = TestClient(
        create_app(runtime=SimpleNamespace(heartbeats=scheduler), auto_start_runtime=False),
        headers=auth_headers(),
    )

    beats = {b["name"]: b for b in client.get("/heartbeat").json()["heartbeats"]}

    sweep, judge = beats["email_sweep"], beats["email_judge"]
    assert sweep["watched"] is True
    assert sweep["last_run"]["output"] == "23 new"
    assert sweep["job"]["missed"] is True and sweep["job"]["state"] == "red"
    assert sweep["job"]["detail"].startswith("Missed ")
    # The detail names the day when it is not today (slots.clock): between 00:00 and
    # 02:00 UTC two hours ago was yesterday, so the expectation uses the same rule.
    assert f"last success {clock(two_hours_ago, now, UTC)}" in sweep["job"]["detail"]
    assert judge["job"]["state"] == "green" and judge["job"]["missed"] is False
    assert judge["job"]["detail"] == f"ran {clock(hour_ago, now, UTC)} ✓ (judged 23 · waiting 0)"
    assert judge["last_run"]["result"]["judged"] == 23
    assert judge["last_success_at"] == hour_ago.isoformat()

    runs = client.get("/heartbeat/runs", params={"name": "email_judge"}).json()
    assert runs["source"] == "kept" and runs["count"] == 1
    assert runs["runs"][0]["result"] == {"judged": 23, "waiting": 0, "unreachable": False}
