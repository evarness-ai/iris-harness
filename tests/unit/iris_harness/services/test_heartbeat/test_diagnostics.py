"""Tests for heartbeat overdue/failure diagnostics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from iris_harness.services.heartbeat.diagnostics import diagnose_heartbeats
from iris_harness.services.heartbeat.models import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatStatus,
)


def test_interval_never_ran_is_reported() -> None:
    created = datetime(2026, 7, 3, 10, 0, tzinfo=UTC)
    now = created + timedelta(minutes=5)
    defs = [
        HeartbeatDefinition(
            name="routine_tick",
            handler="routine_tick",
            schedule="interval:60",
        )
    ]

    issues = diagnose_heartbeats(defs, [], created_at=created, now=now)

    assert len(issues) == 1
    assert issues[0].name == "routine_tick"
    assert issues[0].reason == "never_ran"
    assert issues[0].action == "iris heartbeats trigger routine_tick"


def test_failed_last_run_is_reported() -> None:
    created = datetime(2026, 7, 3, 10, 0, tzinfo=UTC)
    defs = [
        HeartbeatDefinition(
            name="finance_ingest_tick",
            handler="finance_ingest_tick",
            schedule="0 7 * * *",
        )
    ]
    runs = [
        HeartbeatRun(
            name="finance_ingest_tick",
            status=HeartbeatStatus.FAILED,
            started_at=datetime(2026, 7, 3, 7, 0, tzinfo=UTC),
            finished_at=datetime(2026, 7, 3, 7, 0, 10, tzinfo=UTC),
            error="gmail token revoked",
        )
    ]

    issues = diagnose_heartbeats(defs, runs, created_at=created, now=created + timedelta(hours=1))

    assert len(issues) == 1
    assert issues[0].reason == "last_run_failed"
    assert "gmail token revoked" in issues[0].detail


def test_recent_interval_success_not_reported() -> None:
    created = datetime(2026, 7, 3, 10, 0, tzinfo=UTC)
    defs = [
        HeartbeatDefinition(
            name="pressure_tick",
            handler="pressure_tick",
            schedule="interval:30",
        )
    ]
    runs = [
        HeartbeatRun(
            name="pressure_tick",
            status=HeartbeatStatus.SUCCESS,
            started_at=datetime(2026, 7, 3, 10, 0, 25, tzinfo=UTC),
            finished_at=datetime(2026, 7, 3, 10, 0, 25, tzinfo=UTC),
        )
    ]

    issues = diagnose_heartbeats(
        defs,
        runs,
        created_at=created,
        now=datetime(2026, 7, 3, 10, 0, 45, tzinfo=UTC),
    )

    assert issues == []
