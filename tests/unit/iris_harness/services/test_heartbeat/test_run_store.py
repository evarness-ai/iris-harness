"""Kept heartbeat runs (loop-proof D13): durable, cheap, never fatal, no floods."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from iris_harness.services.heartbeat import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatScheduler,
    HeartbeatStatus,
    load_heartbeats,
)
from iris_harness.services.heartbeat.config import HeartbeatConfigError
from iris_harness.services.heartbeat.run_store import HeartbeatRunStore
from iris_harness.services.heartbeat.scheduler import records_every_run

CHICAGO = ZoneInfo("America/Chicago")


def _scheduler(tmp_path: Path, handler, *defs: HeartbeatDefinition) -> HeartbeatScheduler:  # type: ignore[no-untyped-def]
    scheduler = HeartbeatScheduler(
        run_store=HeartbeatRunStore(db_path=tmp_path / "heartbeat_runs.db"), timezone=CHICAGO
    )
    scheduler.register_handler("h", handler)
    for d in defs:
        scheduler.register(d)
    return scheduler


def _ok(definition: HeartbeatDefinition) -> HeartbeatRun:
    return HeartbeatRun(
        name=definition.name,
        status=HeartbeatStatus.SUCCESS,
        output="23 new",
        result={"new": 23, "unreachable": False},
    )


def test_a_run_survives_a_new_store_instance(tmp_path: Path) -> None:
    sweep = HeartbeatDefinition(name="sweep", handler="h", schedule="15 6,12,18 * * *")
    scheduler = _scheduler(tmp_path, _ok, sweep)

    scheduler.trigger_by_name("sweep")

    reopened = HeartbeatRunStore(db_path=tmp_path / "heartbeat_runs.db")
    [kept] = reopened.recent(name="sweep")
    assert kept.status == "success"
    assert kept.output == "23 new"
    assert kept.result == {"new": 23, "unreachable": False}
    assert kept.trigger == "manual"
    assert kept.slot is not None  # the cron slot it served
    assert kept.finished_at is not None and kept.started_at <= kept.finished_at
    assert reopened.first_seen("sweep") is not None
    assert reopened.last_success("sweep") == kept


def test_every_cron_run_is_kept_but_a_fast_tick_keeps_only_changes(tmp_path: Path) -> None:
    outcomes = iter(
        [HeartbeatStatus.SUCCESS] * 5 + [HeartbeatStatus.FAILED] * 3 + [HeartbeatStatus.SUCCESS]
    )

    def handler(definition: HeartbeatDefinition) -> HeartbeatRun:
        return HeartbeatRun(name=definition.name, status=next(outcomes))

    tick = HeartbeatDefinition(name="tick", handler="h", schedule="interval:60")
    scheduler = _scheduler(tmp_path, handler, tick)
    for _ in range(9):
        scheduler.trigger_by_name("tick")

    kept = [r.status for r in reversed(scheduler.kept_runs(name="tick"))]
    assert kept == ["success", "failed", "success"]  # 9 runs, 3 rows
    assert len(scheduler.runs()) == 9  # memory still has every tick

    cron = HeartbeatDefinition(name="cron", handler="h", schedule="*/5 * * * *")
    outcomes = iter([HeartbeatStatus.SUCCESS] * 3)
    scheduler.register(cron)
    for _ in range(3):
        scheduler.trigger_by_name("cron")
    assert len(scheduler.kept_runs(name="cron")) == 3


@pytest.mark.parametrize(
    ("schedule", "record_runs", "expected"),
    [
        ("15 6 * * *", None, True),
        ("interval:600", None, True),
        ("interval:300", None, True),
        ("interval:299", None, False),
        ("interval:60", True, True),
        ("15 6 * * *", False, False),
    ],
)
def test_record_policy(schedule: str, record_runs: bool | None, expected: bool) -> None:
    definition = HeartbeatDefinition(
        name="x", handler="h", schedule=schedule, record_runs=record_runs
    )
    assert records_every_run(definition) is expected


def test_record_runs_is_read_from_heartbeats_yaml(tmp_path: Path) -> None:
    path = tmp_path / "heartbeats.yaml"
    path.write_text(
        "heartbeats:\n"
        "  - {name: a, handler: h, schedule: 'interval:60', record_runs: true}\n"
        "  - {name: b, handler: h, schedule: 'interval:60'}\n",
        encoding="utf-8",
    )
    a, b = load_heartbeats(path)
    assert a.record_runs is True and b.record_runs is None

    path.write_text(
        "heartbeats:\n  - {name: a, handler: h, schedule: 'interval:60', record_runs: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(HeartbeatConfigError):
        load_heartbeats(path)


def test_a_broken_store_never_fails_the_job(tmp_path: Path) -> None:
    (tmp_path / "heartbeat_runs.db").mkdir()  # a directory where the file should be
    sweep = HeartbeatDefinition(name="sweep", handler="h", schedule="15 6 * * *")
    scheduler = _scheduler(tmp_path, _ok, sweep)

    run = scheduler.trigger_by_name("sweep")

    assert run is not None and run.status is HeartbeatStatus.SUCCESS
    assert scheduler.kept_runs(name="sweep") == []  # the read degrades, never raises


def test_a_raising_store_never_fails_the_job(tmp_path: Path) -> None:
    class _Boom(HeartbeatRunStore):
        def record(self, run, *, slot=None):  # type: ignore[no-untyped-def]
            raise RuntimeError("disk full")

    scheduler = HeartbeatScheduler(run_store=_Boom(db_path=tmp_path / "x.db"), timezone=CHICAGO)
    scheduler.register_handler("h", _ok)
    scheduler.register(HeartbeatDefinition(name="sweep", handler="h", schedule="15 6 * * *"))

    run = scheduler.trigger_by_name("sweep")
    assert run is not None and run.status is HeartbeatStatus.SUCCESS


def test_the_scheduler_stamps_when_the_run_really_began(tmp_path: Path) -> None:
    """A handler that builds its run at the end would start it late; the slot check
    compares the real start."""
    late = datetime.now(UTC) + timedelta(minutes=5)

    def handler(definition: HeartbeatDefinition) -> HeartbeatRun:
        return HeartbeatRun(name=definition.name, status=HeartbeatStatus.SUCCESS, started_at=late)

    scheduler = _scheduler(
        tmp_path, handler, HeartbeatDefinition(name="j", handler="h", schedule="15 6 * * *")
    )
    run = scheduler.trigger_by_name("j")
    assert run is not None and run.started_at < late


def test_old_rows_are_pruned(tmp_path: Path) -> None:
    store = HeartbeatRunStore(db_path=tmp_path / "r.db", retention_days=30)
    now = datetime(2026, 9, 26, 12, tzinfo=UTC)
    for days in (40, 10):
        store.record(
            HeartbeatRun(
                name="j", status=HeartbeatStatus.SUCCESS, started_at=now - timedelta(days=days)
            )
        )
    assert store.prune(now=now) == 1
    assert len(store.recent(name="j")) == 1
