"""The ``job_completed`` health check (loop-proof D13) and the watch paging on it."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from iris_harness.services.health import repair as repair_mod
from iris_harness.services.health.incidents import IncidentStore
from iris_harness.services.health.jobs import (
    TARGET,
    job_checks,
    load_watched_jobs,
    parse_jobs,
)
from iris_harness.services.health.models import HealthSnapshot, HealthState
from iris_harness.services.health.watch import HealthWatcher, WatchConfig
from iris_harness.services.heartbeat import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatScheduler,
    HeartbeatStatus,
)
from iris_harness.services.heartbeat.run_store import HeartbeatRunStore

CT = ZoneInfo("America/Chicago")
REPO = Path(__file__).resolve().parents[5]


@pytest.fixture(autouse=True)
def _no_plugin_repairers():  # type: ignore[no-untyped-def]
    repair_mod.clear_repairers()
    yield
    repair_mod.clear_repairers()


def _ct(hour: int, minute: int = 0, day: int = 26) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=CT)


def _heartbeats(tmp_path: Path) -> HeartbeatScheduler:
    store = HeartbeatRunStore(db_path=tmp_path / "heartbeat_runs.db")
    scheduler = HeartbeatScheduler(run_store=store, timezone=CT)
    scheduler.register_handler(
        "h", lambda d: HeartbeatRun(name=d.name, status=HeartbeatStatus.SUCCESS)
    )
    for name, schedule in (
        ("email_sweep", "15 6,12,18 * * *"),
        ("email_judge", "30 6,12,18 * * *"),
    ):
        scheduler.register(HeartbeatDefinition(name=name, handler="h", schedule=schedule))
        with store._connect() as conn:  # backdate the install
            conn.execute("UPDATE heartbeat_jobs SET first_seen = ?", (_ct(0, day=20).isoformat(),))
    return scheduler


def _ran(scheduler: HeartbeatScheduler, name: str, at: datetime, status=HeartbeatStatus.SUCCESS, **kw):  # type: ignore[no-untyped-def]
    assert scheduler.run_store is not None
    scheduler.run_store.record(
        HeartbeatRun(name=name, status=status, started_at=at, finished_at=at, **kw)
    )


JOBS = parse_jobs(
    {"grace_minutes": 30, "watch": ["email_sweep", {"name": "email_judge", "grace_minutes": 45}]}
)


def test_parse_jobs_reads_names_and_graces() -> None:
    assert [(j.name, j.grace) for j in JOBS] == [
        ("email_sweep", timedelta(minutes=30)),
        ("email_judge", timedelta(minutes=45)),
    ]
    assert parse_jobs(None) == () and parse_jobs({"watch": [3]}) == ()


def test_the_shipped_config_watches_the_email_jobs() -> None:
    assert [j.name for j in load_watched_jobs(REPO / "config")] == ["email_sweep", "email_judge"]


def test_one_row_per_job_red_naming_the_missed_slot(tmp_path: Path) -> None:
    heartbeats = _heartbeats(tmp_path)
    _ran(heartbeats, "email_sweep", _ct(6, 15), output="23 new")
    _ran(heartbeats, "email_judge", _ct(6, 30), output="judged 23 · waiting 0")
    _ran(heartbeats, "email_judge", _ct(12, 30), output="judged 4 · waiting 0")

    sweep, judge = job_checks(heartbeats, JOBS, now=_ct(12, 50))

    assert (sweep.target, sweep.subject, sweep.state) == (TARGET, "email_sweep", HealthState.RED)
    assert sweep.detail == "Missed 12:15 — last success 06:15"
    assert sweep.action == "iris heartbeats trigger email_sweep"
    assert (judge.state, judge.detail) == (HealthState.GREEN, "ran 12:30 ✓ (judged 4 · waiting 0)")
    assert judge.action is None


def test_grey_without_a_run_store(tmp_path: Path) -> None:
    heartbeats = HeartbeatScheduler(timezone=CT)
    [row, _] = job_checks(heartbeats, JOBS, now=_ct(12))
    assert row.state is HealthState.GREY


class _Outbox:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, Sequence[str] | None]] = []
        self.urls: list[str | None] = []

    def __call__(
        self, subject: str, body: str, channels: Sequence[str] | None, url: str | None = None
    ) -> None:
        self.sent.append((subject, body, channels))
        self.urls.append(url)


def test_the_watch_pages_the_owner_on_a_missed_slot(tmp_path: Path) -> None:
    heartbeats = _heartbeats(tmp_path)
    _ran(heartbeats, "email_sweep", _ct(6, 15))
    _ran(heartbeats, "email_judge", _ct(12, 30))
    outbox = _Outbox()
    watcher = HealthWatcher(
        store=IncidentStore(tmp_path / "health.db"),
        config=WatchConfig(max_attempts=0),
        notify=outbox,
    )

    now = _ct(12, 50)
    for tick in range(3):
        at = now + timedelta(minutes=tick)
        snapshot = HealthSnapshot(
            checks=tuple(job_checks(heartbeats, JOBS, now=at)), sampled_at=at.isoformat()
        )
        watcher.observe(snapshot, now=at)

    assert len(outbox.sent) == 1
    subject, body, _ = outbox.sent[0]
    assert "job_completed (email_sweep)" in subject + body
    assert "Missed 12:15" in body
    assert "iris heartbeats trigger email_sweep" in body
