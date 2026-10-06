"""Hidden mail: ``judge_reachable`` and the digest's "Email jobs" line (loop-proof D13)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from iris_harness.services.digest import footer
from iris_harness.services.health import service as health_service
from iris_harness.services.health.models import HealthState
from iris_harness.services.heartbeat import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatScheduler,
    HeartbeatStatus,
)
from iris_harness.services.heartbeat.run_store import HeartbeatRunStore
from iris_personal.plugins.email_workflows import job_watch
from iris_personal.plugins.email_workflows.job_watch import (
    JobWatchConfig,
    email_jobs_line,
    judge_reachable_checks,
)
from iris_personal.plugins.email_workflows.judgments import JudgmentStore

CT = ZoneInfo("America/Chicago")
CONFIG = JobWatchConfig.load()
SCHEDULES = {"email_sweep": "15 6,12,18 * * *", "email_judge": "30 6,12,18 * * *"}


def _ct(hour: int, minute: int = 0, day: int = 26) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=CT)


@pytest.fixture
def runs(tmp_path: Path) -> HeartbeatRunStore:
    store = HeartbeatRunStore(db_path=tmp_path / "heartbeat_runs.db")
    for name in SCHEDULES:
        store.note_job(name, now=_ct(0, day=20))
    return store


@pytest.fixture
def judgments(tmp_path: Path) -> JudgmentStore:
    store = JudgmentStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    return store


def _waiting(judgments: JudgmentStore, count: int, since: datetime) -> None:
    judgments.mark_waiting("acct", [f"m{i}" for i in range(count)])
    with judgments._connect() as conn:  # date the queue
        conn.execute(
            "UPDATE email_judgments SET created_at = ?", (since.astimezone(UTC).isoformat(),)
        )


def _run(runs: HeartbeatRunStore, name: str, at: datetime, **kw) -> None:  # type: ignore[no-untyped-def]
    status = kw.pop("status", HeartbeatStatus.SUCCESS)
    runs.record(HeartbeatRun(name=name, status=status, started_at=at, finished_at=at, **kw))


def _unreachable(runs: HeartbeatRunStore, at: datetime) -> None:
    _run(
        runs,
        "email_judge",
        at,
        status=HeartbeatStatus.SKIPPED,
        error="Ollama at the Mac did not answer",
        result={"judged": 0, "waiting": 12, "unreachable": True},
    )


def _check(runs, judgments, *, reachable: bool, now: datetime, probes: list[int] | None = None):  # type: ignore[no-untyped-def]
    def probe() -> bool:
        if probes is not None:
            probes.append(1)
        return reachable

    [row] = judge_reachable_checks(
        runs=runs, judgments=judgments, probe=probe, config=CONFIG, tz=CT, now=now
    )
    assert row.target == "judge_reachable"
    return row


def test_nothing_waiting_is_green_and_does_not_probe(runs, judgments) -> None:  # type: ignore[no-untyped-def]
    probes: list[int] = []
    _unreachable(runs, _ct(6, 30))
    row = _check(runs, judgments, reachable=False, now=_ct(7), probes=probes)
    assert (row.state, row.detail) == (HealthState.GREEN, "nothing waiting")
    assert probes == []


def test_a_judge_run_that_could_not_reach_the_mac_is_red_with_the_waiting_count(
    runs, judgments  # type: ignore[no-untyped-def]
) -> None:
    _waiting(judgments, 12, _ct(6, 14))
    _unreachable(runs, _ct(6, 30))
    row = _check(runs, judgments, reachable=False, now=_ct(6, 35))
    assert row.state is HealthState.RED
    assert row.detail == "Mac unreachable · 12 emails waiting (hidden) · oldest since 06:14"
    assert row.action == "iris heartbeats trigger email_judge"
    # The Mac answers again, but the mail still waits for the next judge run.
    row = _check(runs, judgments, reachable=True, now=_ct(8))
    assert row.state is HealthState.RED and row.detail.startswith("judge could not reach")


def test_a_failed_probe_is_red_once_the_wait_is_past_the_judge_slot(runs, judgments) -> None:  # type: ignore[no-untyped-def]
    _waiting(judgments, 3, _ct(12, 15))
    young = _check(runs, judgments, reachable=False, now=_ct(12, 25))
    assert young.state is HealthState.YELLOW
    assert young.detail == "Mac unreachable · 3 emails waiting for the next judge run"
    old = _check(runs, judgments, reachable=False, now=_ct(12, 50))
    assert old.state is HealthState.RED
    assert old.detail == "Mac unreachable · 3 emails waiting (hidden) · oldest since 12:15"


def test_reachable_but_waiting_longer_than_a_judge_cycle_is_yellow(runs, judgments) -> None:  # type: ignore[no-untyped-def]
    _waiting(judgments, 2, _ct(6, 15))
    assert _check(runs, judgments, reachable=True, now=_ct(9)).state is HealthState.GREEN
    stale = _check(runs, judgments, reachable=True, now=_ct(13, 30))
    assert stale.state is HealthState.YELLOW and "has not drained" in stale.detail


def _line(runs, judgments, start: datetime, end: datetime) -> str | None:  # type: ignore[no-untyped-def]
    return email_jobs_line(
        runs=runs,
        judgments=judgments,
        schedules=lambda: dict(SCHEDULES),
        config=CONFIG,
        tz=CT,
        start=start,
        end=end,
    )


def test_footer_when_every_job_ran(runs, judgments) -> None:  # type: ignore[no-untyped-def]
    for hour in (6, 12, 18):
        _run(runs, "email_sweep", _ct(hour, 15, day=25))
        _run(runs, "email_judge", _ct(hour, 30, day=25))
    line = _line(runs, judgments, _ct(0, day=25), _ct(0))
    assert line == "Email jobs: sweep 3/3 · judge 3/3 · 0 waiting"


def test_footer_when_one_sweep_was_missed_and_the_mac_was_away(runs, judgments) -> None:  # type: ignore[no-untyped-def]
    _run(runs, "email_sweep", _ct(6, 15, day=25))
    _run(runs, "email_sweep", _ct(18, 15, day=25))
    _run(runs, "email_judge", _ct(6, 30, day=25))
    _run(runs, "email_judge", _ct(12, 30, day=25))
    _unreachable(runs, _ct(18, 30, day=25))
    _waiting(judgments, 12, _ct(18, 15, day=25))
    line = _line(runs, judgments, _ct(0, day=25), _ct(0))
    assert line == (
        "Email jobs: sweep 2/3 (12:15 missed) · judge 2/3 · 12 waiting (Mac unreachable)"
    )


def test_no_footer_line_when_no_email_job_is_scheduled(runs, judgments) -> None:  # type: ignore[no-untyped-def]
    assert (
        email_jobs_line(
            runs=runs,
            judgments=judgments,
            schedules=dict,
            config=CONFIG,
            tz=CT,
            start=_ct(0, day=25),
            end=_ct(0),
        )
        is None
    )


def test_register_adds_the_check_and_the_footer_line(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(health_service, "_check_providers", {})
    monkeypatch.setattr(footer, "_LINES", {})
    heartbeats = HeartbeatScheduler(
        run_store=HeartbeatRunStore(db_path=tmp_path / "heartbeat_runs.db"), timezone=CT
    )
    heartbeats.register_handler(
        "h", lambda d: HeartbeatRun(name=d.name, status=HeartbeatStatus.SUCCESS)
    )
    for name, schedule in SCHEDULES.items():
        heartbeats.register(HeartbeatDefinition(name=name, handler="h", schedule=schedule))
    api = SimpleNamespace(
        services=SimpleNamespace(
            data_dir=tmp_path, config_dir=tmp_path, heartbeats=heartbeats, tier_router=None
        ),
        register_footer_line=footer.register_footer_line,
    )
    monkeypatch.setattr(job_watch, "ollama_probe", lambda *a, **k: lambda: False)

    job_watch.register(api)

    [row] = health_service._check_providers["judge_reachable"]()
    assert row.state is HealthState.GREEN  # no mail, nothing hidden
    lines = footer.footer_lines(now=datetime.now(UTC) + timedelta(days=1), tz=CT)
    assert len(lines) == 1 and lines[0].startswith("Email jobs: sweep ")


def test_probe_reads_a_tier_on_an_unknown_provider_as_unreachable() -> None:
    from iris_harness.sdk.llm import provider_root_url

    probe = job_watch.ollama_probe(lambda: provider_root_url("typo"), "/api/version", 1.0)
    assert probe() is False
