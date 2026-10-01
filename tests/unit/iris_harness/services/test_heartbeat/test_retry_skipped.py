"""``retry_skipped_after_minutes``: a skipped run is tried again instead of waiting a slot.

2026-09-28: the Mac restarted at 18:21 and was still in Setup Assistant at 18:30, so the
email_judge run could not reach its model and skipped. The next slot was 06:30, so five
emails stayed hidden overnight. A job that opts in is re-run N minutes after a skip, until
a run does not skip or the next scheduled slot comes first.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from apscheduler.schedulers.background import BackgroundScheduler

from iris_harness.services.heartbeat.config import HeartbeatConfigError, load_heartbeats
from iris_harness.services.heartbeat.models import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatStatus,
)
from iris_harness.services.heartbeat.scheduler import RETRY_JOB_SUFFIX, HeartbeatScheduler


def _judge() -> HeartbeatDefinition:
    """The email judge, with its next scheduled slot always about twelve hours away.

    A retry is deliberately not scheduled when the next slot comes first (that case has
    its own test, with an interval schedule). A fixed cron such as ``30 6,12,18 * * *``
    let the wall clock decide it: in the 30 minutes before each slot these tests found
    no retry job and failed, so the suite went red three times a day. The slot is
    computed from now, and the scheduler is given UTC, so the clock never decides.
    """
    far_hour = (datetime.now(UTC).hour + 12) % 24
    return HeartbeatDefinition(
        name="email_judge",
        handler="judge",
        schedule=f"30 {far_hour} * * *",
        retry_skipped_after_minutes=30,
    )


class _Judge:
    """Skips (Mac unreachable) for the first ``skips`` runs, then succeeds."""

    def __init__(self, skips: int) -> None:
        self.skips = skips
        self.calls = 0

    def __call__(self, definition: HeartbeatDefinition) -> HeartbeatRun:
        self.calls += 1
        status = HeartbeatStatus.SKIPPED if self.calls <= self.skips else HeartbeatStatus.SUCCESS
        return HeartbeatRun(name=definition.name, status=status, finished_at=datetime.now(UTC))


@pytest.fixture
def aps() -> Iterator[BackgroundScheduler]:
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.start(paused=True)
    yield scheduler
    scheduler.shutdown(wait=False)


def _heartbeats(
    aps: BackgroundScheduler, judge: _Judge, definition: HeartbeatDefinition | None = None
) -> HeartbeatScheduler:
    heartbeats = HeartbeatScheduler(scheduler=aps, handlers={"judge": judge}, timezone=UTC)
    heartbeats.register_all([definition or _judge()])
    return heartbeats


def _retry_job(aps: BackgroundScheduler, name: str = "email_judge"):
    return aps.get_job(f"{name}{RETRY_JOB_SUFFIX}")


def test_a_skipped_run_is_retried_after_the_configured_minutes(aps) -> None:
    judge = _Judge(skips=1)
    heartbeats = _heartbeats(aps, judge)
    before = datetime.now(UTC)
    heartbeats.trigger_by_name("email_judge", trigger="schedule")
    job = _retry_job(aps)
    assert job is not None
    assert (
        before + timedelta(minutes=29)
        < job.next_run_time
        <= datetime.now(UTC) + timedelta(minutes=30)
    )
    job.func()  # the retry fires
    assert judge.calls == 2
    assert [r.trigger for r in heartbeats.runs()] == ["schedule", "retry"]
    assert heartbeats.runs()[-1].status == HeartbeatStatus.SUCCESS
    assert _retry_job(aps) is None  # a run that did not skip schedules no more


def test_a_retry_that_skips_again_schedules_the_next_one(aps) -> None:
    judge = _Judge(skips=3)
    _heartbeats(aps, judge).trigger_by_name("email_judge")
    for _ in range(2):
        job = _retry_job(aps)
        assert job is not None
        job.func()
    assert judge.calls == 3
    assert _retry_job(aps) is not None  # still skipping, still retrying


def test_a_run_that_works_is_not_retried(aps) -> None:
    _heartbeats(aps, _Judge(skips=0)).trigger_by_name("email_judge")
    assert _retry_job(aps) is None


def test_jobs_that_do_not_opt_in_are_never_retried(aps) -> None:
    plain = HeartbeatDefinition(name="email_judge", handler="judge", schedule="30 6,12,18 * * *")
    _heartbeats(aps, _Judge(skips=1), plain).trigger_by_name("email_judge")
    assert _retry_job(aps) is None


def test_no_retry_when_the_next_slot_comes_first(aps) -> None:
    every_ten = HeartbeatDefinition(
        name="email_judge", handler="judge", schedule="interval:600", retry_skipped_after_minutes=30
    )
    _heartbeats(aps, _Judge(skips=1), every_ten).trigger_by_name("email_judge")
    assert _retry_job(aps) is None


def test_turning_the_job_off_drops_a_pending_retry(aps) -> None:
    judge = _Judge(skips=1)
    heartbeats = _heartbeats(aps, judge)
    heartbeats.trigger_by_name("email_judge")
    assert _retry_job(aps) is not None
    heartbeats._unschedule("email_judge")  # what turning it off does
    assert _retry_job(aps) is None


def test_a_retry_after_the_job_was_turned_off_does_nothing(aps) -> None:
    judge = _Judge(skips=1)
    heartbeats = _heartbeats(aps, judge)
    heartbeats.trigger_by_name("email_judge")
    func = _retry_job(aps).func
    heartbeats._unschedule("email_judge")
    func()
    assert judge.calls == 1


# --- config -------------------------------------------------------------------------


def _load(tmp_path: Path, value: object) -> list[HeartbeatDefinition]:
    path = tmp_path / "heartbeats.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "heartbeats": [
                    {
                        "name": "email_judge",
                        "handler": "judge",
                        "schedule": "30 6 * * *",
                        "retry_skipped_after_minutes": value,
                    }
                ]
            }
        )
    )
    return load_heartbeats(path)


def test_config_reads_the_retry_minutes(tmp_path: Path) -> None:
    assert _load(tmp_path, 30)[0].retry_skipped_after_minutes == 30


@pytest.mark.parametrize("bad", [0, -5, "30", True, 1.5])
def test_config_refuses_a_bad_retry_value(tmp_path: Path, bad: object) -> None:
    with pytest.raises(HeartbeatConfigError, match="retry_skipped_after_minutes"):
        _load(tmp_path, bad)


_ROOT = Path(__file__).resolve().parents[5]


def test_the_email_judge_retries_every_thirty_minutes() -> None:
    shipped = load_heartbeats(_ROOT / "config" / "heartbeats.yaml")
    judge = next(d for d in shipped if d.name == "email_judge")
    assert judge.retry_skipped_after_minutes == 30


def test_a_run_that_gets_through_cancels_a_pending_retry(aps) -> None:
    judge = _Judge(skips=1)
    heartbeats = _heartbeats(aps, judge)
    heartbeats.trigger_by_name("email_judge", trigger="schedule")  # skips
    assert _retry_job(aps) is not None
    heartbeats.trigger_by_name("email_judge")  # Run now, and the Mac answers
    assert _retry_job(aps) is None
