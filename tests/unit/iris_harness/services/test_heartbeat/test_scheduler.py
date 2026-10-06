"""Tests for HeartbeatScheduler."""

from __future__ import annotations

from datetime import UTC, datetime

from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from iris_harness.services.heartbeat import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatScheduler,
    HeartbeatStatus,
)


def _ok(definition: HeartbeatDefinition) -> HeartbeatRun:
    return HeartbeatRun(
        name=definition.name,
        status=HeartbeatStatus.SUCCESS,
        finished_at=datetime.now(UTC),
        output="ok",
    )


def _boom(definition: HeartbeatDefinition) -> HeartbeatRun:
    raise RuntimeError("kaboom")


def test_register_skips_disabled_definitions() -> None:
    scheduler = HeartbeatScheduler(handlers={"h": _ok})
    definition = HeartbeatDefinition(name="x", handler="h", schedule="interval:60", enabled=False)
    assert scheduler.register(definition) is False


def test_register_skips_unknown_handlers() -> None:
    scheduler = HeartbeatScheduler()
    definition = HeartbeatDefinition(name="x", handler="missing", schedule="interval:60")
    assert scheduler.register(definition) is False


def test_register_returns_true_when_handler_bound() -> None:
    scheduler = HeartbeatScheduler(handlers={"h": _ok})
    definition = HeartbeatDefinition(name="x", handler="h", schedule="interval:60")
    assert scheduler.register(definition) is True


def test_trigger_now_invokes_handler_and_records_run() -> None:
    scheduler = HeartbeatScheduler(handlers={"h": _ok})
    definition = HeartbeatDefinition(name="x", handler="h", schedule="interval:60")
    run = scheduler.trigger_now(definition)
    assert run.status is HeartbeatStatus.SUCCESS
    assert scheduler.runs() == [run]


def test_trigger_now_returns_skipped_when_no_handler() -> None:
    scheduler = HeartbeatScheduler()
    definition = HeartbeatDefinition(name="x", handler="missing", schedule="interval:60")
    run = scheduler.trigger_now(definition)
    assert run.status is HeartbeatStatus.SKIPPED
    assert "no handler" in run.error


def test_trigger_now_captures_handler_exceptions() -> None:
    scheduler = HeartbeatScheduler(handlers={"h": _boom})
    definition = HeartbeatDefinition(name="x", handler="h", schedule="interval:60")
    run = scheduler.trigger_now(definition)
    assert run.status is HeartbeatStatus.FAILED
    assert "kaboom" in run.error


def test_build_trigger_supports_interval_and_cron() -> None:
    scheduler = HeartbeatScheduler(scheduler=None)
    interval = scheduler._build_trigger("interval:120")
    assert isinstance(interval, IntervalTrigger)

    cron = scheduler._build_trigger("0 7 * * *")
    assert isinstance(cron, CronTrigger)


def test_cron_is_read_in_iris_tz_not_the_machine_zone(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """06:15 means 06:15 in IRIS_TZ, whatever the host's zone is (a zone no dev
    machine or VM here runs in, so reading the host's would fail)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    monkeypatch.setenv("IRIS_TZ", "Asia/Kolkata")
    scheduler = HeartbeatScheduler(scheduler=None)
    assert scheduler.timezone() == "Asia/Kolkata"
    cron = scheduler._build_trigger("15 6 * * *")
    fire = cron.get_next_fire_time(None, datetime(2026, 9, 26, 0, 0, tzinfo=ZoneInfo("UTC")))
    assert fire is not None
    assert fire.astimezone(ZoneInfo("UTC")).strftime("%H:%M") == "00:45"  # 06:15 IST


# --- in-flight runs + the trigger label (loop-proof PR 5, the digest's run_first) ---


def test_trigger_by_name_keeps_the_given_trigger() -> None:
    scheduler = HeartbeatScheduler(handlers={"job": _ok})
    scheduler.register(HeartbeatDefinition(name="job", handler="job", schedule="interval:60"))

    assert scheduler.trigger_by_name("job").trigger == "manual"  # type: ignore[union-attr]
    assert scheduler.trigger_by_name("job", trigger="digest").trigger == "digest"  # type: ignore[union-attr]


def test_is_running_while_a_run_is_in_progress_and_idle_after() -> None:
    import threading

    started, release = threading.Event(), threading.Event()

    def slow(definition: HeartbeatDefinition) -> HeartbeatRun:
        started.set()
        release.wait(5)
        raise RuntimeError("fails, still leaves the job idle")

    scheduler = HeartbeatScheduler(handlers={"job": slow})
    scheduler.register(HeartbeatDefinition(name="job", handler="job", schedule="interval:60"))
    worker = threading.Thread(target=scheduler.trigger_by_name, args=("job",))
    worker.start()
    assert started.wait(5)

    assert scheduler.is_running("job")
    assert scheduler.wait_until_idle("job", 0.05) is False
    release.set()
    assert scheduler.wait_until_idle("job", 5) is True
    worker.join(5)
    assert not scheduler.is_running("job")
    assert scheduler.wait_until_idle("never-ran", 0) is True


def test_missing_handlers_log_one_summary_not_a_warning_each(caplog) -> None:  # type: ignore[no-untyped-def]
    """A job whose plugin is not installed is unavailable, not a fault (#110)."""
    scheduler = HeartbeatScheduler(handlers={"h": _ok})
    definitions = [
        HeartbeatDefinition(name="ok", handler="h", schedule="interval:60"),
        HeartbeatDefinition(name="a", handler="gone_a", schedule="interval:60"),
        HeartbeatDefinition(name="b", handler="gone_b", schedule="interval:60"),
        HeartbeatDefinition(name="off", handler="gone_c", schedule="interval:60", enabled=False),
    ]
    with caplog.at_level("DEBUG", logger="iris_harness.services.heartbeat.scheduler"):
        assert scheduler.register_all(definitions) == 1

    assert not [r for r in caplog.records if r.levelname in {"WARNING", "ERROR"}]
    summary = [r for r in caplog.records if r.levelname == "INFO"]
    assert len(summary) == 1
    assert "2 heartbeat(s) unavailable" in summary[0].getMessage()
    assert "a, b" in summary[0].getMessage()
    # still listed, with the reason the app shows
    assert scheduler.unavailable_reason(definitions[1]) is not None
