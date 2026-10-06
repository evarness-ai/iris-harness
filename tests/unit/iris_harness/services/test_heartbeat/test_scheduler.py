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


_SCHED_LOGGER = "iris_harness.services.heartbeat.scheduler"


def _gap(unmounted: dict[str, str]):  # type: ignore[no-untyped-def]
    """A plugin lookup: ``unmounted`` maps plugin -> reason; anything else is mounted."""
    return lambda plugin: unmounted.get(plugin)


def test_unmounted_plugin_heartbeats_log_one_summary_not_a_warning_each(caplog) -> None:  # type: ignore[no-untyped-def]
    """A job whose plugin is not mounted is unavailable, not a fault (#110)."""
    scheduler = HeartbeatScheduler(handlers={"h": _ok})
    scheduler.bind_plugin_gap(_gap({"gone": "not in this profile, or not installed"}))
    definitions = [
        HeartbeatDefinition(name="ok", handler="h", schedule="interval:60"),
        HeartbeatDefinition(name="a", handler="gone_a", schedule="interval:60", plugin="gone"),
        HeartbeatDefinition(name="b", handler="gone_b", schedule="interval:60", plugin="gone"),
        HeartbeatDefinition(
            name="off", handler="gone_c", schedule="interval:60", enabled=False, plugin="gone"
        ),
    ]
    with caplog.at_level("DEBUG", logger=_SCHED_LOGGER):
        assert scheduler.register_all(definitions) == 1

    assert not [r for r in caplog.records if r.levelname in {"WARNING", "ERROR"}]
    summary = [r for r in caplog.records if r.levelname == "INFO"]
    assert len(summary) == 1
    message = summary[0].getMessage()
    assert "2 heartbeat(s) unavailable, plugin not mounted (gone)" in message
    assert "a, b" in message
    # still listed, with the reason the app shows
    reason = scheduler.unavailable_reason(definitions[1])
    assert reason is not None and "plugin gone is not mounted" in reason


def test_mounted_plugin_that_registered_no_handler_warns(caplog) -> None:  # type: ignore[no-untyped-def]
    scheduler = HeartbeatScheduler(handlers={})
    scheduler.bind_plugin_gap(_gap({}))  # everything is mounted
    definition = HeartbeatDefinition(
        name="j", handler="forgotten", schedule="interval:60", plugin="up"
    )
    with caplog.at_level("DEBUG", logger=_SCHED_LOGGER):
        assert scheduler.register_all([definition]) == 0

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "plugin up is mounted but registered no handler forgotten" in warnings[0].getMessage()
    assert not [r for r in caplog.records if r.levelname == "INFO"]  # not in the summary


def test_typo_without_an_owning_plugin_warns(caplog) -> None:  # type: ignore[no-untyped-def]
    scheduler = HeartbeatScheduler(handlers={"real": _ok})
    scheduler.bind_plugin_gap(_gap({"gone": "x"}))
    definition = HeartbeatDefinition(name="j", handler="rael", schedule="interval:60")
    with caplog.at_level("DEBUG", logger=_SCHED_LOGGER):
        assert scheduler.register_all([definition]) == 0

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "unknown handler rael" in warnings[0].getMessage()
    assert not [r for r in caplog.records if r.levelname == "INFO"]


def test_no_plugin_lookup_bound_warns_even_for_a_named_plugin(caplog) -> None:  # type: ignore[no-untyped-def]
    """Nothing says the plugin is absent, so a missing handler is reported as before."""
    scheduler = HeartbeatScheduler(handlers={})
    definition = HeartbeatDefinition(name="j", handler="h", schedule="interval:60", plugin="p")
    with caplog.at_level("DEBUG", logger=_SCHED_LOGGER):
        assert scheduler.register_all([definition]) == 0

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "unknown handler h" in warnings[0].getMessage()
    assert not [r for r in caplog.records if r.levelname == "INFO"]
    assert "no handler 'h' is registered" in (scheduler.unavailable_reason(definition) or "")


def test_plugin_typo_is_a_warning_not_a_quiet_not_installed(caplog) -> None:  # type: ignore[no-untyped-def]
    """`plugin: emial_workflows` is in no list and not mounted: a fault, not an absence."""
    scheduler = HeartbeatScheduler(handlers={})
    scheduler.bind_plugin_gap(_gap({"emial_workflows": "not in this profile, or not installed"}))
    scheduler.bind_known_plugins({"email_workflows", "calendar"})
    typo = HeartbeatDefinition(
        name="j", handler="h", schedule="interval:60", plugin="emial_workflows"
    )
    with caplog.at_level("DEBUG", logger=_SCHED_LOGGER):
        assert scheduler.register_all([typo]) == 0

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "heartbeat j names plugin emial_workflows" in message
    assert "typo in `plugin:`?" in message
    assert not [r for r in caplog.records if r.levelname == "INFO"]  # not in the summary
    reason = scheduler.unavailable_reason(typo)
    assert reason is not None and "not a known plugin (typo?)" in reason


def test_declared_external_plugin_that_is_absent_stays_quiet(caplog) -> None:  # type: ignore[no-untyped-def]
    scheduler = HeartbeatScheduler(handlers={})
    scheduler.bind_plugin_gap(_gap({"calendar": "not in this profile, or not installed"}))
    scheduler.bind_known_plugins({"email_workflows", "calendar"})
    definition = HeartbeatDefinition(
        name="j", handler="h", schedule="interval:60", plugin="calendar"
    )
    with caplog.at_level("DEBUG", logger=_SCHED_LOGGER):
        assert scheduler.register_all([definition]) == 0

    assert not [r for r in caplog.records if r.levelname in {"WARNING", "ERROR"}]
    (summary,) = [r for r in caplog.records if r.levelname == "INFO"]
    assert "1 heartbeat(s) unavailable, plugin not mounted (calendar)" in summary.getMessage()
    assert "plugin calendar is not mounted" in (scheduler.unavailable_reason(definition) or "")


def test_no_declared_owners_means_nothing_is_judged_a_typo(caplog) -> None:  # type: ignore[no-untyped-def]
    """A user file that declares no lists keeps the quiet behaviour for an absent plugin."""
    scheduler = HeartbeatScheduler(handlers={})
    scheduler.bind_plugin_gap(_gap({"anything": "x"}))
    definition = HeartbeatDefinition(
        name="j", handler="h", schedule="interval:60", plugin="anything"
    )
    with caplog.at_level("DEBUG", logger=_SCHED_LOGGER):
        scheduler.register_all([definition])
    assert not [r for r in caplog.records if r.levelname == "WARNING"]
