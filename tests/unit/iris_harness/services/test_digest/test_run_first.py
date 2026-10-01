"""``run_first``: the jobs the digest runs, in order, before it renders (loop-proof PR 5).

Real ``HeartbeatScheduler`` (``scheduler=None``) with fake handlers; job names here are
generic — the shipped ``digest.yaml`` names the real ones (checked last).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import pytest

from iris_harness.services.digest.run_first import (
    JOINED,
    NOT_RUN,
    RunFirstJob,
    RunFirstPlan,
    load_run_first,
    parse_run_first,
    run_first,
)
from iris_harness.services.heartbeat import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatScheduler,
    HeartbeatStatus,
)
from iris_harness.services.heartbeat.run_store import HeartbeatRunStore

_REPO_CONFIG = Path(__file__).resolve().parents[5] / "config"


def _scheduler(
    calls: list[str],
    *,
    names: tuple[str, ...] = ("fetch", "classify"),
    fail: str | None = None,
    raises: str | None = None,
    run_store: HeartbeatRunStore | None = None,
) -> HeartbeatScheduler:
    scheduler = HeartbeatScheduler(run_store=run_store)
    for name in names:

        def handler(definition: HeartbeatDefinition, _name: str = name) -> HeartbeatRun:
            calls.append(_name)
            if _name == raises:
                raise RuntimeError("model endpoint down")
            status = HeartbeatStatus.FAILED if _name == fail else HeartbeatStatus.SUCCESS
            return HeartbeatRun(name=_name, status=status, output=f"{_name} done")

        scheduler.register_handler(name, handler)
        scheduler.register(HeartbeatDefinition(name=name, handler=name, schedule="15 6 * * *"))
    return scheduler


_PLAN = RunFirstPlan(
    jobs=(RunFirstJob("fetch", "IRIS_GATE"), RunFirstJob("classify", "IRIS_GATE")),
    budget_seconds=600,
)


def _on(name: str) -> bool:
    return True


def test_jobs_run_in_the_configured_order() -> None:
    calls: list[str] = []
    outcomes = run_first(_scheduler(calls), _PLAN, setting_on=_on)

    assert calls == ["fetch", "classify"]
    assert [(o.heartbeat, o.status) for o in outcomes] == [
        ("fetch", "success"),
        ("classify", "success"),
    ]


def test_the_order_is_the_plans_not_the_registrations() -> None:
    calls: list[str] = []
    plan = RunFirstPlan(jobs=(RunFirstJob("classify"), RunFirstJob("fetch")))
    run_first(_scheduler(calls), plan, setting_on=_on)
    assert calls == ["classify", "fetch"]


def test_setting_off_runs_neither_job() -> None:
    calls: list[str] = []
    asked: list[str] = []

    def off(name: str) -> bool:
        asked.append(name)
        return False

    outcomes = run_first(_scheduler(calls), _PLAN, setting_on=off)

    assert calls == []
    assert asked == ["IRIS_GATE", "IRIS_GATE"]
    assert all(o.status == NOT_RUN and "IRIS_GATE is off" in o.detail for o in outcomes)


def test_a_job_without_if_setting_always_runs() -> None:
    calls: list[str] = []
    plan = RunFirstPlan(jobs=(RunFirstJob("fetch"),))
    run_first(_scheduler(calls), plan, setting_on=lambda name: False)
    assert calls == ["fetch"]


def test_a_failing_or_raising_job_does_not_stop_the_next(tmp_path: Path) -> None:
    calls: list[str] = []
    store = HeartbeatRunStore(db_path=tmp_path / "heartbeat_runs.db")
    scheduler = _scheduler(calls, raises="fetch", run_store=store)

    outcomes = run_first(scheduler, _PLAN, setting_on=_on)

    assert calls == ["fetch", "classify"]
    assert outcomes[0].status == "failed"
    assert "model endpoint down" in outcomes[0].detail
    assert outcomes[1].status == "success"
    assert store.last("fetch").status == "failed"  # type: ignore[union-attr]


def test_runs_are_kept_like_a_run_now(tmp_path: Path) -> None:
    calls: list[str] = []
    store = HeartbeatRunStore(db_path=tmp_path / "heartbeat_runs.db")
    scheduler = _scheduler(calls, run_store=store)

    run_first(scheduler, _PLAN, setting_on=_on)

    for name in ("fetch", "classify"):
        kept = store.last(name)
        assert kept is not None
        assert kept.status == "success"
        assert kept.trigger == "digest"
    assert [r.name for r in scheduler.runs()] == ["fetch", "classify"]


def test_budget_spent_skips_the_rest() -> None:
    calls: list[str] = []
    ticks = iter([0.0, 0.0, 700.0])  # start, before fetch, before classify
    outcomes = run_first(_scheduler(calls), _PLAN, setting_on=_on, monotonic=lambda: next(ticks))

    assert calls == ["fetch"]
    assert outcomes[1].status == NOT_RUN
    assert "budget" in outcomes[1].detail


def test_unregistered_and_disabled_jobs_are_skipped_with_a_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[str] = []
    scheduler = _scheduler(calls)
    scheduler.update("classify", enabled=False, actor="test")
    plan = RunFirstPlan(
        jobs=(RunFirstJob("missing"), RunFirstJob("classify"), RunFirstJob("fetch"))
    )

    with caplog.at_level(logging.INFO, logger="iris_harness.services.digest.run_first"):
        outcomes = run_first(scheduler, plan, setting_on=_on)

    assert calls == ["fetch"]
    assert [(o.heartbeat, o.status, o.detail) for o in outcomes[:2]] == [
        ("missing", NOT_RUN, "not registered"),
        ("classify", NOT_RUN, "turned off"),
    ]
    assert "missing not run (not registered)" in caplog.text
    assert "classify not run (turned off)" in caplog.text


def test_a_job_already_running_is_waited_for_not_started_twice() -> None:
    calls: list[str] = []
    started, release = threading.Event(), threading.Event()
    scheduler = _scheduler(calls, names=("classify",))

    def slow(definition: HeartbeatDefinition) -> HeartbeatRun:
        calls.append("scheduled")
        started.set()
        release.wait(5)
        return HeartbeatRun(name=definition.name, status=HeartbeatStatus.SUCCESS)

    scheduler.register_handler("classify", slow)
    own = threading.Thread(target=scheduler.trigger_by_name, args=("classify",))
    own.start()
    assert started.wait(5)
    threading.Timer(0.2, release.set).start()

    outcomes = run_first(scheduler, RunFirstPlan(jobs=(RunFirstJob("classify"),)), setting_on=_on)
    own.join(5)

    assert calls == ["scheduled"]
    assert outcomes[0].status == JOINED
    assert not scheduler.is_running("classify")


def test_a_run_still_going_past_the_budget_is_left_alone() -> None:
    calls: list[str] = []
    started, release = threading.Event(), threading.Event()
    scheduler = _scheduler(calls, names=("classify",))

    def slow(definition: HeartbeatDefinition) -> HeartbeatRun:
        started.set()
        release.wait(5)
        return HeartbeatRun(name=definition.name, status=HeartbeatStatus.SUCCESS)

    scheduler.register_handler("classify", slow)
    own = threading.Thread(target=scheduler.trigger_by_name, args=("classify",))
    own.start()
    assert started.wait(5)
    try:
        outcomes = run_first(
            scheduler,
            RunFirstPlan(jobs=(RunFirstJob("classify"),), budget_seconds=0.1),
            setting_on=_on,
        )
    finally:
        release.set()
        own.join(5)

    assert outcomes[0].status == NOT_RUN
    assert "still running" in outcomes[0].detail


def test_a_broken_setting_reader_never_raises() -> None:
    def boom(name: str) -> bool:
        raise RuntimeError("catalog unreadable")

    calls: list[str] = []
    outcomes = run_first(_scheduler(calls), _PLAN, setting_on=boom)
    assert calls == []
    assert [o.status for o in outcomes] == ["failed", "failed"]


# --- config ----------------------------------------------------------------------


def test_parse_drops_bad_entries_and_a_bad_budget() -> None:
    plan = parse_run_first(
        {
            "run_first": [
                {"heartbeat": "fetch", "if_setting": "IRIS_GATE"},
                {"if_setting": "IRIS_GATE"},
                "classify",
                {"heartbeat": "classify", "if_setting": 3},
                {"heartbeat": "tidy"},
            ],
            "run_first_budget_seconds": "soon",
        }
    )
    assert plan.jobs == (RunFirstJob("fetch", "IRIS_GATE"), RunFirstJob("tidy"))
    assert plan.budget_seconds == 240


def test_no_file_runs_nothing(tmp_path: Path) -> None:
    assert load_run_first(tmp_path).jobs == ()


def test_the_shipped_digest_sweeps_then_judges_while_the_judge_is_on() -> None:
    plan = load_run_first(_REPO_CONFIG)
    assert plan.jobs == (
        RunFirstJob("email_sweep", "IRIS_EMAIL_JUDGE"),
        RunFirstJob("email_judge", "IRIS_EMAIL_JUDGE"),
    )
    assert plan.budget_seconds == 240
