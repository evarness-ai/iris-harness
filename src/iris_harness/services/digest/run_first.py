"""Jobs the digest runs, in order, right before it renders (loop-proof PR 5).

``digest.yaml`` names them — the core names none::

    run_first:
      - heartbeat: <job name>
        if_setting: <IRIS_* on/off setting>   # optional: run only while it is on
    run_first_budget_seconds: 240

Each job runs through the heartbeat scheduler's own ``trigger_by_name`` — the path the
Heartbeats screen's "Run now" takes — so it is a real run: kept in the run store,
shown on the Heartbeats screen, counted by the job-health checks. A job is skipped,
with a log line, when its setting is off, it is not registered, it is turned off or it
cannot run on this harness. A job already running (its own schedule got there first)
is waited for, not started again. The total ``budget`` bounds the chain: once it is
spent no further job starts (a running one is never interrupted). Nothing here raises
into the digest: a failed or crashing job is recorded and the digest renders anyway.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BUDGET_SECONDS = 240.0

# Outcome statuses besides a run's own (``success`` / ``failed`` / ``skipped`` ...).
NOT_RUN = "not_run"  # skipped before starting: setting off, unknown, off, over budget
JOINED = "joined"  # a run already in progress was waited for instead


@dataclass(frozen=True)
class RunFirstJob:
    heartbeat: str
    if_setting: str | None = None


@dataclass(frozen=True)
class RunFirstPlan:
    jobs: tuple[RunFirstJob, ...] = ()
    budget_seconds: float = DEFAULT_BUDGET_SECONDS


@dataclass(frozen=True)
class RunFirstOutcome:
    heartbeat: str
    status: str
    detail: str = ""


def parse_run_first(raw: dict[str, Any], *, source: str = "digest.yaml") -> RunFirstPlan:
    """The plan in a loaded ``digest.yaml``; a malformed entry is dropped with a warning."""
    jobs: list[RunFirstJob] = []
    entries = raw.get("run_first") or []
    if not isinstance(entries, list):
        logger.warning("digest: %s run_first is not a list; running nothing first", source)
        entries = []
    for entry in entries:
        name = entry.get("heartbeat") if isinstance(entry, dict) else None
        setting = entry.get("if_setting") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name.strip():
            logger.warning("digest: %s run_first entry %r names no heartbeat", source, entry)
            continue
        if setting is not None and not (isinstance(setting, str) and setting.strip()):
            logger.warning("digest: %s run_first %s: bad if_setting %r", source, name, setting)
            continue
        jobs.append(RunFirstJob(name.strip(), setting.strip() if setting else None))
    budget = DEFAULT_BUDGET_SECONDS
    given = raw.get("run_first_budget_seconds")
    if given is not None:
        if isinstance(given, int | float) and not isinstance(given, bool) and given > 0:
            budget = float(given)
        else:
            logger.warning("digest: %s run_first_budget_seconds %r ignored", source, given)
    return RunFirstPlan(jobs=tuple(jobs), budget_seconds=budget)


def load_run_first(config_dir: Path | None) -> RunFirstPlan:
    """``run_first`` from ``<config_dir>/digest.yaml``; no file or a bad one runs nothing."""
    from .settings import DIGEST_FILE, _config_dir

    path = _config_dir(config_dir) / DIGEST_FILE
    if not path.exists():
        return RunFirstPlan()
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — a bad file must not stop the digest
        logger.warning("digest: could not read %s (%s); running nothing first", path, exc)
        return RunFirstPlan()
    return parse_run_first(loaded if isinstance(loaded, dict) else {}, source=str(path))


def _skip_reason(scheduler: Any, name: str) -> str | None:
    """Why ``name`` cannot be run now, or None when it can."""
    definition = next((d for d in scheduler.all_definitions() if d.name == name), None)
    if definition is None:
        return "not registered"
    if not definition.enabled:
        return "turned off"
    reason = scheduler.unavailable_reason(definition)
    return f"can't run here: {reason}" if reason else None


def run_first(
    scheduler: Any,
    plan: RunFirstPlan,
    *,
    setting_on: Callable[[str], bool],
    trigger: str = "digest",
    monotonic: Callable[[], float] = time.monotonic,
) -> list[RunFirstOutcome]:
    """Run ``plan``'s jobs in order through ``scheduler``; one outcome per job. Never
    raises."""
    outcomes: list[RunFirstOutcome] = []
    started = monotonic()

    def _not_run(name: str, why: str) -> None:
        logger.info("digest run_first: %s not run (%s)", name, why)
        outcomes.append(RunFirstOutcome(name, NOT_RUN, why))

    for job in plan.jobs:
        name = job.heartbeat
        try:
            left = plan.budget_seconds - (monotonic() - started)
            if left <= 0:
                _not_run(name, f"the {plan.budget_seconds:g}s budget is spent")
                continue
            if job.if_setting and not setting_on(job.if_setting):
                _not_run(name, f"{job.if_setting} is off")
                continue
            why = _skip_reason(scheduler, name)
            if why is not None:
                _not_run(name, why)
                continue
            if getattr(scheduler, "is_running", None) and scheduler.is_running(name):
                # Its own schedule got there first: wait for that run, never start a
                # second one beside it. The finished run is this chain's run.
                logger.info("digest run_first: %s is running; waiting up to %.0fs", name, left)
                if scheduler.wait_until_idle(name, left):
                    outcomes.append(RunFirstOutcome(name, JOINED, "waited for the run in progress"))
                else:
                    _not_run(name, "still running when the budget was spent")
                continue
            run = scheduler.trigger_by_name(name, trigger=trigger)
            if run is None:
                _not_run(name, "not registered")
                continue
            status = getattr(run.status, "value", str(run.status))
            outcomes.append(RunFirstOutcome(name, status, run.error or run.output or ""))
            logger.info("digest run_first: %s %s", name, status)
        except Exception as exc:  # a run-first job never blocks the digest
            logger.exception("digest run_first: %s raised; the digest renders anyway", name)
            outcomes.append(RunFirstOutcome(name, "failed", f"{type(exc).__name__}: {exc}"))
    return outcomes


__all__ = [
    "DEFAULT_BUDGET_SECONDS",
    "JOINED",
    "NOT_RUN",
    "RunFirstJob",
    "RunFirstOutcome",
    "RunFirstPlan",
    "load_run_first",
    "parse_run_first",
    "run_first",
]
