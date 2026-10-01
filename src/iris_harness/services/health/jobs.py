"""The ``job_completed`` health check (loop-proof D13, graph §8).

A job that silently stops is invisible: the sweep that did not run at 12:15 leaves no
error anywhere, and mail waits. ``health_watch.yaml`` names the jobs that matter
(``jobs.watch``); for each one this check asks the heartbeat scheduler whether the most
recent scheduled slot had a successful run (``heartbeat/slots.py`` over the kept runs in
``heartbeat_runs.db``, so a restart does not forget):

* green  ``ran 06:15 ✓ (23 new)``
* yellow ``skipped 12:31: <why> — last success 06:31`` (it ran but could not do its work)
* red    ``Missed 12:15 — last success 06:15`` or ``failed 12:16: <error> — …``
* grey   the job is off, cannot run here, or its first slot is still ahead

One row per job (``subject`` = the job name), so each is its own incident; red pages
the owner through the health watch like every other red check. Job names come from
config only — the core knows no job.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from iris_harness.foundation.paths import config_dir as resolve_config_dir
from iris_harness.services.health.models import CheckKind, HealthCheck, HealthState

logger = logging.getLogger(__name__)

TARGET = "job_completed"
DEFAULT_GRACE_MINUTES = 30.0


@dataclass(frozen=True)
class WatchedJob:
    name: str
    grace: timedelta


def parse_jobs(raw: Any) -> tuple[WatchedJob, ...]:
    """``jobs:`` from health_watch.yaml: ``{grace_minutes, watch: [name | {name,
    grace_minutes}]}``. Anything malformed is skipped with a warning."""
    if not isinstance(raw, dict):
        return ()
    try:
        default = float(raw.get("grace_minutes", DEFAULT_GRACE_MINUTES))
    except (TypeError, ValueError):
        default = DEFAULT_GRACE_MINUTES
    jobs: list[WatchedJob] = []
    for entry in raw.get("watch") or ():
        if isinstance(entry, str) and entry.strip():
            jobs.append(WatchedJob(entry.strip(), timedelta(minutes=default)))
        elif isinstance(entry, dict) and str(entry.get("name") or "").strip():
            try:
                grace = float(entry.get("grace_minutes", default))
            except (TypeError, ValueError):
                grace = default
            jobs.append(WatchedJob(str(entry["name"]).strip(), timedelta(minutes=grace)))
        else:
            logger.warning("health_watch.yaml jobs.watch: skipping %r", entry)
    return tuple(jobs)


def load_watched_jobs(config_dir: Path | None = None) -> tuple[WatchedJob, ...]:
    """The ``jobs:`` section of ``health_watch.yaml`` (``foundation.paths.config_dir()``)."""
    base = config_dir or resolve_config_dir()
    path = base / "health_watch.yaml"
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return ()
    return _read_jobs(str(path), mtime)


@lru_cache(maxsize=8)
def _read_jobs(path: str, mtime: int) -> tuple[WatchedJob, ...]:
    """Parsed once per file version: the Heartbeats list asks once per heartbeat."""
    del mtime  # part of the cache key only
    try:
        import yaml

        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read %s: %s — no jobs watched", path, exc)
        return ()
    return parse_jobs(loaded.get("jobs") if isinstance(loaded, dict) else None)


def grace_for(name: str, config_dir: Path | None = None) -> timedelta:
    """The grace a job is judged with: its own, else the section's default."""
    for job in load_watched_jobs(config_dir):
        if job.name == name:
            return job.grace
    return timedelta(minutes=DEFAULT_GRACE_MINUTES)


def job_checks(
    heartbeats: Any,
    jobs: tuple[WatchedJob, ...],
    *,
    now: datetime | None = None,
) -> list[HealthCheck]:
    """One ``job_completed`` row per watched job."""
    checks: list[HealthCheck] = []
    for job in jobs:
        status = heartbeats.job_status(job.name, grace=job.grace, now=now)
        if status is None:
            state, detail = HealthState.GREY, "runs are not kept on this harness"
        else:
            state, detail = HealthState(status.state), status.detail
        checks.append(
            HealthCheck(
                target=TARGET,
                kind=CheckKind.SERVICE,
                state=state,
                detail=detail,
                endpoint="/heartbeat",
                action=(
                    f"iris heartbeats trigger {job.name}"
                    if state in (HealthState.RED, HealthState.YELLOW)
                    else None
                ),
                subject=job.name,
            )
        )
    return checks


def job_completed_provider(
    heartbeats: Any, config_dir: Path | None = None
) -> Callable[[], list[HealthCheck]]:
    """A ``register_check_provider`` callable. The job list is re-read each pass, so an
    edit to health_watch.yaml applies without a restart."""

    def provider() -> list[HealthCheck]:
        return job_checks(heartbeats, load_watched_jobs(config_dir))

    return provider


__all__ = [
    "DEFAULT_GRACE_MINUTES",
    "TARGET",
    "WatchedJob",
    "grace_for",
    "job_checks",
    "job_completed_provider",
    "load_watched_jobs",
    "parse_jobs",
]
