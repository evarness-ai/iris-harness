"""Domain models for heartbeats."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class HeartbeatStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class HeartbeatDefinition:
    """Declarative heartbeat configuration loaded from YAML."""

    name: str
    handler: str
    schedule: str  # cron expression OR "interval:<seconds>"
    enabled: bool = True
    description: str = ""
    params: dict[str, object] = field(default_factory=dict)
    # ``sys.platform`` values this heartbeat can run on; empty means any. A deployment
    # declares it in heartbeats.yaml, so the app can lock a job that needs another
    # machine (the Photos library, EventKit) instead of letting it be turned on.
    platforms: tuple[str, ...] = ()
    # Whether each run is kept in ``heartbeat_runs.db`` (loop-proof D13). ``None`` (the
    # default) decides from the schedule: every cron job and every interval of 5 minutes
    # or more keeps every run; a faster interval (the 30-60 s ticks) keeps only the runs
    # whose status differs from the last one kept, so a tick that works 1,440 times a day
    # is one row, and the run that broke it is the next. ``record_runs:`` in
    # heartbeats.yaml overrides either way.
    record_runs: bool | None = None
    # Run again this many minutes after a run that ended ``skipped``, until one does not
    # (or the next scheduled slot is sooner). Opt-in per job in heartbeats.yaml: a job
    # whose "skipped" means "could not reach what it needs" (email_judge: the Mac asleep
    # or restarting) should not wait a whole slot, with its mail hidden meanwhile.
    retry_skipped_after_minutes: int | None = None
    # The plugin that registers this heartbeat's handler, when it is not the core's. A
    # definition whose handler is missing is then "its plugin is not mounted here" (a
    # profile that leaves it out, an optional package it needs not installed), not a
    # typo: the scheduler skips it quietly and the app shows the reason. Empty means the
    # core owns the handler, so a missing one is a real fault and warns.
    plugin: str = ""


@dataclass
class HeartbeatRun:
    """Result of a single heartbeat invocation."""

    name: str
    status: HeartbeatStatus
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    output: str = ""
    error: str = ""
    # A handler's structured result (counts, flags) beside the one-line ``output``: kept
    # with the run, so a health check or the digest can read it after a restart. Plain
    # JSON values only. See ``heartbeat/run_store.py``.
    result: dict[str, Any] = field(default_factory=dict)
    # ``schedule`` when the scheduler fired it, ``manual`` for a trigger (Run now, the
    # health watch's re-run, the API). Set by the scheduler, not the handler.
    trigger: str = "schedule"
