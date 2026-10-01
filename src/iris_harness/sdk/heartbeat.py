"""Reading whether a scheduled job ran, for a plugin that watches its own jobs.

The scheduler keeps every recorded run under the data dir, so "did the 06:15 sweep
run?" survives a restart. ``heartbeat_runs(data_dir)`` is a read-only view of those
runs (a ``HeartbeatRunHistory``: ``recent``, ``last``, ``last_success``,
``last_failure``, ``first_seen``), each a ``StoredRun``. The scheduler is the one
writer; a host that has recorded nothing yet reads as no runs, and the view never
creates the file.

``tally`` counts the cron slots in a window that a successful run served (a ``Tally``).
``clock`` renders a time the way the Heartbeats screen does ("06:15", "Mon 06:15"); it
is display text, and its format may change, so never parse it.

``load_heartbeats(path)`` reads a ``heartbeats.yaml`` (``config_path("heartbeats.yaml")``)
into ``HeartbeatDefinition`` rows, for a plugin that tells the owner when its job runs;
``describe_schedule`` puts a schedule in plain words ("daily at 06:15"). A malformed
file raises ``HeartbeatConfigError``. Read-only: the owner edits the file.

A heartbeat *handler's* contract (``HeartbeatDefinition`` in, ``HeartbeatRun`` out) is
in ``iris_harness.sdk.types``.
"""

from __future__ import annotations

from iris_harness.services.heartbeat.config import HeartbeatConfigError, load_heartbeats
from iris_harness.services.heartbeat.run_store import (
    HeartbeatRunHistory,
    StoredRun,
    heartbeat_runs,
)
from iris_harness.services.heartbeat.schedule_text import describe_schedule
from iris_harness.services.heartbeat.slots import Tally, clock, tally

__all__ = [
    "HeartbeatConfigError",
    "HeartbeatRunHistory",
    "StoredRun",
    "Tally",
    "clock",
    "describe_schedule",
    "heartbeat_runs",
    "load_heartbeats",
    "tally",
]
