"""Routines learning loop — reflect on per-routine outcomes, propose changes.

The last self-learning gap. Routines are user-authored scheduled invocations; over
time some fail repeatedly or go stale. This analyzer mines the per-routine outcome
ledger (``routine_run`` signals) + the specs and produces *reflections* —
proposals like "this routine keeps failing, review it" or "this routine hasn't run
in weeks". The runtime turns each into a HITL approval-queue request; it never
mutates a routine itself (propose, never auto-apply).

Pure: reads signals + specs, returns reflections. No I/O beyond the store read.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from iris_harness.services.learning.store import LearningMetricsStore
from iris_harness.services.routines.models import RoutineSpec
from iris_harness.services.routines.seeded import is_core_routine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutineReflection:
    """A proposed change to a routine, derived from its measured outcomes."""

    routine_id: str
    title: str
    kind: str  # "failing" | "stale"
    reason: str
    recommendation: str


def reflect_on_routines(
    store: LearningMetricsStore,
    specs: Sequence[RoutineSpec],
    *,
    now: datetime | None = None,
    window: timedelta | None = None,
    min_runs: int = 5,
    failure_rate_threshold: float = 0.5,
    stale_days: int = 30,
    limit: int = 2000,
) -> list[RoutineReflection]:
    """Propose routine changes from the outcome ledger + specs. Best-effort.

    - **failing**: over ``window``, a routine ran at least ``min_runs`` times with a
      failure rate at or above ``failure_rate_threshold``.
    - **stale**: an approved/scheduled routine whose last run is older than
      ``stale_days`` (or that has never run) — it may no longer be wanted.
    """
    moment = now or datetime.now(UTC)
    try:
        rows = store.recent_signals(metric_name="routine_run", limit=limit)
    except Exception:  # analysis must never break a tick
        logger.debug("routine reflection: ledger read failed", exc_info=True)
        rows = []
    if window is not None:
        cutoff = moment - window
        rows = [r for r in rows if r.ts >= cutoff]

    runs: dict[str, int] = defaultdict(int)
    failures: dict[str, int] = defaultdict(int)
    for row in rows:
        rid = str(row.metadata.get("routine_id") or "").strip()
        if not rid:
            continue
        runs[rid] += 1
        if row.value != 1.0:
            failures[rid] += 1

    reflections: list[RoutineReflection] = []
    for spec in specs:
        total = runs.get(spec.id, 0)
        if total >= min_runs:
            rate = failures.get(spec.id, 0) / total
            if rate >= failure_rate_threshold:
                reflections.append(
                    RoutineReflection(
                        routine_id=spec.id,
                        title=spec.title,
                        kind="failing",
                        reason=(
                            f"routine '{spec.title}' failed {failures[spec.id]}/{total} recent runs"
                            f" ({rate:.0%})"
                        ),
                        recommendation="review or disable",
                    )
                )
                continue  # failing dominates staleness
        # A core routine IRIS seeds itself (the morning digest) is never proposed for
        # retirement: it must exist on every install, and a digest that stopped running
        # is a health-watch page, not a "retire it?" card (ADR-0122 §3).
        if spec.is_approved_for_execution and not is_core_routine(spec):
            last = spec.last_run_at
            stale = last is None or (moment - _as_utc(last)) > timedelta(days=stale_days)
            if stale:
                detail = "has never run" if last is None else f"last ran {_as_utc(last).date()}"
                reflections.append(
                    RoutineReflection(
                        routine_id=spec.id,
                        title=spec.title,
                        kind="stale",
                        reason=f"routine '{spec.title}' is approved but {detail}",
                        recommendation="review or retire",
                    )
                )
    return reflections


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
