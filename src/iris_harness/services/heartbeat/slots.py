"""Did a scheduled job run? Slot math over the kept runs (loop-proof D13).

A cron heartbeat has *slots*: the wall-clock times its schedule names, read in the
owner's zone (``IRIS_TZ``), so ``15 6,12,18 * * *`` is 06:15 / 12:15 / 18:15 Central on
a DST day too. :func:`evaluate_job` answers "did the most recent slot run?" from
``heartbeat_runs.db`` — it survives restarts — for the Heartbeats screen, the
``job_completed`` health check and the API; :func:`tally` counts a window's slots for
the digest footer.

The verdict for a job, newest slot first:

* a successful run at/after the slot → **green** ``ran 06:15 ✓ (summary)``;
* the slot's latest run failed → **red** ``failed 12:16: <error>``;
* it ran but skipped (the handler said it could not do its work) → **yellow**;
* no run and the slot is more than ``grace`` old → **red** ``Missed 12:15 — last
  success 06:15``; still inside the grace → the slot before it decides;
* the job is off, cannot run here, or its first slot is still ahead of it (a fresh
  install) → **grey**.

Interval schedules have no wall-clock slots: they are red when no success came within
one interval plus the grace.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from typing import TYPE_CHECKING, Any

from apscheduler.triggers.cron import CronTrigger

if TYPE_CHECKING:
    from .models import HeartbeatDefinition
    from .run_store import HeartbeatRunHistory, HeartbeatRunStore, StoredRun

_INTERVAL = "interval:"
# A run that starts a moment before its slot (clock skew, a manual run at 06:14:30)
# still serves it.
_TOLERANCE = timedelta(minutes=1)
# How far back to look for the most recent slots. A weekly cron still has one.
_LOOKBACK = timedelta(days=8)
# Iteration cap: an every-minute cron over a day is 1,440 slots.
_MAX_SLOTS = 5000
_SUMMARY_CHARS = 80

GREEN, YELLOW, RED, GREY = "green", "yellow", "red", "grey"


def interval_seconds(schedule: str) -> int | None:
    schedule = schedule.strip()
    if not schedule.startswith(_INTERVAL):
        return None
    try:
        seconds = int(schedule[len(_INTERVAL) :])
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def cron_trigger(schedule: str, tz: tzinfo) -> CronTrigger | None:
    """The schedule's cron trigger in ``tz``; None for an interval or a bad cron."""
    if schedule.strip().startswith(_INTERVAL):
        return None
    try:
        return CronTrigger.from_crontab(schedule.strip(), timezone=tz)
    except ValueError:
        return None


def slots_between(schedule: str, start: datetime, end: datetime, tz: tzinfo) -> list[datetime]:
    """Every cron slot in ``[start, end)``, oldest first, as aware datetimes in ``tz``."""
    trigger = cron_trigger(schedule, tz)
    if trigger is None:
        return []
    found: list[datetime] = []
    cursor = start.astimezone(tz)
    while len(found) < _MAX_SLOTS:
        nxt = trigger.get_next_fire_time(None, cursor)
        if nxt is None or nxt >= end:
            break
        found.append(nxt)
        cursor = nxt + timedelta(seconds=1)
    return found


def recent_slots(schedule: str, now: datetime, tz: tzinfo, count: int = 3) -> list[datetime]:
    """The last ``count`` slots at or before ``now``, oldest first."""
    slots = slots_between(schedule, now - _LOOKBACK, now + timedelta(microseconds=1), tz)
    return slots[-count:]


def clock(value: datetime | None, now: datetime, tz: tzinfo) -> str:
    """``12:15`` today, ``Sep 25 12:15`` another day, in the owner's zone."""
    if value is None:
        return "never"
    local = value.astimezone(tz)
    if local.date() == now.astimezone(tz).date():
        return local.strftime("%H:%M")
    return f"{local.strftime('%b')} {local.day} {local.strftime('%H:%M')}"


def run_summary(run: StoredRun) -> str:
    """A run's short summary: ``result["summary"]`` when the handler gave one, else the
    first line of its output, clipped."""
    given = run.result.get("summary") if isinstance(run.result, dict) else None
    if given:
        text = str(given)
    else:
        lines = (run.output or "").strip().splitlines()
        text = lines[0] if lines else ""
    text = " ".join(text.split())
    return text if len(text) <= _SUMMARY_CHARS else text[: _SUMMARY_CHARS - 1] + "…"


@dataclass(frozen=True)
class JobStatus:
    """Did the job's most recent slot run? One verdict, the words to show with it."""

    name: str
    state: str
    detail: str
    slot: datetime | None = None
    missed: bool = False
    last_run: StoredRun | None = None
    last_success: StoredRun | None = None
    last_error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "detail": self.detail,
            "slot": self.slot.astimezone(UTC).isoformat() if self.slot else None,
            "missed": self.missed,
            "ran": self.state in (GREEN, YELLOW) and not self.missed,
            "last_run": self.last_run.as_dict() if self.last_run else None,
            "last_success_at": (
                self.last_success.started_at.astimezone(UTC).isoformat()
                if self.last_success
                else None
            ),
            "last_error": self.last_error,
        }


def evaluate_job(
    definition: HeartbeatDefinition | None,
    store: HeartbeatRunStore,
    *,
    now: datetime,
    tz: tzinfo,
    grace: timedelta,
    unavailable: str | None = None,
    name: str | None = None,
) -> JobStatus:
    """The verdict for one job (see the module docstring)."""
    job = definition.name if definition is not None else (name or "?")
    if definition is None:
        return JobStatus(job, GREY, "not scheduled on this harness")
    if not definition.enabled:
        return JobStatus(job, GREY, "turned off")
    if unavailable:
        return JobStatus(job, GREY, f"can't run here: {unavailable}")

    last_run = store.last(job)
    last_success = store.last_success(job)
    last_failure = store.last_failure(job)
    last_error = ""
    if last_failure is not None and (
        last_success is None or last_failure.started_at > last_success.started_at
    ):
        last_error = last_failure.error or last_failure.output
    common: dict[str, Any] = {
        "last_run": last_run,
        "last_success": last_success,
        "last_error": last_error,
    }

    seconds = interval_seconds(definition.schedule)
    if seconds is not None:
        return _evaluate_interval(job, seconds, store, now, tz, grace, common)

    slots = recent_slots(definition.schedule, now, tz)
    if not slots:
        trigger = cron_trigger(definition.schedule, tz)
        if trigger is None:
            return JobStatus(job, GREY, f"schedule {definition.schedule!r} is not a cron")
        return JobStatus(job, GREY, "no scheduled time in the last 8 days", **common)

    first_seen = store.first_seen(job)
    since_last = (
        f" — last success {clock(last_success.started_at if last_success else None, now, tz)}"
    )
    for index in range(len(slots) - 1, -1, -1):
        slot = slots[index]
        until = slots[index + 1] - _TOLERANCE if index + 1 < len(slots) else None
        runs = store.recent(name=job, since=slot - _TOLERANCE, until=until, limit=50)
        if runs:
            return _verdict(job, slot, runs, now, tz, since_last, common)
        if first_seen is not None and slot < first_seen - _TOLERANCE:
            break  # scheduled before this harness knew the job: not a miss
        if now >= slot + grace:
            detail = f"Missed {clock(slot, now, tz)}{since_last}"
            if last_error:
                detail += f" · last error: {_one_line(last_error)}"
            return JobStatus(job, RED, detail, slot=slot, missed=True, **common)
        # Inside the grace: the slot before this one decides.
    upcoming = _next_slot(definition.schedule, now, tz)
    return JobStatus(
        job,
        GREY,
        f"first run due {clock(upcoming, now, tz)}" if upcoming else "no run yet",
        slot=upcoming,
        **common,
    )


def _verdict(
    job: str,
    slot: datetime,
    runs: list[StoredRun],
    now: datetime,
    tz: tzinfo,
    since_last: str,
    common: dict[str, Any],
) -> JobStatus:
    success = next((r for r in runs if r.ok), None)
    if success is not None:
        summary = run_summary(success)
        detail = f"ran {clock(success.started_at, now, tz)} ✓" + (
            f" ({summary})" if summary else ""
        )
        return JobStatus(job, GREEN, detail, slot=slot, **common)
    newest = runs[0]
    why = _one_line(newest.error or newest.output or newest.status)
    at = clock(newest.started_at, now, tz)
    if newest.status == "failed":
        return JobStatus(job, RED, f"failed {at}: {why}{since_last}", slot=slot, **common)
    return JobStatus(job, YELLOW, f"{newest.status} {at}: {why}{since_last}", slot=slot, **common)


def _evaluate_interval(
    job: str,
    seconds: int,
    store: HeartbeatRunStore,
    now: datetime,
    tz: tzinfo,
    grace: timedelta,
    common: dict[str, Any],
) -> JobStatus:
    last_run: StoredRun | None = common["last_run"]
    last_success: StoredRun | None = common["last_success"]
    every = timedelta(seconds=seconds)
    anchor = last_success.started_at if last_success else store.first_seen(job)
    if anchor is None:
        return JobStatus(job, GREY, "no run yet", **common)
    due = anchor + every
    if last_run is not None and last_run.status == "failed":
        if last_success is None or last_run.started_at > last_success.started_at:
            return JobStatus(
                job,
                RED,
                f"failed {clock(last_run.started_at, now, tz)}: "
                f"{_one_line(last_run.error or last_run.output)}",
                slot=due,
                **common,
            )
    if now >= due + grace:
        return JobStatus(
            job,
            RED,
            f"Missed {clock(due, now, tz)} — last success "
            f"{clock(last_success.started_at if last_success else None, now, tz)}",
            slot=due,
            missed=True,
            **common,
        )
    if last_success is None:
        return JobStatus(job, GREY, f"first run due {clock(due, now, tz)}", slot=due, **common)
    summary = run_summary(last_success)
    detail = f"ran {clock(last_success.started_at, now, tz)} ✓" + (
        f" ({summary})" if summary else ""
    )
    return JobStatus(job, GREEN, detail, slot=due, **common)


def _next_slot(schedule: str, now: datetime, tz: tzinfo) -> datetime | None:
    trigger = cron_trigger(schedule, tz)
    return trigger.get_next_fire_time(None, now.astimezone(tz)) if trigger else None


def _one_line(text: str) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= 200 else flat[:199] + "…"


@dataclass(frozen=True)
class Tally:
    """A window's slots for one job: how many a successful run served (``ran``), and
    which had no run at all (``missed``). A slot whose run failed or skipped is in
    neither: it ran, and did not succeed."""

    ran: int
    expected: int
    missed: tuple[datetime, ...] = ()


def tally(
    schedule: str,
    store: HeartbeatRunHistory,
    name: str,
    start: datetime,
    end: datetime,
    tz: tzinfo,
) -> Tally:
    """Count ``[start, end)``'s cron slots for ``name``.

    A slot is served by a success between it and the next slot (a late catch-up counts:
    the work got done). An interval job has no slots: ``expected`` is 0 and ``ran`` is
    its successful runs in the window. Slots before the job was first scheduled here
    are not counted.
    """
    if interval_seconds(schedule) is not None:
        ok = store.recent(name=name, since=start, until=end, status="success", limit=0)
        return Tally(ran=len(ok), expected=0)
    slots = slots_between(schedule, start, end, tz)
    first_seen = store.first_seen(name)
    if first_seen is not None:
        slots = [s for s in slots if s >= first_seen - _TOLERANCE]
    runs = store.recent(name=name, since=start - _TOLERANCE, until=end, limit=0)
    ran = 0
    missed: list[datetime] = []
    for index, slot in enumerate(slots):
        until = slots[index + 1] - _TOLERANCE if index + 1 < len(slots) else end
        served = [r for r in runs if slot - _TOLERANCE <= r.started_at < until]
        if any(r.ok for r in served):
            ran += 1
        elif not served:
            missed.append(slot)
    return Tally(ran=ran, expected=len(slots), missed=tuple(missed))


__all__ = [
    "GREEN",
    "GREY",
    "RED",
    "YELLOW",
    "JobStatus",
    "Tally",
    "clock",
    "cron_trigger",
    "evaluate_job",
    "interval_seconds",
    "recent_slots",
    "run_summary",
    "slots_between",
    "tally",
]
