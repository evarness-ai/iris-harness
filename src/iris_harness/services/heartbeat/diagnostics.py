"""Heartbeat health diagnostics.

Finds heartbeats that likely did not run when expected, explains why, and
provides a direct command the user can run to recover.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo

from apscheduler.triggers.cron import CronTrigger

from iris_harness.services.heartbeat.models import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatStatus,
)

_INTERVAL_PREFIX = "interval:"
_MIN_GRACE = timedelta(seconds=15)
_CRON_GRACE = timedelta(minutes=2)


@dataclass(frozen=True)
class HeartbeatDiagnostic:
    """One actionable heartbeat scheduling issue."""

    name: str
    schedule: str
    reason: str
    detail: str
    action: str
    last_status: str | None = None
    last_finished_at: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "schedule": self.schedule,
            "reason": self.reason,
            "detail": self.detail,
            "action": self.action,
            "last_status": self.last_status,
            "last_finished_at": self.last_finished_at,
        }


def diagnose_heartbeats(
    definitions: list[HeartbeatDefinition],
    runs: list[HeartbeatRun],
    *,
    created_at: datetime,
    now: datetime | None = None,
    tz: tzinfo | None = None,
) -> list[HeartbeatDiagnostic]:
    """Return heartbeat issues that need user action.

    Rules:
    - Last run failed -> actionable failure.
    - Interval schedules: overdue when last run (or startup) is beyond cadence+grace.
    - Cron schedules: overdue when next scheduled fire after the last run/startup
      has already passed by a small grace window.
    """

    now_utc = now or datetime.now(UTC)
    if tz is None:
        # Cron is read in the owner's zone, as the scheduler reads it (IRIS_TZ).
        from iris_harness.services.digest.settings import iris_timezone

        tz = iris_timezone()
    last_by_name: dict[str, HeartbeatRun] = {}
    for run in runs:
        previous = last_by_name.get(run.name)
        if previous is None or run.started_at > previous.started_at:
            last_by_name[run.name] = run

    issues: list[HeartbeatDiagnostic] = []
    for definition in definitions:
        if not definition.enabled:
            continue

        last = last_by_name.get(definition.name)
        last_finished = _finished_at(last)
        action = f"iris heartbeats trigger {definition.name}"

        if last is not None and last.status is HeartbeatStatus.FAILED:
            issues.append(
                HeartbeatDiagnostic(
                    name=definition.name,
                    schedule=definition.schedule,
                    reason="last_run_failed",
                    detail=(
                        f"last run failed: {last.error.strip()}"
                        if last.error.strip()
                        else "last run failed"
                    ),
                    action=action,
                    last_status=last.status.value,
                    last_finished_at=last_finished.isoformat() if last_finished else None,
                )
            )
            continue

        schedule = definition.schedule.strip()
        if schedule.startswith(_INTERVAL_PREFIX):
            seconds = _parse_interval_seconds(schedule)
            if seconds is None:
                issues.append(
                    HeartbeatDiagnostic(
                        name=definition.name,
                        schedule=definition.schedule,
                        reason="invalid_schedule",
                        detail=f"invalid interval schedule: {definition.schedule}",
                        action=action,
                        last_status=last.status.value if last else None,
                        last_finished_at=last_finished.isoformat() if last_finished else None,
                    )
                )
                continue

            anchor = last_finished or created_at
            elapsed = now_utc - anchor
            grace = max(_MIN_GRACE, timedelta(seconds=max(5, int(seconds * 0.2))))
            if elapsed > timedelta(seconds=seconds) + grace:
                reason = "never_ran" if last is None else "overdue"
                detail = (
                    f"never ran after startup; expected every {seconds}s"
                    if last is None
                    else (
                        f"overdue by {int(elapsed.total_seconds())}s; "
                        f"cadence is every {seconds}s"
                    )
                )
                issues.append(
                    HeartbeatDiagnostic(
                        name=definition.name,
                        schedule=definition.schedule,
                        reason=reason,
                        detail=detail,
                        action=action,
                        last_status=last.status.value if last else None,
                        last_finished_at=last_finished.isoformat() if last_finished else None,
                    )
                )
            continue

        try:
            trigger = CronTrigger.from_crontab(schedule, timezone=tz)
        except ValueError:
            issues.append(
                HeartbeatDiagnostic(
                    name=definition.name,
                    schedule=definition.schedule,
                    reason="invalid_schedule",
                    detail=f"invalid cron schedule: {definition.schedule}",
                    action=action,
                    last_status=last.status.value if last else None,
                    last_finished_at=last_finished.isoformat() if last_finished else None,
                )
            )
            continue

        anchor = last_finished or created_at
        next_due = trigger.get_next_fire_time(anchor, anchor)
        if next_due is None:
            continue
        next_due_utc = next_due.astimezone(UTC)
        if now_utc > next_due_utc + _CRON_GRACE:
            reason = "never_ran" if last is None else "overdue"
            detail = (
                f"never ran; first due at {next_due_utc.isoformat()}"
                if last is None
                else f"missed scheduled run due at {next_due_utc.isoformat()}"
            )
            issues.append(
                HeartbeatDiagnostic(
                    name=definition.name,
                    schedule=definition.schedule,
                    reason=reason,
                    detail=detail,
                    action=action,
                    last_status=last.status.value if last else None,
                    last_finished_at=last_finished.isoformat() if last_finished else None,
                )
            )

    return issues


def _parse_interval_seconds(schedule: str) -> int | None:
    raw = schedule.split(":", 1)[1].strip()
    try:
        seconds = int(raw)
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def _finished_at(run: HeartbeatRun | None) -> datetime | None:
    if run is None:
        return None
    return run.finished_at or run.started_at


__all__ = ["HeartbeatDiagnostic", "diagnose_heartbeats"]
