"""Read-only brief-slot tools over data/tasks.db (Phase 2 Track 2C).

Four slot tools the morning-briefing skill renders:

  list_open_tasks         — things the user needs to do (no wait_for)
  list_due_today          — tasks with due_at on or before today's end
                            (``include_overdue=False``: today's only)
  list_overdue            — ONE short line naming the tasks due before today
  list_resolved_followups — followups whose awaited reply arrived but
                            the user hasn't acknowledged

The list tools return ``list[dict[str, str]]`` shaped for the brief's
item_template substitution; ``list_overdue`` returns its own headed text.
An expired task is never listed — a meeting-prep task once its meeting is over, a
task overdue past ``digest.yaml`` ``expiry.task_overdue_days`` — even before the
expiry sweep has closed it (``iris_harness.services.digest.expiry``). A digest
``title`` is the short one (repeats collapsed, capped); ``full_title`` keeps the
stored one. See
``config/skills/builtin/morning-briefing/manifest.yaml`` for the
matching templates. Each row carries the raw fields (``task_id``, ``due``
as YYYY-MM-DD, ``priority``) for any caller that wants them, plus the
owner-facing ones the digest shows: ``when`` ("due today 17:00", "due
Oct 13", "overdue since Sep 20") and ``flag`` (" · high" when the
priority is above normal, else "").
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

from iris_harness.sdk.tasks import Task, TaskStore, short_title
from iris_harness.sdk.time import iris_timezone
from iris_harness.services.digest.expiry import is_task_expired, load_expiry_policy
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

_DEFAULT_LIMIT = 10
_OVERDUE_TITLE_CAP = 40  # each title in the one "Overdue" line


def _tasks_db_path() -> Path:
    override = os.getenv("IRIS_TASKS_DB")
    if override:
        return Path(override)
    return Path("data/tasks.db")


def _open_store() -> TaskStore:
    store = TaskStore(db_path=_tasks_db_path())
    store.ensure_schema()
    return store


def _format_due(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return dt.strftime("%Y-%m-%d")


def _due_when(dt: datetime | None, now: datetime | None = None) -> str:
    """How a due date reads to the owner, in their zone (``IRIS_TZ``).

    "due today 17:00", "due tomorrow", "due Oct 13", "due Jan 3, 2027",
    "overdue since Sep 20". A time shows only when it is one (not midnight or the
    end of the day, which is how a date-only due is stored), and only for today and
    tomorrow. "" when there is no due date.
    """
    if dt is None:
        return ""
    zone = iris_timezone()
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    local = aware.astimezone(zone)
    today = (now or datetime.now(UTC)).astimezone(zone).date()
    day = local.date()
    clock = local.time().replace(second=0, microsecond=0)
    at = "" if clock in (time(0, 0), time(23, 59)) else f" {clock:%H:%M}"
    if day < today:
        return f"overdue since {day:%b} {day.day}"
    if day == today:
        return f"due today{at}"
    if day == today + timedelta(days=1):
        return f"due tomorrow{at}"
    year = f", {day.year}" if day.year != today.year else ""
    return f"due {day:%b} {day.day}{year}"


def _today_bounds(now: datetime) -> tuple[datetime, datetime]:
    """The owner's local today (``IRIS_TZ``) as ``[start, end]`` instants."""
    zone = iris_timezone()
    day = now.astimezone(zone).date()
    return (
        datetime.combine(day, time.min, tzinfo=zone),
        datetime.combine(day, time.max, tzinfo=zone),
    )


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _unexpired(rows: list[Task], now: datetime) -> list[Task]:
    """``rows`` minus the tasks that have aged out (digest.yaml ``expiry``)."""
    policy = load_expiry_policy()
    return [t for t in rows if not is_task_expired(t, now, policy)]


def _live_tasks(store: TaskStore, now: datetime, *, limit: int) -> list[Task]:
    """Open + doing tasks that have not expired."""
    rows = store.list(status="open", limit=limit)
    rows += store.list(status="doing", limit=limit)
    return _unexpired(rows, now)


def _task_to_brief_row(task: Task) -> dict[str, str]:
    return {
        "task_id": task.id[:8],
        "title": short_title(task.title),
        "full_title": task.title,
        "due": _format_due(task.due_at),
        "priority": str(task.priority),
        "when": _due_when(task.due_at),
        "flag": " · high" if task.priority > 0 else "",
    }


class _LimitInput(BaseModel):
    """Shared input schema — every slot tool takes an optional limit."""

    limit: int = Field(
        default=_DEFAULT_LIMIT,
        ge=1,
        le=100,
        description="Cap on rows returned.",
    )


class _OpenTasksInput(_LimitInput):
    skip_due_by_today: bool = Field(
        default=False,
        description=(
            "Leave out tasks due today or earlier (the digest lists them under "
            "'Due today' and 'Overdue')."
        ),
    )


class _DueTodayInput(_LimitInput):
    include_overdue: bool = Field(
        default=True,
        description="Also list tasks due before today (False: today's only).",
    )


class ListOpenTasksTool(BaseTool):
    """Open tasks the user owns (excludes followups awaiting external signal)."""

    name: str = "list_open_tasks"
    description: str = (
        "Return open tasks ordered by priority desc, due_at asc. Excludes "
        "followup-shaped tasks (wait_for set) so the brief surfaces them "
        "separately. skip_due_by_today leaves out tasks due today or earlier."
    )
    args_schema: type[BaseModel] = _OpenTasksInput

    def _run(
        self, limit: int = _DEFAULT_LIMIT, skip_due_by_today: bool = False
    ) -> list[dict[str, str]]:
        store = _open_store()
        now = datetime.now(UTC)
        _, end_of_today = _today_bounds(now)
        # Coarse over-fetch then python filter — followups are a minority
        # of total open tasks, and TaskStore.list has no wait_for filter.
        actionable = [t for t in _live_tasks(store, now, limit=limit * 3) if t.wait_for is None]
        if skip_due_by_today:
            actionable = [
                t for t in actionable if t.due_at is None or _aware(t.due_at) > end_of_today
            ]
        actionable.sort(key=lambda t: (-t.priority, t.due_at or datetime.max.replace(tzinfo=UTC)))
        return [_task_to_brief_row(t) for t in actionable[:limit]]

    async def _arun(
        self, limit: int = _DEFAULT_LIMIT, skip_due_by_today: bool = False
    ) -> list[dict[str, str]]:
        return self._run(limit, skip_due_by_today)


class ListDueTodayTool(BaseTool):
    """Tasks with due_at on or before end-of-today."""

    name: str = "list_due_today"
    description: str = (
        "Return open or doing tasks whose due_at falls before the end of "
        "today (local timezone). Sorted by due_at ascending. "
        "include_overdue=False lists only tasks due today."
    )
    args_schema: type[BaseModel] = _DueTodayInput

    def _run(
        self, limit: int = _DEFAULT_LIMIT, include_overdue: bool = True
    ) -> list[dict[str, str]]:
        store = _open_store()
        now = datetime.now(UTC)
        start_of_today, end_of_today = _today_bounds(now)
        rows = store.list(status="open", due_before=end_of_today, limit=limit * 3)
        rows += store.list(status="doing", due_before=end_of_today, limit=limit * 3)
        rows = _unexpired(rows, now)
        if not include_overdue:
            rows = [t for t in rows if t.due_at is not None and _aware(t.due_at) >= start_of_today]
        rows.sort(key=lambda t: t.due_at or datetime.max.replace(tzinfo=UTC))
        return [_task_to_brief_row(t) for t in rows[:limit]]

    async def _arun(
        self, limit: int = _DEFAULT_LIMIT, include_overdue: bool = True
    ) -> list[dict[str, str]]:
        return self._run(limit, include_overdue)


def overdue_line(tasks: list[Task], *, limit: int = _DEFAULT_LIMIT) -> str:
    """``## Overdue (2)`` + one bullet naming them ("Renew registration · Call bank")."""
    if not tasks:
        return ""
    names = [short_title(t.title, _OVERDUE_TITLE_CAP) for t in tasks[:limit]]
    more = len(tasks) - len(names)
    line = " · ".join(names) + (f" · +{more} more" if more > 0 else "")
    return f"## Overdue ({len(tasks)})\n- {line}"


class ListOverdueTool(BaseTool):
    """Tasks due before today, as ONE short line (the digest's Today group)."""

    name: str = "list_overdue"
    description: str = (
        "Return the open or doing tasks due before today (local timezone) as one "
        "headed line: '## Overdue (2)' then '- Renew registration · Call bank'. "
        "Oldest first; followups and expired tasks (a prep task after its meeting, "
        "a task overdue past the digest's expiry days) are left out. Empty text "
        "when nothing is overdue."
    )
    args_schema: type[BaseModel] = _LimitInput

    def _run(self, limit: int = _DEFAULT_LIMIT) -> str:
        store = _open_store()
        now = datetime.now(UTC)
        start_of_today, _ = _today_bounds(now)
        rows = store.list(status="open", due_before=start_of_today, limit=1000)
        rows += store.list(status="doing", due_before=start_of_today, limit=1000)
        overdue = [
            t
            for t in _unexpired(rows, now)
            if t.wait_for is None and t.due_at is not None and _aware(t.due_at) < start_of_today
        ]
        overdue.sort(key=lambda t: _aware(t.due_at or now))
        return overdue_line(overdue, limit=limit)

    async def _arun(self, limit: int = _DEFAULT_LIMIT) -> str:
        return self._run(limit)


class ListResolvedFollowupsTool(BaseTool):
    """Followups whose awaited event arrived but the user hasn't closed yet."""

    name: str = "list_resolved_followups"
    description: str = (
        "Return open followup tasks whose wait_for_resolved_at is set but "
        "status is still open or doing — the user has not yet acknowledged."
    )
    args_schema: type[BaseModel] = _LimitInput

    def _run(self, limit: int = _DEFAULT_LIMIT) -> list[dict[str, str]]:
        store = _open_store()
        rows = _live_tasks(store, datetime.now(UTC), limit=limit * 3)
        resolved = [
            t for t in rows if t.wait_for is not None and t.wait_for_resolved_at is not None
        ]
        resolved.sort(
            key=lambda t: t.wait_for_resolved_at or datetime.max.replace(tzinfo=UTC),
            reverse=True,
        )
        out: list[dict[str, str]] = []
        for t in resolved[:limit]:
            row = _task_to_brief_row(t)
            row["from"] = str(t.wait_for.payload.get("from", "")) if t.wait_for else ""
            out.append(row)
        return out

    async def _arun(self, limit: int = _DEFAULT_LIMIT) -> list[dict[str, str]]:
        return self._run(limit)


__all__ = ["ListDueTodayTool", "ListOpenTasksTool", "ListOverdueTool", "ListResolvedFollowupsTool"]

SKILL_TOOLS = [ListOpenTasksTool, ListDueTodayTool, ListOverdueTool, ListResolvedFollowupsTool]
