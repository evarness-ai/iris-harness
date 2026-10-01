"""SQLite-backed persistence for routine specifications."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from iris_harness.foundation.persistence import connect

from .models import (
    RoutineApprovalRequest,
    RoutineApprovalRequestStatus,
    RoutineApprovalStatus,
    RoutineSpec,
    create_routine_approval_request,
)

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS routines (
    id                    TEXT PRIMARY KEY,
    schema_version        TEXT NOT NULL,
    title                 TEXT NOT NULL,
    goal                  TEXT NOT NULL,
    schedule              TEXT NOT NULL,
    template              TEXT NOT NULL,
    source_preferences    TEXT NOT NULL DEFAULT '[]',
    required_capabilities TEXT NOT NULL DEFAULT '[]',
    delivery_channel      TEXT NOT NULL,
    approval_status       TEXT NOT NULL,
    run_count             INTEGER NOT NULL DEFAULT 0,
    success_count         INTEGER NOT NULL DEFAULT 0,
    failure_count         INTEGER NOT NULL DEFAULT 0,
    last_run_at           TEXT,
    promotion_candidate   INTEGER NOT NULL DEFAULT 0,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    metadata              TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_routines_status ON routines(approval_status);
CREATE INDEX IF NOT EXISTS idx_routines_updated_at ON routines(updated_at);

CREATE TABLE IF NOT EXISTS routine_approval_requests (
    id             TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    routine_id     TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    prompt         TEXT NOT NULL,
    status         TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    resolved_at    TEXT,
    metadata       TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_routine_approval_requests_session_status
    ON routine_approval_requests(session_id, status, updated_at);
CREATE INDEX IF NOT EXISTS idx_routine_approval_requests_routine_id
    ON routine_approval_requests(routine_id);
"""


class RoutineStore:
    """Persist routine specs as validated Pydantic records in SQLite."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = connect(self.db_path, row_factory=sqlite3.Row)
        return conn

    @staticmethod
    def _json_tuple(value: tuple[str, ...]) -> str:
        return json.dumps(list(value))

    @staticmethod
    def _load_tuple(value: str) -> tuple[str, ...]:
        raw = json.loads(value or "[]")
        if not isinstance(raw, list):
            return ()
        return tuple(str(item) for item in raw)

    @staticmethod
    def _metadata_blob(metadata: dict[str, Any]) -> str:
        return json.dumps(metadata, sort_keys=True)

    @classmethod
    def _row_to_spec(cls, row: sqlite3.Row) -> RoutineSpec:
        return RoutineSpec(
            schema_version=row["schema_version"],
            id=row["id"],
            title=row["title"],
            goal=row["goal"],
            schedule=row["schedule"],
            template=row["template"],
            source_preferences=cls._load_tuple(row["source_preferences"]),
            required_capabilities=cls._load_tuple(row["required_capabilities"]),
            delivery_channel=row["delivery_channel"],
            approval_status=RoutineApprovalStatus(row["approval_status"]),
            run_count=row["run_count"],
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            last_run_at=(
                datetime.fromisoformat(row["last_run_at"]) if row["last_run_at"] else None
            ),
            promotion_candidate=bool(row["promotion_candidate"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @classmethod
    def _row_to_approval_request(cls, row: sqlite3.Row) -> RoutineApprovalRequest:
        return RoutineApprovalRequest(
            schema_version=row["schema_version"],
            id=row["id"],
            routine_id=row["routine_id"],
            session_id=row["session_id"],
            prompt=row["prompt"],
            status=RoutineApprovalRequestStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            resolved_at=(
                datetime.fromisoformat(row["resolved_at"]) if row["resolved_at"] else None
            ),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def save(self, spec: RoutineSpec) -> RoutineSpec:
        """Upsert a routine and return the persisted spec."""

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO routines (
                    id, schema_version, title, goal, schedule, template,
                    source_preferences, required_capabilities, delivery_channel,
                    approval_status, run_count, success_count, failure_count,
                    last_run_at, promotion_candidate, created_at, updated_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    schema_version=excluded.schema_version,
                    title=excluded.title,
                    goal=excluded.goal,
                    schedule=excluded.schedule,
                    template=excluded.template,
                    source_preferences=excluded.source_preferences,
                    required_capabilities=excluded.required_capabilities,
                    delivery_channel=excluded.delivery_channel,
                    approval_status=excluded.approval_status,
                    run_count=excluded.run_count,
                    success_count=excluded.success_count,
                    failure_count=excluded.failure_count,
                    last_run_at=excluded.last_run_at,
                    promotion_candidate=excluded.promotion_candidate,
                    updated_at=excluded.updated_at,
                    metadata=excluded.metadata
                """,
                (
                    spec.id,
                    spec.schema_version,
                    spec.title,
                    spec.goal,
                    spec.schedule,
                    spec.template,
                    self._json_tuple(spec.source_preferences),
                    self._json_tuple(spec.required_capabilities),
                    spec.delivery_channel,
                    str(spec.approval_status),
                    spec.run_count,
                    spec.success_count,
                    spec.failure_count,
                    spec.last_run_at.isoformat() if spec.last_run_at else None,
                    int(spec.promotion_candidate),
                    spec.created_at.isoformat(),
                    spec.updated_at.isoformat(),
                    self._metadata_blob(spec.metadata),
                ),
            )
        return spec

    def load(self, routine_id: str) -> RoutineSpec | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM routines WHERE id = ?", (routine_id,)).fetchone()
        return self._row_to_spec(row) if row else None

    def list_all(self) -> list[RoutineSpec]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM routines ORDER BY created_at").fetchall()
        return [self._row_to_spec(row) for row in rows]

    def list_by_status(self, status: RoutineApprovalStatus) -> list[RoutineSpec]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM routines WHERE approval_status = ? ORDER BY updated_at DESC",
                (status.value,),
            ).fetchall()
        return [self._row_to_spec(row) for row in rows]

    def find_session_duplicate(self, session_id: str, *, template: str) -> RoutineSpec | None:
        """The most recent IN-PROGRESS draft authored in this session for the same template
        (roadmap "next slice"). Lets the authoring flow reuse its id so re-stating a routine
        mid-session UPDATES the draft instead of accumulating near-duplicates. Scoped to the
        unapproved draft states so a committed routine is never silently overwritten; matched
        per template + session so two distinct routines (e.g. weekday vs weekend brief) in
        different sessions stay separate. ``None`` when there's nothing to update."""
        states = (
            RoutineApprovalStatus.DRAFT.value,
            RoutineApprovalStatus.CLARIFY.value,
            RoutineApprovalStatus.TEMPLATE.value,
        )
        placeholders = ",".join("?" for _ in states)  # fixed "?,?,?" — values parameterized
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM routines "  # noqa: S608 — placeholders is a fixed "?" list
                "WHERE json_extract(metadata, '$.session_id') = ? AND template = ? "
                f"AND approval_status IN ({placeholders}) "
                "ORDER BY updated_at DESC LIMIT 1",
                (session_id, template, *states),
            ).fetchone()
        return self._row_to_spec(row) if row else None

    def list_recent_approved_by_session(
        self,
        session_id: str,
        *,
        since: datetime | None = None,
    ) -> list[RoutineSpec]:
        """Return SCHEDULED routines created in this session, newest first.

        Phase B post-approval refinement: when a user says "deliver to
        telegram" *after* approving a routine, we need to find which
        recent routine they're refining. ``session_id`` lives inside
        each routine's ``metadata`` blob (set during authoring), so
        we fetch by status and filter in Python — the per-session
        count is tiny and avoids a JSON1 dependency.

        ``since`` bounds the lookback window; ``None`` means "any
        time". Callers typically pass the chat's session-start
        timestamp or a fixed N-hour window.
        """

        scheduled = self.list_by_status(RoutineApprovalStatus.SCHEDULED)
        cutoff = _normalize_datetime(since) if since is not None else None
        matches: list[RoutineSpec] = []
        for spec in scheduled:
            if spec.metadata.get("session_id") != session_id:
                continue
            if cutoff is not None and spec.updated_at < cutoff:
                continue
            matches.append(spec)
        return matches

    def list_executable(self) -> list[RoutineSpec]:
        statuses = (
            RoutineApprovalStatus.APPROVED.value,
            RoutineApprovalStatus.SCHEDULED.value,
        )
        query = (
            "SELECT * FROM routines " "WHERE approval_status IN (?, ?) " "ORDER BY updated_at DESC"
        )
        with self._connect() as conn:
            rows = conn.execute(query, statuses).fetchall()
        return [self._row_to_spec(row) for row in rows]

    def list_due(self, *, now: datetime | None = None) -> list[RoutineSpec]:
        """Return executable routines whose schedule is due at ``now``."""

        checked_at = _normalize_datetime(now)
        return [spec for spec in self.list_executable() if _routine_is_due(spec, checked_at)]

    def record_run(
        self,
        routine_id: str,
        *,
        success: bool,
        finished_at: datetime | None = None,
    ) -> RoutineSpec | None:
        spec = self.load(routine_id)
        if spec is None:
            return None
        updated = spec.record_run(success=success, finished_at=finished_at)
        return self.save(updated)

    def delete(self, routine_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM routines WHERE id = ?", (routine_id,))
        return cursor.rowcount > 0

    def clear(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM routines")
        return cursor.rowcount

    def create_approval_request(
        self,
        *,
        routine_id: str,
        session_id: str,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> RoutineApprovalRequest:
        request = create_routine_approval_request(
            routine_id=routine_id,
            session_id=session_id,
            prompt=prompt,
            metadata=metadata,
        )
        return self.save_approval_request(request)

    def save_approval_request(
        self,
        request: RoutineApprovalRequest,
    ) -> RoutineApprovalRequest:
        """Upsert a durable routine approval request."""

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO routine_approval_requests (
                    id, schema_version, routine_id, session_id, prompt, status,
                    created_at, updated_at, resolved_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    schema_version=excluded.schema_version,
                    routine_id=excluded.routine_id,
                    session_id=excluded.session_id,
                    prompt=excluded.prompt,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    resolved_at=excluded.resolved_at,
                    metadata=excluded.metadata
                """,
                (
                    request.id,
                    request.schema_version,
                    request.routine_id,
                    request.session_id,
                    request.prompt,
                    str(request.status),
                    request.created_at.isoformat(),
                    request.updated_at.isoformat(),
                    request.resolved_at.isoformat() if request.resolved_at else None,
                    self._metadata_blob(request.metadata),
                ),
            )
        return request

    def load_approval_request(self, request_id: str) -> RoutineApprovalRequest | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM routine_approval_requests WHERE id = ?",
                (request_id,),
            ).fetchone()
        return self._row_to_approval_request(row) if row else None

    def get_pending_approval_request(self, session_id: str) -> RoutineApprovalRequest | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM routine_approval_requests
                WHERE session_id = ? AND status = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (session_id, RoutineApprovalRequestStatus.PENDING.value),
            ).fetchone()
        return self._row_to_approval_request(row) if row else None

    def list_approval_requests(
        self,
        *,
        status: RoutineApprovalRequestStatus | None = None,
        session_id: str | None = None,
    ) -> list[RoutineApprovalRequest]:
        clauses: list[str] = []
        params: list[str] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        query = "SELECT * FROM routine_approval_requests"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_approval_request(row) for row in rows]

    def resolve_approval_request(
        self,
        request_id: str,
        status: RoutineApprovalRequestStatus,
    ) -> RoutineApprovalRequest | None:
        if status == RoutineApprovalRequestStatus.PENDING:
            raise ValueError("pending approval requests cannot be resolved to pending")
        request = self.load_approval_request(request_id)
        if request is None:
            return None
        return self.save_approval_request(request.with_status(status))

    def supersede_pending_approval_requests(self, session_id: str) -> int:
        resolved_at = _normalize_datetime(None).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE routine_approval_requests
                SET status = ?, updated_at = ?, resolved_at = ?
                WHERE session_id = ? AND status = ?
                """,
                (
                    RoutineApprovalRequestStatus.SUPERSEDED.value,
                    resolved_at,
                    resolved_at,
                    session_id,
                    RoutineApprovalRequestStatus.PENDING.value,
                ),
            )
        return cursor.rowcount


def _normalize_datetime(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC).replace(microsecond=0)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC, microsecond=0)
    return value.astimezone(UTC).replace(microsecond=0)


# Grace window for ``daily:``/``cron:`` schedules: a routine fires if a heartbeat tick
# lands within this many seconds AT/AFTER its scheduled minute. Must exceed the routine
# tick interval (config/heartbeats.yaml ``routine_tick`` = 60s) so at least one tick is
# guaranteed to fall inside it; kept small so firing stays close to the scheduled time.
_DAILY_GRACE_SECONDS = 300


def _routine_zone(spec: RoutineSpec) -> tzinfo | None:
    """The zone a routine's wall-clock schedule is read in; ``None`` = machine-local.

    A routine may pin an IANA zone in ``metadata["timezone"]`` (the seeded digest pins
    ``IRIS_TZ``, because a container's local zone is UTC). An unknown name falls back
    to machine-local rather than breaking every tick.
    """
    name = str(spec.metadata.get("timezone") or "").strip()
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning("routine %s: unknown timezone %r; using machine-local", spec.id, name)
        return None


def _routine_is_due(spec: RoutineSpec, now: datetime) -> bool:
    schedule = spec.schedule.strip().lower()
    if schedule.startswith("interval:"):
        seconds = _parse_interval_seconds(schedule)
        if spec.last_run_at is None:
            return True
        return (now - _normalize_datetime(spec.last_run_at)).total_seconds() >= seconds
    # ``daily:`` / ``cron:`` wall-clock schedules are the USER'S LOCAL time, not UTC —
    # "daily:09:00" means 9am where they are. ``now`` arrives UTC-aware; convert to the
    # routine's pinned zone, else the machine-local zone (the same zone the
    # calendar/day-plan use) before matching.
    zone = _routine_zone(spec)
    local_now = now.astimezone(zone)
    if schedule.startswith("daily:"):
        hour, minute = _parse_daily_time(schedule)
        # ``fold=0``: on the fall-back day a repeated wall time means its FIRST occurrence.
        scheduled = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0, fold=0)
        # Elapsed REAL time since the scheduled instant (compared in UTC). Subtracting
        # two datetimes that share a tzinfo is wall-clock arithmetic, which is wrong on
        # a DST day: a time in the spring-forward gap (02:30) resolves to the instant
        # just after the jump, so the routine still fires once that day.
        scheduled_at = scheduled.astimezone(UTC)
        delta = (now - scheduled_at).total_seconds()
        # Fire within a short grace window AT/AFTER the scheduled minute — wide enough
        # for the heartbeat tick interval to land at least once, but strict: a fully
        # missed window is NOT caught up later in the day.
        if not 0 <= delta < _DAILY_GRACE_SECONDS:
            return False
        if spec.last_run_at is None:
            return True
        # Don't re-fire within the grace window: skip if it already ran at/after today's
        # scheduled instant. A manual run EARLIER in the day (``/routines run`` at 06:00)
        # does not swallow the scheduled one — a digest tested at breakfast still
        # arrives at 07:00. The fall-back repeated hour never re-fires: the scheduled
        # instant is the first occurrence, so the second is past the grace window.
        return _normalize_datetime(spec.last_run_at) < scheduled_at
    if schedule.startswith("cron:"):
        schedule = schedule.split(":", 1)[1].strip()
    if not _cron_matches(schedule, local_now):
        return False
    if spec.last_run_at is None:
        return True
    return _normalize_datetime(spec.last_run_at).astimezone(zone).strftime(
        "%Y-%m-%dT%H:%M"
    ) != local_now.strftime("%Y-%m-%dT%H:%M")


def _parse_interval_seconds(schedule: str) -> int:
    try:
        seconds = int(schedule.split(":", 1)[1])
    except ValueError as exc:
        raise ValueError(f"invalid interval routine schedule {schedule!r}") from exc
    if seconds < 0:
        raise ValueError("interval routine schedule must be non-negative")
    return seconds


def _parse_daily_time(schedule: str) -> tuple[int, int]:
    value = schedule.split(":", 1)[1]
    hour_text, minute_text = value.split(":", 1)
    hour = int(hour_text)
    minute = int(minute_text)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"invalid daily routine schedule {schedule!r}")
    return hour, minute


def _cron_matches(schedule: str, now: datetime) -> bool:
    fields = schedule.split()
    if len(fields) != 5:
        raise ValueError(f"unsupported routine schedule {schedule!r}")
    minute, hour, day, month, day_of_week = fields
    cron_weekday = now.isoweekday() % 7
    return (
        _cron_field_matches(minute, now.minute)
        and _cron_field_matches(hour, now.hour)
        and _cron_field_matches(day, now.day)
        and _cron_field_matches(month, now.month)
        and _cron_field_matches(day_of_week, cron_weekday)
    )


def _cron_field_matches(field: str, value: int) -> bool:
    for part in field.split(","):
        if not part:
            continue
        if part == "*":
            return True
        if part.startswith("*/"):
            step = int(part[2:])
            if step > 0 and value % step == 0:
                return True
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            if int(start_text) <= value <= int(end_text):
                return True
            continue
        if int(part) == value:
            return True
    return False
