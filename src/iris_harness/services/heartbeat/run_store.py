"""Durable heartbeat runs: ``<data_dir>/heartbeat_runs.db`` (loop-proof D13).

The scheduler used to keep its run history in memory, so a restart forgot whether the
06:15 sweep ran — exactly the question the owner asks when mail goes missing. Every run
the scheduler keeps (see ``HeartbeatDefinition.record_runs``) is one row here: job name,
the scheduled slot it served, started / finished, status, the one-line ``output``, the
``error``, the handler's structured ``result`` (JSON) and how it was started.

Why a table of its own and not ``activities.db``: the questions asked of it are "the last
successful run of job X at or after 12:15" and "runs of X in yesterday's window", once a
minute per watched job from ``health_tick``. That wants an index on ``(name,
started_at)`` and one INSERT per run; an Activity is a create-then-transition record
(three writes) whose completion events feed the chat notifier, and ``activities`` is
indexed by status and origin only.

Writes are cheap (one INSERT, WAL) and never fatal: :meth:`record` swallows and logs any
error, so a full disk cannot fail the job it was recording. Rows older than
``retention_days`` are pruned on every 500th write.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from iris_harness.foundation.persistence import connect, data_path

from .models import HeartbeatRun, HeartbeatStatus

logger = logging.getLogger(__name__)

DB_NAME = "heartbeat_runs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS heartbeat_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    status       TEXT NOT NULL,
    trigger      TEXT NOT NULL DEFAULT 'schedule',
    slot         TEXT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    output       TEXT NOT NULL DEFAULT '',
    error        TEXT NOT NULL DEFAULT '',
    result       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_heartbeat_runs_name ON heartbeat_runs(name, started_at);
CREATE INDEX IF NOT EXISTS idx_heartbeat_runs_started ON heartbeat_runs(started_at);
CREATE TABLE IF NOT EXISTS heartbeat_jobs (
    name        TEXT PRIMARY KEY,
    first_seen  TEXT NOT NULL
);
"""

# Output and error are one-line summaries; a handler that returns a novel is clipped.
_TEXT_CAP = 2000
_PRUNE_EVERY = 500


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@dataclass(frozen=True)
class StoredRun:
    """One kept run, as read back."""

    name: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    slot: datetime | None = None
    trigger: str = "schedule"
    output: str = ""
    error: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    id: int = 0

    @property
    def ok(self) -> bool:
        return self.status == HeartbeatStatus.SUCCESS.value

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "trigger": self.trigger,
            "slot": _iso(self.slot),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "output": self.output,
            "error": self.error,
            "result": dict(self.result),
        }

    def to_run(self) -> HeartbeatRun:
        """Back into the scheduler's in-memory shape (for the diagnostics)."""
        return HeartbeatRun(
            name=self.name,
            status=HeartbeatStatus(self.status),
            started_at=self.started_at,
            finished_at=self.finished_at,
            output=self.output,
            error=self.error,
            result=dict(self.result),
            trigger=self.trigger,
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> StoredRun:
        try:
            result = json.loads(row["result"] or "{}")
        except ValueError:
            result = {}
        return cls(
            id=int(row["id"]),
            name=row["name"],
            status=row["status"],
            trigger=row["trigger"] or "schedule",
            slot=_dt(row["slot"]),
            started_at=_dt(row["started_at"]) or datetime.now(UTC),
            finished_at=_dt(row["finished_at"]),
            output=row["output"] or "",
            error=row["error"] or "",
            result=result if isinstance(result, dict) else {},
        )


@dataclass
class HeartbeatRunStore:
    """Append-only run history, one row per kept run."""

    db_path: Path = field(default_factory=lambda: data_path(DB_NAME))
    retention_days: int = 30
    _ready: bool = field(default=False, init=False, repr=False)
    _writes: int = field(default=0, init=False, repr=False)

    def ensure_schema(self) -> None:
        if self._ready:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        self._ready = True

    def _connect(self) -> sqlite3.Connection:
        return connect(self.db_path, row_factory=sqlite3.Row)

    # -- writes ----------------------------------------------------------------

    def record(self, run: HeartbeatRun, *, slot: datetime | None = None) -> bool:
        """Keep ``run``. Returns False (and logs) instead of raising on any error."""
        try:
            self.ensure_schema()
            try:
                result = json.dumps(run.result or {}, default=str)
            except (TypeError, ValueError):
                result = "{}"
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO heartbeat_runs (name, status, trigger, slot, started_at,"
                    " finished_at, output, error, result) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        run.name,
                        run.status.value,
                        run.trigger or "schedule",
                        _iso(slot),
                        _iso(run.started_at),
                        _iso(run.finished_at),
                        (run.output or "")[:_TEXT_CAP],
                        (run.error or "")[:_TEXT_CAP],
                        result,
                    ),
                )
            self._writes += 1
            if self._writes % _PRUNE_EVERY == 0:
                self.prune()
            return True
        except Exception:  # keeping the record must never fail the job
            logger.warning("could not record heartbeat run %s", run.name, exc_info=True)
            return False

    def note_job(self, name: str, *, now: datetime | None = None) -> None:
        """Remember when a job was first scheduled here (the "fresh install" line)."""
        try:
            self.ensure_schema()
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO heartbeat_jobs (name, first_seen) VALUES (?, ?)",
                    (name, _iso(now or datetime.now(UTC))),
                )
        except Exception:
            logger.warning("could not note heartbeat job %s", name, exc_info=True)

    def prune(self, *, now: datetime | None = None) -> int:
        cutoff = (now or datetime.now(UTC)) - timedelta(days=self.retention_days)
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM heartbeat_runs WHERE started_at < ?", (_iso(cutoff),))
        return int(cur.rowcount or 0)

    # -- reads -----------------------------------------------------------------

    def first_seen(self, name: str) -> datetime | None:
        self.ensure_schema()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT first_seen FROM heartbeat_jobs WHERE name = ?", (name,)
            ).fetchone()
        return _dt(row["first_seen"]) if row else None

    def recent(
        self,
        *,
        name: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[StoredRun]:
        """Kept runs, newest first."""
        self.ensure_schema()
        clauses: list[str] = []
        params: list[Any] = []
        if name:
            clauses.append("name = ?")
            params.append(name)
        if since is not None:
            clauses.append("started_at >= ?")
            params.append(_iso(since))
        if until is not None:
            clauses.append("started_at < ?")
            params.append(_iso(until))
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        # `clauses` holds only fixed fragments; values bind through `params`.
        sql = (
            f"SELECT * FROM heartbeat_runs {where} ORDER BY started_at DESC, id DESC"  # noqa: S608
        )
        if limit > 0:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            return [StoredRun.from_row(r) for r in conn.execute(sql, params)]

    def last(self, name: str, *, status: str | None = None) -> StoredRun | None:
        found = self.recent(name=name, status=status, limit=1)
        return found[0] if found else None

    def last_success(self, name: str) -> StoredRun | None:
        return self.last(name, status=HeartbeatStatus.SUCCESS.value)

    def last_failure(self, name: str) -> StoredRun | None:
        return self.last(name, status=HeartbeatStatus.FAILED.value)


class HeartbeatRunHistory(Protocol):
    """The reads of the kept runs: what a plugin watching its own jobs may ask.

    :class:`HeartbeatRunStore` satisfies it; :func:`heartbeat_runs` hands a plugin a
    view that has only these methods, because the scheduler is the one writer.
    """

    def first_seen(self, name: str) -> datetime | None:
        """When ``name`` was first scheduled here, or None."""
        ...

    def recent(
        self,
        *,
        name: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[StoredRun]:
        """Kept runs, newest first (``limit=0``: all of them)."""
        ...

    def last(self, name: str, *, status: str | None = None) -> StoredRun | None:
        """The newest kept run of ``name`` (with ``status``, when given)."""
        ...

    def last_success(self, name: str) -> StoredRun | None:
        """The newest successful run of ``name``."""
        ...

    def last_failure(self, name: str) -> StoredRun | None:
        """The newest failed run of ``name``."""
        ...


@dataclass(frozen=True)
class _ReadOnlyRunHistory:
    """:class:`HeartbeatRunHistory` over one runs database; a missing file is no runs.

    Never creates the database: a host whose scheduler has recorded nothing yet has
    no file, and a reader must not be the one to make it.
    """

    _db_path: Path

    def _store(self) -> HeartbeatRunStore | None:
        return HeartbeatRunStore(db_path=self._db_path) if self._db_path.exists() else None

    def first_seen(self, name: str) -> datetime | None:
        store = self._store()
        return store.first_seen(name) if store is not None else None

    def recent(
        self,
        *,
        name: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[StoredRun]:
        store = self._store()
        if store is None:
            return []
        return store.recent(name=name, since=since, until=until, status=status, limit=limit)

    def last(self, name: str, *, status: str | None = None) -> StoredRun | None:
        store = self._store()
        return store.last(name, status=status) if store is not None else None

    def last_success(self, name: str) -> StoredRun | None:
        return self.last(name, status=HeartbeatStatus.SUCCESS.value)

    def last_failure(self, name: str) -> StoredRun | None:
        return self.last(name, status=HeartbeatStatus.FAILED.value)


def heartbeat_runs(data_dir: Path) -> HeartbeatRunHistory:
    """The kept runs under ``data_dir`` (``<data_dir>/heartbeat_runs.db``), read-only."""
    return _ReadOnlyRunHistory(Path(data_dir) / DB_NAME)


__all__ = [
    "DB_NAME",
    "HeartbeatRunHistory",
    "HeartbeatRunStore",
    "StoredRun",
    "heartbeat_runs",
]
