"""What IRIS keeps, and the one path that deletes it.

Nothing here removes a summary, a confirmed fact, or anything the owner wrote. The
automatic pass only:

- **closes** a session that has been idle (a final summary roll, so cooling never
  discards a span nobody summarized),
- **cools** conversations past ``hot_days`` — the full text and its turn vectors go,
  the session's summary stands in for it,
- **sweeps** vectors whose SQLite row no longer exists,
- **compacts** the database and **rotates** logs (``~/.iris/logs`` holds full prompts,
  so that is a privacy setting as much as a disk one).

Owner actions — deleting a session, a summary, a fact, or everything matching a
search — go through :meth:`RetentionService.forget_matching` and
:meth:`purge_sessions`, both of which preview before they delete.

Every delete goes through one seam that removes the SQLite rows and their vectors
together, because the two drifted apart for months: a forgotten fact kept resurfacing
from ChromaDB, and 2,858 wiki vectors outlived 2,505 pages.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from iris_harness.foundation.process_state import track_globals
from iris_harness.memory.log_archive import count_in_file, scrub_file

logger = logging.getLogger(__name__)

_CONFIG_CACHE: dict[str, Any] | None = None

_FALLBACK: dict[str, Any] = {
    "conversations": {"hot_days": 90, "close_after_idle_days": 1},
    "ephemeral_session_prefixes": [],
    "logs": {
        "compress_after_days": 7,
        "delete_after_days": 30,
        "max_total_mb": 200,
        "rotate_over_mb": 20,
        "service_logs_keep_days": 90,
        "archive": {"enabled": True, "max_mb": 0},
    },
    "housekeeping": {"enabled": True, "vacuum": True},
}


def retention_config() -> dict[str, Any]:
    """Read ``config/memory/retention.yaml`` (cached), falling back to the built-in shape."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    from iris_harness.foundation.paths import config_path, default_config_dir

    path = config_path("memory", "retention.yaml")
    if not path.exists():
        path = default_config_dir() / "memory" / "retention.yaml"
    config = {k: dict(v) if isinstance(v, dict) else list(v) for k, v in _FALLBACK.items()}
    if path.exists():
        try:
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config.update(loaded)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read %s: %s — using built-in retention defaults", path, exc)
    _CONFIG_CACHE = config
    return config


def reset_config_cache() -> None:
    """Drop the cached config (tests, and a reload after editing the YAML)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


def is_ephemeral_session(session_id: str) -> bool:
    """True for playground / eval / test sessions — runs that are not the user's memory.

    1,420 of 3,730 stored turns on the owner's box came from these, and cross-session
    recall served them back as "your past conversations".
    """
    sid = (session_id or "").strip().lower()
    if not sid:
        return False
    prefixes = retention_config().get("ephemeral_session_prefixes") or []
    return any(sid.startswith(str(p).lower()) for p in prefixes if str(p).strip())


@dataclass(frozen=True)
class FlaggedSession:
    """A conversation whose id looks like a test run, with why (``test_session_review``)."""

    session_id: str
    summary_goal: str
    turns: int
    last_activity: str
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "summary_goal": self.summary_goal,
            "turns": self.turns,
            "last_activity": self.last_activity,
            "reasons": list(self.reasons),
        }


def _real_session_patterns() -> list[tuple[re.Pattern[str], str]]:
    """``test_session_review.real_session_patterns``; a bad pattern is skipped, not fatal."""
    review = retention_config().get("test_session_review") or {}
    found: list[tuple[re.Pattern[str], str]] = []
    for raw in review.get("real_session_patterns") or []:
        entry = raw if isinstance(raw, dict) else {"pattern": raw}
        try:
            found.append((re.compile(str(entry.get("pattern", ""))), str(entry.get("what") or "")))
        except re.error as exc:
            logger.warning("retention: bad real_session_pattern %r: %s", raw, exc)
    return found


def real_session_shapes() -> list[str]:
    """What a real conversation's id looks like, in words — shown beside the review."""
    return [what for _p, what in _real_session_patterns() if what]


def flag_test_sessions(store: Any) -> list[FlaggedSession]:
    """Sessions that look like test runs, for the owner to tick and remove (ADR-0119).

    A session is flagged when its id matches none of ``real_session_patterns``. One the
    owner already removed, or a run the ephemeral prefixes already keep out of memory,
    is not offered again. With no patterns configured nothing is flagged: without a
    description of a real id, every conversation would look like a test.
    """
    patterns = _real_session_patterns()
    if not patterns:
        return []
    review = retention_config().get("test_session_review") or {}
    try:
        few = int(review.get("few_turns", 0))
    except (TypeError, ValueError):
        few = 0
    try:
        removed = set(store.removed_session_ids())
    except AttributeError:  # a store without the ledger has removed nothing
        removed = set()
    except Exception as exc:  # the review still lists what it can
        logger.warning(
            "session review: removed-session ledger unreadable (%s); removed sessions may be listed",
            type(exc).__name__,
            exc_info=True,
        )
        removed = set()
    flagged: list[FlaggedSession] = []
    for session_id, last_ts, turns in store.session_activity():
        if session_id in removed or is_ephemeral_session(session_id):
            continue
        if any(p.search(session_id) for p, _what in patterns):
            continue
        reasons = ["its id is not shaped like a real conversation's"]
        if few and turns <= few:
            reasons.append(f"only {turns} turn{'s' if turns != 1 else ''}")
        summary = store.load_conversation_summary(session_id) or ""
        goal = next((line.strip() for line in summary.splitlines() if line.strip()), "")
        flagged.append(FlaggedSession(session_id, goal, turns, str(last_ts), tuple(reasons)))
    return sorted(flagged, key=lambda f: f.last_activity, reverse=True)


@dataclass
class ForgetPreview:
    """What a forget would remove. Nothing is deleted until ``confirm`` is passed."""

    needle: str
    turns: list[tuple[int, str, str, str]] = field(default_factory=list)
    summaries: list[tuple[str, str]] = field(default_factory=list)
    facts: list[tuple[str, str]] = field(default_factory=list)
    # Lines in session logs that mention it — live ones and the encrypted archive.
    # Logs hold full prompts, so a forget that left them would not have forgotten.
    log_lines: int = 0
    archived_log_lines: int = 0

    @property
    def total(self) -> int:
        return (
            len(self.turns)
            + len(self.summaries)
            + len(self.facts)
            + self.log_lines
            + self.archived_log_lines
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "needle": self.needle,
            "total": self.total,
            "turns": [
                {"id": i, "session_id": s, "role": r, "content": c[:200]}
                for i, s, r, c in self.turns
            ],
            "summaries": [{"session_id": s, "summary": t[:200]} for s, t in self.summaries],
            "facts": [{"key": k, "value": v} for k, v in self.facts],
            "log_lines": self.log_lines,
            "archived_log_lines": self.archived_log_lines,
        }


@dataclass
class HousekeepingReport:
    """What one pass did — recorded so a silent cleanup is never a surprise."""

    started_at: str
    dry_run: bool = False
    sessions_closed: int = 0
    # Idle sessions still waiting for their closing summary after this pass hit
    # ``close_max_per_pass``; the next hourly check runs again while this is > 0.
    sessions_close_backlog: int = 0
    # Sessions whose closing roll produced no summary this pass (skipped by the next).
    sessions_close_failed: list[str] = field(default_factory=list)
    sessions_cooled: int = 0
    # Past the window but never summarized — kept, because cold means "the summary
    # stands in for the text" and there is nothing to stand in.
    sessions_kept_unsummarized: int = 0
    turns_deleted: int = 0
    vectors_deleted: int = 0
    orphan_vectors_swept: int = 0
    logs_compressed: int = 0
    # Live service logs (``<service>.log``) copied to a dated .gz and emptied.
    logs_rotated: int = 0
    # Session logs moved into the encrypted monthly archive instead of deleted.
    logs_archived: int = 0
    # Archive months dropped to stay within ``logs.archive.max_mb`` (0 = no budget).
    archive_months_dropped: int = 0
    logs_deleted: int = 0
    log_bytes_reclaimed: int = 0
    vacuumed: bool = False
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "dry_run": self.dry_run,
            "sessions_closed": self.sessions_closed,
            "sessions_close_backlog": self.sessions_close_backlog,
            "sessions_close_failed": list(self.sessions_close_failed),
            "sessions_cooled": self.sessions_cooled,
            "sessions_kept_unsummarized": self.sessions_kept_unsummarized,
            "turns_deleted": self.turns_deleted,
            "vectors_deleted": self.vectors_deleted,
            "orphan_vectors_swept": self.orphan_vectors_swept,
            "logs_compressed": self.logs_compressed,
            "logs_rotated": self.logs_rotated,
            "logs_archived": self.logs_archived,
            "archive_months_dropped": self.archive_months_dropped,
            "logs_deleted": self.logs_deleted,
            "log_bytes_reclaimed": self.log_bytes_reclaimed,
            "vacuumed": self.vacuumed,
            "errors": list(self.errors),
        }


def _log_step_failure(report: HousekeepingReport, step: str, exc: BaseException) -> None:
    """A housekeeping step failed: record it on the report AND log it with its traceback.

    The pass goes on to its other steps either way.
    """
    report.errors.append(f"{step}: {exc}")
    logger.warning("housekeeping step %s failed (%s)", step, type(exc).__name__, exc_info=True)


def housekeeping_due(
    last_run: dict[str, Any] | None, *, min_hours: float, now: datetime | None = None
) -> tuple[bool, str]:
    """Whether the daily pass should run now, and why (or why not).

    Due when it never ran, when the last pass left idle sessions waiting for their
    closing summary, or when ``min_hours`` have passed since the last pass started.
    """
    if last_run is None:
        return (True, "first pass")
    if int(last_run.get("sessions_close_backlog") or 0) > 0:
        return (True, "closing-summary backlog left by the last pass")
    try:
        started = datetime.fromisoformat(str(last_run.get("started_at")))
    except ValueError:
        return (True, "last pass has no readable start time")
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    hours = ((now or datetime.now(UTC)) - started).total_seconds() / 3600
    if hours >= min_hours:
        return (True, f"last pass {hours:.1f}h ago")
    return (False, f"last pass {hours:.1f}h ago; next after {min_hours:g}h")


def _is_log_file(path: Path) -> bool:
    """A file the log rotation may touch: a log, a session log, or an archive of one."""
    if not path.is_file():
        return False
    name = path.name
    return name.endswith((".log", ".jsonl")) or (
        name.endswith(".gz") and (".log" in name or ".jsonl" in name)
    )


def _is_session_log(path: Path) -> bool:
    return path.name.startswith("session-") and path.suffix == ".jsonl"


def _session_id_of(path: Path) -> str:
    return path.name[len("session-") : -len(".jsonl")]


def _copy_truncate(path: Path, now: datetime) -> int:
    """Archive a live log to ``<name>.<stamp>.gz`` and empty it; bytes reclaimed."""
    target = path.with_name(f"{path.name}.{now.strftime('%Y%m%dT%H%M%SZ')}.gz")
    try:
        before = path.stat().st_size
        with path.open("rb") as src, gzip.open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        with path.open("r+b") as fh:
            fh.truncate(0)
        return max(0, before - target.stat().st_size)
    except OSError:
        logger.exception("could not rotate %s", path)
        return 0


class RetentionService:
    """The one place that deletes conversation memory, in SQLite and the index together."""

    def __init__(
        self,
        store: Any,
        semantic_index: Any | None = None,
        sessions: Any | None = None,
        *,
        logs_dir: Path | None = None,
        history_path: Path | None = None,
        archive: Any | None = None,
    ) -> None:
        self._store = store
        self._index = semantic_index
        self._sessions = sessions
        self._logs_dir = logs_dir
        # Where each pass is recorded (JSON lines, newest last). Kept on disk so a
        # restart neither forgets the history nor resets "when did it last run" —
        # the in-memory list alone meant a stack restarted daily never ran the pass.
        self._history_path = history_path
        # Cold storage for session logs past their live window (``log_archive``).
        # None keeps the old behaviour: they are deleted.
        self._archive = archive
        self._history: list[HousekeepingReport] = []

    # -- the seam -------------------------------------------------------

    def _delete_turns(self, row_ids: list[int], *, dry_run: bool) -> tuple[int, int]:
        """Delete turns and their vectors together. Returns ``(rows, vectors)``."""
        if not row_ids or dry_run:
            return (len(row_ids), len(row_ids) if self._index is not None else 0)
        rows = self._store.delete_turns_by_id(row_ids)
        vectors = self._index.drop_turns(row_ids) if self._index is not None else 0
        return (rows, vectors)

    # -- the automatic pass ---------------------------------------------

    def run(self, *, dry_run: bool = False) -> HousekeepingReport:
        """Close idle sessions, cool old ones, sweep, compact, rotate logs."""
        config = retention_config()
        report = HousekeepingReport(started_at=datetime.now(UTC).isoformat(), dry_run=dry_run)
        conv = config.get("conversations") or {}

        try:
            report.sessions_closed, report.sessions_close_backlog = self._close_idle_sessions(
                int(conv.get("close_after_idle_days", 1)),
                max_per_pass=int(conv.get("close_max_per_pass", 20)),
                dry_run=dry_run,
                failed=report.sessions_close_failed,
            )
        except Exception as exc:  # noqa: BLE001
            _log_step_failure(report, "close_idle", exc)

        try:
            cooled, turns, vectors, skipped = self._cool_sessions(
                int(conv.get("hot_days", 90)), dry_run=dry_run
            )
            (
                report.sessions_cooled,
                report.turns_deleted,
                report.vectors_deleted,
                report.sessions_kept_unsummarized,
            ) = (cooled, turns, vectors, skipped)
        except Exception as exc:  # noqa: BLE001
            _log_step_failure(report, "cool", exc)

        try:
            report.orphan_vectors_swept = self._sweep_orphan_vectors(dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001
            _log_step_failure(report, "sweep", exc)

        try:
            self._rotate_logs(config.get("logs") or {}, report, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001
            _log_step_failure(report, "logs", exc)

        if (
            not dry_run
            and (config.get("housekeeping") or {}).get("vacuum", True)
            and report.turns_deleted
        ):
            try:
                self._store.vacuum()
                report.vacuumed = True
            except Exception as exc:  # noqa: BLE001
                _log_step_failure(report, "vacuum", exc)

        self._history.append(report)
        del self._history[:-50]
        self._persist(report)
        logger.info("housekeeping pass: %s", report.as_dict())
        return report

    def _persist(self, report: HousekeepingReport) -> None:
        if self._history_path is None:
            return
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            lines = self._history_lines()[-49:] + [json.dumps(report.as_dict())]
            self._history_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            logger.exception("could not record the housekeeping pass")

    def _history_lines(self) -> list[str]:
        if self._history_path is None or not self._history_path.exists():
            return []
        return [ln for ln in self._history_path.read_text(encoding="utf-8").splitlines() if ln]

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Recorded passes, newest first — from disk when a history path is set."""
        if self._history_path is None:
            return [r.as_dict() for r in self._history[-limit:]][::-1]
        out: list[dict[str, Any]] = []
        for line in reversed(self._history_lines()):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
            if len(out) >= limit:
                break
        return out

    # -- cold storage ------------------------------------------------------

    def archived_logs(self) -> dict[str, Any]:
        """What the session-log archive holds, month by month (oldest first)."""
        if self._archive is None:
            return {"enabled": False, "months": [], "stored_bytes": 0}
        months = [m.as_dict() for m in self._archive.months()]
        return {
            "enabled": True,
            "root": str(self._archive.root),
            "months": months,
            "stored_bytes": self._archive.total_bytes(),
        }

    def restore_logs(self, *, session: str | None = None, month: str | None = None) -> list[str]:
        """Put archived session logs back where the Sessions view reads them."""
        if self._archive is None or self._logs_dir is None:
            raise RuntimeError("no log archive is configured")
        restored = self._archive.restore(self._logs_dir, session=session, month=month)
        return [p.name for p in restored]

    def last_run(self) -> dict[str, Any] | None:
        """The newest real (not dry-run) pass, or None if it has never run."""
        return next((r for r in self.history(limit=50) if not r.get("dry_run")), None)

    def _close_idle_sessions(
        self,
        idle_days: int,
        *,
        max_per_pass: int = 20,
        dry_run: bool,
        failed: list[str] | None = None,
    ) -> tuple[int, int]:
        """Roll a final summary for sessions nobody has touched lately.

        Each roll is a model call, so a pass closes at most ``max_per_pass`` and
        returns ``(closed, backlog)``; the heartbeat runs again while a backlog is
        left (the first pass on this box found 391 sessions waiting). A session counts
        as closed only when a summary was actually written; one that could not be
        summarized goes into ``failed`` and is skipped by the next pass, so it cannot
        hold the same slots every hour.
        """
        if self._sessions is None or idle_days <= 0:
            return (0, 0)
        previous = self.last_run() or {}
        skip = set(previous.get("sessions_close_failed") or [])
        cutoff = datetime.now(UTC) - timedelta(days=idle_days)
        closed = attempts = backlog = 0
        for session_id, last_ts, _count in self._store.session_activity():
            if is_ephemeral_session(session_id):
                continue
            try:
                last = datetime.fromisoformat(last_ts)
            except ValueError:
                continue
            if last.tzinfo is None:
                last = last.replace(tzinfo=UTC)
            if last > cutoff:
                continue
            if self._store.load_conversation_summary(session_id):
                continue  # already has one; cooling will keep it
            if session_id in skip:
                continue  # failed last pass; retried the pass after
            if attempts >= max_per_pass:
                backlog += 1
                continue
            attempts += 1
            if dry_run:
                closed += 1
                continue
            try:
                result = self._close_one(session_id)
            except Exception:
                logger.exception("closing roll failed for %s", session_id)
                result = {}
            if result.get("closed"):
                closed += 1
            elif failed is not None:
                failed.append(session_id)
        return (closed, backlog)

    def _close_one(self, session_id: str) -> dict[str, Any]:
        assert self._sessions is not None  # checked by the caller
        close = getattr(self._sessions, "close_session", None)
        if close is not None:
            return dict(close(session_id))
        # An older sessions object: compact_now summarizes only past its kept window.
        result = dict(self._sessions.compact_now(session_id))
        return {"closed": bool(result.get("compacted")), **result}

    def _cool_sessions(self, hot_days: int, *, dry_run: bool) -> tuple[int, int, int, int]:
        """Past ``hot_days``: drop the text and its vectors, keep the summary.

        Returns ``(cooled, turns, vectors, skipped_unsummarized)``.
        """
        if hot_days <= 0:
            return (0, 0, 0, 0)
        cutoff = datetime.now(UTC) - timedelta(days=hot_days)
        cooled = turns_deleted = vectors_deleted = skipped_unsummarized = 0
        for session_id, last_ts, _count in self._store.session_activity():
            try:
                last = datetime.fromisoformat(last_ts)
            except ValueError:
                continue
            if last.tzinfo is None:
                last = last.replace(tzinfo=UTC)
            if last > cutoff:
                continue
            row_ids = self._store.fetch_turn_ids(session_id)
            if not row_ids:
                continue
            # Cold means "the summary stands in for the text". With no summary there is
            # nothing to stand in, so leave it alone: on the owner's box 253 sessions
            # were past the window and only 10 had ever been summarized — cooling them
            # would have deleted the conversation outright. The closing roll above is
            # what earns a session its summary; a session it could not summarize keeps
            # its text until it can.
            if not self._store.load_conversation_summary(session_id):
                skipped_unsummarized += 1
                continue
            rows, vectors = self._delete_turns(row_ids, dry_run=dry_run)
            cooled += 1
            turns_deleted += rows
            vectors_deleted += vectors
        return (cooled, turns_deleted, vectors_deleted, skipped_unsummarized)

    def _sweep_orphan_vectors(self, *, dry_run: bool) -> int:
        """Drop turn vectors whose SQLite row is gone."""
        if self._index is None:
            return 0
        indexed = self._index.turn_ids()
        if not indexed:
            return 0
        live: set[str] = set()
        for session_id, _ts, _n in self._store.session_activity():
            live.update(str(i) for i in self._store.fetch_turn_ids(session_id))
        orphans = sorted(indexed - live)
        if not orphans or dry_run:
            return len(orphans)
        return int(self._index.drop_turns([int(i) for i in orphans if i.isdigit()]))

    def _rotate_logs(
        self, config: dict[str, Any], report: HousekeepingReport, *, dry_run: bool
    ) -> None:
        """Rotate live service logs, archive session logs, compress and expire the rest.

        Only log files are touched — ``*.log``, ``*.jsonl`` and their ``.gz`` archives;
        the services' ``.pid`` files live here too and used to be compressed and
        deleted like logs (a service up for a week would have lost its pid file).

        - **Session logs** (``session-*.jsonl``) are the chat history the Sessions view
          replays, read uncompressed, so they are never compressed in place. Past
          ``delete_after_days`` a real session's log moves to the encrypted monthly
          archive (owner decision 2026-09-19: keep, locked, instead of delete); a test
          or playground run's log is deleted. A log is removed from here only after
          the archive has stored it.
        - **Live service logs** (``<service>.log``, appended to by ``start_iris.sh``)
          have a fresh mtime forever, so age never rotated them: iris_api.log reached
          100 MB. Past ``rotate_over_mb`` one is copied to ``<name>.<UTC stamp>.gz``
          and emptied in place. The writer holds it with O_APPEND and carries on at
          the new end; lines written between the copy and the truncate are lost
          (logrotate's copytruncate trade-off).
        - **Other logs** (rotated archives, a stopped service's log) are compressed at
          ``compress_after_days`` and deleted at ``service_logs_keep_days``.
        - Over ``max_total_mb`` the oldest go first — session logs to the archive,
          the rest deleted — but never a live ``.log``: unlinking a file a process
          still writes frees nothing.
        """
        logs_dir = self._logs_dir
        if logs_dir is None or not logs_dir.exists():
            return
        compress_after = int(config.get("compress_after_days", 7))
        delete_after = int(config.get("delete_after_days", 30))
        keep_service = int(config.get("service_logs_keep_days", 90))
        max_bytes = int(config.get("max_total_mb", 200)) * 1024 * 1024
        rotate_over = int(config.get("rotate_over_mb", 20)) * 1024 * 1024
        archive_config = config.get("archive") or {}
        archive = self._archive if archive_config.get("enabled", True) else None
        now = datetime.now(UTC)
        to_archive: list[Path] = []

        def drop(path: Path) -> None:
            report.log_bytes_reclaimed += path.stat().st_size
            report.logs_deleted += 1
            if not dry_run:
                path.unlink(missing_ok=True)

        def retire_session_log(path: Path) -> None:
            if archive is not None and not is_ephemeral_session(_session_id_of(path)):
                to_archive.append(path)
            else:
                drop(path)

        for path in [p for p in logs_dir.iterdir() if _is_log_file(p)]:
            stat = path.stat()
            age_days = (now.timestamp() - stat.st_mtime) / 86400
            if _is_session_log(path):
                if age_days >= delete_after:
                    retire_session_log(path)
                continue
            if age_days >= keep_service:
                drop(path)
            elif path.suffix == ".log" and stat.st_size > rotate_over:
                report.logs_rotated += 1
                if not dry_run:
                    report.log_bytes_reclaimed += _copy_truncate(path, now)
            elif path.suffix != ".gz" and age_days >= compress_after:
                report.logs_compressed += 1
                if not dry_run:
                    target = path.with_suffix(path.suffix + ".gz")
                    try:
                        with path.open("rb") as src, gzip.open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        report.log_bytes_reclaimed += max(0, stat.st_size - target.stat().st_size)
                        path.unlink(missing_ok=True)
                    except OSError:
                        logger.exception("could not compress %s", path)

        # Size cap, oldest first, after the age passes — never a live service log.
        leaving = {p.name for p in to_archive}
        remaining = sorted(
            (
                p
                for p in logs_dir.iterdir()
                if _is_log_file(p) and p.suffix != ".log" and p.name not in leaving
            ),
            key=lambda p: p.stat().st_mtime,
        )
        total = sum(
            p.stat().st_size
            for p in logs_dir.iterdir()
            if _is_log_file(p) and p.name not in leaving
        )
        for path in remaining:
            if total <= max_bytes:
                break
            total -= path.stat().st_size
            if _is_session_log(path):
                retire_session_log(path)
            else:
                drop(path)

        if to_archive and archive is not None:
            self._archive_logs(archive, to_archive, report, dry_run=dry_run)
        if archive is not None and not dry_run:
            budget = int(archive_config.get("max_mb", 0)) * 1024 * 1024
            report.archive_months_dropped = archive.enforce_budget(budget)

    def _archive_logs(
        self, archive: Any, paths: list[Path], report: HousekeepingReport, *, dry_run: bool
    ) -> None:
        """Store ``paths`` in the archive, then remove them here — never the other way."""
        if dry_run:
            report.logs_archived += len(paths)
            return
        sizes = {p: p.stat().st_size for p in paths}
        try:
            report.logs_archived += int(archive.add(paths))
        except Exception as exc:  # keep every original if the archive fails
            logger.exception("could not archive %d session log(s)", len(paths))
            report.errors.append(f"archive: {exc}")
            return
        for path in paths:
            path.unlink(missing_ok=True)
            report.log_bytes_reclaimed += sizes[path]

    # -- owner actions ---------------------------------------------------

    def preview_forget(self, needle: str) -> ForgetPreview:
        """Everything a forget would remove — turns, summaries and facts."""
        preview = ForgetPreview(needle=needle)
        if not needle.strip():
            return preview
        preview.turns = self._store.search_turns(needle, include_removed=True)
        preview.summaries = self._store.search_summaries(needle, include_removed=True)
        lowered = needle.lower()
        preview.facts = [
            (f.key, f.value)
            for f in self._store.fetch_all_user_facts()
            if lowered in f.key.lower() or lowered in f.value.lower()
        ]
        preview.log_lines = sum(count_in_file(p, needle) for p in self._session_logs())
        if self._archive is not None:
            preview.archived_log_lines = self._archive.count_matching(needle)
        return preview

    def _session_logs(self) -> list[Path]:
        if self._logs_dir is None or not self._logs_dir.exists():
            return []
        return [p for p in self._logs_dir.iterdir() if p.is_file() and _is_session_log(p)]

    def forget_matching(self, needle: str, *, confirm: bool = False) -> dict[str, Any]:
        """Delete everything matching ``needle``. Returns the preview unless confirmed."""
        preview = self.preview_forget(needle)
        if not confirm:
            return {"deleted": False, "preview": preview.as_dict()}

        turn_ids = [t[0] for t in preview.turns]
        rows, vectors = self._delete_turns(turn_ids, dry_run=False)
        for session_id, _summary in preview.summaries:
            self._store.delete_conversation_summary(session_id)
        forgotten_facts = 0
        if preview.facts:
            from iris_harness.memory.coordinator import FactCoordinator

            coordinator = FactCoordinator(self._store, self._index)
            for key, _value in preview.facts:
                if coordinator.forget(key):
                    forgotten_facts += 1
        log_lines = sum(scrub_file(p, needle) for p in self._session_logs())
        archived = self._archive.scrub(needle) if self._archive is not None else 0
        return {
            "deleted": True,
            "turns": rows,
            "vectors": vectors,
            "summaries": len(preview.summaries),
            "facts": forgotten_facts,
            "log_lines": log_lines,
            "archived_log_lines": archived,
        }

    def purge_sessions(
        self, session_ids: list[str] | None = None, *, confirm: bool = False
    ) -> dict[str, Any]:
        """Remove sessions entirely. Defaults to every ephemeral (test/eval) session."""
        if session_ids is None:
            session_ids = [
                sid for sid, _ts, _n in self._store.session_activity() if is_ephemeral_session(sid)
            ]
        rows = [
            (sid, len(self._store.fetch_turn_ids(sid)))
            for sid in session_ids
            if self._store.fetch_turn_ids(sid)
        ]
        total_turns = sum(n for _sid, n in rows)
        if not confirm:
            return {
                "deleted": False,
                "sessions": [{"session_id": s, "turns": n} for s, n in rows],
                "total_sessions": len(rows),
                "total_turns": total_turns,
            }
        deleted_turns = deleted_vectors = 0
        for session_id, _n in rows:
            ids = self._store.fetch_turn_ids(session_id)
            turns, vectors = self._delete_turns(ids, dry_run=False)
            self._store.delete_conversation_summary(session_id)
            deleted_turns += turns
            deleted_vectors += vectors
        if deleted_turns:
            try:
                self._store.vacuum()
            except Exception:
                logger.exception("vacuum after purge failed")
        return {
            "deleted": True,
            "total_sessions": len(rows),
            "turns": deleted_turns,
            "vectors": deleted_vectors,
        }


__all__ = [
    "FlaggedSession",
    "ForgetPreview",
    "HousekeepingReport",
    "RetentionService",
    "flag_test_sessions",
    "real_session_shapes",
    "is_ephemeral_session",
    "reset_config_cache",
    "retention_config",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_CONFIG_CACHE")
