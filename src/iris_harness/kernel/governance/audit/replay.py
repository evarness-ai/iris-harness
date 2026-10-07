"""Rebuild one session's timeline from the stores and say what is missing (issue #134, stage 5).

The ledger is written by many processes and read back much later, hot and cold. A replay reads
what is stored (the audit ledger, its Parquet archive, the approval queue, the side-effect
ledger, the session log), orders it, hangs the calls on the tree their ``parent_call_id`` edges
make, and then checks every claim the records make about each other:

* a **gap** is proven loss or contradiction: a sequence number no store holds, an archive chunk
  that no longer matches its marker, a call that started and never settled in a process that is
  gone, a parent that does not exist, a witness with no counterpart, a settle with no start;
* a **note** is something a reader should know and that proves nothing is missing: rows from
  before identity existed, a writer that has not said it is finished, a cold tier not read.

Nothing here writes. A record carries identifiers and decisions only (the closed field set of
``audit_view``): never an argument, a result or a reason. The session filter is applied to what
is REPORTED; the checks may look at other sessions' sequence numbers (a writer interleaves
sessions) but never return their rows.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from iris_harness.foundation.observability.audit_view import public_payload
from iris_harness.kernel.governance.audit import sequence, spool

#: What the API allows one request to read: rows per store and seconds in all. A replay of a
#: long session is a CLI job; the endpoint answers within these bounds or says it was cut short.
API_MAX_ROWS = 2000
API_MAX_SECONDS = 5.0

_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

#: Hook points whose pairing means "a call started" / "a call settled".
PRE_CALL = "pre_tool_use"
POST_CALL = "post_tool_use"
#: Decisions at a PRE row after which the call never runs (held for approval, or refused).
_STOPS = frozenset({"deny", "require_approval"})

GAP_CLASSES = (
    "sequence_hole",
    "lost_write",
    "archive_mismatch",
    "open_call",
    "missing_parent",
    "witness_mismatch",
    "orphan_settle",
    "duplicate",
)

_ROW_COLUMNS = (
    "id, ts, run_id, step_id, hook_point, decision, record_id, session_id, turn_id, call_id, "
    "parent_call_id, attempt, replay_of, resumed_from_run, writer_id, writer_seq, kind, "
    "payload_json"
)
_COLD_COLUMNS = tuple(c.strip() for c in _ROW_COLUMNS.split(","))


@dataclass(frozen=True)
class ReplayRecord:
    """One stored fact about the session, as a reader may see it."""

    store: str  # audit | approval | ledger | session_log
    tier: str  # hot | cold | -
    ts: str
    kind: str | None
    hook_point: str | None
    decision: str | None
    record_id: str | None
    writer_id: str | None
    writer_seq: int | None
    run_id: str | None
    step_id: int | None
    turn_id: str | None
    call_id: str | None
    parent_call_id: str | None
    attempt: int | None
    replay_of: str | None
    resumed_from_run: str | None
    tool: str | None = None
    caller: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass(frozen=True)
class Gap:
    """Proven loss or contradiction. ``evidence`` names record ids, never content."""

    cls: str
    subject: str
    detail: str
    evidence: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "class": self.cls,
            "subject": self.subject,
            "detail": self.detail,
            "evidence": list(self.evidence),
        }

    def __str__(self) -> str:
        return f"{self.cls}: {self.detail}"


@dataclass(frozen=True)
class Note:
    """Something to know that proves nothing is missing."""

    code: str
    subject: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "subject": self.subject, "detail": self.detail}


@dataclass(frozen=True)
class Replay:
    session_id: str
    records: tuple[ReplayRecord, ...]
    tree: tuple[dict[str, Any], ...]
    gaps: tuple[Gap, ...]
    notes: tuple[Note, ...]
    #: False when a row cap or the time cap cut a read short: gap detection is then skipped
    #: (a partial read cannot prove anything missing) and ``gaps`` is empty by construction.
    complete: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "complete": self.complete,
            "truncated": not self.complete,
            "records": [r.as_dict() for r in self.records],
            "tree": list(self.tree),
            "gaps": [g.as_dict() for g in self.gaps],
            "notes": [n.as_dict() for n in self.notes],
        }


@dataclass(frozen=True)
class ReplaySources:
    """Where the stores are. ``None`` for a store means it is not read (a note says so)."""

    audit_db: Path
    archive_root: Path | None = None
    approvals_db: Path | None = None
    ledger_db: Path | None = None
    session_log_dir: Path | None = None

    @classmethod
    def default(cls, audit_db: Path | None = None) -> ReplaySources:
        from iris_harness.foundation.observability.session_log import session_log_dir
        from iris_harness.kernel.governance.approvals.store import default_approvals_db_path
        from iris_harness.kernel.governance.audit.log import _default_audit_db_path
        from iris_harness.kernel.governance.side_effects.store import default_ledger_db_path

        try:
            from iris_harness.kernel.governance.audit.archive.writer import default_archive_root

            archive: Path | None = default_archive_root()
        except ModuleNotFoundError:  # pyarrow is optional
            archive = None
        return cls(
            audit_db=audit_db or _default_audit_db_path(),
            archive_root=archive,
            approvals_db=default_approvals_db_path(),
            ledger_db=default_ledger_db_path(),
            session_log_dir=session_log_dir(),
        )


class _Budget:
    """Row and time caps shared by every read of one replay."""

    def __init__(self, max_rows: int | None, max_seconds: float | None) -> None:
        self.max_rows = max_rows
        self.deadline = None if max_seconds is None else time.monotonic() + max_seconds
        self.max_seconds = max_seconds
        self.cut = False

    def remaining(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.monotonic())

    def expired(self) -> bool:
        left = self.remaining()
        if left is not None and left <= 0.0:
            self.cut = True
            return True
        return False


def replay_session(
    session_id: str,
    *,
    turn: str | None = None,
    run: str | None = None,
    sources: ReplaySources | None = None,
    include_archive: bool = True,
    max_rows: int | None = None,
    max_seconds: float | None = None,
) -> Replay:
    """Rebuild ``session_id`` (optionally one turn or one run of it) from the stores."""
    if not _SESSION_ID.match(session_id) or ".." in session_id:
        raise ValueError("not a session id")  # it names a file in the session log directory
    src = sources or ReplaySources.default()
    budget = _Budget(max_rows, max_seconds)
    try:
        return _replay(session_id, turn, run, src, include_archive, budget)
    except sqlite3.OperationalError:
        if not budget.cut:  # not the budget stopping a statement: a real failure
            raise
        note = Note("truncated", "replay", "the time cap stopped a read; gaps were not judged")
        return Replay(session_id, (), (), (), (note,), complete=False)


def _replay(
    session_id: str,
    turn: str | None,
    run: str | None,
    src: ReplaySources,
    include_archive: bool,
    budget: _Budget,
) -> Replay:
    notes: list[Note] = []
    gaps: list[Gap] = []

    log_events = _read_session_log(src, session_id, budget, notes)
    hot = _hot_session_rows(src.audit_db, session_id, budget)
    since, until = _window([r["ts"] for r in hot] + [e["ts"] for e in log_events])

    cold: list[dict[str, Any]] = []
    cold_read = False
    files: list[Path] = []
    if include_archive and src.archive_root is not None and src.archive_root.exists():
        files, cold, cold_read = _cold_session_rows(
            src, session_id, since, until, budget, notes, gaps
        )
    elif not include_archive:
        notes.append(Note("cold_tier_not_read", "archive", "the archive was not asked for"))

    rows, dupes = _merge(hot, cold)
    gaps.extend(
        Gap(
            "duplicate",
            f"record {rid}",
            "the same record id was stored twice with different content",
            (rid,),
        )
        for rid in dupes
    )

    if budget.cut:  # a partial read proves nothing missing: say so and stop
        notes.append(
            Note("truncated", "replay", "a row or time cap cut a read short; gaps were not judged")
        )
        records = tuple(_record(r) for r in _sort(rows) if _selected(r, turn, run))
        return Replay(session_id, records, (), (), tuple(notes), complete=False)

    ctx = _Context(src, budget, since, until, files, cold_read, notes)
    gaps.extend(_sequence_gaps(ctx, rows))
    gaps.extend(_archive_gaps(ctx))
    gaps.extend(_call_gaps(ctx, rows, log_events, session_id))
    gaps.extend(_lineage_gaps(ctx, rows, session_id))
    gaps.extend(_writer_notes(ctx, rows))
    if any(r["record_id"] is None for r in rows):
        n = sum(1 for r in rows if r["record_id"] is None)
        notes.append(
            Note("pre_identity_era", "audit", f"{n} row(s) were written before records had ids")
        )
    if budget.cut:
        notes.append(
            Note("truncated", "replay", "the time cap cut a check short; gaps were not judged")
        )
        records = tuple(_record(r) for r in _sort(rows) if _selected(r, turn, run))
        return Replay(session_id, records, (), (), tuple(notes), complete=False)

    selected = [r for r in _sort(rows) if _selected(r, turn, run)]
    records = tuple(_record(r) for r in selected)
    return Replay(
        session_id=session_id,
        records=records,
        tree=tuple(_tree(selected)),
        gaps=tuple(gaps),
        notes=tuple(notes),
    )


# ------------------------------------------------------------------------------ reading


def _window(stamps: Iterable[str]) -> tuple[datetime | None, datetime | None]:
    parsed: list[datetime] = []
    for raw in stamps:
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            continue
        parsed.append(dt if dt.tzinfo else dt.replace(tzinfo=UTC))
    if not parsed:
        return None, None
    margin = timedelta(seconds=1)
    return min(parsed) - margin, max(parsed) + margin


#: VM instructions between two looks at the clock. A statement that scans a large table does
#: not return to Python, so the wall-clock budget can only stop it from inside SQLite.
_PROGRESS_OPS = 10_000


def _connect_ro(path: Path, budget: _Budget | None = None) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    if budget is not None and budget.deadline is not None:
        # Returning non-zero aborts the running statement ("interrupted"); the budget is
        # marked cut, and ``replay_session`` turns that into an incomplete result.
        conn.set_progress_handler(lambda: 1 if budget.expired() else 0, _PROGRESS_OPS)
    return conn


def _columns_of(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _row_dict(row: sqlite3.Row, present: set[str], tier: str) -> dict[str, Any]:
    out = {name: (row[name] if name in present else None) for name in _COLD_COLUMNS}
    out["tier"] = tier
    return out


def _hot_session_rows(db: Path, session_id: str, budget: _Budget) -> list[dict[str, Any]]:
    conn = _connect_ro(db, budget)
    if conn is None:
        return []
    with closing(conn):
        present = _columns_of(conn, "audit_log")
        select = ", ".join(c if c in present else f"NULL AS {c}" for c in _COLD_COLUMNS)
        by_col = "session_id = ? OR " if "session_id" in present else ""
        sql = (
            f"SELECT {select} FROM audit_log WHERE {by_col}"  # noqa: S608 - fixed names
            "("
            + ("session_id IS NULL AND " if by_col else "")
            + "json_extract(payload_json, '$.session_id') = ?) "
            "ORDER BY ts, id"
        )
        params = ([session_id] if by_col else []) + [session_id]
        limit = None if budget.max_rows is None else budget.max_rows + 1
        if limit is not None:
            sql += f" LIMIT {limit}"
        rows = [_row_dict(r, set(_COLD_COLUMNS), "hot") for r in conn.execute(sql, params)]
    if budget.max_rows is not None and len(rows) > budget.max_rows:
        budget.cut = True
        rows = rows[: budget.max_rows]
    return rows


def _cold_session_rows(
    src: ReplaySources,
    session_id: str,
    since: datetime | None,
    until: datetime | None,
    budget: _Budget,
    notes: list[Note],
    gaps: list[Gap],
) -> tuple[list[Path], list[dict[str, Any]], bool]:
    try:
        from iris_harness.kernel.governance.audit.archive.markers import chunks_in_window
        from iris_harness.kernel.governance.audit.archive.query import read_cold_rows
    except ModuleNotFoundError:
        notes.append(Note("cold_tier_not_read", "archive", "pyarrow or duckdb is not installed"))
        return [], [], False
    assert src.archive_root is not None
    if since is None and until is None:
        notes.append(
            Note(
                "cold_window_unbounded",
                "archive",
                "no live row or session log fixed the session's time window: every chunk was read",
            )
        )
    files = chunks_in_window(src.audit_db, src.archive_root, since, until)
    if not files:
        return [], [], True
    limit = None if budget.max_rows is None else budget.max_rows + 1
    try:
        rows, cut = read_cold_rows(
            files,
            columns=_COLD_COLUMNS,
            where="session_id = ? OR (session_id IS NULL AND "
            "json_extract_string(payload_json, '$.session_id') = ?)",
            params=[session_id, session_id],
            limit=limit,
            timeout_s=budget.remaining(),
        )
    except Exception as exc:  # noqa: BLE001 - an unreadable chunk is the finding, not a crash
        gaps.append(
            Gap(
                "archive_mismatch",
                "archive",
                f"the archive could not be read ({type(exc).__name__}); " "run `iris audit verify`",
            )
        )
        return files, [], False
    if cut or (budget.max_rows is not None and len(rows) > budget.max_rows):
        budget.cut = True
        rows = rows[: budget.max_rows] if budget.max_rows is not None else []
    out = []
    for r in rows:
        r = dict(r)
        ts = r.get("ts")
        r["ts"] = ts.isoformat() if isinstance(ts, datetime) else str(ts)
        r["tier"] = "cold"
        out.append(r)
    return files, out, True


def _read_session_log(
    src: ReplaySources, session_id: str, budget: _Budget, notes: list[Note]
) -> list[dict[str, Any]]:
    if src.session_log_dir is None:
        return []
    path = src.session_log_dir / f"session-{session_id}.jsonl"
    if not path.exists():
        notes.append(Note("store_absent", "session_log", "no session log file for this session"))
        return []
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if budget.max_rows is not None and len(events) >= budget.max_rows:
                budget.cut = True
                break
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and isinstance(event.get("ts"), str):
                events.append(event)
    return events


def _merge(
    hot: list[dict[str, Any]], cold: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Hot first, then cold rows not already seen (by record id, else by row id)."""
    seen: dict[str, dict[str, Any]] = {}
    dupes: list[str] = []
    out: list[dict[str, Any]] = []
    for row in (*hot, *cold):
        key = row["record_id"] or (f"id:{row['id']}" if row["id"] is not None else None)
        if key is None:
            out.append(row)
            continue
        prior = seen.get(key)
        if prior is None:
            seen[key] = row
            out.append(row)
        elif (prior["payload_json"], prior["hook_point"], prior["writer_seq"]) != (
            row["payload_json"],
            row["hook_point"],
            row["writer_seq"],
        ):
            dupes.append(str(row["record_id"] or key))
    return out, dupes


def _sort(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (str(r["ts"]), str(r["writer_id"] or ""), r["writer_seq"] or 0, r["id"] or 0),
    )


def _selected(row: Mapping[str, Any], turn: str | None, run: str | None) -> bool:
    if turn is not None and row["turn_id"] != turn:
        return False
    return run is None or row["run_id"] == run


def _record(row: Mapping[str, Any]) -> ReplayRecord:
    pub = public_payload(row["payload_json"])
    return ReplayRecord(
        store="audit",
        tier=str(row.get("tier") or "hot"),
        ts=str(row["ts"]),
        kind=row["kind"],
        hook_point=row["hook_point"],
        decision=row["decision"],
        record_id=row["record_id"],
        writer_id=row["writer_id"],
        writer_seq=row["writer_seq"],
        run_id=row["run_id"],
        step_id=row["step_id"],
        turn_id=row["turn_id"] or pub.get("turn_id"),
        call_id=row["call_id"] or pub.get("call_id"),
        parent_call_id=row["parent_call_id"] or pub.get("parent_call_id"),
        attempt=row["attempt"] if row["attempt"] is not None else pub.get("attempt"),
        replay_of=row["replay_of"] or pub.get("replay_of"),
        resumed_from_run=row["resumed_from_run"] or pub.get("resumed_from_run"),
        tool=pub.get("tool_name"),
        caller=pub.get("caller"),
    )


# ------------------------------------------------------------------------------ checks


@dataclass
class _Context:
    src: ReplaySources
    budget: _Budget
    since: datetime | None
    until: datetime | None
    files: list[Path]
    cold_read: bool
    notes: list[Note]


def _writer_rows(
    ctx: _Context,
    sql_hot: str,
    params: Sequence[Any],
    where_cold: str,
    cold_params: Sequence[Any],
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Rows of one writer from both tiers; the cold part only when the archive was read."""
    out: list[dict[str, Any]] = []
    conn = _connect_ro(ctx.src.audit_db, ctx.budget)
    if conn is not None:
        with closing(conn):
            if "writer_id" in _columns_of(conn, "audit_log"):
                out.extend(dict(r) for r in conn.execute(sql_hot, list(params)))
    if ctx.cold_read and ctx.files:
        from iris_harness.kernel.governance.audit.archive.query import read_cold_rows

        try:
            rows, cut = read_cold_rows(
                ctx.files,
                columns=columns,
                where=where_cold,
                params=cold_params,
                timeout_s=ctx.budget.remaining(),
            )
        except Exception:  # noqa: BLE001 - reported once by the session read
            return out
        if cut:
            ctx.budget.cut = True
        out.extend(rows)
    return out


def _sequence_gaps(ctx: _Context, rows: list[dict[str, Any]]) -> list[Gap]:
    """Holes in the sequence of every writer that wrote to the session."""
    spans: dict[str, tuple[int, int]] = {}
    for r in rows:
        if r["writer_id"] and r["writer_seq"] is not None:
            lo, hi = spans.get(r["writer_id"], (r["writer_seq"], r["writer_seq"]))
            spans[r["writer_id"]] = (min(lo, r["writer_seq"]), max(hi, r["writer_seq"]))
    if not spans:
        return []
    mine = {(r["writer_id"], r["writer_seq"]) for r in rows}
    pending = _spooled(ctx.src.audit_db)
    archived = _marker_ranges(ctx)
    gaps: list[Gap] = []
    for writer, (lo, hi) in sorted(spans.items()):
        if ctx.budget.expired():
            return gaps
        present_rows = _writer_rows(
            ctx,
            "SELECT writer_seq FROM audit_log WHERE writer_id = ? AND writer_seq BETWEEN ? AND ?",
            [writer, lo, hi],
            "writer_id = ? AND writer_seq BETWEEN ? AND ?",
            [writer, lo, hi],
            ("writer_seq",),
        )
        present = {int(p["writer_seq"]) for p in present_rows} | {
            s for (w, s) in mine if w == writer
        }
        declared = _declared_losses(ctx, writer)
        for seq in range(lo, hi + 1):
            if seq in present or (writer, seq) in pending:
                continue
            if not ctx.cold_read and any(a <= seq <= b for a, b in archived.get(writer, ())):
                continue  # moved to the archive and the archive was not read: unknowable
            if seq in declared:
                gaps.append(
                    Gap(
                        "lost_write",
                        f"writer {writer} seq {seq}",
                        f"writer {writer} seq {seq} was lost when written and the writer said so",
                        (f"{writer}:{seq}",),
                    )
                )
            else:
                where = (
                    "between rows of this session"
                    if _bounded(rows, writer, seq)
                    else ("session unknown")
                )
                gaps.append(
                    Gap(
                        "sequence_hole",
                        f"writer {writer} seq {seq}",
                        f"writer {writer} seq {seq} is in no store ({where})",
                        (f"{writer}:{seq}",),
                    )
                )
    if archived and not ctx.cold_read:
        ctx.notes.append(
            Note(
                "cold_tier_not_read",
                "archive",
                "compacted sequence ranges were not checked because the archive was not read",
            )
        )
    return gaps


def _bounded(rows: list[dict[str, Any]], writer: str, seq: int) -> bool:
    before = any(r["writer_id"] == writer and (r["writer_seq"] or 0) < seq for r in rows)
    after = any(r["writer_id"] == writer and (r["writer_seq"] or 0) > seq for r in rows)
    return before and after


def _declared_losses(ctx: _Context, writer: str) -> set[int]:
    out: set[int] = set()
    found = _writer_rows(
        ctx,
        "SELECT payload_json FROM audit_log WHERE writer_id = ? AND kind = 'gap'",
        [writer],
        "writer_id = ? AND kind = 'gap'",
        [writer],
        ("payload_json",),
    )
    for item in found:
        try:
            payload = json.loads(item["payload_json"])
            out.update(range(int(payload["first_missing"]), int(payload["last_missing"]) + 1))
        except (ValueError, KeyError, TypeError):
            continue
    return out


def _spooled(audit_db: Path) -> set[tuple[str, int]]:
    try:
        records, _ = spool.read_valid(spool.spool_path_for(audit_db))
    except OSError:
        return set()
    return {
        (str(r["writer_id"]), int(r["writer_seq"]))
        for r in records
        if r.get("writer_id") and r.get("writer_seq") is not None
    }


def _marker_ranges(ctx: _Context) -> dict[str, list[tuple[int, int]]]:
    try:
        from iris_harness.kernel.governance.audit.archive.markers import read_markers
    except ModuleNotFoundError:
        return {}
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for marker in read_markers(ctx.src.audit_db):
        for w in marker.writers:
            out[str(w["writer_id"])].append((int(w["min_seq"]), int(w["max_seq"])))
    return out


def _archive_gaps(ctx: _Context) -> list[Gap]:
    if not ctx.cold_read or ctx.src.archive_root is None:
        return []
    from iris_harness.kernel.governance.audit.archive.markers import check_window

    return [
        Gap("archive_mismatch", problem.split(":", 1)[0], problem, (problem.split(":", 1)[0],))
        for problem in check_window(ctx.src.audit_db, ctx.src.archive_root, ctx.since, ctx.until)
    ]


#: A recycled pid's process was born later than the writer: more than this many seconds apart
#: is a different process.
_START_TOLERANCE_S = 2.0


def _pid_alive(pid: int, started: float | None = None) -> bool:
    """Is the process that wrote ``writer.start`` still running?

    With the start time the row recorded, a recycled pid (another process now has the number)
    is told from the writer: the creation times differ. A row from before the start time was
    recorded falls back to the pid alone, which reads a recycled pid as alive (the call then
    stays an ``in_flight`` note instead of an ``open_call`` gap: never a false gap).
    """
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a core dependency
        psutil = None
    if psutil is not None:
        try:
            proc = psutil.Process(pid)
            if proc.status() == psutil.STATUS_ZOMBIE:
                return False
            if started is not None:
                return bool(abs(proc.create_time() - started) < _START_TOLERANCE_S)
            return True
        except psutil.NoSuchProcess:
            return False
        except psutil.AccessDenied:
            return True
        except psutil.Error:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _writer_states(ctx: _Context, writers: Iterable[str]) -> dict[str, str]:
    """``live`` (process running, no close), ``closed`` or ``gone`` (no close, process gone)."""
    states: dict[str, str] = {}
    for writer in writers:
        found = _writer_rows(
            ctx,
            "SELECT hook_point, payload_json FROM audit_log WHERE writer_id = ? AND kind = 'writer'",
            [writer],
            "writer_id = ? AND kind = 'writer'",
            [writer],
            ("hook_point", "payload_json"),
        )
        if any(f["hook_point"] == sequence.WRITER_CLOSE for f in found):
            states[writer] = "closed"
            continue
        pid = None
        started: float | None = None
        for f in found:
            if f["hook_point"] == sequence.WRITER_START:
                try:
                    body = json.loads(f["payload_json"])
                    pid = int(body.get("pid"))
                    started = float(body["started"]) if body.get("started") is not None else None
                except (ValueError, TypeError):
                    pid = None
        if pid is None:
            states[writer] = "unknown"
        else:
            states[writer] = "live" if _pid_alive(pid, started) else "gone"
    return states


def _writer_notes(ctx: _Context, rows: list[dict[str, Any]]) -> list[Gap]:
    writers = sorted({r["writer_id"] for r in rows if r["writer_id"]})
    for writer, state in _writer_states(ctx, writers).items():
        if state == "gone":
            ctx.notes.append(
                Note(
                    "writer_ended_without_close",
                    f"writer {writer}",
                    f"writer {writer} ended without close: tail unknown",
                )
            )
        elif state == "live":
            ctx.notes.append(
                Note("writer_open", f"writer {writer}", f"writer {writer} is still running")
            )
        elif state == "unknown":
            ctx.notes.append(
                Note(
                    "writer_unknown",
                    f"writer {writer}",
                    f"writer {writer} has no start row: its state is unknown",
                )
            )
    return []


def _call_gaps(
    ctx: _Context,
    rows: list[dict[str, Any]],
    log_events: list[dict[str, Any]],
    session_id: str,
) -> list[Gap]:
    gaps: list[Gap] = []
    pre: dict[str, list[dict[str, Any]]] = defaultdict(list)
    post: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cid = r["call_id"] or public_payload(r["payload_json"]).get("call_id")
        if not cid:
            continue
        if r["hook_point"] == PRE_CALL:
            pre[cid].append(r)
        elif r["hook_point"] == POST_CALL:
            post[cid].append(r)
    started = {
        e["payload"]["call_id"]
        for e in log_events
        if e.get("kind") == "tool.invoke.start"
        and isinstance(e.get("payload"), dict)
        and e["payload"].get("call_id")
    }
    ended = {
        e["payload"]["call_id"]
        for e in log_events
        if e.get("kind") == "tool.invoke.end"
        and isinstance(e.get("payload"), dict)
        and e["payload"].get("call_id")
    }
    ledger = _ledger_rows(ctx, {r["run_id"] for r in rows if r["run_id"]})
    pending_ledger = {x["call_id"] for x in ledger if x["call_id"] and x["status"] == "pending"}
    states = _writer_states(
        ctx, {r["writer_id"] for rs in pre.values() for r in rs if r["writer_id"]}
    )

    for cid, pres in pre.items():
        if cid in post or any(p["decision"] in _STOPS for p in pres):
            continue
        writer = next((p["writer_id"] for p in pres if p["writer_id"]), None)
        state = states.get(writer or "", "unknown")
        witnessed = (cid in started and cid not in ended) or cid in pending_ledger
        tool = public_payload(pres[0]["payload_json"]).get("tool_name", "?")
        if state == "live":
            ctx.notes.append(Note("in_flight", f"call {cid}", f"call {cid} ({tool}) is running"))
        elif witnessed and state in ("gone", "closed"):
            gaps.append(
                Gap(
                    "open_call",
                    f"call {cid}",
                    f"call {cid} ({tool}) started and never settled",
                    (cid,),
                )
            )
        else:
            ctx.notes.append(
                Note(
                    "unsettled_call",
                    f"call {cid}",
                    f"call {cid} ({tool}) has no settle row and no witness",
                )
            )
    for cid in post:
        if cid not in pre:
            gaps.append(
                Gap("orphan_settle", f"call {cid}", f"call {cid} settled with no start row", (cid,))
            )
    known = set(pre) | set(post)
    for cid in sorted(started - known):
        gaps.append(
            Gap(
                "witness_mismatch",
                f"call {cid}",
                f"the session log shows call {cid} start with no audit row",
                (cid,),
            )
        )
    for item in ledger:
        cid = item["call_id"]
        if cid and cid not in known and not _call_exists(ctx, cid):
            gaps.append(
                Gap(
                    "witness_mismatch",
                    f"call {cid}",
                    f"the side-effect ledger has call {cid} with no audit row",
                    (cid,),
                )
            )
    for appr in _approval_rows(ctx, session_id):
        cid = appr["call_id"]
        if cid and cid not in known and not _call_exists(ctx, cid):
            gaps.append(
                Gap(
                    "witness_mismatch",
                    f"approval {appr['approval_id']}",
                    f"approval {appr['approval_id']} names call {cid} with no audit row",
                    (str(appr["approval_id"]), cid),
                )
            )
    return gaps


def _lineage_gaps(ctx: _Context, rows: list[dict[str, Any]], session_id: str) -> list[Gap]:
    gaps: list[Gap] = []
    calls = {r["call_id"] for r in rows if r["call_id"]}
    runs = {r["run_id"] for r in rows if r["run_id"]}
    seen: set[tuple[str, str]] = set()
    for r in rows:
        checks: tuple[tuple[str, Any, Callable[[str], bool]], ...] = (
            ("parent_call_id", r["parent_call_id"], lambda v: v in calls or _call_exists(ctx, v)),
            ("replay_of", r["replay_of"], lambda v: v in calls or _call_exists(ctx, v)),
            ("resumed_from_run", r["resumed_from_run"], lambda v: v in runs or _run_exists(ctx, v)),
        )
        for field_, value, exists in checks:
            if value and (field_, value) not in seen and not exists(value):
                seen.add((field_, value))
                gaps.append(
                    Gap(
                        "missing_parent",
                        f"{field_} {value}",
                        f"{field_} {value} names a record that does not exist",
                        (r["record_id"] or str(r["id"]),),
                    )
                )
    return gaps


def _call_exists(ctx: _Context, call_id: str) -> bool:
    found = _writer_rows(
        ctx,
        "SELECT 1 AS x FROM audit_log WHERE call_id = ? LIMIT 1",
        [call_id],
        "call_id = ?",
        [call_id],
        ("call_id",),
    )
    return bool(found)


def _run_exists(ctx: _Context, run_id: str) -> bool:
    found = _writer_rows(
        ctx,
        "SELECT 1 AS x FROM audit_log WHERE run_id = ? LIMIT 1",
        [run_id],
        "run_id = ?",
        [run_id],
        ("run_id",),
    )
    return bool(found)


def _ledger_rows(ctx: _Context, runs: set[str]) -> list[dict[str, Any]]:
    if ctx.src.ledger_db is None or not runs:
        return []
    conn = _connect_ro(ctx.src.ledger_db, ctx.budget)
    if conn is None:
        ctx.notes.append(Note("store_absent", "side_effect_ledger", "no ledger file"))
        return []
    with closing(conn):
        if "call_id" not in _columns_of(conn, "side_effect_ledger"):
            return []
        marks = ",".join("?" for _ in runs)
        sql = f"SELECT call_id, status, run_id FROM side_effect_ledger WHERE run_id IN ({marks})"  # noqa: S608
        return [dict(r) for r in conn.execute(sql, sorted(runs))]


def _approval_rows(ctx: _Context, session_id: str) -> list[dict[str, Any]]:
    if ctx.src.approvals_db is None:
        return []
    conn = _connect_ro(ctx.src.approvals_db, ctx.budget)
    if conn is None:
        ctx.notes.append(Note("store_absent", "approvals", "no approval queue file"))
        return []
    with closing(conn):
        if "call_id" not in _columns_of(conn, "approval_queue"):
            return []
        return [
            dict(r)
            for r in conn.execute(
                "SELECT approval_id, call_id, status FROM approval_queue WHERE session_id = ?",
                [session_id],
            )
        ]


# ------------------------------------------------------------------------------ the tree


def _tree(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn -> run -> step -> call, with nested calls under the call they ran inside."""
    calls: dict[str, dict[str, Any]] = {}
    for r in rows:
        cid = r["call_id"]
        if not cid:
            continue
        node = calls.setdefault(
            cid,
            {
                "call_id": cid,
                "tool": None,
                "attempt": None,
                "replay_of": None,
                "parent_call_id": None,
                "run_id": r["run_id"],
                "step_id": r["step_id"],
                "turn_id": r["turn_id"],
                "resumed_from_run": None,
                "children": [],
            },
        )
        pub = public_payload(r["payload_json"])
        node["tool"] = node["tool"] or pub.get("tool_name")
        for key in ("attempt", "replay_of", "parent_call_id", "resumed_from_run"):
            if node[key] is None and r[key] is not None:
                node[key] = r[key]
    roots: list[dict[str, Any]] = []
    for node in calls.values():
        parent = calls.get(node["parent_call_id"] or "")
        (parent["children"] if parent else roots).append(node)
    turns: dict[str | None, dict[str | None, dict[Any, list[dict[str, Any]]]]] = {}
    for node in roots:
        turns.setdefault(node["turn_id"], {}).setdefault(node["run_id"], {}).setdefault(
            node["step_id"], []
        ).append(node)
    return [
        {
            "turn_id": turn,
            "runs": [
                {
                    "run_id": run,
                    "steps": [{"step_id": step, "calls": cs} for step, cs in by_step.items()],
                }
                for run, by_step in by_run.items()
            ],
        }
        for turn, by_run in turns.items()
    ]


__all__ = [
    "API_MAX_ROWS",
    "API_MAX_SECONDS",
    "GAP_CLASSES",
    "Gap",
    "Note",
    "Replay",
    "ReplayRecord",
    "ReplaySources",
    "replay_session",
]
