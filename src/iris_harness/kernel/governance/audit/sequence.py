"""Which writer wrote a ledger row, and in what order (issue #134, stage 4).

A ledger that cannot show what is missing cannot prove it is complete. Every process that
writes to an audit database is a *writer*: it mints a ``writer_id`` (a ULID) and numbers its
rows 1, 2, 3 ... (``writer_seq``). The number is taken BEFORE the write is attempted, so a
write that fails leaves a hole in ``(writer_id, writer_seq)``; the hole is the evidence, and
the next row that does land records it in a ``gap`` row (see ``AuditLog``).

The registry is process-wide and keyed by the database file, because rows reach one database
from many ``AuditLog`` instances (the kernel's, the approvals queue's, a plugin's through the
SDK). One lock guards it: ``AuditLog.record`` is called from worker threads. A ``fork`` gives
the child a NEW writer id and a counter from 1 -- two processes must never number rows under
one id -- and the registry lock is held across the fork so the child never inherits it locked.

Nothing here touches SQLite or the disk; it is the counters only.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from iris_harness.foundation.ids import new_ulid
from iris_harness.foundation.process_state import track_globals

logger = logging.getLogger(__name__)

#: ``kind`` values of the rows the store writes about itself. A row of a hook firing has NULL.
KIND_GAP = "gap"
KIND_WRITER = "writer"
KIND_COMPACTION = "compaction"  # written by the archive (stage 4b)
#: ``hook_point`` of the writer's own rows.
WRITER_START = "writer.start"
WRITER_CLOSE = "writer.close"
#: ``writer.start`` takes sequence 0, so the first row of a hook firing is 1.
START_SEQ = 0
#: The most gap rows one successful write carries; the rest wait for the next write.
MAX_GAPS_PER_WRITE = 16


@dataclass
class Missing:
    """One sequence number whose write failed, and what became of its record."""

    seq: int
    cause: str  # an exception class name, never a message
    spooled: bool


@dataclass
class Writer:
    """The counters of this process's writer for one database."""

    writer_id: str
    next_seq: int = 1
    started: bool = False
    missing: list[Missing] = field(default_factory=list)


@dataclass(frozen=True)
class GapRange:
    """A run of consecutive missing sequence numbers, for one ``gap`` row."""

    first: int
    last: int
    causes: dict[str, int]
    spooled: int
    lost: int
    items: tuple[Missing, ...] = ()

    def as_payload(self) -> dict[str, object]:
        return {
            "first_missing": self.first,
            "last_missing": self.last,
            "count": self.last - self.first + 1,
            "causes": dict(sorted(self.causes.items())),
            "spooled": self.spooled,
            "lost": self.lost,
        }


@dataclass(frozen=True)
class WriteStats:
    """What this process knows about its own audit writes (the health row reads it)."""

    lost: int
    spooled: int
    last_cause: str | None


_lock = threading.RLock()
_writers: dict[str, Writer] = {}
_lost = 0
_spooled = 0
_last_cause: str | None = None
_close_hook: Callable[[str, Writer], None] | None = None
_installed = False


def _key(db_path: object) -> str:
    return os.path.abspath(os.fspath(db_path))  # type: ignore[call-overload]


def _install() -> None:
    """Register the fork and exit handlers, once per process image."""
    global _installed
    if _installed:
        return
    _installed = True
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(
            before=_lock.acquire,
            after_in_parent=_lock.release,
            after_in_child=_after_fork_in_child,
        )
    atexit.register(_close_all)


def _after_fork_in_child() -> None:
    global _lost, _spooled, _last_cause
    # New process, new writers: ids are re-minted lazily on the next write.
    _writers.clear()
    _lost = 0
    _spooled = 0
    _last_cause = None
    _lock.release()


def writer_for(db_path: object) -> Writer:
    """This process's writer for ``db_path`` (created on first use)."""
    key = _key(db_path)
    with _lock:
        _install()
        writer = _writers.get(key)
        if writer is None:
            writer = _writers[key] = Writer(writer_id=new_ulid())
        return writer


def allocate(db_path: object) -> tuple[Writer, int]:
    """Take the next sequence number. Called BEFORE the write is attempted."""
    with _lock:
        writer = writer_for(db_path)
        seq = writer.next_seq
        writer.next_seq += 1
        return writer, seq


def mark_started(db_path: object, writer: Writer) -> None:
    with _lock:
        if _writers.get(_key(db_path)) is writer:
            writer.started = True


def note_failure(
    db_path: object,
    writer: Writer,
    seq: int,
    exc: BaseException,
    *,
    spooled: bool,
    own: bool = False,
) -> None:
    """Remember that ``seq`` did not reach the database (and whether the spool took it).

    ``own``: the number belonged to a row the store writes about itself (a gap row), which is
    regenerated, not an audit event lost; the hole is recorded but not counted as a loss.
    """
    global _lost, _spooled, _last_cause
    cause = exc.__class__.__name__
    with _lock:
        if _writers.get(_key(db_path)) is writer:
            writer.missing.append(Missing(seq=seq, cause=cause, spooled=spooled))
        _last_cause = cause
        if own:
            return
        if spooled:
            _spooled += 1
        else:
            _lost += 1


def claim_gaps(db_path: object, writer: Writer) -> list[GapRange]:
    """Take the missing sequence numbers of ``writer`` to record, as runs of consecutive numbers.

    The numbers leave the writer: a concurrent write cannot record the same gap twice. A
    write that then fails hands them back with :func:`restore_gaps`.
    """
    with _lock:
        if _writers.get(_key(db_path)) is not writer or not writer.missing:
            return []
        ordered = sorted(writer.missing, key=lambda m: m.seq)
        runs: list[list[Missing]] = []
        for item in ordered:
            if runs and item.seq == runs[-1][-1].seq + 1:
                runs[-1].append(item)
            else:
                runs.append([item])
        taken = runs[:MAX_GAPS_PER_WRITE]
        claimed = {m.seq for run in taken for m in run}
        writer.missing = [m for m in writer.missing if m.seq not in claimed]
    out: list[GapRange] = []
    for run in taken:
        causes: dict[str, int] = {}
        for item in run:
            causes[item.cause] = causes.get(item.cause, 0) + 1
        spooled = sum(1 for item in run if item.spooled)
        out.append(
            GapRange(
                first=run[0].seq,
                last=run[-1].seq,
                causes=causes,
                spooled=spooled,
                lost=len(run) - spooled,
                items=tuple(run),
            )
        )
    return out


def restore_gaps(db_path: object, writer: Writer, gaps: list[GapRange]) -> None:
    """Hand back gaps whose ``gap`` row did not commit, so the next write records them."""
    with _lock:
        if _writers.get(_key(db_path)) is writer:
            for gap in gaps:
                writer.missing.extend(gap.items)


def stats() -> WriteStats:
    """This process's count of audit writes that were spooled and that were lost."""
    with _lock:
        return WriteStats(lost=_lost, spooled=_spooled, last_cause=_last_cause)


def set_close_hook(hook: Callable[[str, Writer], None] | None) -> None:
    """Install what ``atexit`` calls per writer (``AuditLog`` sets it; tests may replace it)."""
    global _close_hook
    with _lock:
        _close_hook = hook


def _close_all() -> None:
    """Best effort at exit: let each started writer say it closed. Never raises."""
    with _lock:
        hook = _close_hook
        items = [(k, w) for k, w in _writers.items() if w.started]
    if hook is None:
        return
    for key, writer in items:
        try:
            hook(key, writer)
        except Exception:  # noqa: BLE001 - exit must not fail on a missing ledger
            logger.debug("audit writer.close not written for %s", key, exc_info=True)


def reset_for_tests() -> None:
    """Forget every writer and counter (a test that wants a fresh process)."""
    global _lost, _spooled, _last_cause
    with _lock:
        _writers.clear()
        _lost = 0
        _spooled = 0
        _last_cause = None


track_globals(__name__, "_lost", "_spooled", "_last_cause", "_writers")

__all__ = [
    "KIND_COMPACTION",
    "KIND_GAP",
    "KIND_WRITER",
    "MAX_GAPS_PER_WRITE",
    "START_SEQ",
    "WRITER_CLOSE",
    "WRITER_START",
    "GapRange",
    "Missing",
    "WriteStats",
    "Writer",
    "allocate",
    "claim_gaps",
    "mark_started",
    "note_failure",
    "restore_gaps",
    "reset_for_tests",
    "set_close_hook",
    "stats",
    "writer_for",
]
