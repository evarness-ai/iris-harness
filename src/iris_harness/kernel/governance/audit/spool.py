"""The local spool for an audit row the database would not take (issue #134, stage 4).

When ``audit.db`` cannot be written (locked, disk full, read-only) a row that does not guard
an effect is not dropped: it is appended here, one JSON line, and put into the database
later. A row that does guard an effect is refused only when the spool cannot take it either
(``AuditLog.record`` returns 0 for a spooled row; the kernel reads it).

Permission model: the spool is a sibling of the database it belongs to, in the same
directory, created ``0600`` like the database file (the directory is the governance data
dir). Whoever can write the spool can already write ``audit.db``; the spool adds no new
principal. What it must not add is a way to *forge or alter* history, so the drain is strict:

* a line is accepted only if it has exactly the closed field set below with the right types;
  anything else is a malformed line: it is counted, moved to ``<spool>.rejected`` and skipped,
  never raised;
* a line is replayed with ``INSERT OR IGNORE``: a ``record_id`` (or a ``(writer_id,
  writer_seq)``) that the ledger already holds changes nothing, so replaying a line twice, or
  a line naming an existing row, is a no-op. A spool line can add a row that is missing; it
  cannot rewrite one that is there.

The content of a line is the same argument-free whitelist the database row holds (the kernel
stamps ids, counts and decision text, never tool arguments), so the spool is no more
sensitive than the database it sits beside.

Appends and drains both hold an exclusive ``flock`` on ``<spool>.lock`` so a drain never
reads a half-written line and never drops one appended while it ran.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # POSIX only; on a platform without it the lock degrades to the single-process case.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_ULID = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")

#: The closed field set of a spool line, with the types each must have.
_STR = (str,)
_OPT_STR = (str, type(None))
_OPT_INT = (int, type(None))
_FIELDS: dict[str, tuple[type, ...]] = {
    "record_id": _STR,
    "writer_id": _STR,
    "writer_seq": (int,),
    "kind": _OPT_STR,
    "ts": _STR,
    "run_id": _STR,
    "step_id": _OPT_INT,
    "agent_type": _STR,
    "hook_point": _STR,
    "plugin": _STR,
    "decision": _STR,
    "classification": _OPT_STR,
    "tier": _OPT_STR,
    "cost_usd": (int, float, type(None)),
    "severity": _STR,
    "reason": _STR,
    "payload": (dict,),
}


def spool_path_for(db_path: Path) -> Path:
    """``audit.db`` -> ``audit-spool.jsonl`` beside it (``<stem>-spool.jsonl``)."""
    return db_path.with_name(f"{db_path.stem}-spool.jsonl")


@dataclass(frozen=True)
class SpoolState:
    """What is waiting in the spool, for health and the status screens."""

    pending: int
    rejected: int


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    lock_path = path.with_name(path.name + ".lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def append(path: Path, record: Mapping[str, Any]) -> None:
    """Append one line and make it durable (``fsync``). Raises when the line cannot be kept."""
    line = json.dumps(record, default=str, sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path):
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)


def state(path: Path) -> SpoolState:
    """How many lines wait, and how many were rejected as malformed (cheap; never raises)."""
    return SpoolState(pending=_count_lines(path), rejected=_count_lines(_rejected(path)))


def _count_lines(path: Path) -> int:
    try:
        with open(path, "rb") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def _rejected(path: Path) -> Path:
    return path.with_name(path.name + ".rejected")


def validate(line: object) -> dict[str, Any] | None:
    """The parsed line when it is exactly a spool record, else ``None``."""
    if not isinstance(line, dict) or set(line) != set(_FIELDS):
        return None
    for name, types in _FIELDS.items():
        value = line[name]
        if not isinstance(value, types) or (isinstance(value, bool) and bool not in types):
            return None
    if not _ULID.fullmatch(line["record_id"]) or not _ULID.fullmatch(line["writer_id"]):
        return None
    if line["writer_seq"] < 0:
        return None
    return line


def read_valid(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Every acceptable line, and the raw text of every line that is not. Never raises."""
    good: list[dict[str, Any]] = []
    bad: list[str] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return good, bad
    for text in raw.splitlines():
        if not text.strip():
            continue
        try:
            parsed = validate(json.loads(text))
        except ValueError:
            parsed = None
        if parsed is None:
            bad.append(text)
        else:
            good.append(parsed)
    return good, bad


def drain(path: Path, apply: Any) -> int:
    """Replay the spool through ``apply(records) -> bool`` and drop what it took.

    ``apply`` puts the valid records into the database (idempotently) and returns whether the
    transaction committed. On ``False`` the spool is left exactly as it was. Malformed lines
    are moved to ``<spool>.rejected`` either way. Returns the number of records replayed.
    """
    if not path.exists():
        return 0
    with _locked(path):
        good, bad = read_valid(path)
        if bad:
            with open(_rejected(path), "a", encoding="utf-8") as fh:
                fh.write("\n".join(bad) + "\n")
            with contextlib.suppress(OSError):
                os.chmod(_rejected(path), 0o600)
            logger.warning("audit spool: %d malformed line(s) set aside, not replayed", len(bad))
        if good and not apply(good):
            if bad:
                _rewrite(path, good)
            return 0
        _rewrite(path, [])
        return len(good)


def _rewrite(path: Path, records: list[dict[str, Any]]) -> None:
    """Replace the spool with exactly ``records`` (temp file + rename, ``0600``)."""
    if not records:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        return
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        for record in records:
            os.write(
                fd,
                (
                    json.dumps(record, default=str, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode("utf-8"),
            )
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


__all__ = ["SpoolState", "append", "drain", "read_valid", "spool_path_for", "state", "validate"]
