"""SQLite hot-tier compaction into Parquet cold-tier archive.

One run: write and fsync the chunks, then ONE SQLite transaction inserts the ``compaction``
marker and deletes the archived rows (``AuditLog.write_compaction``). A crash between the two
leaves chunks no marker names; the next run adopts them (their rows are all still hot) or moves
them to ``<archive>/.orphans/<compaction id>/`` (reversible, never unlinked). Only files named
for a compaction run are ever adopted or moved: a chunk of the older shape, or any other file,
is left exactly where it is.
"""

from __future__ import annotations

import fcntl
import os
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq

from iris_harness.foundation.ids import new_ulid
from iris_harness.kernel.governance.audit.archive.markers import (
    keys_of,
    marker_payload,
    read_markers,
    scan_archive,
)
from iris_harness.kernel.governance.audit.archive.writer import (
    CHUNK_NAME_RE,
    CHUNK_TMP_RE,
    AuditArchive,
    ChunkInfo,
    describe_chunk,
)
from iris_harness.kernel.governance.audit.log import (
    AuditLog,
    AuditRow,
    CompactionConflict,
    _row_to_audit,
)
from iris_harness.kernel.governance.audit.sequence import KIND_COMPACTION

#: Where a chunk that cannot be adopted goes, under the archive root. No reader's glob sees it.
QUARANTINE_DIR = ".orphans"
LOCK_NAME = ".compact.lock"


@dataclass(frozen=True)
class CompactResult:
    selected_rows: int
    archived_rows: int
    deleted_rows: int
    partition_count: int
    cutoff_ts: str
    adopted_chunks: int = 0
    quarantined_chunks: int = 0


class AuditCompactor:
    """Move old audit rows from SQLite to Parquet, then delete from SQLite."""

    def __init__(
        self,
        *,
        audit_log: AuditLog,
        archive: AuditArchive,
        retention_days: int = 30,
        _after_chunks: Callable[[], None] | None = None,
    ) -> None:
        if retention_days <= 0:
            raise ValueError("retention_days must be > 0")
        self._audit_log = audit_log
        self._archive = archive
        self._retention_days = retention_days
        self._after_chunks = _after_chunks  # test seam: the moment a crash would leave orphans

    def compact(self, *, now: datetime | None = None) -> CompactResult:
        current = (now or datetime.now(UTC)).astimezone(UTC)
        cutoff_iso = (current - timedelta(days=self._retention_days)).isoformat()

        with self._locked():
            adopted, quarantined = self._recover_orphans()
            rows = self._select_rows_older_than(cutoff_iso)
            if not rows:
                return CompactResult(0, 0, 0, 0, cutoff_iso, adopted, quarantined)

            compaction_id = new_ulid()
            chunks = self._archive.write_chunks(rows, compaction_id=compaction_id)
            if self._after_chunks is not None:
                self._after_chunks()
            payload = marker_payload(
                compaction_id=compaction_id,
                cutoff_ts=cutoff_iso,
                rows=rows,
                chunks=chunks,
                adopted=False,
            )
            try:
                deleted = self._audit_log.write_compaction(payload, archived=keys_of(rows))
            except CompactionConflict:
                for chunk in chunks:
                    self._quarantine(self._archive.root / chunk.file, compaction_id)
                raise
            return CompactResult(
                selected_rows=len(rows),
                archived_rows=sum(c.rows for c in chunks),
                deleted_rows=deleted,
                partition_count=len(chunks),
                cutoff_ts=cutoff_iso,
                adopted_chunks=adopted,
                quarantined_chunks=quarantined,
            )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """One compaction at a time per archive, so a chunk mid-run is never taken for an orphan."""
        fd = os.open(self._archive.root / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)  # closing releases the lock

    def _recover_orphans(self) -> tuple[int, int]:
        """Adopt or quarantine chunks of an interrupted run; ``(adopted, quarantined)`` chunks."""
        root = self._archive.root
        files = scan_archive(root)
        named = {
            str(c.get("file")) for m in read_markers(self._audit_log.db_path) for c in m.chunks
        }
        adopted = quarantined = 0
        for path in files.temp_chunks:  # never published: a half-written file
            match = CHUNK_TMP_RE.match(path.name)
            self._quarantine(path, match.group("cid") if match else "unknown")
            quarantined += 1

        groups: dict[str, list[Path]] = defaultdict(list)
        for path in files.run_chunks:
            match = CHUNK_NAME_RE.match(path.name)
            if match and path.relative_to(root).as_posix() not in named:
                groups[match.group("cid")].append(path)

        for compaction_id, paths in sorted(groups.items()):
            good: list[tuple[Path, list[AuditRow]]] = []
            for path in paths:
                try:
                    good.append((path, _rows_of_chunk(path)))
                except Exception:  # noqa: BLE001 - an unreadable orphan is quarantined, not lost
                    self._quarantine(path, compaction_id)
                    quarantined += 1
            if not good:
                continue
            rows = [row for _, chunk_rows in good for row in chunk_rows]
            infos: list[ChunkInfo] = [describe_chunk(root, p, chunk_rows) for p, chunk_rows in good]
            payload = marker_payload(
                compaction_id=compaction_id,
                cutoff_ts=None,
                rows=rows,
                chunks=infos,
                adopted=True,
            )
            try:
                self._audit_log.write_compaction(payload, archived=keys_of(rows))
                adopted += len(good)
            except CompactionConflict:  # some row is not hot any more: do not trust the chunk
                for path, _ in good:
                    self._quarantine(path, compaction_id)
                quarantined += len(good)
        return adopted, quarantined

    def _quarantine(self, path: Path, compaction_id: str) -> None:
        """Move ``path`` under ``.orphans/<compaction id>/`` keeping its partition directory."""
        root = self._archive.root
        dest = root / QUARANTINE_DIR / compaction_id / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        for directory in (root / QUARANTINE_DIR, root / QUARANTINE_DIR / compaction_id):
            os.chmod(directory, 0o700)
        n = 0
        while dest.exists():
            n += 1
            dest = dest.with_name(f"{path.name}.{n}")
        os.replace(path, dest)

    def _select_rows_older_than(self, cutoff_iso: str) -> tuple[AuditRow, ...]:
        with closing(sqlite3.connect(self._audit_log.db_path)) as conn, conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT *
                FROM audit_log
                WHERE ts < ? AND kind IS NOT ?
                ORDER BY ts ASC, id ASC
                """,
                (cutoff_iso, KIND_COMPACTION),
            ).fetchall()
        return tuple(_row_to_audit(row) for row in rows)


def _rows_of_chunk(path: Path) -> list[AuditRow]:
    """The rows a run chunk holds, as ``AuditRow``; raises when the file cannot be read."""
    table = pq.read_table(path)
    out: list[AuditRow] = []
    for rec in table.to_pylist():
        ts = rec["ts"]
        out.append(
            AuditRow(
                id=int(rec["id"]),
                ts=ts.isoformat() if isinstance(ts, datetime) else str(ts),
                run_id=rec["run_id"],
                step_id=rec["step_id"],
                agent_type=rec["agent_type"],
                hook_point=rec["hook_point"],
                plugin=rec["plugin"],
                decision=rec["decision"],
                classification=rec["classification"],
                tier=rec["tier"],
                cost_usd=rec["cost_usd"],
                severity=rec["severity"],
                reason=rec["reason"],
                payload_json=rec["payload_json"],
                record_id=rec["record_id"],
                session_id=rec["session_id"],
                turn_id=rec["turn_id"],
                call_id=rec["call_id"],
                parent_call_id=rec["parent_call_id"],
                attempt=rec["attempt"],
                replay_of=rec["replay_of"],
                resumed_from_run=rec["resumed_from_run"],
                writer_id=rec["writer_id"],
                writer_seq=rec["writer_seq"],
                kind=rec["kind"],
            )
        )
    return out
