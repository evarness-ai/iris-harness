"""Parquet+zstd writer for the cold audit archive."""

from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from iris_harness.foundation.ids import new_ulid
from iris_harness.foundation.paths import governance_data_dir
from iris_harness.kernel.governance.audit.log import AuditRow

# Override slot (``None``: resolved from ``IRIS_HOME`` on every use, never frozen at
# import -- a process that relocates the home after importing IRIS writes into the
# new one). Set it to point the default somewhere else outright (tests).
DEFAULT_ARCHIVE_ROOT: Path | None = None


def default_archive_root() -> Path:
    """``DEFAULT_ARCHIVE_ROOT`` when set, else ``<governance data dir>/audit-archive``."""
    return (
        DEFAULT_ARCHIVE_ROOT
        if DEFAULT_ARCHIVE_ROOT is not None
        else governance_data_dir() / "audit-archive"
    )


@dataclass(frozen=True, order=True)
class ArchivePartition:
    """Partition key ``year=YYYY/month=MM`` for one audit row timestamp."""

    year: int
    month: int

    @property
    def relative_path(self) -> Path:
        return Path(f"year={self.year:04d}") / f"month={self.month:02d}"


#: A chunk written by a compaction run is named for the run: ``audit-<compaction ULID>-<n>.parquet``.
#: Only a file with this name can be adopted or quarantined by a later run; a chunk of the older
#: shape (``audit-<uuid4 hex>.parquet``) or any other file in the archive is never touched.
CHUNK_NAME_RE = re.compile(r"^audit-(?P<cid>[0-9A-HJKMNP-TV-Z]{26})-(?P<n>\d+)\.parquet$")
#: The not-yet-published name of a chunk being written; no reader's glob matches it.
CHUNK_TMP_RE = re.compile(r"^\.audit-(?P<cid>[0-9A-HJKMNP-TV-Z]{26})-(?P<n>\d+)\.parquet\.tmp$")


@dataclass(frozen=True)
class WriterRange:
    """The sequence numbers one writer holds in a chunk: lowest, highest and how many rows."""

    writer_id: str
    min_seq: int
    max_seq: int
    count: int


@dataclass(frozen=True)
class ChunkInfo:
    """What a published chunk holds, for the compaction marker that accounts for it."""

    file: str  # relative to the archive root, ``year=YYYY/month=MM/<name>``
    sha256: str
    rows: int
    id_min: int
    id_max: int
    ts_min: str
    ts_max: str
    writers: tuple[WriterRange, ...] = field(default_factory=tuple)


class AuditArchive:
    """Write ``AuditRow`` batches to partitioned Parquet chunk files."""

    def __init__(self, root: Path | None = None, *, compression: str = "zstd") -> None:
        self.root = (root or default_archive_root()).expanduser().resolve()
        self._compression = compression
        self._ensure_root()

    def partition_for(self, ts: datetime) -> ArchivePartition:
        utc_ts = ts.astimezone(UTC)
        return ArchivePartition(year=utc_ts.year, month=utc_ts.month)

    def write(self, rows: Iterable[AuditRow]) -> dict[ArchivePartition, int]:
        """Write ``rows`` as chunks of one new compaction run; the rows per partition."""
        chunks = self.write_chunks(rows, compaction_id=new_ulid())
        counts: dict[ArchivePartition, int] = {}
        for chunk in chunks:
            year, month = Path(chunk.file).parts[0], Path(chunk.file).parts[1]
            counts[ArchivePartition(int(year[5:]), int(month[6:]))] = chunk.rows
        return counts

    def write_chunks(
        self, rows: Iterable[AuditRow], *, compaction_id: str
    ) -> tuple[ChunkInfo, ...]:
        """Write ``rows`` as one chunk per month, named for ``compaction_id``; describe each.

        A chunk is written under a temporary name, fsynced, and only then renamed into place,
        so a name a reader (or a recovery pass) sees is always a complete file.
        """
        grouped: dict[ArchivePartition, list[AuditRow]] = defaultdict(list)
        for row in rows:
            grouped[self.partition_for(_parse_ts(row.ts))].append(row)

        infos: list[ChunkInfo] = []
        for n, (partition, part_rows) in enumerate(sorted(grouped.items())):
            partition_dir = self.root / partition.relative_path
            self._ensure_partition_dir(partition_dir)
            name = f"audit-{compaction_id}-{n}.parquet"
            tmp_path = partition_dir / f".{name}.tmp"
            final_path = partition_dir / name
            pq.write_table(
                _table_from_rows(part_rows),
                tmp_path,
                compression=self._compression,
                use_dictionary=[
                    "agent_type",
                    "hook_point",
                    "plugin",
                    "decision",
                    "classification",
                    "tier",
                    "severity",
                    "kind",
                ],
            )
            _fsync_file(tmp_path)
            os.replace(tmp_path, final_path)
            _fsync_dir(partition_dir)
            infos.append(describe_chunk(self.root, final_path, part_rows))

        return tuple(infos)

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    @staticmethod
    def _ensure_partition_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)


def audit_archive_schema() -> pa.Schema:
    dict_str = pa.dictionary(pa.int32(), pa.string())
    return pa.schema(
        [
            ("ts", pa.timestamp("ms", tz="UTC")),
            ("run_id", pa.string()),
            ("step_id", pa.int32()),
            ("agent_type", dict_str),
            ("hook_point", dict_str),
            ("plugin", dict_str),
            ("decision", dict_str),
            ("classification", dict_str),
            ("tier", dict_str),
            ("cost_usd", pa.float64()),
            ("severity", dict_str),
            ("reason", pa.string()),
            ("payload_json", pa.string()),
            # Identity and completeness columns (issue #134, stages 3 and 4). A chunk written
            # before they existed lacks them; the query view reads such a chunk with NULLs.
            ("id", pa.int64()),
            ("record_id", pa.string()),
            ("session_id", pa.string()),
            ("turn_id", pa.string()),
            ("call_id", pa.string()),
            ("parent_call_id", pa.string()),
            ("attempt", pa.int32()),
            ("replay_of", pa.string()),
            ("resumed_from_run", pa.string()),
            ("writer_id", pa.string()),
            ("writer_seq", pa.int64()),
            ("kind", dict_str),
        ]
    )


def _table_from_rows(rows: list[AuditRow]) -> pa.Table:
    data: dict[str, pa.Array] = {
        "ts": pa.array([_parse_ts(row.ts) for row in rows], type=pa.timestamp("ms", tz="UTC")),
        "run_id": pa.array([row.run_id for row in rows], type=pa.string()),
        "step_id": pa.array([row.step_id for row in rows], type=pa.int32()),
        "agent_type": pa.array(
            [row.agent_type for row in rows],
            type=pa.dictionary(pa.int32(), pa.string()),
        ),
        "hook_point": pa.array(
            [row.hook_point for row in rows],
            type=pa.dictionary(pa.int32(), pa.string()),
        ),
        "plugin": pa.array(
            [row.plugin for row in rows],
            type=pa.dictionary(pa.int32(), pa.string()),
        ),
        "decision": pa.array(
            [row.decision for row in rows],
            type=pa.dictionary(pa.int32(), pa.string()),
        ),
        "classification": pa.array(
            [row.classification for row in rows],
            type=pa.dictionary(pa.int32(), pa.string()),
        ),
        "tier": pa.array(
            [row.tier for row in rows],
            type=pa.dictionary(pa.int32(), pa.string()),
        ),
        "cost_usd": pa.array([row.cost_usd for row in rows], type=pa.float64()),
        "severity": pa.array(
            [row.severity for row in rows],
            type=pa.dictionary(pa.int32(), pa.string()),
        ),
        "reason": pa.array([row.reason for row in rows], type=pa.string()),
        "payload_json": pa.array([row.payload_json for row in rows], type=pa.string()),
        "id": pa.array([row.id for row in rows], type=pa.int64()),
        "record_id": pa.array([row.record_id for row in rows], type=pa.string()),
        "session_id": pa.array([row.session_id for row in rows], type=pa.string()),
        "turn_id": pa.array([row.turn_id for row in rows], type=pa.string()),
        "call_id": pa.array([row.call_id for row in rows], type=pa.string()),
        "parent_call_id": pa.array([row.parent_call_id for row in rows], type=pa.string()),
        "attempt": pa.array([row.attempt for row in rows], type=pa.int32()),
        "replay_of": pa.array([row.replay_of for row in rows], type=pa.string()),
        "resumed_from_run": pa.array([row.resumed_from_run for row in rows], type=pa.string()),
        "writer_id": pa.array([row.writer_id for row in rows], type=pa.string()),
        "writer_seq": pa.array([row.writer_seq for row in rows], type=pa.int64()),
        "kind": pa.array([row.kind for row in rows], type=pa.dictionary(pa.int32(), pa.string())),
    }
    return pa.Table.from_arrays(
        [data[name] for name in audit_archive_schema().names],
        schema=audit_archive_schema(),
    )


def writer_ranges(rows: Iterable[AuditRow]) -> tuple[WriterRange, ...]:
    """Per writer, the lowest and highest sequence number and the row count (sorted by id)."""
    seen: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row.writer_id is not None and row.writer_seq is not None:
            seen[row.writer_id].append(row.writer_seq)
    return tuple(
        WriterRange(writer_id=w, min_seq=min(seqs), max_seq=max(seqs), count=len(seqs))
        for w, seqs in sorted(seen.items())
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_chunk(root: Path, path: Path, rows: list[AuditRow]) -> ChunkInfo:
    stamps = sorted(_parse_ts(row.ts).isoformat() for row in rows)
    ids = [row.id for row in rows]
    return ChunkInfo(
        file=path.relative_to(root).as_posix(),
        sha256=sha256_of(path),
        rows=len(rows),
        id_min=min(ids),
        id_max=max(ids),
        ts_min=stamps[0],
        ts_max=stamps[-1],
        writers=writer_ranges(rows),
    )


def _parse_ts(raw: str) -> datetime:
    text = raw.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as fh:
        os.fsync(fh.fileno())


def _fsync_dir(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, os.O_RDONLY | flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
