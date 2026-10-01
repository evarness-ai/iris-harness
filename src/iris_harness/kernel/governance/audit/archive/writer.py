"""Parquet+zstd writer for the cold audit archive."""

from __future__ import annotations

import os
import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

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
        grouped: dict[ArchivePartition, list[AuditRow]] = defaultdict(list)
        for row in rows:
            grouped[self.partition_for(_parse_ts(row.ts))].append(row)

        if not grouped:
            return {}

        counts: dict[ArchivePartition, int] = {}
        for partition, part_rows in grouped.items():
            partition_dir = self.root / partition.relative_path
            self._ensure_partition_dir(partition_dir)
            chunk_name = f"audit-{uuid.uuid4().hex}.parquet"
            chunk_path = partition_dir / chunk_name
            table = _table_from_rows(part_rows)
            pq.write_table(
                table,
                chunk_path,
                compression=self._compression,
                use_dictionary=[
                    "agent_type",
                    "hook_point",
                    "plugin",
                    "decision",
                    "classification",
                    "tier",
                    "severity",
                ],
            )
            _fsync_file(chunk_path)
            _fsync_dir(partition_dir)
            counts[partition] = len(part_rows)

        return counts

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
    }
    return pa.Table.from_arrays(
        [data[name] for name in audit_archive_schema().names],
        schema=audit_archive_schema(),
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
