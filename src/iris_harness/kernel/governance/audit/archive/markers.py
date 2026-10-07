"""The compaction marker: what a compaction run moved to the cold archive, for reconciliation.

A compaction is two stores changing: Parquet chunks appear, hot rows disappear. Neither store
alone can say "nothing was lost", so the run writes ONE ``kind='compaction'`` row in the same
SQLite transaction that deletes the rows it archived. The row names the chunks (file, SHA-256,
row count), the id range, and per writer the lowest and highest sequence number and the count,
so a reader can subtract the archived ranges from a writer's sequence and still see a hole
for a row that is in neither store (issue #134, stage 4b).

Markers are never archived: the account of a compaction must stay where the compaction can be
checked against it. One small row per run stays in the hot tier.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from iris_harness.kernel.governance.audit.archive.writer import (
    CHUNK_NAME_RE,
    CHUNK_TMP_RE,
    ChunkInfo,
    sha256_of,
    writer_ranges,
)
from iris_harness.kernel.governance.audit.log import AuditRow
from iris_harness.kernel.governance.audit.sequence import KIND_COMPACTION


def marker_payload(
    *,
    compaction_id: str,
    cutoff_ts: str | None,
    rows: Sequence[AuditRow],
    chunks: Sequence[ChunkInfo],
    adopted: bool,
) -> dict[str, Any]:
    """The marker's payload for ``rows`` archived into ``chunks``."""
    kinds: dict[str, int] = {}
    for row in rows:
        key = row.kind or "event"
        kinds[key] = kinds.get(key, 0) + 1
    ranges = writer_ranges(rows)
    ids = [row.id for row in rows]
    return {
        "compaction_id": compaction_id,
        "cutoff_ts": cutoff_ts,
        "adopted": adopted,
        "row_count": len(rows),
        "id_range": [min(ids), max(ids)] if ids else [],
        "pre_identity_rows": sum(1 for row in rows if row.record_id is None),
        "pre_sequence_rows": sum(1 for row in rows if row.writer_id is None),
        "kinds": dict(sorted(kinds.items())),
        "chunks": [
            {
                "file": c.file,
                "sha256": c.sha256,
                "rows": c.rows,
                "ts_min": c.ts_min,
                "ts_max": c.ts_max,
            }
            for c in chunks
        ],
        "writers": [
            {"writer_id": r.writer_id, "min_seq": r.min_seq, "max_seq": r.max_seq, "count": r.count}
            for r in ranges
        ],
    }


@dataclass(frozen=True)
class Marker:
    """One compaction marker row, parsed."""

    row_id: int
    compaction_id: str
    adopted: bool
    row_count: int
    chunks: tuple[dict[str, Any], ...]
    writers: tuple[dict[str, Any], ...]
    payload: dict[str, Any] = field(default_factory=dict)


def read_markers(db_path: Path) -> list[Marker]:
    """Every compaction marker in the hot ledger, oldest first (empty for an older database)."""
    if not db_path.exists():
        return []
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        try:
            raw = conn.execute(
                "SELECT id, payload_json FROM audit_log WHERE kind = ? ORDER BY id",
                (KIND_COMPACTION,),
            ).fetchall()
        except sqlite3.OperationalError:  # no kind column: a database before stage 4
            return []
    markers: list[Marker] = []
    for row_id, payload_json in raw:
        try:
            payload = json.loads(payload_json)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        markers.append(
            Marker(
                row_id=int(row_id),
                compaction_id=str(payload.get("compaction_id", "")),
                adopted=bool(payload.get("adopted", False)),
                row_count=int(payload.get("row_count", 0)),
                chunks=tuple(payload.get("chunks") or ()),
                writers=tuple(payload.get("writers") or ()),
                payload=payload,
            )
        )
    return markers


@dataclass(frozen=True)
class ArchiveFiles:
    """The files under an archive root, sorted by what a compaction may do with them."""

    run_chunks: tuple[Path, ...]  # audit-<ulid>-<n>.parquet: written by a stage 4b compaction
    temp_chunks: tuple[Path, ...]  # a chunk that never finished publishing
    other: tuple[Path, ...]  # legacy chunks and anything else: never adopted, moved or removed


def scan_archive(root: Path) -> ArchiveFiles:
    run: list[Path] = []
    temp: list[Path] = []
    other: list[Path] = []
    for path in sorted(root.glob("year=*/month=*/*")):
        if not path.is_file():
            continue
        if CHUNK_NAME_RE.match(path.name):
            run.append(path)
        elif CHUNK_TMP_RE.match(path.name):
            temp.append(path)
        elif path.suffix == ".parquet":
            other.append(path)
    return ArchiveFiles(tuple(run), tuple(temp), tuple(other))


@dataclass(frozen=True)
class VerifyReport:
    """What ``verify_archive`` found. Empty ``problems`` and ``unreadable`` means reconciled."""

    markers: int
    chunks_checked: int
    problems: tuple[str, ...]
    orphans: tuple[str, ...]  # run chunks no marker names yet (the next compaction handles them)
    unreadable: tuple[str, ...]  # a chunk that cannot be read, legacy or not: reported, untouched

    @property
    def clean(self) -> bool:
        return not (self.problems or self.unreadable)


def verify_archive(db_path: Path, root: Path) -> VerifyReport:
    """Check every marker against the chunks it names, and list files nothing accounts for.

    Per marker: each chunk exists and has the recorded SHA-256 and row count; the chunk rows
    add up to ``row_count``; the per-writer ranges recomputed from the chunks equal the
    recorded ones. A chunk that cannot be read is reported and left exactly where it is,
    whatever its name. Read-only.
    """
    import pyarrow.parquet as pq

    markers = read_markers(db_path)
    problems: list[str] = []
    unreadable: list[str] = []
    named: set[str] = set()
    checked = 0
    for marker in markers:
        total = 0
        seen: dict[str, list[int]] = {}
        for chunk in marker.chunks:
            rel = str(chunk.get("file", ""))
            named.add(rel)
            path = root / rel
            label = f"{marker.compaction_id}:{rel}"
            if not path.is_file():
                problems.append(f"{label}: chunk is missing")
                continue
            checked += 1
            if sha256_of(path) != chunk.get("sha256"):
                problems.append(f"{label}: sha256 differs from the marker")
            try:
                table = pq.read_table(path, columns=["writer_id", "writer_seq"])
            except Exception:  # noqa: BLE001 - any read failure is the finding
                unreadable.append(rel)
                continue
            if table.num_rows != chunk.get("rows"):
                problems.append(f"{label}: {table.num_rows} rows, marker says {chunk.get('rows')}")
            total += table.num_rows
            for wid, seq in zip(
                table.column("writer_id").to_pylist(),
                table.column("writer_seq").to_pylist(),
                strict=True,
            ):
                if wid is not None and seq is not None:
                    seen.setdefault(wid, []).append(seq)
        if total != marker.row_count:
            problems.append(
                f"{marker.compaction_id}: chunks hold {total} rows, marker says {marker.row_count}"
            )
        recorded = {
            w["writer_id"]: (w["min_seq"], w["max_seq"], w["count"]) for w in marker.writers
        }
        found = {w: (min(s), max(s), len(s)) for w, s in seen.items()}
        if recorded != found and not any(
            p.startswith(f"{marker.compaction_id}:") for p in problems
        ):
            problems.append(
                f"{marker.compaction_id}: writer sequence ranges differ from the marker"
            )
    files = scan_archive(root)
    for path in files.other:
        try:
            pq.read_metadata(path)
        except Exception:  # noqa: BLE001
            unreadable.append(path.relative_to(root).as_posix())
    orphans = tuple(
        p.relative_to(root).as_posix()
        for p in files.run_chunks
        if p.relative_to(root).as_posix() not in named
    )
    return VerifyReport(
        markers=len(markers),
        chunks_checked=checked,
        problems=tuple(problems),
        orphans=orphans,
        unreadable=tuple(sorted(set(unreadable))),
    )


def keys_of(rows: Iterable[AuditRow]) -> list[tuple[int, str | None]]:
    """``(id, record_id)`` of each row: what the delete is checked against."""
    return [(row.id, row.record_id) for row in rows]
