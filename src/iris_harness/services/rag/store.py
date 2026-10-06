"""SQLite registry for document sources + chunks → ``data/rag.db`` (RAG R0).

Holds the canonical chunk text (for citation rendering + the keyword-search
fallback when the vector index is unavailable) and a per-source content hash
for incremental sync. Mirrors the EmailStore/CalendarStore conventions: lazy
``ensure_schema``, idempotent upsert, ``sqlite3.Row``.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from iris_harness.foundation.clock import utc_now
from iris_harness.foundation.persistence import connect, data_path
from iris_harness.services.rag.models import DocumentChunk, DocumentSource, SourceKind
from iris_harness.services.rag.sensitivity import ratchet

# Columns added after R0 — applied additively so existing rag.db files migrate.
_SOURCE_MIGRATIONS = (
    ("tags", "TEXT NOT NULL DEFAULT '[]'"),
    ("links", "TEXT NOT NULL DEFAULT '[]'"),
    ("mtime", "REAL NOT NULL DEFAULT 0"),
    # FMX8: the source's classification. Backfilled from its chunks when added (see
    # ``_backfill_source_classification``); None = never classified (pre-FMX8 ingest).
    ("classification", "TEXT"),
)
_CHUNK_MIGRATIONS = (
    ("page", "INTEGER"),  # R2: 1-based PDF page number
    ("classification", "TEXT"),  # FMX8: source sensitivity at ingest time
)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_dt(value: str | None) -> datetime:
    return datetime.fromisoformat(value) if value else utc_now()


@dataclass
class DocumentStore:
    db_path: Path = field(default_factory=lambda: data_path("rag.db"))

    def ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            added: set[tuple[str, str]] = set()
            for table, migrations in (
                ("document_sources", _SOURCE_MIGRATIONS),
                ("document_chunks", _CHUNK_MIGRATIONS),
            ):
                cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
                for name, ddl in migrations:
                    if name not in cols:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
                        added.add((table, name))
            if ("document_sources", "classification") in added:
                _backfill_source_classification(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = connect(self.db_path, row_factory=sqlite3.Row)
        return conn

    # ── sources ───────────────────────────────────────────────────────

    def get_source(self, source_id: str) -> DocumentSource | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM document_sources WHERE id = ?", (source_id,)
            ).fetchone()
        return _row_to_source(row) if row else None

    def upsert_source(
        self,
        *,
        id: str,
        path: str,
        kind: SourceKind,
        title: str,
        content_sha: str,
        tags: tuple[str, ...] = (),
        links: tuple[str, ...] = (),
        mtime: float = 0.0,
        classification: str | None = None,
    ) -> DocumentSource:
        """Insert or update a source. An omitted (None) ``classification`` keeps the label
        already stored: a label is only ever replaced by another label."""
        now = utc_now()
        existing = self.get_source(id)
        added = existing.added_at if existing else now
        tags_json = json.dumps(list(tags))
        links_json = json.dumps(list(links))
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO document_sources (id, path, kind, title, content_sha, added_at, "
                "last_synced_at, tags, links, mtime, classification) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET path=excluded.path, kind=excluded.kind, "
                "title=excluded.title, content_sha=excluded.content_sha, "
                "last_synced_at=excluded.last_synced_at, tags=excluded.tags, "
                "links=excluded.links, mtime=excluded.mtime, "
                "classification=COALESCE(excluded.classification, document_sources.classification)",
                (
                    id,
                    path,
                    kind,
                    title,
                    content_sha,
                    _iso(added),
                    _iso(now),
                    tags_json,
                    links_json,
                    mtime,
                    classification,
                ),
            )
        return DocumentSource(
            id=id,
            path=path,
            kind=kind,
            title=title,
            content_sha=content_sha,
            added_at=added,
            last_synced_at=now,
            tags=tuple(tags),
            links=tuple(links),
            mtime=mtime,
            classification=(
                classification
                if classification is not None or existing is None
                else existing.classification
            ),
        )

    def list_sources(self) -> list[DocumentSource]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM document_sources ORDER BY path").fetchall()
        return [_row_to_source(r) for r in rows]

    def delete_source(self, source_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM document_chunks WHERE source_id = ?", (source_id,))
            conn.execute("DELETE FROM document_sources WHERE id = ?", (source_id,))

    # ── chunks ────────────────────────────────────────────────────────

    def replace_chunks(self, source_id: str, chunks: Iterable[DocumentChunk]) -> int:
        rows = [
            (
                c.id,
                c.source_id,
                c.source_path,
                c.title,
                c.chunk_index,
                c.text,
                c.page,
                c.classification,
            )
            for c in chunks
        ]
        with self._connect() as conn:
            conn.execute("DELETE FROM document_chunks WHERE source_id = ?", (source_id,))
            conn.executemany(
                "INSERT INTO document_chunks (id, source_id, source_path, title, chunk_index, "
                "text, page, classification) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM document_chunks WHERE id = ?", (chunk_id,)).fetchone()
        return _row_to_chunk(row) if row else None

    def iter_chunks(self) -> Iterable[DocumentChunk]:
        """Every stored chunk, in a stable order (source, then position).

        The canonical chunk set the vector index is rebuilt from. The rows are read in
        one ``fetchall()`` and the connection is closed before the first chunk is
        yielded; only the row-to-chunk conversion is lazy. That is deliberate: a cursor
        held open for a rebuild would keep a read transaction on ``rag.db`` across every
        embedding upsert (slow), which can block a running server's writes. Memory cost
        is one row per chunk (text only, no vectors), which is small next to the index.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM document_chunks ORDER BY source_id, chunk_index"
            ).fetchall()
        return (_row_to_chunk(r) for r in rows)

    def count_all_chunks(self) -> int:
        """Total number of stored chunks across every source."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM document_chunks").fetchone()
        return int(row["n"])

    def count_chunks(self, source_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM document_chunks WHERE source_id = ?", (source_id,)
            ).fetchone()
        return int(row["n"])

    def keyword_search(self, query: str, *, limit: int = 5) -> list[tuple[DocumentChunk, float]]:
        """Token-overlap fallback used when the vector index is unavailable.

        Scores each chunk by how many distinct query tokens it contains
        (case-insensitive). Deterministic; ties broken by chunk id.
        """
        tokens = [t for t in _tokenize(query) if t]
        if not tokens:
            return []
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM document_chunks").fetchall()
        scored: list[tuple[DocumentChunk, float]] = []
        for row in rows:
            chunk = _row_to_chunk(row)
            haystack = f"{chunk.title}\n{chunk.text}".lower()
            hits = sum(1 for t in set(tokens) if t in haystack)
            if hits:
                scored.append((chunk, hits / len(set(tokens))))
        scored.sort(key=lambda cs: (-cs[1], cs[0].id))
        return scored[:limit]


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def _row_to_source(row: sqlite3.Row) -> DocumentSource:
    keys = row.keys()
    return DocumentSource(
        id=row["id"],
        path=row["path"],
        kind=row["kind"],
        title=row["title"],
        content_sha=row["content_sha"],
        added_at=_parse_dt(row["added_at"]),
        last_synced_at=_parse_dt(row["last_synced_at"]),
        tags=tuple(json.loads(row["tags"])) if "tags" in keys and row["tags"] else (),
        links=tuple(json.loads(row["links"])) if "links" in keys and row["links"] else (),
        mtime=float(row["mtime"]) if "mtime" in keys and row["mtime"] is not None else 0.0,
        classification=row["classification"] if "classification" in keys else None,
    )


def _backfill_source_classification(conn: sqlite3.Connection) -> None:
    """Give each existing source the most sensitive label its chunks carry.

    Runs once, when the per-source column is added to an older ``rag.db``. A source whose
    chunks carry no label stays None (it was ingested before classification existed).
    """
    labels: dict[str, list[str]] = {}
    for row in conn.execute(
        "SELECT DISTINCT source_id, classification FROM document_chunks "
        "WHERE classification IS NOT NULL"
    ):
        labels.setdefault(row["source_id"], []).append(row["classification"])
    conn.executemany(
        "UPDATE document_sources SET classification = ? WHERE id = ?",
        [(ratchet(*found), sid) for sid, found in labels.items()],
    )


def _row_to_chunk(row: sqlite3.Row) -> DocumentChunk:
    keys = row.keys()
    return DocumentChunk(
        id=row["id"],
        source_id=row["source_id"],
        source_path=row["source_path"],
        title=row["title"],
        chunk_index=row["chunk_index"],
        text=row["text"],
        page=row["page"] if "page" in keys and row["page"] is not None else None,
        classification=(row["classification"] if "classification" in keys else None),
    )


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS document_sources (
    id             TEXT PRIMARY KEY,
    path           TEXT NOT NULL,
    kind           TEXT NOT NULL,
    title          TEXT NOT NULL,
    content_sha    TEXT NOT NULL,
    added_at       TEXT NOT NULL,
    last_synced_at TEXT NOT NULL,
    tags           TEXT NOT NULL DEFAULT '[]',
    links          TEXT NOT NULL DEFAULT '[]',
    mtime          REAL NOT NULL DEFAULT 0,
    classification TEXT
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL,
    source_path TEXT NOT NULL,
    title       TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    page        INTEGER,
    classification TEXT
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_source ON document_chunks(source_id);
"""


__all__ = ["DocumentStore"]
