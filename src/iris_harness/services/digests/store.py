"""Stored digests: the full web copy of every brief the harness delivered.

Telegram gets the digest in chunks and the phone gets a three-line headline;
neither is the digest. The one full, re-readable copy lives here, and the push
notification's tap lands on it (``/digest/<id>``). Before this store the body
lived only in the in-memory ``HeartbeatRun.output`` and was gone on restart
(loop-proof plan PR 2, graph §7 "web: full, stored").

One row per render: the markdown body exactly as rendered (``iris:`` action links
included, so the web view can turn them into buttons), and the sections that
failed to build, so a partial digest says so wherever it is read (D6).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from iris_harness.foundation.clock import utc_now_iso
from iris_harness.foundation.paths import data_dir
from iris_harness.foundation.persistence.sqlite import connect
from iris_harness.foundation.process_state import track_globals

logger = logging.getLogger(__name__)

DEFAULT_DB_FILENAME = "digests.db"

#: Rows kept. A daily digest plus the odd manual run: ~three months of mornings.
DEFAULT_KEEP = 120

_SCHEMA = """
CREATE TABLE IF NOT EXISTS digests (
    id               TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    heartbeat        TEXT NOT NULL DEFAULT '',
    skill_id         TEXT NOT NULL DEFAULT '',
    subject          TEXT NOT NULL DEFAULT '',
    body             TEXT NOT NULL,
    failed_sections  TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_digests_created ON digests(created_at);
"""


@dataclass(frozen=True)
class FailedSection:
    """A section that raised while rendering: named in the digest, kept here."""

    name: str
    title: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "title": self.title, "reason": self.reason}


@dataclass(frozen=True)
class StoredDigest:
    id: str
    created_at: str
    body: str
    heartbeat: str = ""
    skill_id: str = ""
    subject: str = ""
    failed_sections: tuple[FailedSection, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "heartbeat": self.heartbeat,
            "skill_id": self.skill_id,
            "subject": self.subject,
            "body": self.body,
            "failed_sections": [f.as_dict() for f in self.failed_sections],
        }


def default_db_path() -> Path:
    """``$IRIS_DATA_DIR/digests.db`` — the data volume, beside ``settings.db``."""
    return data_dir() / DEFAULT_DB_FILENAME


def _row_to_digest(row: sqlite3.Row) -> StoredDigest:
    try:
        raw = json.loads(row["failed_sections"] or "[]")
    except json.JSONDecodeError:
        raw = []
    failed = tuple(
        FailedSection(
            name=str(item.get("name", "")),
            title=str(item.get("title", "")),
            reason=str(item.get("reason", "")),
        )
        for item in raw
        if isinstance(item, dict)
    )
    return StoredDigest(
        id=row["id"],
        created_at=row["created_at"],
        body=row["body"],
        heartbeat=row["heartbeat"],
        skill_id=row["skill_id"],
        subject=row["subject"],
        failed_sections=failed,
    )


class DigestStore:
    """SQLite store of rendered digests, newest first."""

    def __init__(self, db_path: Path | None = None, *, keep: int = DEFAULT_KEEP) -> None:
        self.db_path = db_path or default_db_path()
        self.keep = keep
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = connect(self.db_path, row_factory=sqlite3.Row)  # WAL + busy_timeout
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def save(
        self,
        body: str,
        *,
        heartbeat: str = "",
        skill_id: str = "",
        subject: str = "",
        failed_sections: tuple[FailedSection, ...] = (),
    ) -> StoredDigest:
        """Store one render and prune past ``keep``. Returns the stored row."""
        digest = StoredDigest(
            id=uuid.uuid4().hex,
            created_at=utc_now_iso(),
            body=body,
            heartbeat=heartbeat,
            skill_id=skill_id,
            subject=subject,
            failed_sections=tuple(failed_sections),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO digests
                    (id, created_at, heartbeat, skill_id, subject, body, failed_sections)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    digest.id,
                    digest.created_at,
                    digest.heartbeat,
                    digest.skill_id,
                    digest.subject,
                    digest.body,
                    json.dumps([f.as_dict() for f in digest.failed_sections]),
                ),
            )
            if self.keep > 0:
                conn.execute(
                    """
                    DELETE FROM digests WHERE id NOT IN (
                        SELECT id FROM digests ORDER BY created_at DESC, rowid DESC LIMIT ?
                    )
                    """,
                    (self.keep,),
                )
        return digest

    def get(self, digest_id: str) -> StoredDigest | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM digests WHERE id = ?", (digest_id,)).fetchone()
        return _row_to_digest(row) if row else None

    def latest(self, *, skill_id: str | None = None) -> StoredDigest | None:
        """The newest digest, optionally of one brief skill."""
        query = "SELECT * FROM digests"
        params: tuple[str, ...] = ()
        if skill_id:
            query += " WHERE skill_id = ?"
            params = (skill_id,)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return _row_to_digest(row) if row else None

    def list(self, *, limit: int = 20) -> tuple[StoredDigest, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM digests ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        return tuple(_row_to_digest(row) for row in rows)


_shared_store: DigestStore | None = None


def shared_digest_store() -> DigestStore:
    """The process-wide store, opened on first use.

    Shared by the brief handler (writes) and the API routes (reads) so both see
    one file; lazy so a harness that never sends a digest never creates it.
    """
    global _shared_store  # one process-wide store
    if _shared_store is None or _shared_store.db_path != default_db_path():
        _shared_store = DigestStore()
    return _shared_store


__all__ = [
    "DEFAULT_DB_FILENAME",
    "DigestStore",
    "FailedSection",
    "StoredDigest",
    "default_db_path",
    "shared_digest_store",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_shared_store")
