"""SideEffectLedger — SQLite-backed ledger of irreversible tool side-effects.

Story 12.gov-4.10 / design §10.3.

Each row records one side effect. A plain write is recorded by the PostToolUse hook, after
the call. A high-risk call (destructive, or a write the owner approves per call) is
written *before* it runs by the PreToolUse hook as a ``pending`` row, and the PostToolUse
hook finalises that same row (``finalize``), so a process that dies mid-call leaves
evidence of the attempt. At resume time, ``pending(run_id)`` returns rows whose probes haven't
confirmed completion yet; the resume flow then decides to skip,
re-execute, or enqueue an approval.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_harness.foundation.paths import governance_data_dir

logger = logging.getLogger(__name__)

# Override slot (``None``: resolved from ``IRIS_HOME`` on every use, never frozen at
# import -- a process that relocates the home after importing IRIS writes into the
# new one). Set it to point the default somewhere else outright (tests).
DEFAULT_LEDGER_DB_PATH: Path | None = None


def default_ledger_db_path() -> Path:
    """``DEFAULT_LEDGER_DB_PATH`` when set, else ``<governance data dir>/side_effects.db``."""
    return (
        DEFAULT_LEDGER_DB_PATH
        if DEFAULT_LEDGER_DB_PATH is not None
        else governance_data_dir() / "side_effects.db"
    )


_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

ProbeStatus = str  # "pending" | "completed" | "not_completed" | "ambiguous" | "error"


@dataclass(frozen=True)
class SideEffectRow:
    side_effect_id: str
    run_id: str
    step_id: int
    tool: str
    verification_probe: str
    probe_metadata: dict[str, Any]
    status: ProbeStatus
    completed_at: str | None
    error: str | None

    @property
    def probe_subject(self) -> str:
        """What the probe checks: the effect's own id (a commit SHA, a PR URL) when the
        recorder knew it (``probe_metadata["subject"]``), else the row's key."""
        subject = self.probe_metadata.get("subject")
        return subject if isinstance(subject, str) and subject else self.side_effect_id


class SideEffectLedger:
    """Open-or-create the side-effect ledger; enforces append-only semantics."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or default_ledger_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if not self.db_path.exists():
            fd = os.open(str(self.db_path), os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
        else:
            try:
                os.chmod(str(self.db_path), 0o600)
            except OSError as exc:  # pragma: no cover
                logger.warning("could not chmod %s to 0o600: %s", self.db_path, exc)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        schema = _SCHEMA_PATH.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(schema)
            conn.commit()

    @staticmethod
    def _row(r: sqlite3.Row) -> SideEffectRow:
        try:
            meta = json.loads(r["probe_metadata"] or "{}")
        except json.JSONDecodeError:
            meta = {}
        return SideEffectRow(
            side_effect_id=r["side_effect_id"],
            run_id=r["run_id"],
            step_id=r["step_id"],
            tool=r["tool"],
            verification_probe=r["verification_probe"],
            probe_metadata=meta if isinstance(meta, dict) else {},
            status=r["status"],
            completed_at=r["completed_at"],
            error=r["error"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        run_id: str,
        step_id: int,
        tool: str,
        verification_probe: str,
        probe_metadata: dict[str, Any] | None = None,
        side_effect_id: str | None = None,
    ) -> str:
        """Insert a new pending side-effect row; returns the side_effect_id.

        ``side_effect_id`` is the caller's key for the effect (the ledger hook's
        ``<run_id>:<step_id>:<tool_call_id>``); a key already recorded is left as it is,
        so recording one call twice is a no-op. Without one, a fresh UUID is minted.
        """
        side_effect_id = side_effect_id or str(uuid.uuid4())
        meta_json = json.dumps(probe_metadata or {}, default=str, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO side_effect_ledger
                    (side_effect_id, run_id, step_id, tool,
                     verification_probe, probe_metadata, status)
                VALUES (?, ?, ?, ?, ?, ?, 'pending')
                """,
                (side_effect_id, run_id, step_id, tool, verification_probe, meta_json),
            )
            conn.commit()
        logger.debug(
            "side_effect recorded %s run=%s step=%d tool=%s probe=%s",
            side_effect_id,
            run_id,
            step_id,
            tool,
            verification_probe,
        )
        return side_effect_id

    def get(self, side_effect_id: str) -> SideEffectRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM side_effect_ledger WHERE side_effect_id = ?",
                (side_effect_id,),
            ).fetchone()
        return self._row(row) if row else None

    def pending(self, run_id: str) -> list[SideEffectRow]:
        """Return rows for run_id whose status is not 'completed'."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM side_effect_ledger
                WHERE run_id = ? AND status != 'completed'
                ORDER BY step_id, rowid
                """,
                (run_id,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def list_by_run(self, run_id: str) -> list[SideEffectRow]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM side_effect_ledger WHERE run_id = ? ORDER BY step_id, rowid",
                (run_id,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def set_status(
        self,
        side_effect_id: str,
        *,
        status: ProbeStatus,
        error: str | None = None,
    ) -> None:
        """Update a row's status (and optionally error)."""
        completed_at = datetime.now(UTC).isoformat() if status == "completed" else None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE side_effect_ledger
                   SET status = ?, completed_at = ?, error = ?
                 WHERE side_effect_id = ?
                """,
                (status, completed_at, error, side_effect_id),
            )
            conn.commit()

    def finalize(
        self,
        side_effect_id: str,
        *,
        status: ProbeStatus,
        error: str | None = None,
        verification_probe: str | None = None,
        probe_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Settle a row written before its call ran; ``False`` when there is no such row.

        ``status`` is the outcome (``completed`` / ``error``); ``error`` only ever names
        the exception class, never its message. ``verification_probe`` and
        ``probe_metadata`` replace the pre-call values when the call's result named the
        effect (a commit SHA, a PR URL): the metadata is merged over the row's own, so the
        pre-call keys survive.
        """
        completed_at = datetime.now(UTC).isoformat() if status == "completed" else None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT probe_metadata, verification_probe FROM side_effect_ledger "
                "WHERE side_effect_id = ?",
                (side_effect_id,),
            ).fetchone()
            if row is None:
                return False
            try:
                merged = json.loads(row["probe_metadata"] or "{}")
            except json.JSONDecodeError:
                merged = {}
            if not isinstance(merged, dict):
                merged = {}
            merged.update(probe_metadata or {})
            conn.execute(
                """
                UPDATE side_effect_ledger
                   SET status = ?, completed_at = ?, error = ?,
                       verification_probe = ?, probe_metadata = ?
                 WHERE side_effect_id = ?
                """,
                (
                    status,
                    completed_at,
                    error,
                    row["verification_probe"] if verification_probe is None else verification_probe,
                    json.dumps(merged, default=str, sort_keys=True),
                    side_effect_id,
                ),
            )
            conn.commit()
        return True
