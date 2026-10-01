"""Approval-taught exemptions for the ``goal_drift`` signal.

The signal halts a run when a thought looks unlike the task. When the person says
"that was fine", nothing remembered it, so the same shape of thought halted the same
shape of question again on the next turn. This store is the memory: a flagged thought
is written as a *candidate* when it halts, and the person's approval promotes it to an
exemption that later runs can match.

Be clear about what this is: **a governance control learning not to fire.** An
exemption is a standing grant, so the matching rule is deliberately narrow on three
axes at once, and every use of one is audited with the id of the exemption that
allowed it.

* **Scope.** An exemption belongs to the task that earned it. It applies only when the
  new run's original task embeds close to the approved one, so "inbox thoughts are
  fine when I ask how my day looks" does not become "inbox thoughts are fine".
* **Keywords, not distance.** The thought must share the approved thought's content
  words. Embedding proximity is what mis-fired in the first place; asking for the same
  words back is a rule you can read in the audit and predict before it runs.
* **Floor.** A single shared word never exempts anything. At least
  ``_MIN_SHARED_KEYWORDS`` must match, and they must cover ``_MIN_COVERAGE`` of the
  approved thought's keywords.

Rejections are recorded too. A thought the person refused must never be promoted by a
later approval of a different step in the same run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from iris_harness.foundation.paths import governance_data_dir

logger = logging.getLogger(__name__)


def default_db_path() -> Path:
    """Where the exemptions live.

    Resolved on every call, never snapshotted into a module constant: a constant
    computed at import time cannot see ``IRIS_HOME`` or the explicit override that a
    test (or an operator moving the profile) sets afterwards, and a store that silently
    keeps writing to the old location is how a governance grant ends up in a file
    nobody is looking at.
    """
    override = os.getenv("IRIS_GOVERNANCE_DRIFT_EXEMPTIONS_DB_PATH", "").strip()
    if override:
        return Path(override)
    return governance_data_dir() / "drift_exemptions.db"


_MIN_SHARED_KEYWORDS = 2
_MIN_COVERAGE = 0.6

_WORD = re.compile(r"[a-z0-9']+")

# Filler that carries no topic. Kept short on purpose: a long list is a place for a
# content word to hide, and every word removed here is one fewer the rule can require.
_STOPWORDS = frozenset("""
    a an the and or but if then than that this these those to of in on at for with from
    by as is are was were be been being am do does did doing have has had having i me my
    we our you your it its they them their he she his her not no nor so such can could
    will would shall should may might must need needs let lets now next also more most
    some any all each both few other into over under again further once here there when
    where why how what which who whom about against between through during before after
    above below up down out off only own same too very just
    need to i should i will let me now i
    """.split())


def extract_keywords(text: str) -> frozenset[str]:
    """The content words of ``text``: lowercased, de-duplicated, filler removed.

    Deterministic and dependency-free on purpose. A rule the person can predict before
    it runs is worth more here than a cleverer one they cannot.
    """
    return frozenset(
        word for word in _WORD.findall(text.lower()) if len(word) >= 3 and word not in _STOPWORDS
    )


@dataclass(frozen=True)
class DriftExemption:
    """One approved (or still pending) thought shape."""

    exemption_id: str
    run_id: str
    step_id: int
    original_task: str
    task_vector: tuple[float, ...]
    keywords: frozenset[str]
    thought: str
    status: str
    created_at: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS drift_exemptions (
    exemption_id  TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    step_id       INTEGER NOT NULL,
    original_task TEXT NOT NULL,
    task_vector   TEXT NOT NULL,
    keywords      TEXT NOT NULL,
    thought       TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected')),
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_drift_exemptions_run ON drift_exemptions (run_id);
CREATE INDEX IF NOT EXISTS idx_drift_exemptions_status ON drift_exemptions (status);
"""


class DriftExemptionStore:
    """SQLite-backed store of thought shapes a person has approved."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record_candidate(
        self,
        *,
        run_id: str,
        step_id: int,
        original_task: str,
        task_vector: Sequence[float],
        thought: str,
    ) -> str:
        """Write the thought that just halted a run, pending the person's answer."""
        exemption_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO drift_exemptions (exemption_id, run_id, step_id, original_task,"
                " task_vector, keywords, thought, status, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                (
                    exemption_id,
                    run_id,
                    step_id,
                    original_task,
                    json.dumps(list(task_vector)),
                    json.dumps(sorted(extract_keywords(thought))),
                    thought,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return exemption_id

    def settle_run(self, run_id: str, *, approved: bool) -> int:
        """Promote (or bury) every candidate this run raised. Returns the row count.

        Only ``pending`` rows move, so a rejection is permanent: a later approval of
        some other step in the same run cannot resurrect a refused thought.
        """
        status = "approved" if approved else "rejected"
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE drift_exemptions SET status = ? WHERE run_id = ? AND status = 'pending'",
                (status, run_id),
            )
            return int(cur.rowcount)

    def approved(self) -> list[DriftExemption]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM drift_exemptions WHERE status = 'approved' ORDER BY created_at"
            ).fetchall()
        return [_row_to_exemption(r) for r in rows]

    def all_rows(self) -> list[DriftExemption]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM drift_exemptions ORDER BY created_at").fetchall()
        return [_row_to_exemption(r) for r in rows]

    def clear(self) -> int:
        """Forget every exemption. The person can always take the grant back."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM drift_exemptions")
            return int(cur.rowcount)


def _row_to_exemption(row: sqlite3.Row) -> DriftExemption:
    return DriftExemption(
        exemption_id=row["exemption_id"],
        run_id=row["run_id"],
        step_id=int(row["step_id"]),
        original_task=row["original_task"],
        task_vector=tuple(json.loads(row["task_vector"])),
        keywords=frozenset(json.loads(row["keywords"])),
        thought=row["thought"],
        status=row["status"],
        created_at=row["created_at"],
    )


def keyword_match(exemption: DriftExemption, thought_keywords: frozenset[str]) -> float | None:
    """Coverage of ``exemption``'s keywords by the thought, or None if it falls short.

    Returns the fraction so the audit can record *how* well it matched, not just that
    it did.
    """
    if not exemption.keywords:
        return None
    shared = exemption.keywords & thought_keywords
    if len(shared) < _MIN_SHARED_KEYWORDS:
        return None
    coverage = len(shared) / len(exemption.keywords)
    return coverage if coverage >= _MIN_COVERAGE else None


__all__ = [
    "DriftExemption",
    "DriftExemptionStore",
    "default_db_path",
    "extract_keywords",
    "keyword_match",
]
