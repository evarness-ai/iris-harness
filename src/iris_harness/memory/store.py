"""SQLite-backed memory store primitives for PRD-05 bootstrap.

This module intentionally keeps the first slice minimal and typed so pipeline
personas can iterate safely in follow-up runs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from iris_harness.foundation.persistence import data_path, sqlite_conn, with_locked_retry
from iris_harness.foundation.persistence.sqlite import add_columns_if_missing
from iris_harness.memory.fact_statements import FactStatements, Link
from iris_harness.memory.vocabulary import Vocabulary
from memris.graph import MemoryGraph

logger = logging.getLogger(__name__)

# A read that leaves out what the owner removed (ADR-0119): the SQL is a constant.
_NOT_REMOVED = (
    "AND session_id NOT IN " "(SELECT target_id FROM memory_removals WHERE kind = 'session')"
)


@dataclass(frozen=True)
class LessonProposal:
    """Something IRIS learned about how to work, waiting for the owner to say yes.

    ``trigger`` is when it applies (an intent, or the words that bring it up) and
    ``lesson`` is what to do instead. Approving one writes a behavior file, which is
    the mechanism that already reaches the prompt — no second injection path.
    """

    id: int
    trigger: str
    lesson: str
    evidence: str
    source: str
    created_at: datetime
    status: str = "pending"
    resolved_at: datetime | None = None


@dataclass(frozen=True)
class FactProposal:
    """A fact the extractor suggested, waiting for the owner to say yes or no.

    ``current_value`` is what was stored for this key when the proposal was raised, so
    a review reads "changed?" instead of silently overwriting a confirmed fact.
    """

    id: str  # the proposed statement's id (memris plan PR 2c)
    key: str
    value: str
    confidence: float
    source: str
    evidence: str
    current_value: str | None
    created_at: datetime
    status: str = "pending"
    resolved_at: datetime | None = None
    seen_count: int = 1
    # Who it is about when that is not the owner — someone one hop away (memris PR 3b).
    subject: str | None = None


@dataclass(frozen=True)
class UserFact:
    """A single learned fact about the user."""

    key: str
    value: str
    confidence: float
    source: str
    first_seen: datetime
    last_confirmed: datetime
    times_confirmed: int = 1
    # Owner-confirmed (PR: fact confirmation). Only confirmed facts are recalled into a
    # prompt. Confidence never sets this: the store held `name=ollama` at 1.0 and
    # `employer=Department of Justice` at 0.9, both mined from content the user was
    # merely discussing. A human says yes, or it stays a proposal.
    confirmed: bool = False
    # Runtime-only recall annotation (NOT persisted): set by the retriever's quality
    # filter when a fact's confidence is in the low/uncertain band, so the prompt
    # renderer can mark it "(unconfirmed)" instead of presenting it as authoritative.
    uncertain: bool = False
    # The statement this fact is (memris). A key can hold several values — two cards —
    # and an edit or a forget must be able to name ONE of them, not the key.
    statement_id: str | None = None


@dataclass(frozen=True)
class FactHistoryEntry:
    """One change to a user fact, derived from its statement chain (memris PR 2c-ii)."""

    id: str  # the statement id; "<id>#forget" for the forget event
    key: str
    old_value: str | None
    old_confidence: float | None
    new_value: str | None
    new_confidence: float | None
    source: str
    reason: str  # capture | supersede | correct | forget | restore
    changed_at: datetime


@dataclass(frozen=True)
class FactContradiction:
    """A same-key value conflict, surfaced for human review (memris PR 2c-ii)."""

    id: str  # the statement that replaced or was refused
    key: str
    stored_value: str
    stored_confidence: float | None
    incoming_value: str
    incoming_confidence: float | None
    resolution: str  # superseded | blocked
    source: str
    detected_at: datetime
    acknowledged: bool
    seen_count: int = 1


@dataclass(frozen=True)
class LearningSignal:
    """A single learning event from user interaction."""

    id: str
    signal_type: str
    domain: str
    agent_type: str
    query: str
    context: str
    outcome: str
    improvement_hint: str | None
    timestamp: datetime


#: Columns added to a table after it first shipped, by table. ``seen_count`` lets repeated
#: identical conflicts and proposals dedup (bump) instead of inserting a fresh row every turn;
#: ``confirmed`` arrived with owner-confirmed recall and defaults to 0 on purpose (the facts
#: already in a store were mined without anyone reviewing them, and stop reaching prompts until
#: they are); ``turn_origin`` (#145) is nullable: NULL is unknown, never backfilled.
_LATER_COLUMNS: dict[str, dict[str, str]] = {
    "user_fact_contradictions": {"seen_count": "INTEGER NOT NULL DEFAULT 1"},
    "user_facts": {"confirmed": "INTEGER NOT NULL DEFAULT 0"},
    "fact_proposals": {"seen_count": "INTEGER NOT NULL DEFAULT 1"},
    "conversations": {"turn_origin": "TEXT"},
}


@dataclass
class MemoryStore:
    """Small bootstrap persistence layer for memory facts and signals."""

    db_path: Path = field(default_factory=lambda: data_path("memory.db"))
    # User facts live as memris statements (memris plan PR 2b). The legacy user_facts
    # table is kept, unread, as the archive the one-time migration came from.
    _fact_statements: FactStatements | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _facts_migrated: bool = field(default=False, init=False, repr=False, compare=False)
    _columns_ready_for: tuple[int, int] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def memory_graph(self) -> MemoryGraph:
        """The memris graph behind user facts — read it, resolve against it, review merges.

        Runs the schema step first, so a database's one-time fact migration has happened
        before anyone reads statements (the Map once drew no facts on a first open).
        """
        self.ensure_schema()
        return self._facts().graph

    def fact_statements(self) -> FactStatements:
        """User facts as memris statements — the view the removal rules read keys from."""
        self.ensure_schema()
        return self._facts()

    def _facts(self) -> FactStatements:
        if self._fact_statements is None:
            self._fact_statements = FactStatements(self.db_path)
        return self._fact_statements

    def _migrate_facts_once(self) -> None:
        """Move legacy user_facts onto statements the first time this database opens.

        Owner decision (2026-09-18): automatic, with a backup — a step people have to
        remember is a step that gets skipped. A no-op once done, and on a fresh database.
        """
        if self._facts_migrated:
            return
        from iris_harness.memory.ontology import memory_ontology
        from iris_harness.memory.statement_migration import apply_migration

        base = self.db_path.parent
        result = apply_migration(
            self.db_path,
            memory_ontology(),
            out_dir=base / "memris-migration",
            backup_dir=base / "backups",
        )
        if result.applied:
            logger.warning(
                "memory: moved user facts onto memris statements (%d statements); backup %s, report %s",
                result.statements,
                result.backup,
                result.report,
            )
        self._facts_migrated = True

    def ensure_schema(self) -> None:
        """Create required tables if they do not exist, then move legacy facts once."""
        self._ensure_tables()
        self._migrate_facts_once()

    def _ensure_tables(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite_conn(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_confirmed TEXT NOT NULL,
                    times_confirmed INTEGER NOT NULL DEFAULT 1
                )
                """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_signals (
                    id TEXT PRIMARY KEY,
                    signal_type TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    query TEXT NOT NULL,
                    context TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    improvement_hint TEXT,
                    timestamp TEXT NOT NULL
                )
                """)
            # Conversation persistence — raw turns + per-session compacted summary.
            # Raw turns are append-only; the summary is updated whenever the compactor runs.
            # On reload we inject: summary (1 line) + last N raw turns → bounded context.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role      TEXT NOT NULL,
                    content   TEXT NOT NULL,
                    ts        TEXT NOT NULL,
                    turn_origin TEXT
                )
                """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id, id)"
            )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    session_id TEXT PRIMARY KEY,
                    summary    TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """)
            # Append-only audit trail of every user-fact value change (capture /
            # supersede / correct / forget / restore). The store's trust backbone:
            # nothing IRIS "believes" about the user is lost or unexplained, and a
            # change is always reversible (restore re-activates the prior value).
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_fact_history (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    key             TEXT NOT NULL,
                    old_value       TEXT,
                    old_confidence  REAL,
                    new_value       TEXT,
                    new_confidence  REAL,
                    source          TEXT NOT NULL,
                    reason          TEXT NOT NULL,
                    changed_at      TEXT NOT NULL
                )
                """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fact_history_key ON user_fact_history(key, id)"
            )
            # Detected same-key value conflicts, for human review. Distinct from history
            # (applied changes): this also captures BLOCKED conflicts that never changed
            # the fact, so contradicting info is never silently lost.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_fact_contradictions (
                    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                    key                  TEXT NOT NULL,
                    stored_value         TEXT NOT NULL,
                    stored_confidence    REAL,
                    incoming_value       TEXT NOT NULL,
                    incoming_confidence  REAL,
                    resolution           TEXT NOT NULL,
                    source               TEXT NOT NULL,
                    detected_at          TEXT NOT NULL,
                    acknowledged         INTEGER NOT NULL DEFAULT 0,
                    seen_count           INTEGER NOT NULL DEFAULT 1
                )
                """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fact_contra_ack "
                "ON user_fact_contradictions(acknowledged, id)"
            )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fact_proposals (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    key          TEXT NOT NULL,
                    value        TEXT NOT NULL,
                    confidence   REAL NOT NULL,
                    source       TEXT NOT NULL,
                    evidence     TEXT NOT NULL DEFAULT '',
                    current_value TEXT,
                    created_at   TEXT NOT NULL,
                    status       TEXT NOT NULL DEFAULT 'pending',
                    resolved_at  TEXT
                )
                """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fact_proposals_status "
                "ON fact_proposals(status, created_at)"
            )
            # Lessons wait in the same kind of queue as facts: an approved one becomes a
            # behavior file, which is the mechanism that already reaches the prompt.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lesson_proposals (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger     TEXT NOT NULL,
                    lesson      TEXT NOT NULL,
                    evidence    TEXT NOT NULL DEFAULT '',
                    source      TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'pending',
                    resolved_at TEXT
                )
                """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lesson_proposals_status "
                "ON lesson_proposals(status, created_at)"
            )
            # What the owner removed from memory (ADR-0119): the Removed list, and the
            # marks the read paths check — a removed session, a suppressed name. An
            # entity's own mark lives in memris; its row here keeps its name suppressed.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_removals (
                    id          TEXT PRIMARY KEY,
                    kind        TEXT NOT NULL,
                    target_id   TEXT NOT NULL,
                    label       TEXT NOT NULL,
                    name_key    TEXT,
                    removed_at  TEXT NOT NULL,
                    cascade     TEXT NOT NULL DEFAULT '[]',
                    permanent   INTEGER NOT NULL DEFAULT 0
                )
                """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_removals_target "
                "ON memory_removals(kind, target_id)"
            )
            conn.commit()
        self._add_later_columns()

    def _add_later_columns(self) -> None:
        """Add the columns that arrived after their tables first shipped (``_LATER_COLUMNS``).

        Each goes in on a connection of its own under ``BEGIN IMMEDIATE``
        (``add_columns_if_missing``). They used to be a ``PRAGMA table_info`` read followed by
        an ``ALTER`` inside the connection's transaction, which fails with ``database is
        locked`` / ``duplicate column name`` when two processes open an older ``memory.db`` at
        once (the API, the CLI, a heartbeat); the busy timeout does not help. The nullable,
        no-default rule is the policy for NEW identity columns (``turn_origin``); the others
        keep the declaration they always had, so an old row reads exactly as before.

        Done once per database file this instance has seen (its device and inode), not on every
        call: ``ensure_schema`` runs before every store method.
        """
        try:
            stat = self.db_path.stat()
        except OSError:
            return
        identity = (stat.st_dev, stat.st_ino)
        if self._columns_ready_for == identity:
            return
        for table, columns in _LATER_COLUMNS.items():
            add_columns_if_missing(self.db_path, table, columns)
        self._columns_ready_for = identity

    @with_locked_retry
    def upsert_user_fact(self, fact: UserFact) -> None:
        """Persist or update a user fact — as a memris statement (memris plan PR 2b).

        A new value only REPLACES an existing one when its confidence is at least the
        stored fact's — so a low-confidence mis-extraction ("blog"="site", 0.3) can't
        clobber a higher-confidence grounded fact ("blog"="web3notes.example", 0.9); the
        refused value is kept as a refused statement, so review still sees it (PR 2c-ii).
        Re-confirming the same value refreshes its count and timestamps (issue 0021).
        A key with no property in the ontology raises FactKeyError.
        """
        self.ensure_schema()
        self._facts().put(fact)

    @with_locked_retry
    def append_learning_signal(self, signal: LearningSignal) -> None:
        """Persist a learning signal event."""
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO learning_signals(
                    id, signal_type, domain, agent_type, query, context, outcome, improvement_hint, timestamp
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.id,
                    signal.signal_type,
                    signal.domain,
                    signal.agent_type,
                    signal.query,
                    signal.context,
                    signal.outcome,
                    signal.improvement_hint,
                    signal.timestamp.isoformat(),
                ),
            )
            conn.commit()

    def validate_user_fact(self, fact: UserFact) -> bool:
        """Validate a user fact's data integrity."""
        if not fact.key or not isinstance(fact.key, str):
            return False
        if not fact.value or not isinstance(fact.value, str):
            return False
        if not (0.0 <= fact.confidence <= 1.0):
            return False
        if not fact.source or not isinstance(fact.source, str):
            return False
        if not isinstance(fact.first_seen, datetime):
            return False
        if not isinstance(fact.last_confirmed, datetime):
            return False
        if not isinstance(fact.times_confirmed, int) or fact.times_confirmed < 1:
            return False
        return True

    _FACT_COLUMNS = (
        "key, value, confidence, source, first_seen, last_confirmed, times_confirmed, confirmed"
    )

    @staticmethod
    def _fact_row(row: tuple[Any, ...]) -> UserFact:
        return UserFact(
            key=row[0],
            value=row[1],
            confidence=row[2],
            source=row[3],
            first_seen=datetime.fromisoformat(row[4]),
            last_confirmed=datetime.fromisoformat(row[5]),
            times_confirmed=row[6],
            confirmed=bool(row[7]),
        )

    def fetch_user_fact(self, key: str) -> UserFact | None:
        """Retrieve a user fact by key (confirmed or not — callers decide)."""
        self.ensure_schema()
        return self._facts().get(key)

    def fetch_all_user_facts(self, *, confirmed_only: bool = False) -> list[UserFact]:
        """Retrieve user facts. ``confirmed_only`` is what recall uses."""
        self.ensure_schema()
        return self._facts().all(confirmed_only=confirmed_only)

    def fetch_fact_projections(self, *, confirmed_only: bool = False) -> list[UserFact]:
        """One fact per key, for the homes that hold one line per key.

        The recall index and USER.md's auto block keep a key once. A key holding
        several values (two cards) is one fact here whose value lists them all, oldest
        first — keyed dicts over ``fetch_all_user_facts`` silently kept only the last.
        """
        by_key: dict[str, list[UserFact]] = {}
        for fact in self.fetch_all_user_facts(confirmed_only=confirmed_only):
            by_key.setdefault(fact.key, []).append(fact)
        merged: list[UserFact] = []
        for key, facts in sorted(by_key.items()):
            if len(facts) == 1:
                merged.append(facts[0])
                continue
            facts.sort(key=lambda f: (f.first_seen, f.value))
            merged.append(
                UserFact(
                    key=key,
                    value=", ".join(dict.fromkeys(f.value for f in facts)),
                    confidence=min(f.confidence for f in facts),
                    source=facts[0].source,
                    first_seen=facts[0].first_seen,
                    last_confirmed=max(f.last_confirmed for f in facts),
                    times_confirmed=sum(f.times_confirmed for f in facts),
                    confirmed=all(f.confirmed for f in facts),
                )
            )
        return merged

    # ------------------------------------------------------------------
    # Fact proposals — the review queue behind owner-confirmed recall
    # ------------------------------------------------------------------

    @with_locked_retry
    def add_fact_proposal(
        self,
        *,
        key: str,
        value: str,
        confidence: float,
        source: str,
        evidence: str = "",
        subject: str | None = None,
        subject_class: str | None = None,
    ) -> str | None:
        """Queue a proposed fact for review. Returns its id, or None when redundant.

        ``subject``: the name of whom it is about when not the owner (one hop away,
        already checked in scope by capture), resolved as a ``subject_class`` entity.

        A proposal is a proposed statement (memris plan PR 2c). Redundant = the same
        key/value is already confirmed (that just bumps its count), so repeating
        something you already told IRIS does not fill the queue. An identical pending
        proposal returns the existing id, its seen count bumped.
        """
        self.ensure_schema()
        return self._facts().propose(
            key,
            value,
            confidence=confidence,
            source=source,
            evidence=evidence,
            subject=subject,
            subject_class=subject_class,
        )

    def subject_in_scope(self, name: str, links: list[Link], *, max_hops: int) -> str | None:
        """The class of ``name`` when it is within ``max_hops`` of the owner, else None.

        ``links`` are relations stated in the same message (memris PR 3b); a confirmed
        relation counts as well. See :meth:`FactStatements.reachable`.
        """
        self.ensure_schema()
        return self._facts().reachable(name, links, max_hops=max_hops)

    def vocabulary(self) -> Vocabulary:
        """The learned vocabulary (memris PR 7): words memory picked up, as data."""
        self.ensure_schema()
        facts = self._facts()
        return Vocabulary(facts.graph, facts._key_for)

    def learned_fact_keys(self) -> list[str]:
        """Fact keys that are active learned terms — offered to the extractor like any key."""
        self.ensure_schema()
        return sorted(t.name.partition(":")[2] for t in self._facts().graph.store.terms("active"))

    def fetch_fact_proposals(
        self, *, status: str = "pending", limit: int = 200
    ) -> list[FactProposal]:
        """The review queue for one status, newest first."""
        self.ensure_schema()
        return self._facts().proposals(status=status, limit=limit)

    def fetch_fact_proposal(self, proposal_id: str) -> FactProposal | None:
        """One proposal by id, whatever its status."""
        self.ensure_schema()
        return self._facts().proposal(proposal_id)

    @with_locked_retry
    def resolve_fact_proposal(self, proposal_id: str, status: str) -> bool:
        """Mark a proposal approved / rejected / expired. Nothing is deleted."""
        self.ensure_schema()
        return self._facts().resolve(proposal_id, status)

    @with_locked_retry
    def expire_fact_proposals(self, *, older_than_days: int = 30) -> int:
        """Expire proposals nobody reviewed, so the queue does not grow forever."""
        self.ensure_schema()
        return self._facts().expire(older_than=now_utc() - timedelta(days=older_than_days))

    @with_locked_retry
    def set_fact_confirmed(self, key: str, confirmed: bool = True) -> bool:
        """Flip the owner-confirmed flag on a stored fact."""
        self.ensure_schema()
        return self._facts().set_confirmed(key, confirmed)

    def count_pending_review(self) -> int:
        """Proposed facts awaiting the owner — what review owes the user."""
        self.ensure_schema()
        return self._facts().count_unconfirmed()

    # ------------------------------------------------------------------
    # Lesson proposals — captured from evidence, approved into behaviors
    # ------------------------------------------------------------------

    @with_locked_retry
    def add_lesson_proposal(
        self, *, trigger: str, lesson: str, source: str, evidence: str = ""
    ) -> int | None:
        """Queue a lesson. Returns the row id, or the existing one when it repeats."""
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            existing = conn.execute(
                "SELECT id FROM lesson_proposals WHERE trigger = ? AND lesson = ? "
                "AND status = 'pending'",
                (trigger, lesson),
            ).fetchone()
            if existing:
                return int(existing[0])
            cursor = conn.execute(
                """
                INSERT INTO lesson_proposals(trigger, lesson, evidence, source, created_at, status)
                VALUES(?, ?, ?, ?, ?, 'pending')
                """,
                (trigger, lesson, evidence[:800], source, now_utc().isoformat()),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)

    def fetch_lesson_proposals(
        self, *, status: str = "pending", limit: int = 100
    ) -> list[LessonProposal]:
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, trigger, lesson, evidence, source, created_at, status, resolved_at "
                "FROM lesson_proposals WHERE status = ? ORDER BY id DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        return [
            LessonProposal(
                id=int(r[0]),
                trigger=r[1],
                lesson=r[2],
                evidence=r[3] or "",
                source=r[4],
                created_at=datetime.fromisoformat(r[5]),
                status=r[6],
                resolved_at=datetime.fromisoformat(r[7]) if r[7] else None,
            )
            for r in rows
        ]

    def fetch_lesson_proposal(self, proposal_id: int) -> LessonProposal | None:
        for status in ("pending", "approved", "rejected", "expired"):
            for proposal in self.fetch_lesson_proposals(status=status, limit=500):
                if proposal.id == proposal_id:
                    return proposal
        return None

    @with_locked_retry
    def resolve_lesson_proposal(self, proposal_id: int, status: str) -> bool:
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE lesson_proposals SET status = ?, resolved_at = ? "
                "WHERE id = ? AND status = 'pending'",
                (status, now_utc().isoformat(), proposal_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    @with_locked_retry
    def expire_lesson_proposals(self, *, older_than_days: int = 30) -> int:
        self.ensure_schema()
        cutoff = (now_utc() - timedelta(days=older_than_days)).isoformat()
        with sqlite_conn(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE lesson_proposals SET status = 'expired', resolved_at = ? "
                "WHERE status = 'pending' AND created_at < ?",
                (now_utc().isoformat(), cutoff),
            )
            conn.commit()
            return int(cursor.rowcount)

    # ------------------------------------------------------------------
    # Retention — see config/memory/retention.yaml
    # ------------------------------------------------------------------

    def session_activity(self) -> list[tuple[str, str, int]]:
        """``(session_id, last_ts, turn_count)`` for every session holding turns."""
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            return [
                (str(r[0]), str(r[1]), int(r[2]))
                for r in conn.execute(
                    "SELECT session_id, MAX(ts), COUNT(*) FROM conversations GROUP BY session_id"
                ).fetchall()
            ]

    def fetch_turn_ids(self, session_id: str) -> list[int]:
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            return [
                int(r[0])
                for r in conn.execute(
                    "SELECT id FROM conversations WHERE session_id = ?", (session_id,)
                ).fetchall()
            ]

    @with_locked_retry
    def delete_session_turns(self, session_id: str) -> list[int]:
        """Delete a session's turns. Returns the row ids, so the caller drops their vectors.

        The summary is NOT touched: cooling a session replaces its text with its summary.
        Use :meth:`delete_conversation_summary` to remove that too (an owner action).
        """
        ids = self.fetch_turn_ids(session_id)
        if not ids:
            return []
        with sqlite_conn(self.db_path) as conn:
            conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
            conn.commit()
        return ids

    @with_locked_retry
    def delete_conversation_summary(self, session_id: str) -> bool:
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM conversation_summaries WHERE session_id = ?", (session_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def search_turns(
        self, needle: str, *, limit: int = 200, include_removed: bool = False
    ) -> list[tuple[int, str, str, str]]:
        """``(id, session_id, role, content)`` for turns containing ``needle``.

        Turns of a session the owner removed are left out (ADR-0119) unless
        ``include_removed`` — forgetting by needle must still find them to delete them.
        """
        self.ensure_schema()
        if not needle.strip():
            return []
        where = "" if include_removed else _NOT_REMOVED
        sql = f"SELECT id, session_id, role, content FROM conversations WHERE content LIKE ? {where} ORDER BY id DESC LIMIT ?"  # noqa: S608 — a constant
        with sqlite_conn(self.db_path) as conn:
            rows = conn.execute(sql, (f"%{needle}%", limit)).fetchall()
        return [(int(r[0]), str(r[1]), str(r[2]), str(r[3])) for r in rows]

    def search_summaries(
        self, needle: str, *, limit: int = 100, include_removed: bool = False
    ) -> list[tuple[str, str]]:
        """``(session_id, summary)`` for summaries containing ``needle`` — a removed
        session's left out unless ``include_removed``."""
        self.ensure_schema()
        if not needle.strip():
            return []
        where = "" if include_removed else _NOT_REMOVED
        sql = f"SELECT session_id, summary FROM conversation_summaries WHERE summary LIKE ? {where} LIMIT ?"  # noqa: S608 — a constant
        with sqlite_conn(self.db_path) as conn:
            rows = conn.execute(sql, (f"%{needle}%", limit)).fetchall()
        return [(str(r[0]), str(r[1])) for r in rows]

    # ------------------------------------------------------------------
    # Removals (ADR-0119) — the ledger; memory/removal.py holds the rules
    # ------------------------------------------------------------------

    _REMOVAL_COLUMNS = "id, kind, target_id, label, name_key, removed_at, cascade, permanent"

    @staticmethod
    def _removal_row(r: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "id": str(r[0]),
            "kind": str(r[1]),
            "target_id": str(r[2]),
            "label": str(r[3]),
            "name_key": r[4],
            "removed_at": str(r[5]),
            "cascade": json.loads(r[6] or "[]"),
            "permanent": bool(r[7]),
        }

    @with_locked_retry
    def add_removal(self, row: dict[str, Any]) -> None:
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            conn.execute(
                f"INSERT INTO memory_removals({self._REMOVAL_COLUMNS}) VALUES (?,?,?,?,?,?,?,?)",  # noqa: S608
                (
                    row["id"],
                    row["kind"],
                    row["target_id"],
                    row["label"],
                    row.get("name_key"),
                    row["removed_at"],
                    json.dumps(row.get("cascade") or []),
                    int(bool(row.get("permanent"))),
                ),
            )
            conn.commit()

    def removals(self) -> list[dict[str, Any]]:
        """Every ledger row, newest first."""
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT {self._REMOVAL_COLUMNS} FROM memory_removals "  # noqa: S608
                "ORDER BY removed_at DESC, id DESC"
            ).fetchall()
        return [self._removal_row(r) for r in rows]

    def get_removal(self, removal_id: str) -> dict[str, Any] | None:
        return next((r for r in self.removals() if r["id"] == removal_id), None)

    @with_locked_retry
    def delete_removal(self, removal_id: str) -> bool:
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM memory_removals WHERE id = ?", (removal_id,))
            conn.commit()
            return cursor.rowcount > 0

    @with_locked_retry
    def mark_removal_permanent(self, removal_id: str) -> None:
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            conn.execute("UPDATE memory_removals SET permanent = 1 WHERE id = ?", (removal_id,))
            conn.commit()

    def removed_session_ids(self) -> set[str]:
        """Sessions the owner removed: out of the Map, recall and the chat list."""
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT target_id FROM memory_removals WHERE kind = 'session'"
            ).fetchall()
        return {str(r[0]) for r in rows}

    def suppressed_name_keys(self) -> set[str]:
        """Folded names a summary mention must not bring back (removed names and
        removed or deleted entities)."""
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT name_key FROM memory_removals WHERE name_key IS NOT NULL"
            ).fetchall()
        return {str(r[0]) for r in rows}

    @with_locked_retry
    def delete_turns_by_id(self, ids: list[int]) -> int:
        if not ids:
            return 0
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            cursor = conn.executemany("DELETE FROM conversations WHERE id = ?", [(i,) for i in ids])
            conn.commit()
            return int(cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else len(ids))

    def vacuum(self) -> None:
        """Reclaim space after deletes (SQLite keeps the pages otherwise)."""
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            conn.execute("VACUUM")

    def fetch_learning_signals(self, filters: dict[str, Any] | None = None) -> list[LearningSignal]:
        """Retrieve learning signals with optional filters."""
        self.ensure_schema()
        query = "SELECT id, signal_type, domain, agent_type, query, context, outcome, improvement_hint, timestamp FROM learning_signals"
        params = []

        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(f"{key} = ?")
                params.append(value)
            query += " WHERE " + " AND ".join(conditions)

        with sqlite_conn(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return [
                LearningSignal(
                    id=row[0],
                    signal_type=row[1],
                    domain=row[2],
                    agent_type=row[3],
                    query=row[4],
                    context=row[5],
                    outcome=row[6],
                    improvement_hint=row[7],
                    timestamp=datetime.fromisoformat(row[8]),
                )
                for row in cursor.fetchall()
            ]

    # ------------------------------------------------------------------
    # Conversation persistence
    # ------------------------------------------------------------------

    @with_locked_retry
    def save_conversation_turns(
        self,
        session_id: str,
        turns: list[tuple[str, str]],
        *,
        assistant_origin: str | None = None,
    ) -> None:
        """Append (role, content) pairs for a session. Ignores empty lists.

        ``assistant_origin`` is the ``turn_origin`` stored on the assistant rows only
        (``"external"`` when the run read third-party text; None is unknown). A user row never
        carries one: the owner's own words are not provenance-tracked.
        """
        if not turns:
            return
        self.ensure_schema()
        now = datetime.now(UTC).isoformat()
        with sqlite_conn(self.db_path) as conn:
            conn.executemany(
                "INSERT INTO conversations(session_id, role, content, ts, turn_origin) "
                "VALUES(?, ?, ?, ?, ?)",
                [
                    (
                        session_id,
                        role,
                        content,
                        now,
                        assistant_origin if role == "assistant" else None,
                    )
                    for role, content in turns
                ],
            )
            conn.commit()

    def load_recent_turns(self, session_id: str, *, limit: int = 10) -> list[tuple[str, str]]:
        """Return the last ``limit`` (role, content) pairs for a session, oldest first."""
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT role, content FROM (
                    SELECT role, content, id
                    FROM conversations
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (session_id, limit),
            )
            return [(row[0], row[1]) for row in cursor.fetchall()]

    def load_recent_turns_with_origin(
        self, session_id: str, *, limit: int = 10
    ) -> list[tuple[str, str, str | None]]:
        """:meth:`load_recent_turns` with each row's ``turn_origin`` (None = unknown: a row
        written before the column, or by a producer that did not say)."""
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT role, content, turn_origin FROM (
                    SELECT role, content, turn_origin, id
                    FROM conversations
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (session_id, limit),
            )
            return [(row[0], row[1], row[2]) for row in cursor.fetchall()]

    def turn_origins(self, row_ids: Sequence[str | int]) -> dict[str, str | None]:
        """``turn_origin`` of each row id, one query. An id that is not a row (a Chroma id with
        no SQLite row) is left out of the result, which reads as unknown."""
        keys: list[int] = []
        for row_id in row_ids:
            try:
                keys.append(int(row_id))
            except (TypeError, ValueError):
                continue
        if not keys:
            return {}
        self.ensure_schema()
        marks = ", ".join("?" for _ in keys)
        sql = f"SELECT id, turn_origin FROM conversations WHERE id IN ({marks})"  # noqa: S608 - placeholders only
        with sqlite_conn(self.db_path) as conn:
            rows = conn.execute(sql, keys).fetchall()
        return {str(r[0]): r[1] for r in rows}

    @with_locked_retry
    def save_conversation_summary(self, session_id: str, summary: str) -> None:
        """Upsert the compacted summary for a session."""
        if not summary:
            return
        self.ensure_schema()
        now = datetime.now(UTC).isoformat()
        with sqlite_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO conversation_summaries(session_id, summary, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary=excluded.summary,
                    updated_at=excluded.updated_at
                """,
                (session_id, summary, now),
            )
            conn.commit()

    def conversation_row_ts(self, row_id: str | int) -> str | None:
        """The ``ts`` of one conversation row, or ``None`` when there is no such row."""
        try:
            key = int(row_id)
        except (TypeError, ValueError):
            return None
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            row = conn.execute("SELECT ts FROM conversations WHERE id = ?", (key,)).fetchone()
            return row[0] if row else None

    def load_conversation_summary(self, session_id: str) -> str:
        """Return the stored summary for a session, or empty string if none."""
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT summary FROM conversation_summaries WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            return row[0] if row else ""

    @with_locked_retry
    def save_conversation_turns_and_get_ids(
        self,
        session_id: str,
        turns: list[tuple[str, str]],
        *,
        assistant_origin: str | None = None,
    ) -> list[int]:
        """Append turns and return their assigned SQLite row IDs (same order as input).

        ``assistant_origin``: see :meth:`save_conversation_turns`."""
        if not turns:
            return []
        self.ensure_schema()
        now = datetime.now(UTC).isoformat()
        with sqlite_conn(self.db_path) as conn:
            conn.executemany(
                "INSERT INTO conversations(session_id, role, content, ts, turn_origin) "
                "VALUES(?, ?, ?, ?, ?)",
                [
                    (
                        session_id,
                        role,
                        content,
                        now,
                        assistant_origin if role == "assistant" else None,
                    )
                    for role, content in turns
                ],
            )
            last_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.commit()
        first_id = last_id - len(turns) + 1
        return list(range(first_id, last_id + 1))

    def load_turns_since(self, min_id: int = 0) -> list[tuple[int, str, str, str]]:
        """Return all turns with id > min_id as (id, session_id, role, content)."""
        self.ensure_schema()
        with sqlite_conn(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, session_id, role, content FROM conversations WHERE id > ? ORDER BY id ASC",
                (min_id,),
            )
            return [(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()]

    @with_locked_retry
    def delete_user_fact(
        self,
        key: str,
        *,
        source: str = "user:forget",
        reason: str = "forget",
        statement_id: str | None = None,
    ) -> bool:
        """Forget a user fact: its statement is retracted (reason ``forgot``).

        Returns True if a fact existed. The value survives in the retracted statement,
        so the history shows the forget and ``restore_user_fact`` can bring it back.
        ``statement_id`` forgets that one value of a key that holds several (one of
        two cards); without it, every current value of the key goes.
        """
        self.ensure_schema()
        return bool(self._facts().forget(key, statement_id=statement_id))

    @with_locked_retry
    def correct_user_fact(
        self,
        key: str,
        value: str,
        *,
        confidence: float = 1.0,
        source: str = "user:correction",
        reason: str = "correct",
        statement_id: str | None = None,
    ) -> bool:
        """Explicitly set a fact's value, overriding the confidence gate.

        Unlike ``upsert_user_fact`` (auto-capture, confidence-gated), a user correction
        always wins; the new statement carries reason ``corrected``, which the history
        reads back. Returns True if it replaced an existing value (vs. created new). A
        key with no property raises FactKeyError.
        """
        self.ensure_schema()
        now = datetime.now(UTC)
        prior = self._facts().get(key)
        fact = UserFact(
            key=key,
            value=value,
            confidence=confidence,
            source=source,
            first_seen=prior.first_seen if prior else now,
            last_confirmed=now,
            times_confirmed=(prior.times_confirmed + 1) if prior else 1,
            confirmed=prior.confirmed if prior else False,
        )
        if statement_id is not None and self._facts().replace(
            key, statement_id, fact, reason="corrected"
        ):
            return True
        self._facts().put(fact, force=True, reason="corrected")
        return prior is not None

    @with_locked_retry
    def restore_user_fact(self, key: str) -> str | None:
        """Undo the last change to a fact by re-activating its prior value.

        The prior value is the newest history event carrying one (the value active just
        before the last forget / correction / supersession); it is stored again with
        reason ``restored``. Returns it, or None if there is nothing to restore.
        """
        self.ensure_schema()
        facts = self._facts()
        previous = next((e for e in facts.history(key) if e.old_value is not None), None)
        if previous is None or previous.old_value is None:
            return None
        now = datetime.now(UTC)
        current = facts.get(key)
        facts.put(
            UserFact(
                key=key,
                value=previous.old_value,
                confidence=previous.old_confidence if previous.old_confidence is not None else 1.0,
                source="user:restore",
                first_seen=current.first_seen if current else now,
                last_confirmed=now,
                times_confirmed=1,
                confirmed=current.confirmed if current else False,
            ),
            force=True,
            reason="restored",
        )
        return previous.old_value

    def fetch_fact_history(self, key: str, *, limit: int = 50) -> list[FactHistoryEntry]:
        """How a fact changed, most recent first — read from its statement chain."""
        self.ensure_schema()
        return self._facts().history(key)[:limit]

    # ------------------------------------------------------------------
    # History retention (human-reviewed; never auto-deletes)
    # ------------------------------------------------------------------

    def count_fact_history(self) -> int:
        """Total number of history events (across all facts)."""
        self.ensure_schema()
        return len(self._facts().history())

    def fetch_history_retention_candidates(
        self, *, older_than_days: int = 180, limit: int = 1000
    ) -> list[FactHistoryEntry]:
        """History older than the retention window — the human review queue.

        OLDEST first. Only history of statements that no longer hold (superseded,
        forgotten): a current fact is never a pruning candidate. Deletes nothing.
        """
        self.ensure_schema()
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
        return self._facts().retention_candidates(older_than=cutoff, limit=limit)

    @with_locked_retry
    def prune_history_entries(self, ids: list[str]) -> int:
        """Delete owner-chosen history for good (statements that no longer hold only)."""
        self.ensure_schema()
        return self._facts().prune(ids)

    # ------------------------------------------------------------------
    # Contradiction review (detected same-key conflicts)
    # ------------------------------------------------------------------

    def fetch_contradictions(
        self, *, include_acknowledged: bool = False, limit: int = 200
    ) -> list[FactContradiction]:
        """Same-key value conflicts for review (most recent first), read from the chain."""
        self.ensure_schema()
        return self._facts().contradictions(include_reviewed=include_acknowledged, limit=limit)

    @with_locked_retry
    def acknowledge_contradictions(self, ids: list[str]) -> int:
        """Mark contradictions as reviewed (clears them from the default queue)."""
        if not ids:
            return 0
        self.ensure_schema()
        return self._facts().acknowledge(ids)


def now_utc() -> datetime:
    """Return current UTC timestamp for memory records."""
    return datetime.now(UTC)
