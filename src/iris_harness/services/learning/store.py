"""SQLite-backed persistence for learning signals and experiments.

The store keeps two simple tables:

``signals``
    one row per observation emitted by ``LearningSignalCollector`` after every
    chat response (or whatever subsystem feeds it). Used by the ``learning_tick``
    heartbeat to compute baselines + measurements.

``experiments``
    one row per :class:`iris_harness.services.learning.models.Experiment` ever created, with full
    lifecycle (status, baseline, current_metric, etc.). Provides durability for
    autonomous experiments across process restarts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from iris_harness.foundation.persistence import connect
from iris_harness.foundation.persistence.sqlite import ensure_columns
from iris_harness.services.learning.models import Experiment, ExperimentStatus

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    source TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    success INTEGER NOT NULL,
    latency_ms REAL,
    metadata_json TEXT,
    session_id TEXT,
    turn_id TEXT,
    trace_id TEXT,
    span_id TEXT,
    resolved_tier TEXT,
    resolved_agent TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_metric_ts ON signals(metric_name, ts);

-- Durable, process-independent monotonic counters (e.g. signals_dropped_total).
-- The drop counter must survive even when a signal write itself fails, so it is
-- a separate single-row-per-name table written on its own guarded path.
CREATE TABLE IF NOT EXISTS meta_counters (
    name TEXT PRIMARY KEY,
    value INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    variant_description TEXT NOT NULL,
    config_changes_json TEXT NOT NULL,
    baseline_metric REAL NOT NULL,
    current_metric REAL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    evaluated_at TEXT,
    evaluation_window_hours INTEGER NOT NULL,
    rollback_config_json TEXT NOT NULL,
    strategy_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_strategy ON experiments(strategy_name);

-- Latest learning-analyst run (ADR-0069 #4 slice 2). A single replaceable row:
-- the analyst is advisory and re-runs on a heartbeat, so only the most recent
-- analysis matters (no per-recommendation ack/status workflow in v1).
CREATE TABLE IF NOT EXISTS learning_analysis (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload_json TEXT NOT NULL,
    generated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS behavior_pattern_proposals (
    pattern_id   TEXT PRIMARY KEY,
    text         TEXT NOT NULL,
    confidence   TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    status       TEXT NOT NULL DEFAULT 'pending',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_behavior_status
    ON behavior_pattern_proposals(status, created_at);

CREATE TABLE IF NOT EXISTS user_behavior_signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    subject     TEXT NOT NULL DEFAULT '',
    detail      TEXT NOT NULL DEFAULT '',
    session_id  TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_user_behavior_kind
    ON user_behavior_signals(kind, id);

CREATE TABLE IF NOT EXISTS intentions (
    intention_id    TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    summary         TEXT NOT NULL DEFAULT '',
    supporting_json TEXT NOT NULL DEFAULT '[]',
    status          TEXT NOT NULL DEFAULT 'proposed',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_intention_status
    ON intentions(status, created_at);
"""


@dataclass(frozen=True)
class BehaviorProposal:
    """A proposed recurring behavior pattern awaiting human review (HITL)."""

    pattern_id: str
    text: str
    confidence: str  # low | medium | high
    evidence: tuple[str, ...]
    status: str  # pending | approved | rejected
    created_at: str


@dataclass(frozen=True)
class IntentionProposal:
    """A proposed longitudinal goal ('what the user is working toward'), pending review."""

    intention_id: str
    title: str
    summary: str
    supporting: tuple[str, ...]
    status: str  # proposed | active | dismissed
    created_at: str


@dataclass(frozen=True)
class UserBehaviorSignal:
    """One observation of the user STEERING the assistant — confirming/dismissing a
    proposed habit, or correcting/forgetting a fact. Ground truth about the user, distinct
    from model-performance signals; the substrate for the longitudinal intention model."""

    id: int
    kind: str  # pattern_confirmed | pattern_dismissed | fact_corrected | fact_forgotten
    subject: str
    detail: str
    session_id: str
    created_at: str


@dataclass(frozen=True)
class SignalRecord:
    """Immutable snapshot of a stored signal row."""

    id: int
    ts: datetime
    source: str
    metric_name: str
    value: float
    success: bool
    latency_ms: float | None
    metadata: dict[str, Any]
    # Correlation + integrity fields (learning-observability.md §4.1). Nullable
    # so legacy rows and out-of-band callers keep working; populated by the
    # runtime so every signal is auditable back to its turn / trace.
    session_id: str | None = None
    turn_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    resolved_tier: str | None = None
    resolved_agent: str | None = None


@dataclass(frozen=True)
class MetricAggregate:
    """Aggregate statistics for a metric over a time window."""

    metric_name: str
    sample_count: int
    average: float
    success_rate: float
    latest_ts: datetime | None


class LearningMetricsStore:
    """SQLite store for learning signals + experiment lifecycle persistence."""

    def __init__(self, *, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    # Columns added after the original signals table shipped; ensure_schema()
    # ALTERs them in for pre-existing learning.db files (idempotent).
    _SIGNAL_MIGRATION_COLUMNS = (
        "session_id",
        "turn_id",
        "trace_id",
        "span_id",
        "resolved_tier",
        "resolved_agent",
    )

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        # Columns must exist before any index that references them, so a pre-L1 signals table is
        # migrated before the turn_id index is built. On a connection of its own under BEGIN
        # IMMEDIATE (#201): a read of ``table_info`` then an ``ALTER`` raised "duplicate column
        # name" for the process that lost a race to open an older learning.db.
        ensure_columns(
            self.db_path,
            "signals",
            dict.fromkeys(self._SIGNAL_MIGRATION_COLUMNS, "TEXT"),  # a fixed allow-list
            indexes=("CREATE INDEX IF NOT EXISTS idx_signals_turn ON signals(turn_id)",),
        )

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def record_signal(
        self,
        *,
        source: str,
        metric_name: str,
        value: float,
        success: bool,
        latency_ms: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        ts: datetime | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        resolved_tier: str | None = None,
        resolved_agent: str | None = None,
    ) -> int:
        moment = ts or datetime.now(UTC)
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO signals(ts, source, metric_name, value, success,"
                " latency_ms, metadata_json, session_id, turn_id, trace_id, span_id,"
                " resolved_tier, resolved_agent)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    moment.isoformat(),
                    source,
                    metric_name,
                    float(value),
                    1 if success else 0,
                    None if latency_ms is None else float(latency_ms),
                    json.dumps(dict(metadata or {}), sort_keys=True),
                    session_id,
                    turn_id,
                    trace_id,
                    span_id,
                    resolved_tier,
                    resolved_agent,
                ),
            )
            inserted_id = int(cursor.lastrowid or 0)
            return inserted_id

    _SIGNAL_COLUMNS = (
        "id, ts, source, metric_name, value, success, latency_ms, metadata_json,"
        " session_id, turn_id, trace_id, span_id, resolved_tier, resolved_agent"
    )

    def recent_signals(self, *, metric_name: str, limit: int = 100) -> list[SignalRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {self._SIGNAL_COLUMNS} FROM signals WHERE metric_name = ?"  # noqa: S608 — column list is a fixed constant
                " ORDER BY ts DESC LIMIT ?",
                (metric_name, int(limit)),
            ).fetchall()
        return [_row_to_signal(row) for row in rows]

    def signals_in_window(
        self,
        *,
        metric_names: Sequence[str],
        window: timedelta | None = None,
        now: datetime | None = None,
    ) -> list[SignalRecord]:
        """Return full signal rows for the given metrics within a recent window.

        The read substrate for the learning-intelligence layer: it groups these
        rows in Python (by ``metadata['intent']`` + ``resolved_tier``) to build
        the per-cell outcome matrix and the escalation-judge accuracy figures,
        which a flat ``aggregate_metric`` can't express. Newest first.
        """
        names = list(metric_names)
        if not names:
            return []
        moment = now or datetime.now(UTC)
        placeholders = ",".join("?" for _ in names)
        params: list[Any] = list(names)
        clause = f"WHERE metric_name IN ({placeholders})"
        if window is not None:
            clause += " AND ts >= ?"
            params.append((moment - window).isoformat())
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {self._SIGNAL_COLUMNS} FROM signals "  # noqa: S608 — columns + clause are fixed/allow-listed
                + clause
                + " ORDER BY ts DESC",
                params,
            ).fetchall()
        return [_row_to_signal(row) for row in rows]

    def aggregate_metric(
        self,
        *,
        metric_name: str,
        window: timedelta | None = None,
        now: datetime | None = None,
    ) -> MetricAggregate:
        moment = now or datetime.now(UTC)
        params: list[Any] = [metric_name]
        clause = "WHERE metric_name = ?"
        if window is not None:
            clause += " AND ts >= ?"
            params.append((moment - window).isoformat())
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*), AVG(value), AVG(success), MAX(ts) FROM signals "  # noqa: S608 — clause is composed from a fixed allow-list
                + clause,
                params,
            ).fetchone()
        sample_count = int(row[0] or 0)
        average = float(row[1]) if row[1] is not None else 0.0
        success_rate = float(row[2]) if row[2] is not None else 0.0
        latest_ts = _parse_ts(row[3]) if row[3] else None
        return MetricAggregate(
            metric_name=metric_name,
            sample_count=sample_count,
            average=average,
            success_rate=success_rate,
            latest_ts=latest_ts,
        )

    # ------------------------------------------------------------------
    # Meta counters + health (learning-observability.md §4.1, §4.4)
    # ------------------------------------------------------------------

    def increment_counter(self, name: str, *, by: int = 1) -> int:
        """Atomically bump a durable monotonic counter and return the new value."""
        moment = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO meta_counters(name, value, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(name) DO UPDATE SET"
                "   value = value + excluded.value, updated_at = excluded.updated_at",
                (name, int(by), moment),
            )
            row = conn.execute("SELECT value FROM meta_counters WHERE name = ?", (name,)).fetchone()
        return int(row[0]) if row else 0

    def counter(self, name: str) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta_counters WHERE name = ?", (name,)).fetchone()
        return int(row[0]) if row else 0

    def all_counters(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT name, value FROM meta_counters").fetchall()
        return {str(name): int(value) for name, value in rows}

    def metric_volume(
        self, *, window: timedelta | None = None, now: datetime | None = None
    ) -> dict[str, int]:
        """Signal count per ``metric_name`` (optionally within a recent window).

        ``now`` is injectable for deterministic tests — the window is measured back
        from it, not from the wall clock (so fixed-timestamp fixtures don't expire).
        """
        params: list[Any] = []
        clause = ""
        if window is not None:
            clause = " WHERE ts >= ?"
            params.append(((now or datetime.now(UTC)) - window).isoformat())
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT metric_name, COUNT(*) FROM signals"  # noqa: S608 — clause is a fixed literal
                + clause
                + " GROUP BY metric_name",
                params,
            ).fetchall()
        return {str(name): int(count) for name, count in rows}

    def experiment_status_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM experiments GROUP BY status"
            ).fetchall()
        return {str(status): int(count) for status, count in rows}

    def health_summary(
        self, *, window: timedelta | None = None, now: datetime | None = None
    ) -> dict[str, Any]:
        """One-call meta-observability view: 'is self-learning working?'.

        Combines persisted drop counters, per-metric signal volume, and the
        experiment lifecycle tally so an operator can answer the question
        without reading the DB (learning-observability.md §4.4, D2). ``now`` is
        injectable so the window is measured from a fixed moment in tests.
        """
        counters = self.all_counters()
        volume = self.metric_volume(window=window, now=now)
        recorded = sum(volume.values())
        dropped = int(counters.get("signals_dropped_total", 0))
        total = recorded + dropped
        return {
            "signal_volume": volume,
            "signals_recorded_total": recorded,
            "signals_dropped_total": dropped,
            "drop_rate": (dropped / total) if total else 0.0,
            "counters": counters,
            "experiments": self.experiment_status_counts(),
        }

    # ------------------------------------------------------------------
    # Learning-analyst snapshot (ADR-0069 #4 slice 2)
    # ------------------------------------------------------------------

    def save_analysis(self, payload: dict[str, Any]) -> None:
        """Persist the latest learning-analyst run, replacing any prior one."""
        generated_at = str(payload.get("generated_at") or datetime.now(UTC).isoformat())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO learning_analysis(id, payload_json, generated_at)"
                " VALUES (1, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET"
                "   payload_json = excluded.payload_json,"
                "   generated_at = excluded.generated_at",
                (json.dumps(payload, sort_keys=True), generated_at),
            )

    def latest_analysis(self) -> dict[str, Any] | None:
        """Return the most recent learning-analyst payload, or None if never run."""
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM learning_analysis WHERE id = 1").fetchone()
        if not row:
            return None
        try:
            parsed = json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    # ------------------------------------------------------------------
    # Behavior-pattern proposals (HITL review queue; propose-only)
    # ------------------------------------------------------------------

    def propose_behavior_pattern(
        self, pattern_id: str, text: str, confidence: str, evidence: list[str]
    ) -> bool:
        """Record a proposed behavior pattern for review. No-op if already present
        (same pattern_id) so re-mining the same habit doesn't flood the queue.
        Returns True if a new proposal was inserted.
        """
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO behavior_pattern_proposals(pattern_id, text, confidence,"
                " evidence_json, status, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, 'pending', ?, ?)"
                " ON CONFLICT(pattern_id) DO NOTHING",
                (pattern_id, text, confidence, json.dumps(evidence), now, now),
            )
            return cur.rowcount > 0

    def list_behavior_proposals(self, *, status: str = "pending") -> list[BehaviorProposal]:
        """Return behavior-pattern proposals with the given status (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT pattern_id, text, confidence, evidence_json, status, created_at"
                " FROM behavior_pattern_proposals WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        out: list[BehaviorProposal] = []
        for r in rows:
            try:
                evidence = tuple(json.loads(r[3]))
            except (json.JSONDecodeError, TypeError):
                evidence = ()
            out.append(
                BehaviorProposal(
                    pattern_id=r[0],
                    text=r[1],
                    confidence=r[2],
                    evidence=evidence,
                    status=r[4],
                    created_at=r[5],
                )
            )
        return out

    def get_behavior_proposal(self, pattern_id: str) -> BehaviorProposal | None:
        with self._connect() as conn:
            r = conn.execute(
                "SELECT pattern_id, text, confidence, evidence_json, status, created_at"
                " FROM behavior_pattern_proposals WHERE pattern_id = ?",
                (pattern_id,),
            ).fetchone()
        if not r:
            return None
        try:
            evidence = tuple(json.loads(r[3]))
        except (json.JSONDecodeError, TypeError):
            evidence = ()
        return BehaviorProposal(
            pattern_id=r[0],
            text=r[1],
            confidence=r[2],
            evidence=evidence,
            status=r[4],
            created_at=r[5],
        )

    def resolve_behavior_proposal(self, pattern_id: str, status: str) -> bool:
        """Set a proposal's status (e.g. 'approved' / 'rejected'). Returns True if it existed."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE behavior_pattern_proposals SET status = ?, updated_at = ?"
                " WHERE pattern_id = ?",
                (status, now, pattern_id),
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # User-behavior signals (the user steering the assistant)
    # ------------------------------------------------------------------

    def record_user_behavior_signal(
        self, kind: str, subject: str = "", detail: str = "", session_id: str = ""
    ) -> None:
        """Append one user-steering signal. Best-effort; never raises into the caller."""
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO user_behavior_signals(kind, subject, detail, session_id, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (kind, subject, detail, session_id, datetime.now(UTC).isoformat()),
                )
        except Exception:  # signal capture must never break a user action
            logger.debug("user-behavior signal capture failed (kind=%s)", kind, exc_info=True)

    def list_user_behavior_signals(
        self, *, kind: str | None = None, limit: int = 100
    ) -> list[UserBehaviorSignal]:
        """Return recent user-behavior signals (newest first), optionally filtered by kind."""
        sql = "SELECT id, kind, subject, detail, session_id, created_at FROM user_behavior_signals"
        params: list[Any] = []
        if kind:
            sql += " WHERE kind = ?"
            params.append(kind)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            UserBehaviorSignal(
                id=r[0], kind=r[1], subject=r[2], detail=r[3], session_id=r[4], created_at=r[5]
            )
            for r in rows
        ]

    def user_behavior_summary(self) -> dict[str, int]:
        """Counts of user-behavior signals by kind — a compact view of how the user steers."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT kind, COUNT(*) FROM user_behavior_signals GROUP BY kind ORDER BY 2 DESC"
            ).fetchall()
        return {r[0]: int(r[1]) for r in rows}

    # ------------------------------------------------------------------
    # Intentions (longitudinal goals; HITL review queue, propose-only)
    # ------------------------------------------------------------------

    def propose_intention(
        self, intention_id: str, title: str, summary: str, supporting: list[str]
    ) -> bool:
        """Record a proposed intention. No-op if the id is already present (any status),
        so re-rolling the same goal doesn't flood the queue. Returns True if inserted."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO intentions(intention_id, title, summary, supporting_json,"
                " status, created_at, updated_at) VALUES (?, ?, ?, ?, 'proposed', ?, ?)"
                " ON CONFLICT(intention_id) DO NOTHING",
                (intention_id, title, summary, json.dumps(supporting), now, now),
            )
            return cur.rowcount > 0

    def list_intentions(self, *, status: str = "proposed") -> list[IntentionProposal]:
        """Return intentions with the given status (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT intention_id, title, summary, supporting_json, status, created_at"
                " FROM intentions WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        return [self._intention_row(r) for r in rows]

    def get_intention(self, intention_id: str) -> IntentionProposal | None:
        with self._connect() as conn:
            r = conn.execute(
                "SELECT intention_id, title, summary, supporting_json, status, created_at"
                " FROM intentions WHERE intention_id = ?",
                (intention_id,),
            ).fetchone()
        return self._intention_row(r) if r else None

    def resolve_intention(self, intention_id: str, status: str) -> bool:
        """Set an intention's status ('active' / 'dismissed'). Returns True if it existed."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE intentions SET status = ?, updated_at = ? WHERE intention_id = ?",
                (status, now, intention_id),
            )
            return cur.rowcount > 0

    @staticmethod
    def _intention_row(r: tuple[Any, ...]) -> IntentionProposal:
        try:
            supporting = tuple(json.loads(r[3]))
        except (json.JSONDecodeError, TypeError):
            supporting = ()
        return IntentionProposal(
            intention_id=r[0],
            title=r[1],
            summary=r[2],
            supporting=supporting,
            status=r[4],
            created_at=r[5],
        )

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def upsert_experiment(
        self,
        experiment: Experiment,
        *,
        strategy_name: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO experiments(id, domain, hypothesis, variant_description,"
                " config_changes_json, baseline_metric, current_metric, status,"
                " created_at, started_at, evaluated_at, evaluation_window_hours,"
                " rollback_config_json, strategy_name)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET"
                "   domain=excluded.domain,"
                "   hypothesis=excluded.hypothesis,"
                "   variant_description=excluded.variant_description,"
                "   config_changes_json=excluded.config_changes_json,"
                "   baseline_metric=excluded.baseline_metric,"
                "   current_metric=excluded.current_metric,"
                "   status=excluded.status,"
                "   started_at=excluded.started_at,"
                "   evaluated_at=excluded.evaluated_at,"
                "   evaluation_window_hours=excluded.evaluation_window_hours,"
                "   rollback_config_json=excluded.rollback_config_json,"
                "   strategy_name=COALESCE(excluded.strategy_name, strategy_name)",
                (
                    experiment.id,
                    experiment.domain,
                    experiment.hypothesis,
                    experiment.variant_description,
                    json.dumps(experiment.config_changes, sort_keys=True),
                    float(experiment.baseline_metric),
                    None if experiment.current_metric is None else float(experiment.current_metric),
                    experiment.status,
                    experiment.created_at.isoformat(),
                    experiment.started_at.isoformat() if experiment.started_at else None,
                    experiment.evaluated_at.isoformat() if experiment.evaluated_at else None,
                    int(experiment.evaluation_window_hours),
                    json.dumps(experiment.rollback_config, sort_keys=True),
                    strategy_name,
                ),
            )

    def list_experiments(
        self,
        *,
        status: ExperimentStatus | None = None,
        strategy_name: str | None = None,
    ) -> list[Experiment]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if strategy_name is not None:
            clauses.append("strategy_name = ?")
            params.append(strategy_name)
        sql = (
            "SELECT id, domain, hypothesis, variant_description, config_changes_json,"
            " baseline_metric, current_metric, status, created_at, started_at,"
            " evaluated_at, evaluation_window_hours, rollback_config_json"
            " FROM experiments"
        )
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_experiment(row) for row in rows]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def _row_to_signal(row: tuple[Any, ...]) -> SignalRecord:
    return SignalRecord(
        id=int(row[0]),
        ts=_parse_ts(row[1]) or datetime.now(UTC),
        source=str(row[2]),
        metric_name=str(row[3]),
        value=float(row[4]),
        success=bool(row[5]),
        latency_ms=None if row[6] is None else float(row[6]),
        metadata=json.loads(row[7]) if row[7] else {},
        session_id=None if row[8] is None else str(row[8]),
        turn_id=None if row[9] is None else str(row[9]),
        trace_id=None if row[10] is None else str(row[10]),
        span_id=None if row[11] is None else str(row[11]),
        resolved_tier=None if row[12] is None else str(row[12]),
        resolved_agent=None if row[13] is None else str(row[13]),
    )


def _row_to_experiment(row: tuple[Any, ...]) -> Experiment:
    return Experiment(
        id=str(row[0]),
        domain=str(row[1]),
        hypothesis=str(row[2]),
        variant_description=str(row[3]),
        config_changes=json.loads(row[4]) if row[4] else {},
        baseline_metric=float(row[5]),
        current_metric=None if row[6] is None else float(row[6]),
        status=str(row[7]),  # type: ignore[arg-type]
        created_at=_parse_ts(row[8]) or datetime.now(UTC),
        started_at=_parse_ts(row[9]),
        evaluated_at=_parse_ts(row[10]),
        evaluation_window_hours=int(row[11]),
        rollback_config=json.loads(row[12]) if row[12] else {},
    )


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(str(value))
