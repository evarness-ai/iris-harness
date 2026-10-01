"""Email judgments: one row per judged email (loop-proof PR 5, plan D11).

The judge (``judge.py``) writes a row per non-promo email: its bucket, confidence, the
figures it read, the model and tier, and how long it took. An email the judge could not
reach yet (the Mac asleep, over the per-sweep cap) is a ``waiting`` row, judged first
next time. The owner's correction — a Gmail relabel, the Action Center card, chat or the
web list — writes ``owner_bucket`` on the same row (:meth:`JudgmentStore.correct`); the
row's **effective bucket** is the owner's when set, else the judge's. A row with no
correction after 7 days counts as accepted (the accuracy line, PR 8).

``label_bucket`` records the IRIS/* label last written to Gmail, so the label step
(``judge_labels.py``) writes only rows whose effective bucket moved.

The table lives in ``email.db`` beside ``emails`` (one message id space). The bucket
names come from ``judge.yaml``; the store keeps them as plain strings.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_harness.sdk.persistence import connect, data_path

# Row status. ``judged`` rows carry a bucket; ``waiting`` rows do not yet.
WAITING = "waiting"
JUDGED = "judged"

# Where a correction came from (``owner_source``).
CORRECTION_SOURCES = ("gmail", "card", "chat", "web", "cli")

# The owner may also say an email is promo: it loses its IRIS/* label and the sender's
# mail is not judged again (the promo track owns it).
PROMO = "promo"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS email_judgments (
    message_id     TEXT PRIMARY KEY,
    account_id     TEXT NOT NULL,
    status         TEXT NOT NULL,
    bucket         TEXT,
    confidence     REAL,
    fields         TEXT NOT NULL DEFAULT '{}',
    reason         TEXT NOT NULL DEFAULT '',
    model          TEXT NOT NULL DEFAULT '',
    tier           TEXT NOT NULL DEFAULT '',
    latency_ms     INTEGER,
    run_id         TEXT NOT NULL DEFAULT '',
    error          TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    judged_at      TEXT,
    owner_bucket   TEXT,
    owner_source   TEXT,
    corrected_at   TEXT,
    label_bucket   TEXT,
    labelled_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_email_judgments_status ON email_judgments(status, created_at);
CREATE INDEX IF NOT EXISTS idx_email_judgments_judged ON email_judgments(judged_at);
CREATE INDEX IF NOT EXISTS idx_email_judgments_corrected ON email_judgments(corrected_at);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


@dataclass(frozen=True)
class Judgment:
    """One row of ``email_judgments``."""

    message_id: str
    account_id: str
    status: str
    bucket: str | None = None
    confidence: float | None = None
    fields: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    model: str = ""
    tier: str = ""
    latency_ms: int | None = None
    run_id: str = ""
    error: str = ""
    created_at: str = ""
    judged_at: str | None = None
    owner_bucket: str | None = None
    owner_source: str | None = None
    corrected_at: str | None = None
    label_bucket: str | None = None
    labelled_at: str | None = None

    @property
    def effective_bucket(self) -> str | None:
        """The owner's bucket when corrected, else the judge's."""
        return self.owner_bucket or self.bucket

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Judgment:
        data = dict(row)
        data["fields"] = json.loads(data.get("fields") or "{}")
        return cls(**data)


@dataclass(frozen=True)
class Correction:
    """What :meth:`JudgmentStore.correct` changed (``previous`` is the effective bucket
    before it)."""

    judgment: Judgment
    previous: str | None
    changed: bool


@dataclass
class JudgmentStore:
    """CRUD over ``email_judgments`` in ``email.db``."""

    db_path: Path = field(default_factory=lambda: data_path("email.db"))

    def ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return connect(self.db_path, row_factory=sqlite3.Row)

    # -- writes ---------------------------------------------------------------

    def mark_waiting(self, account_id: str, message_ids: Iterable[str]) -> int:
        """Queue emails for judging. A message that already has a row is left alone.
        Returns how many rows were added."""
        now = _now()
        added = 0
        with self._connect() as conn:
            for mid in message_ids:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO email_judgments "
                    "(message_id, account_id, status, created_at) VALUES (?, ?, ?, ?)",
                    (mid, account_id, WAITING, now),
                )
                added += cur.rowcount
        return added

    def record(
        self,
        message_id: str,
        account_id: str,
        *,
        bucket: str,
        confidence: float | None,
        fields: dict[str, Any] | None = None,
        reason: str = "",
        model: str = "",
        tier: str = "",
        latency_ms: int | None = None,
        run_id: str = "",
        error: str = "",
    ) -> Judgment:
        """Write the judge's verdict (creating the row if it was never queued). An owner
        correction already on the row is kept."""
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO email_judgments (message_id, account_id, status, created_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(message_id) DO NOTHING",
                (message_id, account_id, WAITING, now),
            )
            conn.execute(
                "UPDATE email_judgments SET status = ?, bucket = ?, confidence = ?, "
                "fields = ?, reason = ?, model = ?, tier = ?, latency_ms = ?, run_id = ?, "
                "error = ?, judged_at = ? WHERE message_id = ?",
                (
                    JUDGED,
                    bucket,
                    confidence,
                    json.dumps(fields or {}, sort_keys=True),
                    reason,
                    model,
                    tier,
                    latency_ms,
                    run_id,
                    error,
                    now,
                    message_id,
                ),
            )
        judgment = self.get(message_id)
        assert judgment is not None
        return judgment

    def correct(self, message_id: str, bucket: str, *, source: str) -> Correction | None:
        """The owner's bucket for one email. ``None`` when the email has no row.
        Setting the bucket it already has changes nothing (``changed=False``)."""
        if source not in CORRECTION_SOURCES:
            raise ValueError(f"unknown correction source {source!r}")
        before = self.get(message_id)
        if before is None:
            return None
        previous = before.effective_bucket
        if previous == bucket:
            return Correction(judgment=before, previous=previous, changed=False)
        with self._connect() as conn:
            conn.execute(
                "UPDATE email_judgments SET owner_bucket = ?, owner_source = ?, "
                "corrected_at = ? WHERE message_id = ?",
                (bucket, source, _now(), message_id),
            )
        after = self.get(message_id)
        assert after is not None
        return Correction(judgment=after, previous=previous, changed=True)

    def mark_labelled(self, message_ids: Iterable[str], bucket: str | None) -> None:
        """Record the IRIS/* label now on these emails in Gmail (``None`` = none)."""
        now = _now()
        with self._connect() as conn:
            conn.executemany(
                "UPDATE email_judgments SET label_bucket = ?, labelled_at = ? "
                "WHERE message_id = ?",
                [(bucket, now, mid) for mid in message_ids],
            )

    # -- reads ----------------------------------------------------------------

    def get(self, message_id: str) -> Judgment | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM email_judgments WHERE message_id = ?", (message_id,)
            ).fetchone()
        return Judgment.from_row(row) if row else None

    def waiting(self, limit: int | None = None) -> list[Judgment]:
        """Rows still to judge, oldest first (waiting rows drain before new mail)."""
        sql = "SELECT * FROM email_judgments WHERE status = ? ORDER BY created_at, message_id"
        params: list[Any] = [WAITING]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            return [Judgment.from_row(r) for r in conn.execute(sql, params)]

    def count_waiting(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM email_judgments WHERE status = ?", (WAITING,)
            ).fetchone()
        return int(row[0])

    def labels_due(self, account_id: str | None = None) -> list[Judgment]:
        """Judged rows whose Gmail label differs from their effective bucket."""
        sql = (
            "SELECT * FROM email_judgments WHERE status = ? "
            "AND COALESCE(owner_bucket, bucket) IS NOT COALESCE(label_bucket, '') "
            "AND NOT (COALESCE(owner_bucket, bucket) = ? AND label_bucket IS NULL)"
        )
        params: list[Any] = [JUDGED, PROMO]
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            return [Judgment.from_row(r) for r in conn.execute(sql + " ORDER BY judged_at", params)]

    def judged_in_run(self, run_id: str) -> list[Judgment]:
        """Rows the judge wrote under ``run_id`` (``run_judge(run_id=...)``), oldest
        first: the durable record of what one run judged, however often it resumed."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM email_judgments WHERE status = ? AND run_id = ? "
                "ORDER BY judged_at, message_id",
                (JUDGED, run_id),
            )
            return [Judgment.from_row(r) for r in rows]

    def judged_between(self, start: datetime, end: datetime) -> list[Judgment]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM email_judgments WHERE status = ? AND judged_at >= ? "
                "AND judged_at < ? ORDER BY judged_at",
                (JUDGED, _iso(start), _iso(end)),
            )
            return [Judgment.from_row(r) for r in rows]

    def corrected_between(self, start: datetime, end: datetime) -> list[Judgment]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM email_judgments WHERE corrected_at >= ? AND corrected_at < ? "
                "ORDER BY corrected_at",
                (_iso(start), _iso(end)),
            )
            return [Judgment.from_row(r) for r in rows]

    def recent(
        self, *, bucket: str | None = None, limit: int = 50, since: datetime | None = None
    ) -> list[Judgment]:
        """Judged rows newest first, optionally by effective bucket."""
        sql = "SELECT * FROM email_judgments WHERE status = ?"
        params: list[Any] = [JUDGED]
        if bucket is not None:
            sql += " AND COALESCE(owner_bucket, bucket) = ?"
            params.append(bucket)
        if since is not None:
            sql += " AND judged_at >= ?"
            params.append(_iso(since))
        sql += " ORDER BY judged_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return [Judgment.from_row(r) for r in conn.execute(sql, params)]

    def owner_buckets_for_sender(self, from_address: str, limit: int = 5) -> list[tuple[str, str]]:
        """The owner's corrections for mail from this sender, newest first, as
        ``(subject, owner_bucket)`` — the judge's hints (the learning effect, D17 S2)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT e.subject, j.owner_bucket FROM email_judgments j "
                "JOIN emails e ON e.id = j.message_id "
                "WHERE j.owner_bucket IS NOT NULL AND lower(e.from_address) = lower(?) "
                "ORDER BY j.corrected_at DESC LIMIT ?",
                (from_address, limit),
            )
            return [(r[0] or "", r[1]) for r in rows]

    def is_promo_sender(self, from_address: str) -> bool:
        """The owner said mail from this sender is promo: it is not judged again."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM email_judgments j JOIN emails e ON e.id = j.message_id "
                "WHERE j.owner_bucket = ? AND lower(e.from_address) = lower(?) LIMIT 1",
                (PROMO, from_address),
            ).fetchone()
        return row is not None

    def missing(self, message_ids: Sequence[str]) -> list[str]:
        """The ids with no row yet."""
        if not message_ids:
            return []
        with self._connect() as conn:
            marks = ",".join("?" for _ in message_ids)
            have = {
                r[0]
                for r in conn.execute(
                    f"SELECT message_id FROM email_judgments WHERE message_id IN ({marks})",  # noqa: S608 — placeholders only
                    list(message_ids),
                )
            }
        return [m for m in message_ids if m not in have]
