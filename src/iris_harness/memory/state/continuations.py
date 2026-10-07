"""ContinuationStore — who this conversation owes an answer to (ADR-0106).

A *continuation* records that some owner — an agent or a plugin intercept — put a
question to the user and is waiting for the reply. It is the state that was missing
when a research plan's "yes go head with the plan" was claimed by the file
organizer's approval intercept: every deterministic short-circuit could see that
*an* approvable thing existed, none could see *who had actually asked*.

Storage only. Policy — when to supersede, who may claim a turn, how an answer is
routed back — belongs to the registry in the core (M5.C2); this module just keeps
rows honest.

**Why a separate table, not extra columns on ``checkpoints``.** ADR-0106 sketched
these as "continuation columns" on the existing row. In the writing they do not fit:
a checkpoint is loop state keyed by ``(run_id, step_id)``, while a continuation is
conversational ownership keyed by ``session_id``, and a Tier-A continuation has no
run at all. Sharing the row would force synthetic run ids and make every existing
``list``/``get_latest``/``sweep_expired`` query filter out rows it never expected.
Same database file (ADR-0105: do not add stores), separate table, one honest shape
each. A Tier-B continuation points *at* a checkpoint through ``run_id``/``step_id``.

The one-pending-continuation-per-session invariant (ADR-0106 decision 5) is a
partial unique index, not a convention — a second ``open`` for a live session raises
``ContinuationConflictError`` rather than quietly creating an ambiguity, and the
registry supersedes first.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from iris_harness.foundation.persistence.sqlite import connect, ensure_columns
from iris_harness.memory.state.store import default_checkpoint_db_path

logger = logging.getLogger(__name__)

# A confirmation payload is a handful of fields describing one action, not a document.
# Capped for the same reason the checkpoint payload is: a governance table should not be
# where an unbounded blob ends up.
_MAX_PAYLOAD_BYTES = 64 * 1024

# ADR-0106 D3: lifetime follows the session, with the checkpoint TTL as the backstop.
# Resume re-grounds by restating the question, which is what makes a span this long
# safe; without that restatement the honest default would be hours.
DEFAULT_CONTINUATION_TTL = timedelta(days=7)

# ``choice`` is a question whose answers were enumerated when it was asked: "which of
# these did you mean? 1. ... 2. ...". The options ride in the payload, so a reply of
# "1" resolves to the thing that was actually on screen rather than to whatever a
# re-run of the search returns now.
CONTINUATION_KINDS = frozenset({"question", "approval", "choice"})
CONTINUATION_STATUSES = frozenset({"pending", "answered", "superseded", "expired"})


class ContinuationConflictError(RuntimeError):
    """A session already has a pending continuation (supersede it first)."""


@dataclass(frozen=True)
class Continuation:
    """One pending — or since-decided — question a session owes an answer to."""

    continuation_id: str
    session_id: str
    owner: str
    kind: str
    question: str
    intent: str
    run_id: str | None
    step_id: int | None
    status: str
    created_at: str
    decided_at: str | None
    expires_at: str
    # ADR-0108 follow-up — the payload seam. A `Continuation` records *who asked*; some
    # questions also carry *what happens if you say yes*. `_pending_confirmations` held
    # that second half in an in-memory dict keyed by session, which is the asymmetry
    # ADR-0106 decision 6 warned about pointing the other way: the answer was durable
    # and the thing it would execute died with the process, so a restart turned a
    # pending "approve" into a message that resolved nothing and routed on as if the
    # question had never been asked.
    #
    # `executor_kind` names the registered ConfirmationExecutor that runs it, so the
    # payload is opaque here — governance owns ownership and durability, never what the
    # action means.
    executor_kind: str | None = None
    payload: dict[str, Any] | None = None

    @property
    def is_executable(self) -> bool:
        """True when answering this does something, not just routes somewhere."""
        return bool(self.executor_kind)

    @property
    def choices(self) -> tuple[dict[str, Any], ...]:
        """The options a ``choice`` continuation offered, in the order shown."""
        if self.kind != "choice" or not self.payload:
            return ()
        raw = self.payload.get("choices")
        if not isinstance(raw, list):
            return ()
        return tuple(item for item in raw if isinstance(item, dict))

    @property
    def is_resumable_run(self) -> bool:
        """True when a halted loop run backs this (Tier B), not just a re-prompt."""
        return self.run_id is not None and self.step_id is not None


@dataclass
class ContinuationStore:
    """SQLite store for continuations, colocated with the checkpoint spine."""

    db_path: Path = field(default_factory=default_checkpoint_db_path)
    ttl: timedelta = DEFAULT_CONTINUATION_TTL

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Write ──────────────────────────────────────────────────────────────

    def open(
        self,
        *,
        session_id: str,
        owner: str,
        kind: str = "approval",
        question: str = "",
        intent: str = "",
        run_id: str | None = None,
        step_id: int | None = None,
        executor_kind: str | None = None,
        payload: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> Continuation:
        """Record that ``owner`` asked ``session_id`` something and is waiting.

        ``executor_kind`` + ``payload`` are the payload seam: the action to run if the
        answer is yes, and the name of the registered executor that knows how. The
        payload must be JSON-serialisable — that is the price of it outliving the
        process, and the reason it is checked here rather than discovered at read time,
        when the question has already been put to the user and the answer is on its way.

        A ``choice`` is the one kind whose payload is not an action: it carries the
        enumerated options (``{"choices": [...]}``) and nothing runs them, so it takes a
        payload *without* an executor and refuses one with. Every other payload still
        needs its executor.

        Raises :class:`ValueError` for an unknown kind, an empty session id, a payload
        that will not serialise, or a payload that does not fit its kind, and
        :class:`ContinuationConflictError` when the session already has one pending.
        """
        if not session_id:
            raise ValueError("a continuation must belong to a session")
        if kind not in CONTINUATION_KINDS:
            raise ValueError(f"unknown continuation kind: {kind!r}")
        if kind == "choice":
            choices = (payload or {}).get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("a choice continuation needs a non-empty 'choices' list")
            if executor_kind:
                raise ValueError("a choice continuation selects; it has no executor_kind")
        elif payload is not None and not executor_kind:
            raise ValueError("a payload needs an executor_kind to run it")
        payload_json = _encode_payload(payload)
        when = now or datetime.now(UTC)
        continuation_id = uuid.uuid4().hex[:12]
        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO continuations(
                        continuation_id, session_id, owner, kind, question, intent,
                        run_id, step_id, status, created_at, decided_at, expires_at,
                        executor_kind, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, NULL, ?, ?, ?)
                    """,
                    (
                        continuation_id,
                        session_id,
                        owner,
                        kind,
                        question,
                        intent,
                        run_id,
                        step_id,
                        when.isoformat(),
                        (when + self.ttl).isoformat(),
                        executor_kind,
                        payload_json,
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                raise ContinuationConflictError(
                    f"session {session_id} already has a pending continuation; "
                    "supersede it before opening another"
                ) from exc
        opened = self.get(continuation_id)
        assert opened is not None  # just written
        return opened

    def set_status(self, continuation_id: str, status: str, *, now: datetime | None = None) -> None:
        """Advance one continuation's lifecycle. Non-pending statuses stamp
        ``decided_at`` so a superseded question stays auditable rather than vanishing."""
        if status not in CONTINUATION_STATUSES:
            raise ValueError(f"unknown continuation status: {status!r}")
        stamp = None if status == "pending" else (now or datetime.now(UTC)).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE continuations SET status = ?, decided_at = ? WHERE continuation_id = ?",
                (status, stamp, continuation_id),
            )
            conn.commit()

    def supersede(self, session_id: str, *, now: datetime | None = None) -> int:
        """Mark this session's pending continuation superseded. Returns rows changed.

        The row is marked, never deleted: a question the user was asked and then had
        replaced is part of the conversation's record.
        """
        stamp = (now or datetime.now(UTC)).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE continuations SET status = 'superseded', decided_at = ? "
                "WHERE session_id = ? AND status = 'pending'",
                (stamp, session_id),
            )
            conn.commit()
        return int(cur.rowcount or 0)

    def expire_stale(self, *, now: datetime | None = None) -> int:
        """Expire pending continuations past their TTL. Returns rows changed."""
        when = now or datetime.now(UTC)
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE continuations SET status = 'expired', decided_at = ? "
                "WHERE status = 'pending' AND expires_at < ?",
                (when.isoformat(), when.isoformat()),
            )
            conn.commit()
        return int(cur.rowcount or 0)

    # ── Read ───────────────────────────────────────────────────────────────

    def pending_for(self, session_id: str, *, now: datetime | None = None) -> Continuation | None:
        """This session's open continuation, or None.

        Expires past-TTL rows **first**, so the TTL cannot be honoured on one surface
        and forgotten on another — the arrangement that let a months-dead organize
        plan still read as open to the chat path (PR #412).

        Never falls back to another session, however few are pending globally. That
        inference is the incident this whole design exists to remove: a continuation
        the user did not see in *this* conversation is not one they can be answering.
        """
        if not session_id:
            return None
        self.expire_stale(now=now)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM continuations WHERE session_id = ? AND status = 'pending'",
                (session_id,),
            ).fetchone()
        return _row_to_continuation(row) if row is not None else None

    def get(self, continuation_id: str) -> Continuation | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM continuations WHERE continuation_id = ?",
                (continuation_id,),
            ).fetchone()
        return _row_to_continuation(row) if row is not None else None

    def history_for(self, session_id: str) -> tuple[Continuation, ...]:
        """Every continuation this session has had, oldest first (for inspection)."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM continuations WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return tuple(_row_to_continuation(row) for row in rows)

    # ── Internal ───────────────────────────────────────────────────────────

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = connect(self.db_path)  # WAL + busy_timeout; the caller still commits
        try:
            try:
                os.chmod(self.db_path, 0o600)
            except OSError as exc:  # pragma: no cover - non-POSIX or perms issue
                logger.warning("could not chmod %s to 0o600: %s", self.db_path, exc)
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS continuations (
                    continuation_id TEXT PRIMARY KEY,
                    session_id      TEXT NOT NULL,
                    owner           TEXT NOT NULL,
                    kind            TEXT NOT NULL,
                    question        TEXT NOT NULL DEFAULT '',
                    intent          TEXT NOT NULL DEFAULT '',
                    run_id          TEXT,
                    step_id         INTEGER,
                    status          TEXT NOT NULL DEFAULT 'pending',
                    created_at      TEXT NOT NULL,
                    decided_at      TEXT,
                    expires_at      TEXT NOT NULL,
                    executor_kind   TEXT,
                    payload_json    TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_continuations_session
                    ON continuations(session_id, status);
                -- ADR-0106 decision 5, as a constraint rather than a convention:
                -- at most one pending continuation per session.
                CREATE UNIQUE INDEX IF NOT EXISTS idx_continuations_one_pending
                    ON continuations(session_id) WHERE status = 'pending';
                """)
            conn.commit()
        # A table created before the payload seam has neither column, and CREATE TABLE IF NOT
        # EXISTS leaves it that way. Both nullable, so every existing row stays valid and simply
        # answers "nothing to execute". On a connection of its own under BEGIN IMMEDIATE (#201).
        ensure_columns(
            self.db_path,
            "continuations",
            {"executor_kind": "TEXT", "payload_json": "TEXT"},
        )


def _row_to_continuation(row: sqlite3.Row) -> Continuation:
    return Continuation(
        continuation_id=str(row["continuation_id"]),
        session_id=str(row["session_id"]),
        owner=str(row["owner"]),
        kind=str(row["kind"]),
        question=str(row["question"]),
        intent=str(row["intent"]),
        run_id=None if row["run_id"] is None else str(row["run_id"]),
        step_id=None if row["step_id"] is None else int(row["step_id"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        decided_at=None if row["decided_at"] is None else str(row["decided_at"]),
        expires_at=str(row["expires_at"]),
        executor_kind=_optional_column(row, "executor_kind"),
        payload=_decode_payload(_optional_column(row, "payload_json")),
    )


def _optional_column(row: sqlite3.Row, name: str) -> str | None:
    """A column a pre-migration database may not have at all."""
    if name not in row.keys():
        return None
    value = row[name]
    return None if value is None else str(value)


def _encode_payload(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    try:
        encoded = json.dumps(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"continuation payload must be JSON-serialisable: {exc}") from exc
    if len(encoded.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"continuation payload is {len(encoded)} bytes, over the "
            f"{_MAX_PAYLOAD_BYTES}-byte cap"
        )
    return encoded


def _decode_payload(raw: str | None) -> dict[str, Any] | None:
    """Never raises: an undecodable payload costs the action, not the conversation."""
    if not raw:
        return None
    try:
        decoded = json.loads(raw)
    except ValueError:
        logger.warning("continuation payload is not decodable JSON; ignoring it")
        return None
    return decoded if isinstance(decoded, dict) else None


__all__ = [
    "CONTINUATION_KINDS",
    "CONTINUATION_STATUSES",
    "DEFAULT_CONTINUATION_TTL",
    "Continuation",
    "ContinuationConflictError",
    "ContinuationStore",
]
