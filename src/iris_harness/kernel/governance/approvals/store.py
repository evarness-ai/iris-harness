"""ApprovalStore — SQLite-backed approval queue (design §11.1)."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from iris_harness.foundation.paths import governance_data_dir

logger = logging.getLogger(__name__)

# Override slot (``None``: resolved from ``IRIS_HOME`` on every use, never frozen at
# import -- a process that relocates the home after importing IRIS writes into the
# new one). Set it to point the default somewhere else outright (tests).
DEFAULT_APPROVALS_DB_PATH: Path | None = None


def default_approvals_db_path() -> Path:
    """``DEFAULT_APPROVALS_DB_PATH`` when set, else ``<governance data dir>/approvals.db``."""
    return (
        DEFAULT_APPROVALS_DB_PATH
        if DEFAULT_APPROVALS_DB_PATH is not None
        else governance_data_dir() / "approvals.db"
    )


_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class ApprovalNotFoundError(LookupError):
    """No approval row with the given ID exists."""


class ApprovalAlreadyAnsweredError(ValueError):
    """respond() called on a row that is not pending."""


ApprovalId = str  # UUID str


@dataclass(frozen=True)
class ApprovalItem:
    """One tool call an approval pins (ADR-0118 decision 2).

    The arguments are held as canonical JSON (sorted keys), so "is this the call that
    was approved?" is an exact string comparison and nothing about the item can change
    after the row is written.
    """

    tool: str
    args_json: str

    @classmethod
    def of(cls, tool: str, args: dict[str, Any]) -> ApprovalItem:
        return cls(tool=tool, args_json=json.dumps(args, sort_keys=True, default=str))

    @property
    def args(self) -> dict[str, Any]:
        value = json.loads(self.args_json)
        return value if isinstance(value, dict) else {}

    def render(self) -> str:
        return f"{self.tool} {self.args_json}"


@dataclass(frozen=True)
class ApprovalCard:
    """How a destructive-tool approval reads to the owner (ADR-0118 step 4).

    The plugin's ``describe`` output (title + one line per item), the declared undo
    tool and window, and what the owner asked for. Frozen with the row: the card the
    owner approves is the card that was written, never re-derived later.
    """

    title: str
    lines: tuple[str, ...] = ()
    undo_tool: str | None = None
    undo_window_days: int | None = None
    asked: str | None = None
    # What approving does: ``destructive`` (data loss) or ``write`` — a write declared
    # ``approval: pinned``, such as sending an email (ADR-0118 amendment). Rows written
    # before the amendment have no value and are destructive.
    effect: str = "destructive"

    def undo_sentence(self) -> str:
        if self.undo_tool is None:
            return "This cannot be undone."
        if self.undo_window_days:
            return f"Reversible for {self.undo_window_days} days (undo: {self.undo_tool})."
        return f"Reversible (undo: {self.undo_tool})."

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "lines": list(self.lines),
            "undo_tool": self.undo_tool,
            "undo_window_days": self.undo_window_days,
            "asked": self.asked,
            "effect": self.effect,
        }


def _card_to_json(card: ApprovalCard | None) -> str | None:
    return None if card is None else json.dumps(card.to_dict())


def _card_from_json(raw: str | None) -> ApprovalCard | None:
    if raw is None:
        return None
    try:
        d = json.loads(raw)
        days = d.get("undo_window_days")
        return ApprovalCard(
            title=str(d["title"]),
            lines=tuple(str(line) for line in d.get("lines") or ()),
            undo_tool=str(d["undo_tool"]) if d.get("undo_tool") else None,
            undo_window_days=int(days) if days else None,
            asked=str(d["asked"]) if d.get("asked") else None,
            effect="write" if d.get("effect") == "write" else "destructive",
        )
    except (ValueError, KeyError, TypeError):
        logger.warning("approval card did not decode; showing the raw call instead")
        return None


def _items_to_json(items: tuple[ApprovalItem, ...] | None) -> str | None:
    if items is None:
        return None
    return json.dumps([{"tool": i.tool, "args_json": i.args_json} for i in items])


def _items_from_json(raw: str | None) -> tuple[ApprovalItem, ...] | None:
    if raw is None:
        return None
    try:
        decoded = json.loads(raw)
        return tuple(
            ApprovalItem(tool=str(d["tool"]), args_json=str(d["args_json"])) for d in decoded
        )
    except (ValueError, KeyError, TypeError):
        # An unreadable item list can authorise nothing: an empty tuple matches no call.
        logger.warning("approval items did not decode; treating as none approved")
        return ()


@dataclass(frozen=True)
class ApprovalRow:
    approval_id: str
    run_id: str
    checkpoint_id: str | None
    signal: str
    context_summary: str
    requested_at: str
    channel: str
    status: str
    responded_at: str | None
    response_actor: str | None
    timeout_at: str
    policy_on_timeout: str
    # The conversation this approval belongs to. Needed to tell the user when it
    # lapses: a notice has to land somewhere, and deriving the session from the
    # nullable `checkpoint_id` pointer only works for approvals that got linked.
    # Nullable, so rows written before this column stay valid — ADR-0106 C1 added
    # `session_id` to `checkpoints` for the same reason and in the same way.
    session_id: str | None = None
    # The tool calls a destructive-tool approval pins (ADR-0118). ``None`` for every
    # other approval (evaluator halts), which is also how a row says which kind it is.
    items: tuple[ApprovalItem, ...] | None = None
    # How it reads to the owner (ADR-0118 step 4); None when there is nothing better
    # than the raw call, and for every evaluator approval.
    card: ApprovalCard | None = None
    # Who asked, for a call made from code (``plugin:<name>``, ``core:<workflow>``) that
    # the harness runs itself once the owner approves (plugin-capabilities decision 1).
    # None for a loop's approval, which a resumed run executes instead.
    caller: str | None = None
    # When the executor claimed the approved call to run it. Set once, atomically
    # (``claim_execution``), so an approved call runs at most once.
    executed_at: str | None = None
    # The call id (ULID) of the HELD attempt that raised this approval (#134). The approved
    # re-execution is a new call with its own id and records this one as ``held_call_id``,
    # so both attempts join. None for a row written before the column, and for an approval
    # no tool call raised (evaluator halts).
    call_id: str | None = None

    @property
    def is_deferred_call(self) -> bool:
        """A code caller's pinned call, run by the harness on approval (no run resumes)."""
        return self.caller is not None and self.items is not None

    def lapse_consequence(self) -> str:
        """What an unanswered lapse leaves behind, in words every channel's notice uses.

        A loop's approval halted a run, which stays halted; a code caller's approval
        halted nothing — its call simply never runs — so saying "the run stays halted"
        about one would describe a run that does not exist.
        """
        if self.is_deferred_call:
            tool = self.items[0].tool if self.items else "the call"
            return f"the call to {tool} (asked by {self.caller}) was not run"
        return "the run stays halted"


class ApprovalStore:
    """Open-or-create the approvals DB; enforces immutability at the API level."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or default_approvals_db_path()
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
            except OSError as exc:  # pragma: no cover - non-POSIX or perms issue
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
            # A database written before the timeout notice has no session_id, and
            # CREATE TABLE IF NOT EXISTS leaves it that way. Nullable, so pre-existing
            # rows stay valid and simply answer "no session" — they get the channel
            # notice when they lapse and no in-chat one.
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(approval_queue)")}
            if "session_id" not in columns:
                conn.execute("ALTER TABLE approval_queue ADD COLUMN session_id TEXT")
            # ADR-0118: the pinned calls of a destructive-tool approval. Nullable for
            # the same reason: every older row is an evaluator approval and has none.
            if "items_json" not in columns:
                conn.execute("ALTER TABLE approval_queue ADD COLUMN items_json TEXT")
            if "card_json" not in columns:
                conn.execute("ALTER TABLE approval_queue ADD COLUMN card_json TEXT")
            # Plugin-capabilities decision 1: a code caller's approved call. Nullable for
            # the same reason again: no older row is one.
            if "caller" not in columns:
                conn.execute("ALTER TABLE approval_queue ADD COLUMN caller TEXT")
            if "executed_at" not in columns:
                conn.execute("ALTER TABLE approval_queue ADD COLUMN executed_at TEXT")
            # #134: the id of the held call attempt. Nullable (older rows have none).
            # Several processes open this file; the one that loses the race to ALTER
            # sees "duplicate column name", which is the state it wanted.
            if "call_id" not in columns:
                try:
                    conn.execute("ALTER TABLE approval_queue ADD COLUMN call_id TEXT")
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc).lower():
                        raise
            conn.commit()

    @staticmethod
    def _row(r: sqlite3.Row) -> ApprovalRow:
        return ApprovalRow(
            approval_id=r["approval_id"],
            run_id=r["run_id"],
            checkpoint_id=r["checkpoint_id"],
            signal=r["signal"],
            context_summary=r["context_summary"],
            requested_at=r["requested_at"],
            channel=r["channel"],
            status=r["status"],
            responded_at=r["responded_at"],
            response_actor=r["response_actor"],
            timeout_at=r["timeout_at"],
            policy_on_timeout=r["policy_on_timeout"],
            session_id=(r["session_id"] if "session_id" in r.keys() else None),
            items=_items_from_json(r["items_json"] if "items_json" in r.keys() else None),
            card=_card_from_json(r["card_json"] if "card_json" in r.keys() else None),
            caller=(r["caller"] if "caller" in r.keys() else None),
            executed_at=(r["executed_at"] if "executed_at" in r.keys() else None),
            call_id=(r["call_id"] if "call_id" in r.keys() else None),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        run_id: str,
        checkpoint_id: str | None,
        signal: str,
        context_summary: str,
        *,
        channel: str = "cli",
        timeout_minutes: int = 10,
        policy_on_timeout: str = "fail_closed",
        session_id: str | None = None,
        items: tuple[ApprovalItem, ...] | None = None,
        card: ApprovalCard | None = None,
        caller: str | None = None,
        call_id: str | None = None,
    ) -> ApprovalId:
        approval_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        timeout_at = now + timedelta(minutes=timeout_minutes)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO approval_queue
                    (approval_id, run_id, checkpoint_id, signal, context_summary,
                     requested_at, channel, status, timeout_at, policy_on_timeout,
                     session_id, items_json, card_json, caller, call_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    run_id,
                    checkpoint_id,
                    signal,
                    context_summary,
                    now.isoformat(),
                    channel,
                    timeout_at.isoformat(),
                    policy_on_timeout,
                    session_id,
                    _items_to_json(items),
                    _card_to_json(card),
                    caller,
                    call_id,
                ),
            )
            conn.commit()
        logger.info("enqueued approval %s for run %s signal=%s", approval_id, run_id, signal)
        return approval_id

    def get(self, approval_id: str) -> ApprovalRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM approval_queue WHERE approval_id = ?", (approval_id,)
            ).fetchone()
        return self._row(row) if row else None

    def list_pending(self, due_only: bool = False) -> list[ApprovalRow]:
        now_iso = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            if due_only:
                rows = conn.execute(
                    "SELECT * FROM approval_queue WHERE status='pending' AND timeout_at <= ?"
                    " ORDER BY requested_at",
                    (now_iso,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM approval_queue WHERE status='pending' ORDER BY requested_at"
                ).fetchall()
        return [self._row(r) for r in rows]

    def pending_for_run(self, run_id: str) -> ApprovalRow | None:
        """The approval still gating ``run_id``, or None. Read-only.

        Asked by anything about to continue a halted run: a resume that ignored an
        unanswered approval would step straight past the human the evaluator stopped
        the run to consult, which is the whole point of the queue.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM approval_queue WHERE run_id = ? AND status = 'pending'"
                " ORDER BY requested_at DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        return self._row(row) if row is not None else None

    def list_by_status(self, status: str) -> list[ApprovalRow]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM approval_queue WHERE status = ? ORDER BY requested_at",
                (status,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def respond(
        self,
        approval_id: str,
        *,
        status: str,
        actor: str,
    ) -> ApprovalRow:
        if status not in ("approved", "rejected"):
            raise ValueError(f"status must be 'approved' or 'rejected', got {status!r}")
        now_iso = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            # Conditional on ``pending``, and checked by rowcount: two answers racing from
            # two processes (the API and a CLI) cannot both land, so nothing an approval
            # gates can be set off twice by one decision.
            cursor = conn.execute(
                """
                UPDATE approval_queue
                   SET status = ?, responded_at = ?, response_actor = ?
                 WHERE approval_id = ? AND status = 'pending'
                """,
                (status, now_iso, actor, approval_id),
            )
            conn.commit()
            if cursor.rowcount != 1:
                row = conn.execute(
                    "SELECT status FROM approval_queue WHERE approval_id = ?", (approval_id,)
                ).fetchone()
                if row is None:
                    raise ApprovalNotFoundError(approval_id)
                raise ApprovalAlreadyAnsweredError(
                    f"approval {approval_id!r} already has status {row['status']!r}"
                )
            updated = conn.execute(
                "SELECT * FROM approval_queue WHERE approval_id = ?", (approval_id,)
            ).fetchone()
        return self._row(updated)

    def claim_execution(self, approval_id: str) -> ApprovalRow | None:
        """Claim an approved call to run it; the row, or None when it may not run now.

        One conditional write: the row must be ``approved`` and not claimed before. Only
        one claimant wins however many race, so an approved call runs at most once — a
        second approve, a retried request or a second process gets None. A rejected,
        expired or pending row is never claimable.
        """
        now_iso = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE approval_queue
                   SET executed_at = ?
                 WHERE approval_id = ? AND status = 'approved' AND executed_at IS NULL
                """,
                (now_iso, approval_id),
            )
            conn.commit()
            if cursor.rowcount != 1:
                return None
            row = conn.execute(
                "SELECT * FROM approval_queue WHERE approval_id = ?", (approval_id,)
            ).fetchone()
        return self._row(row) if row is not None else None

    def set_checkpoint(self, approval_id: str, checkpoint_id: str) -> ApprovalRow:
        """Link a queued approval to the checkpoint that answering it resumes.

        The column has existed since Phase 3 and has always been NULL in practice:
        the evaluator enqueues at ``PostStep``, *before* ``agentic_core`` writes the
        checkpoint, so it has nothing to pass and there was no setter to fill in
        afterwards. Without the link an approval is a decision about a run nobody can
        find again — which is why "approve" has never been able to continue anything.

        Deliberately allowed after the row is answered: the write is bookkeeping about
        where the run lives, not a change to the decision.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT approval_id FROM approval_queue WHERE approval_id = ?", (approval_id,)
            ).fetchone()
            if row is None:
                raise ApprovalNotFoundError(approval_id)
            conn.execute(
                "UPDATE approval_queue SET checkpoint_id = ? WHERE approval_id = ?",
                (checkpoint_id, approval_id),
            )
            conn.commit()
            updated = conn.execute(
                "SELECT * FROM approval_queue WHERE approval_id = ?", (approval_id,)
            ).fetchone()
        return self._row(updated)

    def expire_stale(self) -> list[ApprovalRow]:
        """Time out every overdue pending row; return the rows that transitioned.

        Returns the rows, not a count, because the caller's real job is to *tell
        someone*: an approval that lapses in silence leaves the run it halted looking
        as though it were still being considered. Selecting inside the same
        transaction as the UPDATE is what makes that safe to act on — the write only
        touches ``status='pending'``, so a second sweep transitions nothing and
        returns nothing, and a notice is therefore sent exactly once per lapse
        however many sweeps race.
        """
        now_iso = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            due = conn.execute(
                "SELECT * FROM approval_queue WHERE status='pending' AND timeout_at <= ?"
                " ORDER BY requested_at",
                (now_iso,),
            ).fetchall()
            if not due:
                return []
            conn.execute(
                """
                UPDATE approval_queue
                   SET status = 'timed_out'
                 WHERE status = 'pending' AND timeout_at <= ?
                """,
                (now_iso,),
            )
            conn.commit()
        # The rows are already in hand from the SELECT above, so they are returned with
        # the status the UPDATE just gave them rather than read back. One less query,
        # and no dynamically built `IN (...)` clause to have to reason about.
        expired = [replace(self._row(r), status="timed_out") for r in due]
        logger.warning("expired %d stale approval(s)", len(expired))
        return expired
