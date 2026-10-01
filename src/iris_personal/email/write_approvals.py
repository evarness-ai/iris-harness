"""The mailbox-write approval gate: one per account, for every mail provider (R4).

OSS plan R4: nothing touches the owner's mailbox before they approve it (the label
preview step of email setup). "Touches" means any write a provider makes -- a label or
keyword, a move to Trash, a restore. This module is the one place that approval lives,
whatever the provider: a row per ``email_accounts`` id naming the approval that granted
it (``approval_ref``: the approval-queue row id, or the command that recorded it).

* The approver -- the onboarding flow's step-6 approval, or a one-time command for an
  account connected before onboarding existed -- calls :func:`approve_mailbox_writes`.
* A provider calls :func:`require_mailbox_writes` first thing in every write method,
  before it opens a connection; with no row it raises ``PermissionError``, which the
  judge's label sync and the trash tools already report as "cannot modify this mail".
* :func:`grant_mailbox_writes` is the audited grant every approver records it with (the
  writes command, setup's step 6, the demo for its own synthetic account): a governance
  audit row first (plugin ``email_write_approvals``, hook ``mailbox_writes``), then the
  approval naming it -- ``<ref> [audit #<row>]``. No audit row, no approval (R14).
* :func:`revoke_mailbox_writes` takes it back, with an audit row of its own (decision
  ``deny``) when there was an approval to take; :func:`mailbox_writes_approved` reads it.
* :func:`mailbox_write` is how a provider makes a write: the approval check first, then
  the write, then one audit row of what reached the mailbox (hook
  ``mailbox_write_performed``: the account in the payload, the kind of write and how
  many messages or labels it changed). Those rows are the proof bundle's mailbox-write
  observations (invariant 2), so ``iris governance check`` and the web see every write.

No ledger ``reason`` written here names an address: a reason says "a gmail account"
(:func:`iris_personal.email.accounts.ledger_account_label`), and the account id lives in
the payload, which the proof bundle pseudonymises.
* ``iris email writes approve|status|revoke`` (the ``email_workflows`` plugin) is the
  owner's command line over it, for any provider; a refusal names the exact command
  (:func:`approve_command`).

Stored in ``mailbox_write_approvals.db`` under ``sdk.persistence.data_dir()``, so it
follows ``IRIS_DATA_DIR`` / ``IRIS_HOME`` and a test never touches the owner's file.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_harness.sdk.persistence import data_dir, sqlite_conn
from iris_personal.email.accounts import ledger_account_label

logger = logging.getLogger(__name__)

DB_FILENAME = "mailbox_write_approvals.db"
#: The ``plugin`` of every audit row this module writes, and the ``hook_point`` of a
#: grant's (decision ``allow``) or a revoke's (decision ``deny``).
AUDIT_PLUGIN = "email_write_approvals"
AUDIT_HOOK = "mailbox_writes"
#: The ``hook_point`` of the row a write that reached a mailbox leaves (decision
#: ``allow``): the proof bundle's mailbox-write observation. Never an approval.
WRITE_HOOK = "mailbox_write_performed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mailbox_write_approvals (
    account_id   TEXT PRIMARY KEY,
    approval_ref TEXT NOT NULL,
    approved_at  TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class WriteApproval:
    account_id: str
    approval_ref: str
    approved_at: str


class WriteApprovalStore:
    """The approvals table. ``db_path`` defaults to the data dir's file (tests pass one)."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    @property
    def db_path(self) -> Path:
        return self._db_path or data_dir() / DB_FILENAME

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite_conn(self.db_path, row_factory=sqlite3.Row) as conn:
            conn.executescript(_SCHEMA)
            yield conn

    def approve(self, account_id: str, approval_ref: str) -> WriteApproval:
        if not account_id.strip() or not approval_ref.strip():
            raise ValueError("an account id and an approval reference are required")
        approval = WriteApproval(
            account_id=account_id.strip(),
            approval_ref=approval_ref.strip(),
            approved_at=datetime.now(UTC).isoformat(),
        )
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO mailbox_write_approvals (account_id, approval_ref, approved_at)"
                " VALUES (?, ?, ?) ON CONFLICT(account_id) DO UPDATE SET"
                " approval_ref = excluded.approval_ref, approved_at = excluded.approved_at",
                (approval.account_id, approval.approval_ref, approval.approved_at),
            )
        return approval

    def revoke(self, account_id: str) -> bool:
        """Remove ``account_id``'s approval; True when there was one."""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM mailbox_write_approvals WHERE account_id = ?", (account_id,)
            )
            return cur.rowcount > 0

    def get(self, account_id: str) -> WriteApproval | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM mailbox_write_approvals WHERE account_id = ?", (account_id,)
            ).fetchone()
        if row is None:
            return None
        return WriteApproval(row["account_id"], row["approval_ref"], row["approved_at"])


def approve_mailbox_writes(
    account_id: str, approval_ref: str, *, store: WriteApprovalStore | None = None
) -> WriteApproval:
    """Record the owner's approval to change ``account_id``'s mailbox."""
    return (store or WriteApprovalStore()).approve(account_id, approval_ref)


def _ledger(audit_log: Any) -> Any:
    if audit_log is not None:
        return audit_log
    from iris_harness.sdk.audit import AuditLog, audit_db_path

    return AuditLog(db_path=audit_db_path())


def grant_mailbox_writes(
    account_id: str,
    ref: str,
    *,
    actor: str,
    run_id: str,
    agent_type: str,
    step_id: int | None = None,
    audit_log: Any = None,
    store: WriteApprovalStore | None = None,
) -> tuple[WriteApproval, int]:
    """Record an approval of ``account_id``'s mailbox writes the audited way: the
    ledger row first, then the approval ``<ref> [audit #<row>]``. Returns both. When the
    audit row cannot be written this raises and nothing is approved. The row's reason
    names the provider, not the address; who approved (``actor``) and the account are in
    the payload."""
    row = int(
        _ledger(audit_log).record(
            run_id=run_id,
            step_id=step_id,
            agent_type=agent_type,
            hook_point=AUDIT_HOOK,
            plugin=AUDIT_PLUGIN,
            decision="allow",
            severity="info",
            reason=f"mailbox writes approved for {ledger_account_label(account_id)}",
            payload={"account": account_id, "actor": actor, "ref": ref},
        )
    )
    return approve_mailbox_writes(account_id, f"{ref} [audit #{row}]", store=store), row


def revoke_mailbox_writes(
    account_id: str,
    *,
    actor: str,
    agent_type: str,
    run_id: str | None = None,
    audit_log: Any = None,
    store: WriteApprovalStore | None = None,
) -> int | None:
    """Take back ``account_id``'s approval, then record that in the ledger (decision
    ``deny``). Less permission never waits on a ledger: the approval is gone first, and
    a ledger that cannot be written is logged, not raised. Returns the audit row id, or
    None when there was no approval to take (no row) or the row could not be written."""
    if not (store or WriteApprovalStore()).revoke(account_id):
        return None
    try:
        return int(
            _ledger(audit_log).record(
                run_id=run_id or f"mailbox-writes-{uuid.uuid4().hex[:12]}",
                step_id=None,
                agent_type=agent_type,
                hook_point=AUDIT_HOOK,
                plugin=AUDIT_PLUGIN,
                decision="deny",
                severity="info",
                reason=f"mailbox writes revoked for {ledger_account_label(account_id)}",
                payload={"account": account_id, "actor": actor, "ref": None},
            )
        )
    except Exception:  # the revoke stands; the gap is logged
        logger.error("mailbox writes revoked, but the audit row failed", exc_info=True)
        return None


def mailbox_write_approval(
    account_id: str, *, store: WriteApprovalStore | None = None
) -> WriteApproval | None:
    return (store or WriteApprovalStore()).get(account_id)


def mailbox_writes_approved(account_id: str, *, store: WriteApprovalStore | None = None) -> bool:
    return mailbox_write_approval(account_id, store=store) is not None


def require_mailbox_writes(
    account_id: str, what: str, *, store: WriteApprovalStore | None = None
) -> None:
    """Raise ``PermissionError`` unless ``account_id`` has a write approval. ``what`` is
    the write in the owner's words ("change labels", "move mail to Trash")."""
    if not mailbox_writes_approved(account_id, store=store):
        raise PermissionError(
            f"IRIS has no approval to change {account_id} yet, so it did not {what}. "
            f"To allow it, run `{approve_command(account_id)}` once (email setup's "
            "label-preview approval records the same thing)."
        )


class WriteTally:
    """What one :func:`mailbox_write` changed: ``add`` each batch as it lands."""

    def __init__(self) -> None:
        self.count = 0

    def add(self, count: int) -> None:
        if count > 0:
            self.count += count


@contextmanager
def mailbox_write(
    account_id: str,
    what: str,
    *,
    op: str,
    store: WriteApprovalStore | None = None,
    audit_log: Any = None,
) -> Iterator[WriteTally]:
    """Gate and record one mailbox write. :func:`require_mailbox_writes` first (no
    approval: ``PermissionError``, nothing written); the body makes the write and
    ``add``s what reached the mailbox as each batch lands; then one audit row
    (:func:`record_mailbox_write`) for the total -- also when the body raises part way,
    so a partial write is observed too. ``op`` names the kind of write (``trash``,
    ``restore``, ``label``, ``create_label``, ``restore_labels``)."""
    require_mailbox_writes(account_id, what, store=store)
    tally = WriteTally()
    try:
        yield tally
    finally:
        if tally.count:
            record_mailbox_write(account_id, op, tally.count, audit_log=audit_log)


def record_mailbox_write(
    account_id: str, op: str, count: int, *, audit_log: Any = None
) -> int | None:
    """One ledger row for ``count`` changes of kind ``op`` that reached ``account_id``'s
    mailbox (hook :data:`WRITE_HOOK`). The account is in the payload only; the reason
    names its provider. Returns the row id. The write has already happened, so a ledger
    that cannot be written is logged as an error, not raised: raising would report a
    write that reached the mailbox as one that failed."""
    try:
        return int(
            _ledger(audit_log).record(
                run_id=f"mailbox-write-{uuid.uuid4().hex[:12]}",
                step_id=None,
                agent_type="mail_provider",
                hook_point=WRITE_HOOK,
                plugin=AUDIT_PLUGIN,
                decision="allow",
                severity="info",
                reason=(f"mailbox write: {op} x{count} on {ledger_account_label(account_id)}"),
                payload={"account": account_id, "op": op, "count": count},
            )
        )
    except Exception:  # see the docstring
        logger.error("a mailbox write (%s x%d) has no audit row", op, count, exc_info=True)
        return None


def approve_command(account_id: str) -> str:
    """The one-time command that approves ``account_id``'s mailbox writes: what every
    refusal and the System Health row tell the owner to run."""
    return f"iris email writes approve --account {account_id}"


__all__ = [
    "AUDIT_HOOK",
    "AUDIT_PLUGIN",
    "DB_FILENAME",
    "WriteApproval",
    "WRITE_HOOK",
    "WriteApprovalStore",
    "WriteTally",
    "approve_command",
    "approve_mailbox_writes",
    "grant_mailbox_writes",
    "mailbox_write",
    "mailbox_write_approval",
    "mailbox_writes_approved",
    "record_mailbox_write",
    "require_mailbox_writes",
    "revoke_mailbox_writes",
]
