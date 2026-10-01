"""Tests for ApprovalStore (story 12.gov-4.7)."""

from __future__ import annotations

import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_harness.kernel.governance.approvals import (
    ApprovalAlreadyAnsweredError,
    ApprovalNotFoundError,
    ApprovalQueue,
    ApprovalStore,
)


def _store(tmp_path: Path) -> ApprovalStore:
    return ApprovalStore(db_path=tmp_path / "approvals.db")


# ---------------------------------------------------------------------------
# AC-1: DB file created with 0o600 permissions
# ---------------------------------------------------------------------------


def test_db_created_with_0o600_perms(tmp_path: Path) -> None:
    db_path = tmp_path / "approvals.db"
    ApprovalStore(db_path=db_path)
    mode = stat.S_IMODE(os.stat(db_path).st_mode)
    assert mode == 0o600


# ---------------------------------------------------------------------------
# AC-2: enqueue returns a UUID string and persists a pending row
# ---------------------------------------------------------------------------


def test_enqueue_returns_uuid_and_creates_pending_row(tmp_path: Path) -> None:
    store = _store(tmp_path)
    aid = store.enqueue("run-1", None, "require_approval", "agent wants to run rm -rf")
    assert len(aid) == 36
    row = store.get(aid)
    assert row is not None
    assert row.status == "pending"
    assert row.run_id == "run-1"
    assert row.signal == "require_approval"
    assert row.context_summary == "agent wants to run rm -rf"
    assert row.channel == "cli"
    assert row.policy_on_timeout == "fail_closed"
    assert row.responded_at is None
    assert row.response_actor is None


def test_enqueue_with_checkpoint_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    aid = store.enqueue("run-2", "chk-abc", "goal_drift", "drifted")
    row = store.get(aid)
    assert row is not None
    assert row.checkpoint_id == "chk-abc"


def test_enqueue_custom_timeout_and_policy(tmp_path: Path) -> None:
    store = _store(tmp_path)
    before = datetime.now(UTC)
    aid = store.enqueue(
        "run-3",
        None,
        "cost_budget",
        "over budget",
        timeout_minutes=30,
        policy_on_timeout="fail_open",
    )
    row = store.get(aid)
    assert row is not None
    assert row.policy_on_timeout == "fail_open"
    timeout_dt = datetime.fromisoformat(row.timeout_at)
    assert timeout_dt > before + timedelta(minutes=29)
    assert timeout_dt < before + timedelta(minutes=31)


# ---------------------------------------------------------------------------
# AC-3: list_pending returns pending rows (with optional due_only filter)
# ---------------------------------------------------------------------------


def test_list_pending_returns_pending_rows(tmp_path: Path) -> None:
    store = _store(tmp_path)
    aid1 = store.enqueue("r1", None, "s1", "ctx1")
    aid2 = store.enqueue("r2", None, "s2", "ctx2")
    pending = store.list_pending()
    ids = {r.approval_id for r in pending}
    assert aid1 in ids
    assert aid2 in ids


def test_list_pending_excludes_non_pending(tmp_path: Path) -> None:
    store = _store(tmp_path)
    aid = store.enqueue("r1", None, "s1", "ctx")
    store.respond(aid, status="approved", actor="cli:user")
    pending = store.list_pending()
    assert all(r.approval_id != aid for r in pending)


def test_list_pending_due_only_returns_expired(tmp_path: Path) -> None:
    store = _store(tmp_path)
    aid = store.enqueue("r1", None, "s1", "ctx", timeout_minutes=-1)
    due = store.list_pending(due_only=True)
    ids = {r.approval_id for r in due}
    assert aid in ids


def test_list_pending_due_only_excludes_future(tmp_path: Path) -> None:
    store = _store(tmp_path)
    aid = store.enqueue("r1", None, "s1", "ctx", timeout_minutes=60)
    due = store.list_pending(due_only=True)
    assert all(r.approval_id != aid for r in due)


# ---------------------------------------------------------------------------
# AC-4: respond transitions status and records actor + timestamp
# ---------------------------------------------------------------------------


def test_respond_approve_transitions_status(tmp_path: Path) -> None:
    store = _store(tmp_path)
    aid = store.enqueue("r1", None, "s1", "ctx")
    row = store.respond(aid, status="approved", actor="cli:alice")
    assert row.status == "approved"
    assert row.response_actor == "cli:alice"
    assert row.responded_at is not None


def test_respond_reject_transitions_status(tmp_path: Path) -> None:
    store = _store(tmp_path)
    aid = store.enqueue("r1", None, "s1", "ctx")
    row = store.respond(aid, status="rejected", actor="cli:bob")
    assert row.status == "rejected"
    assert row.response_actor == "cli:bob"


def test_respond_invalid_status_raises_value_error(tmp_path: Path) -> None:
    store = _store(tmp_path)
    aid = store.enqueue("r1", None, "s1", "ctx")
    with pytest.raises(ValueError, match="status must be"):
        store.respond(aid, status="maybe", actor="cli:user")


# ---------------------------------------------------------------------------
# AC-5: double-respond raises ApprovalAlreadyAnsweredError
# ---------------------------------------------------------------------------


def test_respond_twice_raises_already_answered(tmp_path: Path) -> None:
    store = _store(tmp_path)
    aid = store.enqueue("r1", None, "s1", "ctx")
    store.respond(aid, status="approved", actor="cli:user")
    with pytest.raises(ApprovalAlreadyAnsweredError):
        store.respond(aid, status="rejected", actor="cli:attacker")


# ---------------------------------------------------------------------------
# AC-6: get returns None for unknown IDs; ApprovalNotFoundError on respond
# ---------------------------------------------------------------------------


def test_get_unknown_returns_none(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.get("00000000-0000-0000-0000-000000000000") is None


def test_respond_unknown_raises_not_found(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ApprovalNotFoundError):
        store.respond("00000000-0000-0000-0000-000000000000", status="approved", actor="cli:x")


# ---------------------------------------------------------------------------
# AC-7: expire_stale marks timed-out rows and returns the rows it transitioned
# ---------------------------------------------------------------------------


def test_expire_stale_marks_timed_out_rows(tmp_path: Path) -> None:
    store = _store(tmp_path)
    aid = store.enqueue("r1", None, "s1", "ctx", timeout_minutes=-1)
    expired = store.expire_stale()
    assert len(expired) == 1
    # The rows, not a count — the caller's real job is to tell someone, and it needs
    # the run, the signal and the session to do that.
    assert expired[0].approval_id == aid
    assert expired[0].status == "timed_out"
    row = store.get(aid)
    assert row is not None
    assert row.status == "timed_out"


def test_expire_stale_ignores_future_rows(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.enqueue("r1", None, "s1", "ctx", timeout_minutes=60)
    assert store.expire_stale() == []


def test_expire_stale_ignores_already_responded(tmp_path: Path) -> None:
    store = _store(tmp_path)
    aid = store.enqueue("r1", None, "s1", "ctx", timeout_minutes=-1)
    store.respond(aid, status="approved", actor="cli:user")
    assert store.expire_stale() == []


def test_expire_stale_only_counts_newly_expired(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.enqueue("r1", None, "s1", "ctx1", timeout_minutes=-1)
    store.enqueue("r2", None, "s2", "ctx2", timeout_minutes=-1)
    store.enqueue("r3", None, "s3", "ctx3", timeout_minutes=60)
    assert len(store.expire_stale()) == 2
    # A second sweep transitions nothing, which is what makes "announce once per
    # lapse" safe however many sweeps race.
    assert store.expire_stale() == []


# ---------------------------------------------------------------------------
# list_by_status query helper
# ---------------------------------------------------------------------------


def test_list_by_status_returns_correct_rows(tmp_path: Path) -> None:
    store = _store(tmp_path)
    aid1 = store.enqueue("r1", None, "s1", "ctx1")
    aid2 = store.enqueue("r2", None, "s2", "ctx2")
    store.respond(aid1, status="approved", actor="cli:user")
    approved = store.list_by_status("approved")
    assert len(approved) == 1
    assert approved[0].approval_id == aid1
    pending = store.list_by_status("pending")
    assert len(pending) == 1
    assert pending[0].approval_id == aid2


# ---------------------------------------------------------------------------
# ApprovalQueue facade — audit integration
# ---------------------------------------------------------------------------


def test_approval_queue_enqueue_and_respond(tmp_path: Path) -> None:
    from iris_harness.kernel.governance.audit import AuditLog

    audit = AuditLog(db_path=tmp_path / "audit.db")
    queue = ApprovalQueue(db_path=tmp_path / "approvals.db", audit_log=audit)
    aid = queue.enqueue("run-q1", None, "require_approval", "wants network access")
    assert len(aid) == 36
    row = queue.get(aid)
    assert row is not None
    assert row.status == "pending"
    audit_rows = audit.query(decision="require_approval")
    assert len(audit_rows) == 1
    assert "require_approval" in audit_rows[0].reason

    queue.respond(aid, status="approved", actor="cli:user")
    approved_rows = audit.query(decision="approved")
    assert len(approved_rows) == 1


def test_approval_queue_expire_stale_writes_audit(tmp_path: Path) -> None:
    from iris_harness.kernel.governance.audit import AuditLog

    audit = AuditLog(db_path=tmp_path / "audit.db")
    queue = ApprovalQueue(db_path=tmp_path / "approvals.db", audit_log=audit)
    queue.enqueue("run-exp", None, "s1", "ctx", timeout_minutes=-1)
    expired = queue.expire_stale()
    assert len(expired) == 1
    timed_out = audit.query(decision="timed_out")
    assert len(timed_out) == 1
    # Keyed to the run that stalled, not to "system" — otherwise `iris run inspect`
    # on the affected run shows nothing about why it never finished.
    assert timed_out[0].run_id == "run-exp"


def test_approval_queue_no_audit_log(tmp_path: Path) -> None:
    queue = ApprovalQueue(db_path=tmp_path / "approvals.db")
    aid = queue.enqueue("run-noaudit", None, "s1", "ctx")
    row = queue.respond(aid, status="rejected", actor="cli:user")
    assert row.status == "rejected"


# ---------------------------------------------------------------------------
# ApprovalRow is a frozen dataclass (immutable)
# ---------------------------------------------------------------------------


def test_approval_row_is_frozen(tmp_path: Path) -> None:
    store = _store(tmp_path)
    aid = store.enqueue("r1", None, "s1", "ctx")
    row = store.get(aid)
    assert row is not None
    with pytest.raises((AttributeError, TypeError)):
        row.status = "approved"  # type: ignore[misc]


# ── ADR-0118: pinned items ─────────────────────────────────────────────────────


def test_pinned_items_round_trip_and_compare_by_canonical_arguments(tmp_path: Path) -> None:
    from iris_harness.kernel.governance.approvals.store import ApprovalItem, ApprovalStore

    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    item = ApprovalItem.of("trash_email", {"ids": ["m1"], "folder": "inbox"})
    approval_id = store.enqueue("run-1", None, "sig", "ctx", items=(item,))

    row = store.get(approval_id)
    assert row is not None and row.items == (item,)
    assert row.items[0].args == {"ids": ["m1"], "folder": "inbox"}
    # Key order does not matter; any other difference does.
    assert ApprovalItem.of("trash_email", {"folder": "inbox", "ids": ["m1"]}) == item
    assert ApprovalItem.of("trash_email", {"folder": "inbox", "ids": ["m1", "m2"]}) != item
    # An evaluator approval pins nothing.
    other = store.get(store.enqueue("run-2", None, "sig", "ctx"))
    assert other is not None and other.items is None


def test_an_old_database_gains_the_items_column(tmp_path: Path) -> None:
    import sqlite3

    from iris_harness.kernel.governance.approvals.store import ApprovalStore

    db = tmp_path / "approvals.db"
    with sqlite3.connect(db) as conn:  # the table as it was before ADR-0118
        conn.execute(
            "CREATE TABLE approval_queue (approval_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,"
            " checkpoint_id TEXT, signal TEXT NOT NULL, context_summary TEXT NOT NULL,"
            " requested_at TEXT NOT NULL, channel TEXT NOT NULL DEFAULT 'cli',"
            " status TEXT NOT NULL DEFAULT 'pending', responded_at TEXT, response_actor TEXT,"
            " timeout_at TEXT NOT NULL, policy_on_timeout TEXT NOT NULL DEFAULT 'fail_closed',"
            " session_id TEXT)"
        )
        conn.execute(
            "INSERT INTO approval_queue (approval_id, run_id, signal, context_summary,"
            " requested_at, timeout_at) VALUES ('old', 'r', 's', 'c', '2026-01-01', '2026-01-02')"
        )
    store = ApprovalStore(db_path=db)
    old = store.get("old")
    assert old is not None and old.items is None


def test_unreadable_items_approve_nothing(tmp_path: Path) -> None:
    import sqlite3

    from iris_harness.kernel.governance.approvals.store import ApprovalItem, ApprovalStore

    db = tmp_path / "approvals.db"
    store = ApprovalStore(db_path=db)
    approval_id = store.enqueue(
        "run-1", None, "sig", "ctx", items=(ApprovalItem.of("trash_email", {"ids": ["m1"]}),)
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE approval_queue SET items_json = 'not json' WHERE approval_id = ?",
            (approval_id,),
        )
    row = store.get(approval_id)
    assert row is not None and row.items == ()  # matches no call


def test_the_card_round_trips_and_an_old_database_gains_its_column(tmp_path: Path) -> None:
    import sqlite3

    from iris_harness.kernel.governance.approvals.store import ApprovalCard, ApprovalStore

    db = tmp_path / "approvals.db"
    with sqlite3.connect(db) as conn:  # before step 4: items, no card
        conn.execute(
            "CREATE TABLE approval_queue (approval_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,"
            " checkpoint_id TEXT, signal TEXT NOT NULL, context_summary TEXT NOT NULL,"
            " requested_at TEXT NOT NULL, channel TEXT NOT NULL DEFAULT 'cli',"
            " status TEXT NOT NULL DEFAULT 'pending', responded_at TEXT, response_actor TEXT,"
            " timeout_at TEXT NOT NULL, policy_on_timeout TEXT NOT NULL DEFAULT 'fail_closed',"
            " session_id TEXT, items_json TEXT)"
        )
    store = ApprovalStore(db_path=db)
    card = ApprovalCard(
        title="Trash 2 emails",
        lines=("Deals — Store X", "Sale — Shop Y"),
        undo_tool="restore_email",
        undo_window_days=30,
        asked="clean up the promos",
    )
    row = store.get(store.enqueue("run-1", None, "Trash 2 emails", "ctx", card=card))
    assert row is not None and row.card == card
    assert card.undo_sentence() == "Reversible for 30 days (undo: restore_email)."
    assert ApprovalCard(title="x").undo_sentence() == "This cannot be undone."


def test_an_unreadable_card_falls_back_to_none(tmp_path: Path) -> None:
    import sqlite3

    from iris_harness.kernel.governance.approvals.store import ApprovalCard, ApprovalStore

    db = tmp_path / "approvals.db"
    store = ApprovalStore(db_path=db)
    approval_id = store.enqueue("run-1", None, "s", "c", card=ApprovalCard(title="t"))
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE approval_queue SET card_json = '{bad' WHERE approval_id = ?", (approval_id,)
        )
    row = store.get(approval_id)
    assert row is not None and row.card is None  # the surfaces fall back to the raw call


def test_an_old_database_gains_the_caller_and_claim_columns(tmp_path: Path) -> None:
    """Plugin-capabilities decision 1: a code caller's approval, claimed once."""
    import sqlite3

    from iris_harness.kernel.governance.approvals.store import ApprovalItem, ApprovalStore

    db = tmp_path / "approvals.db"
    with sqlite3.connect(db) as conn:  # before decision 1: a card, no caller
        conn.execute(
            "CREATE TABLE approval_queue (approval_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,"
            " checkpoint_id TEXT, signal TEXT NOT NULL, context_summary TEXT NOT NULL,"
            " requested_at TEXT NOT NULL, channel TEXT NOT NULL DEFAULT 'cli',"
            " status TEXT NOT NULL DEFAULT 'pending', responded_at TEXT, response_actor TEXT,"
            " timeout_at TEXT NOT NULL, policy_on_timeout TEXT NOT NULL DEFAULT 'fail_closed',"
            " session_id TEXT, items_json TEXT, card_json TEXT)"
        )
    store = ApprovalStore(db_path=db)
    items = (ApprovalItem.of("add_note", {"text": "milk"}),)
    aid = store.enqueue("run-1", None, "s", "ctx", items=items, caller="plugin:p")
    loop_aid = store.enqueue("run-2", None, "s", "ctx", items=items)
    row = store.get(aid)
    assert row is not None and row.caller == "plugin:p" and row.is_deferred_call
    assert not store.get(loop_aid).is_deferred_call  # type: ignore[union-attr]

    assert store.claim_execution(aid) is None  # pending: never claimable
    store.respond(aid, status="approved", actor="t")
    claimed = store.claim_execution(aid)
    assert claimed is not None and claimed.executed_at is not None
    assert store.claim_execution(aid) is None  # once


def test_respond_is_conditional_on_pending(tmp_path: Path) -> None:
    from iris_harness.kernel.governance.approvals.store import ApprovalStore

    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    aid = store.enqueue("run-1", None, "s", "ctx")
    store.respond(aid, status="approved", actor="a")
    with pytest.raises(ApprovalAlreadyAnsweredError):
        store.respond(aid, status="rejected", actor="b")
    row = store.get(aid)
    assert row is not None and (row.status, row.response_actor) == ("approved", "a")
    with pytest.raises(ApprovalNotFoundError):
        store.respond("missing", status="approved", actor="a")
