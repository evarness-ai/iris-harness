"""CLI tests for the approvals sub-command (story 12.gov-4.7)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from iris_harness.cli.approvals import approvals_app
from iris_harness.kernel.governance.approvals import ApprovalStore

runner = CliRunner()


def _populated_store(tmp_path: Path) -> tuple[ApprovalStore, str, str]:
    """Return (store, pending_id, approved_id)."""
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    pending_id = store.enqueue("run-cli-1", None, "require_approval", "agent wants network")
    approved_id = store.enqueue("run-cli-2", None, "cost_budget", "over spend limit")
    store.respond(approved_id, status="approved", actor="cli:test-setup")
    return store, pending_id, approved_id


def _patch_store(store: ApprovalStore):
    return patch("iris_harness.cli.approvals._store", return_value=store)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_pending_shows_table(tmp_path: Path) -> None:
    store, pending_id, _ = _populated_store(tmp_path)
    with _patch_store(store):
        result = runner.invoke(approvals_app, ["list"])
    assert result.exit_code == 0
    # Rich may truncate long values in narrow terminals; check UUID prefix and partial signal
    assert pending_id[:8] in result.output
    assert "requ" in result.output  # truncated "require_approval"


def test_list_pending_empty(tmp_path: Path) -> None:
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    with _patch_store(store):
        result = runner.invoke(approvals_app, ["list"])
    assert result.exit_code == 0
    assert "No pending" in result.output


def test_list_by_status_approved(tmp_path: Path) -> None:
    store, _, approved_id = _populated_store(tmp_path)
    with _patch_store(store):
        result = runner.invoke(approvals_app, ["list", "--status", "approved"])
    assert result.exit_code == 0
    assert approved_id[:8] in result.output


def test_list_by_status_no_rows(tmp_path: Path) -> None:
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    with _patch_store(store):
        result = runner.invoke(approvals_app, ["list", "--status", "rejected"])
    assert result.exit_code == 0
    assert "No rejected" in result.output


def test_list_due_only(tmp_path: Path) -> None:
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    overdue_id = store.enqueue("run-due", None, "s1", "overdue ctx", timeout_minutes=-1)
    future_id = store.enqueue("run-future", None, "s2", "future ctx", timeout_minutes=60)
    with _patch_store(store):
        result = runner.invoke(approvals_app, ["list", "--due"])
    assert result.exit_code == 0
    assert overdue_id[:8] in result.output
    assert future_id[:8] not in result.output


# ---------------------------------------------------------------------------
# approve
# ---------------------------------------------------------------------------


def test_approve_succeeds(tmp_path: Path) -> None:
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    aid = store.enqueue("run-ap", None, "s1", "ctx")
    with _patch_store(store):
        result = runner.invoke(approvals_app, ["approve", aid])
    assert result.exit_code == 0
    assert "Approved" in result.output
    row = store.get(aid)
    assert row is not None
    assert row.status == "approved"


def test_approve_with_reason(tmp_path: Path) -> None:
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    aid = store.enqueue("run-ap2", None, "s1", "ctx")
    with _patch_store(store):
        result = runner.invoke(approvals_app, ["approve", aid, "--reason", "LGTM"])
    assert result.exit_code == 0
    row = store.get(aid)
    assert row is not None
    assert "LGTM" in (row.response_actor or "")


def test_approve_not_found(tmp_path: Path) -> None:
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    with _patch_store(store):
        result = runner.invoke(approvals_app, ["approve", "00000000-0000-0000-0000-000000000000"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_approve_already_answered(tmp_path: Path) -> None:
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    aid = store.enqueue("run-aa", None, "s1", "ctx")
    store.respond(aid, status="approved", actor="cli:setup")
    with _patch_store(store):
        result = runner.invoke(approvals_app, ["approve", aid])
    assert result.exit_code == 1
    assert "already" in result.output.lower() or "Error" in result.output


# ---------------------------------------------------------------------------
# reject
# ---------------------------------------------------------------------------


def test_reject_succeeds(tmp_path: Path) -> None:
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    aid = store.enqueue("run-rj", None, "s1", "ctx")
    with _patch_store(store):
        result = runner.invoke(approvals_app, ["reject", aid])
    assert result.exit_code == 0
    assert "Rejected" in result.output
    row = store.get(aid)
    assert row is not None
    assert row.status == "rejected"


def test_reject_with_reason(tmp_path: Path) -> None:
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    aid = store.enqueue("run-rj2", None, "s1", "ctx")
    with _patch_store(store):
        result = runner.invoke(approvals_app, ["reject", aid, "--reason", "too risky"])
    assert result.exit_code == 0
    row = store.get(aid)
    assert row is not None
    assert "too risky" in (row.response_actor or "")


def test_reject_not_found(tmp_path: Path) -> None:
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    with _patch_store(store):
        result = runner.invoke(approvals_app, ["reject", "00000000-0000-0000-0000-000000000000"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_reject_already_answered(tmp_path: Path) -> None:
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    aid = store.enqueue("run-rja", None, "s1", "ctx")
    store.respond(aid, status="rejected", actor="cli:setup")
    with _patch_store(store):
        result = runner.invoke(approvals_app, ["reject", aid])
    assert result.exit_code == 1
    assert "already" in result.output.lower() or "Error" in result.output
