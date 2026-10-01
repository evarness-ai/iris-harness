"""The mailbox-write approval gate (R4): one row per account, any provider."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_personal.email import write_approvals as wa


def test_no_row_means_no_writes_and_a_clear_error(tmp_path: Path) -> None:
    store = wa.WriteApprovalStore(tmp_path / "a.db")
    assert wa.mailbox_writes_approved("imap:a@x.test", store=store) is False
    with pytest.raises(PermissionError, match="no approval to change imap:a@x.test"):
        wa.require_mailbox_writes("imap:a@x.test", "change labels", store=store)


def test_approve_is_per_account_and_revocable(tmp_path: Path) -> None:
    store = wa.WriteApprovalStore(tmp_path / "a.db")
    approval = wa.approve_mailbox_writes("imap:a@x.test", "approval-7", store=store)
    assert approval.approval_ref == "approval-7"
    wa.require_mailbox_writes("imap:a@x.test", "change labels", store=store)
    assert wa.mailbox_writes_approved("gmail:a@x.test", store=store) is False
    got = wa.mailbox_write_approval("imap:a@x.test", store=store)
    assert got is not None and got.approval_ref == "approval-7"
    wa.revoke_mailbox_writes("imap:a@x.test", actor="test", agent_type="test", store=store)
    assert wa.mailbox_writes_approved("imap:a@x.test", store=store) is False


def test_an_approval_needs_a_reference(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        wa.approve_mailbox_writes(
            "imap:a@x.test", " ", store=wa.WriteApprovalStore(tmp_path / "a.db")
        )


def test_the_default_store_follows_the_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path / "data"))
    wa.approve_mailbox_writes("imap:a@x.test", "approval-1")
    assert (tmp_path / "data" / wa.DB_FILENAME).exists()
    assert wa.mailbox_writes_approved("imap:a@x.test")
