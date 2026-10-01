"""The mailbox-write ledger rows (R4 + R14): approvals, revokes and the writes themselves.

* No row's free text names the mailbox address: a reason says "a gmail account"; the
  account id lives in the payload, which the proof bundle pseudonymises. Swept over every
  row the grant (command, library), the revoke (command, library, IMAP logout), the
  sweep's refusal and the writes leave.
* Every write that reaches a mailbox leaves one ``mailbox_write_performed`` row, and
  ``check_window`` reads those as invariant 2's observations: a write after an approval
  holds, one without an approval is a violation.

Synthetic addresses only; the ledger, approvals and accounts live in this test's dirs.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from iris_harness.kernel.governance.audit.proof_bundle import (
    MAILBOX_WRITE_APPROVED,
    check_window,
    export_bundle,
)
from iris_harness.main import app
from iris_harness.sdk.audit import AuditLog, audit_db_path
from iris_personal.email.accounts import EmailAccountStore, ledger_account_label
from iris_personal.email.write_approvals import (
    AUDIT_HOOK,
    AUDIT_PLUGIN,
    WRITE_HOOK,
    grant_mailbox_writes,
    mailbox_write,
    record_mailbox_write,
    require_mailbox_writes,
    revoke_mailbox_writes,
)
from iris_personal.plugins.email_workflows.demo.provider import DemoMailProvider
from iris_personal.plugins.email_workflows.judge_config import JudgeConfig
from iris_personal.plugins.email_workflows.judge_labels import sync_labels
from iris_personal.plugins.email_workflows.judgments import JudgmentStore

ADDRESS = "casey.synthetic@example.test"
GMAIL = f"gmail:{ADDRESS}"
IMAP = f"imap:{ADDRESS}"
DEMO = f"demo:{ADDRESS}"
LOCAL = ADDRESS.split("@", 1)[0]


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "audit.db"))
    return tmp_path


def _ledger() -> AuditLog:
    return AuditLog(db_path=audit_db_path())


def _window() -> dict[str, Any]:
    now = datetime.now(UTC)
    return check_window(_ledger(), since=now - timedelta(hours=1), until=now + timedelta(hours=1))


def _mailbox_invariant(result: dict[str, Any]) -> dict[str, Any]:
    return next(i for i in result["invariants"] if i["id"] == MAILBOX_WRITE_APPROVED)


def _assert_no_address_outside_the_payload(rows: tuple[Any, ...]) -> None:
    for row in rows:
        text = " ".join(
            str(value)
            for value in (
                row.reason,
                row.run_id,
                row.agent_type,
                row.hook_point,
                row.plugin,
                row.decision,
                row.severity,
                row.classification,
                row.tier,
            )
        )
        assert LOCAL not in text, f"row {row.id} ({row.hook_point}) names the address: {text}"


def test_ledger_account_label_names_the_provider_only() -> None:
    assert ledger_account_label(GMAIL) == "a gmail account"
    assert ledger_account_label(IMAP) == "an imap account"
    assert ledger_account_label(ADDRESS) == "an account"  # no provider: say nothing more
    assert ledger_account_label(f"{ADDRESS}:x") == "an account"


def test_no_ledger_row_names_the_address(home: Path) -> None:
    accounts = EmailAccountStore()
    accounts.ensure_schema()
    accounts.add(provider="gmail", address=ADDRESS)
    runner = CliRunner()

    # The owner's commands: approve, revoke (their terminal still shows the address).
    approved = runner.invoke(app, ["email", "writes", "approve", "--account", GMAIL, "--yes"])
    assert approved.exit_code == 0, approved.output
    revoked = runner.invoke(app, ["email", "writes", "revoke", "--account", GMAIL])
    assert revoked.exit_code == 0, revoked.output
    assert ADDRESS in revoked.output

    # The sweep's label sync, refused (no approval): logged, nothing in the ledger.
    judgments = JudgmentStore(db_path=home / "judgments.db")
    judgments.ensure_schema()
    config = JudgeConfig.load(None)
    judgments.record("demo-0001", DEMO, bucket=config.keys[0], confidence=0.9)
    provider = DemoMailProvider(state_path=home / "demo.json")
    refused = sync_labels(judgments, config, {DEMO: provider})
    assert refused.written == 0 and "no approval" in refused.errors[0]
    with pytest.raises(PermissionError):
        require_mailbox_writes(IMAP, "move mail to Trash")

    # The library: grant, write, revoke -- for an IMAP and the demo account.
    for account in (IMAP, DEMO):
        grant_mailbox_writes(account, "test", actor="test", run_id="r-1", agent_type="test")
    done = sync_labels(judgments, config, {DEMO: provider})
    assert done.written == 1
    with mailbox_write(IMAP, "move mail to Trash", op="trash") as tally:
        tally.add(2)
    for account in (IMAP, DEMO):
        assert revoke_mailbox_writes(account, actor="test", agent_type="test") is not None

    rows = _ledger().query()
    by_kind = sorted((r.hook_point, r.decision) for r in rows)
    assert by_kind == sorted(
        [(AUDIT_HOOK, "allow")] * 3 + [(AUDIT_HOOK, "deny")] * 3 + [(WRITE_HOOK, "allow")] * 2
    )
    assert {r.plugin for r in rows} == {AUDIT_PLUGIN}
    _assert_no_address_outside_the_payload(rows)
    # ... and the account is in every payload, for the bundle to pseudonymise.
    assert {json.loads(r.payload_json)["account"] for r in rows} == {GMAIL, IMAP, DEMO}
    assert ADDRESS not in json.dumps(export_bundle(_ledger()))


def test_a_revoke_with_nothing_to_revoke_writes_no_row() -> None:
    assert revoke_mailbox_writes(GMAIL, actor="test", agent_type="test") is None
    assert _ledger().query() == ()


def test_a_write_after_its_approval_holds_from_check_window() -> None:
    grant_mailbox_writes(GMAIL, "test", actor="owner", run_id="r-1", agent_type="test")
    with mailbox_write(GMAIL, "change labels", op="label") as tally:
        tally.add(3)
        tally.add(2)
    [row] = [r for r in _ledger().query() if r.hook_point == WRITE_HOOK]
    assert json.loads(row.payload_json) == {"account": GMAIL, "op": "label", "count": 5}
    assert row.reason == "mailbox write: label x5 on a gmail account"

    result = _window()
    mail = _mailbox_invariant(result)
    assert result["ok"] is True and mail["ok"] is True
    assert mail["evidence"] == {"approval_rows": 1, "writes_observed": 5}
    assert LOCAL not in json.dumps(result)


def test_a_write_without_an_approval_is_a_violation() -> None:
    record_mailbox_write(GMAIL, "trash", 2)  # what a provider that skipped the gate leaves
    mail = _mailbox_invariant(_window())
    assert mail["ok"] is False and mail["violation_count"] == 1
    assert LOCAL not in json.dumps(mail)


def test_a_revoke_row_is_never_an_approval() -> None:
    """A revoke row (decision deny) authorises no account, not even another one."""
    grant_mailbox_writes(GMAIL, "test", actor="owner", run_id="r-1", agent_type="test")
    revoke_mailbox_writes(GMAIL, actor="owner", agent_type="test")
    record_mailbox_write(IMAP, "label", 1)
    assert _mailbox_invariant(_window())["violation_count"] == 1


def test_a_write_after_a_revoke_is_a_violation_until_approved_again() -> None:
    """Invariant 2 is time-ordered: the approval in force at the write's time decides.
    A write that skipped the gate after a revoke fails; after a re-approval, a write
    through the gate holds again."""
    grant_mailbox_writes(GMAIL, "test", actor="owner", run_id="r-1", agent_type="test")
    revoke_mailbox_writes(GMAIL, actor="owner", agent_type="test")
    record_mailbox_write(GMAIL, "label", 1)  # what a provider that skipped the gate leaves
    mail = _mailbox_invariant(_window())
    assert mail["violation_count"] == 1 and "revoked" in mail["violations"][0]

    grant_mailbox_writes(GMAIL, "again", actor="owner", run_id="r-2", agent_type="test")
    with mailbox_write(GMAIL, "change labels", op="label") as tally:
        tally.add(2)
    mail = _mailbox_invariant(_window())
    assert mail["violation_count"] == 1  # the one after the revoke, and only it
    assert LOCAL not in json.dumps(mail)


def test_a_refused_write_records_nothing() -> None:
    with pytest.raises(PermissionError), mailbox_write(GMAIL, "change labels", op="label"):
        raise AssertionError("the body never runs without an approval")
    assert _ledger().query() == ()


def test_a_partial_write_is_recorded_when_the_body_raises() -> None:
    grant_mailbox_writes(GMAIL, "test", actor="owner", run_id="r-1", agent_type="test")
    with pytest.raises(RuntimeError), mailbox_write(GMAIL, "change labels", op="label") as tally:
        tally.add(1000)
        raise RuntimeError("the second batch failed")
    [row] = [r for r in _ledger().query() if r.hook_point == WRITE_HOOK]
    assert json.loads(row.payload_json)["count"] == 1000


def test_a_write_that_changed_nothing_records_nothing() -> None:
    grant_mailbox_writes(GMAIL, "test", actor="owner", run_id="r-1", agent_type="test")
    with mailbox_write(GMAIL, "change labels", op="label"):
        pass
    assert [r.hook_point for r in _ledger().query()] == [AUDIT_HOOK]


def test_a_ledger_that_cannot_be_written_never_fails_a_write_that_happened(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Broken:
        def record(self, **_kw: Any) -> int:
            raise OSError("disk full")

    with caplog.at_level("ERROR"):
        assert record_mailbox_write(GMAIL, "trash", 1, audit_log=Broken()) is None
    assert "has no audit row" in caplog.text and LOCAL not in caplog.text
