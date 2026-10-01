"""``iris email writes approve|status|revoke`` and the System Health row (OSS plan R4).

Accounts, approvals and the audit ledger all live in this test's own directories.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.main import app
from iris_harness.sdk.audit import AuditLog, audit_db_path
from iris_harness.sdk.health import CheckKind, HealthState
from iris_personal.email.accounts import EmailAccountStore
from iris_personal.email.write_approvals import (
    approve_mailbox_writes,
    mailbox_write_approval,
    mailbox_writes_approved,
)
from iris_personal.plugins.email_workflows.cli_writes import AUDIT_PLUGIN
from iris_personal.plugins.email_workflows.writes_health import TARGET, write_approval_checks

GMAIL = "gmail:owner@example.com"
IMAP = "imap:other@example.org"


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "audit.db"))
    store = EmailAccountStore()
    store.ensure_schema()
    store.add(provider="gmail", address="owner@example.com")
    store.add(provider="imap", address="other@example.org")
    return tmp_path


def _run(*args: str, input: str | None = None) -> tuple[int, str]:
    result = CliRunner().invoke(app, ["email", "writes", *args], input=input)
    return result.exit_code, result.output


def _audit_rows() -> list[tuple[str, str]]:
    rows = AuditLog(db_path=audit_db_path()).query(plugin=AUDIT_PLUGIN)
    return [(r.decision, json.loads(r.payload_json)["account"]) for r in rows]


def test_approve_asks_first_and_no_changes_nothing() -> None:
    code, out = _run("approve", "--account", GMAIL, input="n\n")
    assert code == 1
    assert "Approve mailbox writes" in out
    assert not mailbox_writes_approved(GMAIL)
    assert _audit_rows() == []


def test_approve_by_address_records_the_approval_and_an_audit_row() -> None:
    code, out = _run("approve", "--account", "owner@example.com", "--ref", "backfill", input="y\n")
    assert code == 0, out
    approval = mailbox_write_approval(GMAIL)
    assert approval is not None
    assert _audit_rows() == [("allow", GMAIL)]
    row_id = AuditLog(db_path=audit_db_path()).query(plugin=AUDIT_PLUGIN)[0].id
    assert approval.approval_ref == f"backfill [audit #{row_id}]"
    assert not mailbox_writes_approved(IMAP)  # per account


def test_approve_yes_does_not_ask() -> None:
    code, out = _run("approve", "--account", IMAP, "--yes")
    assert code == 0, out
    assert "Approve mailbox writes" not in out
    approval = mailbox_write_approval(IMAP)
    assert approval is not None and approval.approval_ref.startswith("iris email writes approve")


def test_status_shows_each_account() -> None:
    approve_mailbox_writes(GMAIL, "setup step 6")
    code, out = _run("status")
    assert code == 0
    assert "approved" in out and "setup step 6" in out
    assert "not approved" in out and "writes blocked" in out
    assert f"iris email writes approve --account {IMAP}" in out.replace("\n", "")

    code, out = _run("status", "--account", IMAP)
    assert GMAIL not in out and "not approved" in out


def test_revoke_takes_it_back_with_an_audit_row() -> None:
    _run("approve", "--account", GMAIL, "--yes")
    code, out = _run("revoke", "--account", GMAIL)
    assert code == 0, out
    assert not mailbox_writes_approved(GMAIL)
    assert _audit_rows() == [("allow", GMAIL), ("deny", GMAIL)]

    code, out = _run("revoke", "--account", GMAIL)  # nothing left to revoke
    assert code == 0 and "nothing to revoke" in out
    assert len(_audit_rows()) == 2


def test_the_ledger_reasons_name_the_provider_never_the_address() -> None:
    """The owner's terminal shows the address; the ledger's free text does not (the
    account id stays in the payload, which the proof bundle pseudonymises)."""
    code, out = _run("approve", "--account", IMAP, "--yes")
    assert code == 0 and IMAP in out
    _run("revoke", "--account", IMAP)
    rows = AuditLog(db_path=audit_db_path()).query(plugin=AUDIT_PLUGIN)
    assert [r.reason for r in rows] == [
        "mailbox writes approved for an imap account",
        "mailbox writes revoked for an imap account",
    ]
    assert all(json.loads(r.payload_json)["account"] == IMAP for r in rows)


@pytest.mark.parametrize("command", ["approve", "revoke", "status"])
def test_an_unknown_account_exits_2_and_changes_nothing(command: str) -> None:
    extra = ["--yes"] if command == "approve" else []
    code, out = _run(command, "--account", "nobody@example.com", *extra)
    assert code == 2
    assert "no mailbox account" in out and GMAIL in out
    assert _audit_rows() == []


def test_an_address_on_two_accounts_asks_for_the_id() -> None:
    EmailAccountStore().add(provider="imap", address="owner@example.com")
    code, out = _run("approve", "--account", "owner@example.com", "--yes")
    assert code == 2
    assert "more than one account" in out and "pass the account id" in out
    assert not mailbox_writes_approved(GMAIL)


# ─── System Health ──────────────────────────────────────────────────────────


def test_health_warns_for_each_mounted_mailbox_without_approval() -> None:
    approve_mailbox_writes(IMAP, "test")
    calendar = "gcalendar:owner@example.com"  # shares the table; not a mailbox

    def accounts() -> list[tuple[str, str]]:
        return [
            (GMAIL, "owner@example.com"),
            (IMAP, "other@example.org"),
            (calendar, "owner@example.com"),
        ]

    def provider_for(account_id: str) -> object | None:
        return None if account_id == calendar else object()

    rows = write_approval_checks(accounts=accounts, provider_for=provider_for)

    assert len(rows) == 1
    row = rows[0]
    assert (row.target, row.kind, row.state) == (TARGET, CheckKind.CREDENTIAL, HealthState.YELLOW)
    assert row.subject == "owner@example.com"
    assert row.action == f"iris email writes approve --account {GMAIL}"
    assert "not approved" in row.detail


def test_health_has_no_row_once_approved() -> None:
    approve_mailbox_writes(GMAIL, "test")
    rows = write_approval_checks(
        accounts=lambda: [(GMAIL, "owner@example.com")], provider_for=lambda a: object()
    )
    assert rows == []


def test_the_plugin_registers_the_health_row(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_personal.email import providers
    from iris_personal.plugins.email_workflows import plugin

    checks: dict[str, object] = {}

    class _Api:
        def subscribe(self, *_a: object, **_k: object) -> None: ...
        def register_heartbeat(self, *_a: object, **_k: object) -> None: ...
        def register_api_router(self, *_a: object, **_k: object) -> None: ...
        def register_credential_check(self, name: str, check: object) -> None:
            checks[name] = check

    monkeypatch.setattr(plugin, "_register_agent", lambda _api: None)
    monkeypatch.setattr("iris_personal.plugins.email_workflows.tools.register", lambda _api: None)
    monkeypatch.setattr(
        "iris_personal.plugins.email_workflows.judge_surfaces.register", lambda _api: None
    )
    monkeypatch.setattr(providers, "mail_provider_for", lambda account_id: object())
    plugin.setup(_Api())  # type: ignore[arg-type]

    rows = checks["email_write_approvals"](False)  # type: ignore[operator]
    assert {r.subject for r in rows} == {"owner@example.com", "other@example.org"}
