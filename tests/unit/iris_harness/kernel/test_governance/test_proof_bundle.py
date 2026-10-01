"""The R14 proof bundle (kernel/governance/audit/proof_bundle.py).

Export cuts the ledger down to ids, decisions, classifications, tiers and digests (no
content, accounts pseudonymised), in the documented schema; verify checks the format,
the integrity and each invariant offline, and fails a bundle that breaks any of them.
The CLI is the same two functions.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from iris_harness.cli.governance import governance_app
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.audit.proof_bundle import (
    EVERY_CALL_AUDITED,
    INVARIANTS,
    MAILBOX_WRITE_APPROVED,
    MAILBOX_WRITE_PERFORMED_HOOK,
    NO_PRIVATE_TO_CLOUD,
    SCHEMA_FILE,
    MailboxWrite,
    Observations,
    check_window,
    content_sha256,
    export_bundle,
    observations_from_json,
    observations_from_session_logs,
    verify_bundle,
)

ACCOUNT = "gmail:owner@example.com"
CANARY = "CANARY-CONTENT-7f3a"
T0 = datetime(2026, 9, 30, 9, 0, tzinfo=UTC)


def _at(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


def _record(log: AuditLog, minute: int, hook_point: str, **kw: Any) -> int:
    fields: dict[str, Any] = {
        "run_id": "run-1",
        "step_id": None,
        "agent_type": "email",
        "plugin": "test",
        "decision": "allow",
        "severity": "info",
        "reason": f"why: {CANARY}",
        "payload": {"session_id": "s-1", "prompt": CANARY},
    }
    fields.update(kw)
    return log.record(hook_point=hook_point, ts=_at(minute), **fields)


@pytest.fixture()
def ledger(tmp_path: Path) -> AuditLog:
    """A good run: an approval, two governed model calls, an answer."""
    log = AuditLog(db_path=tmp_path / "audit.db")
    _record(
        log,
        0,
        "mailbox_writes",
        run_id="setup",
        plugin="email_write_approvals",
        payload={"account": ACCOUNT, "actor": "owner", "ref": "label preview"},
    )
    for step in (0, 1):
        _record(
            log, 10 + step, "pre_llm_call", step_id=step, classification="personal", tier="tier_1"
        )
    _record(log, 12, "pre_response")
    return log


GOOD = Observations(
    model_calls=(("s-1", "tier1"), ("s-1", "tier1")),
    answers=("s-1",),
    mailbox_writes=(MailboxWrite(ACCOUNT, 3, at=_at(20).isoformat()),),
)


def test_a_good_run_exports_a_bundle_that_verifies(ledger: AuditLog) -> None:
    bundle = export_bundle(ledger, observations=GOOD, subject="email-onboarding")
    assert verify_bundle(bundle) == []
    assert [i["id"] for i in bundle["invariants"]] == list(INVARIANTS)
    assert len(bundle["ledger"]) == 4


def test_the_export_matches_the_schema_file(ledger: AuditLog) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    jsonschema.validate(export_bundle(ledger, observations=GOOD), schema)


def test_no_content_and_no_raw_account_leaves(ledger: AuditLog) -> None:
    text = json.dumps(export_bundle(ledger, observations=GOOD))
    assert CANARY not in text  # neither the row's reason nor its payload's text
    assert "owner@example.com" not in text and "actor" not in text


def test_an_account_is_one_pseudonym_inside_a_bundle_and_unlinkable_across(
    ledger: AuditLog,
) -> None:
    first, second = (export_bundle(ledger, observations=GOOD) for _ in range(2))
    [approval] = [r for r in first["ledger"] if r["hook_point"] == "mailbox_writes"]
    assert approval["account_ref"] == first["observations"]["mailbox_writes"][0]["account_ref"]
    assert approval["account_ref"] != next(
        r["account_ref"] for r in second["ledger"] if r["hook_point"] == "mailbox_writes"
    )


def test_an_approval_before_the_window_still_counts(ledger: AuditLog) -> None:
    bundle = export_bundle(ledger, since=_at(5), observations=GOOD)
    assert any(r["hook_point"] == "mailbox_writes" for r in bundle["ledger"])
    assert verify_bundle(bundle) == []


def test_a_run_scope_keeps_only_that_run(ledger: AuditLog) -> None:
    _record(ledger, 30, "pre_llm_call", run_id="other", step_id=0)
    bundle = export_bundle(ledger, run_ids=["run-1"])
    assert {r["run_id"] for r in bundle["ledger"]} == {"run-1", "setup"}


# -- each invariant bites ------------------------------------------------------------------


def _ids(bundle: dict[str, Any]) -> set[str]:
    return {v.invariant for v in verify_bundle(bundle)}


def test_private_data_allowed_to_the_cloud_fails(ledger: AuditLog) -> None:
    _record(ledger, 13, "pre_llm_call", step_id=2, classification="secret", tier="tier_3")
    assert _ids(export_bundle(ledger, observations=GOOD)) == {NO_PRIVATE_TO_CLOUD}


def test_private_data_refused_at_the_cloud_is_fine(ledger: AuditLog) -> None:
    _record(
        ledger, 13, "pre_llm_call", step_id=2, classification="personal", tier="tier_3",
        decision="deny",
    )  # fmt: skip
    assert verify_bundle(export_bundle(ledger, observations=GOOD)) == []


def test_a_write_with_no_approval_row_fails(tmp_path: Path) -> None:
    log = AuditLog(db_path=tmp_path / "audit.db")
    bundle = export_bundle(
        log, observations=Observations(mailbox_writes=(MailboxWrite(ACCOUNT, 2),))
    )
    assert _ids(bundle) == {MAILBOX_WRITE_APPROVED}


def test_another_accounts_approval_does_not_count(ledger: AuditLog) -> None:
    other = Observations(mailbox_writes=(MailboxWrite("imap:someone@example.org", 1),))
    assert _ids(export_bundle(ledger, observations=other)) == {MAILBOX_WRITE_APPROVED}


def test_a_write_before_its_approval_fails(ledger: AuditLog) -> None:
    early = Observations(mailbox_writes=(MailboxWrite(ACCOUNT, 1, at=_at(-5).isoformat()),))
    assert _ids(export_bundle(ledger, observations=early)) == {MAILBOX_WRITE_APPROVED}


def test_an_unaudited_model_call_fails(ledger: AuditLog) -> None:
    three = Observations(model_calls=(("s-1", None),) * 3)
    assert _ids(export_bundle(ledger, observations=three)) == {EVERY_CALL_AUDITED}


def test_a_call_with_no_session_is_counted_against_every_row(ledger: AuditLog) -> None:
    assert (
        verify_bundle(export_bundle(ledger, observations=Observations(((None, None),) * 2))) == []
    )
    assert _ids(export_bundle(ledger, observations=Observations(((None, None),) * 3))) == {
        EVERY_CALL_AUDITED
    }


def test_an_answer_with_no_pre_response_row_fails(ledger: AuditLog) -> None:
    other = Observations(answers=("s-2",))
    assert _ids(export_bundle(ledger, observations=other)) == {EVERY_CALL_AUDITED}


# -- format and integrity ------------------------------------------------------------------


def test_an_edit_that_does_not_recompute_the_digest_fails(ledger: AuditLog) -> None:
    bundle = export_bundle(ledger, observations=GOOD)
    bundle["ledger"].pop()  # drop the pre_response row, quietly
    assert "integrity" in _ids(bundle)


def test_content_smuggled_into_a_row_is_rejected(ledger: AuditLog) -> None:
    bundle = export_bundle(ledger, observations=GOOD)
    bundle["ledger"][0]["reason"] = CANARY
    bundle["content_sha256"] = content_sha256(bundle)
    assert _ids(bundle) == {"format"}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), 2),
        (("format",), "something-else"),
        (("invariants", 0, "statement"), "Anything goes."),
        (("observations", "model_calls"), "many"),
    ],
)
def test_a_malformed_bundle_fails_format(
    ledger: AuditLog, path: tuple[Any, ...], value: Any
) -> None:
    bundle = copy.deepcopy(export_bundle(ledger, observations=GOOD))
    target: Any = bundle
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    bundle["content_sha256"] = content_sha256(bundle)
    assert _ids(bundle) == {"format"}


# -- observations --------------------------------------------------------------------------


def test_session_logs_give_calls_and_answers_never_text(tmp_path: Path) -> None:
    events = [
        {"kind": "llm_call", "session_id": "s-1", "ts": _at(1).isoformat(), "tier": "tier1",
         "input_messages": [{"content": CANARY}]},
        {"kind": "agent_response", "session_id": "s-1", "ts": _at(2).isoformat(),
         "response": CANARY},
        {"kind": "llm_call", "session_id": "s-1", "ts": _at(-60).isoformat(), "tier": "tier1"},
        {"kind": "tool_run", "session_id": "s-1", "ts": _at(3).isoformat()},
    ]  # fmt: skip
    (tmp_path / "session-s-1.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\nnot json\n", encoding="utf-8"
    )
    seen = observations_from_session_logs(tmp_path, since=_at(0))
    assert seen == Observations(model_calls=(("s-1", "tier1"),), answers=("s-1",))
    assert observations_from_session_logs(tmp_path / "missing") == Observations()


def test_observations_from_json() -> None:
    raw = {"mailbox_writes": [{"account": ACCOUNT, "count": 3}], "answers": [{"session_id": "s"}]}
    seen = observations_from_json(raw)
    assert seen.mailbox_writes == (MailboxWrite(ACCOUNT, 3),) and seen.answers == ("s",)


# -- mailbox writes the email library recorded in the ledger -------------------------------


def _write_row(log: AuditLog, minute: int, account: str = ACCOUNT, count: int = 3) -> int:
    """What ``write_approvals.record_mailbox_write`` leaves once a write reached a mailbox."""
    return _record(
        log,
        minute,
        MAILBOX_WRITE_PERFORMED_HOOK,
        run_id="mailbox-write-1",
        plugin="email_write_approvals",
        reason="mailbox write: label x3 on a gmail account",
        payload={"account": account, "op": "label", "count": count},
    )


def test_a_recorded_write_after_its_approval_is_observed_and_holds(ledger: AuditLog) -> None:
    _write_row(ledger, 20)
    bundle = export_bundle(ledger)
    [write] = bundle["observations"]["mailbox_writes"]
    assert write["count"] == 3 and write["at"] == _at(20).isoformat()
    [approval] = [r for r in bundle["ledger"] if r["hook_point"] == "mailbox_writes"]
    assert write["account_ref"] == approval["account_ref"]
    assert verify_bundle(bundle) == []
    assert "owner@example.com" not in json.dumps(bundle)


def test_a_recorded_write_without_an_approval_fails(tmp_path: Path) -> None:
    log = AuditLog(db_path=tmp_path / "audit.db")
    _write_row(log, 20)
    assert _ids(export_bundle(log)) == {MAILBOX_WRITE_APPROVED}


def test_a_recorded_write_before_its_approval_fails(ledger: AuditLog) -> None:
    _write_row(ledger, -5)
    assert _ids(export_bundle(ledger)) == {MAILBOX_WRITE_APPROVED}


def test_a_recorded_write_is_never_an_approval(tmp_path: Path) -> None:
    """Two write rows, no approval row: the earlier write does not authorise the later."""
    log = AuditLog(db_path=tmp_path / "audit.db")
    _write_row(log, 1)
    _write_row(log, 2)
    assert len(verify_bundle(export_bundle(log))) == 2


def test_a_revoke_row_is_not_an_approval(tmp_path: Path) -> None:
    log = AuditLog(db_path=tmp_path / "audit.db")
    _record(
        log, 0, "mailbox_writes", run_id="revoke", plugin="email_write_approvals",
        decision="deny", payload={"account": ACCOUNT, "actor": "owner", "ref": None},
    )  # fmt: skip
    _write_row(log, 5)
    assert _ids(export_bundle(log)) == {MAILBOX_WRITE_APPROVED}


def test_check_window_reads_the_recorded_writes(ledger: AuditLog) -> None:
    _write_row(ledger, 20, count=4)
    result = check_window(ledger, since=_at(-60), until=_at(60))
    [mail] = [i for i in result["invariants"] if i["id"] == MAILBOX_WRITE_APPROVED]
    assert mail["ok"] is True
    assert mail["evidence"] == {"approval_rows": 1, "writes_observed": 4}

    _write_row(ledger, 21, account="imap:someone@example.org", count=1)
    result = check_window(ledger, since=_at(-60), until=_at(60))
    [mail] = [i for i in result["invariants"] if i["id"] == MAILBOX_WRITE_APPROVED]
    assert mail["ok"] is False and mail["violation_count"] == 1
    assert "someone@example.org" not in json.dumps(result)


def test_a_write_outside_the_window_is_not_observed(ledger: AuditLog) -> None:
    _write_row(ledger, 20)
    bundle = export_bundle(ledger, since=_at(30))
    assert bundle["observations"]["mailbox_writes"] == []


def test_caller_observations_add_to_the_recorded_writes(ledger: AuditLog) -> None:
    _write_row(ledger, 20, count=2)
    extra = Observations(mailbox_writes=(MailboxWrite(ACCOUNT, 5, at=_at(25).isoformat()),))
    bundle = export_bundle(ledger, observations=extra)
    assert sorted(w["count"] for w in bundle["observations"]["mailbox_writes"]) == [2, 5]
    assert verify_bundle(bundle) == []


# -- invariant 2 is time-ordered: the latest approval row before a write decides ------------


def _revoke_row(log: AuditLog, minute: int) -> int:
    """What ``write_approvals.revoke_mailbox_writes`` leaves (decision deny)."""
    return _record(
        log,
        minute,
        "mailbox_writes",
        run_id="revoke",
        plugin="email_write_approvals",
        decision="deny",
        payload={"account": ACCOUNT, "actor": "owner", "ref": None},
    )


def _approve_row(log: AuditLog, minute: int) -> int:
    return _record(
        log,
        minute,
        "mailbox_writes",
        run_id="approve",
        plugin="email_write_approvals",
        payload={"account": ACCOUNT, "actor": "owner", "ref": "cli"},
    )


def test_approve_then_write_holds(ledger: AuditLog) -> None:
    _write_row(ledger, 20)
    assert verify_bundle(export_bundle(ledger)) == []


def test_approve_revoke_then_write_is_a_violation(ledger: AuditLog) -> None:
    """The fixture approves at minute 0; a revoke at 10 means the write at 20 has none."""
    _revoke_row(ledger, 10)
    _write_row(ledger, 20)
    [violation] = verify_bundle(export_bundle(ledger))
    assert violation.invariant == MAILBOX_WRITE_APPROVED
    assert "revoked" in violation.detail


def test_approve_revoke_approve_then_write_holds(ledger: AuditLog) -> None:
    _revoke_row(ledger, 10)
    _approve_row(ledger, 15)
    _write_row(ledger, 20)
    assert verify_bundle(export_bundle(ledger)) == []


def test_a_write_before_the_revoke_still_holds(ledger: AuditLog) -> None:
    """A revoke takes back what follows it, never a write that already happened."""
    _write_row(ledger, 5)
    _revoke_row(ledger, 10)
    assert verify_bundle(export_bundle(ledger)) == []


def test_each_write_is_judged_by_the_row_in_force_at_its_time(ledger: AuditLog) -> None:
    _write_row(ledger, 5)  # approved (minute 0)
    _revoke_row(ledger, 10)
    _write_row(ledger, 12)  # revoked
    _approve_row(ledger, 15)
    _write_row(ledger, 20)  # approved again
    violations = verify_bundle(export_bundle(ledger))
    assert [v.invariant for v in violations] == [MAILBOX_WRITE_APPROVED]
    assert _at(12).isoformat() in violations[0].detail


def test_a_caller_observed_write_after_a_revoke_is_a_violation(ledger: AuditLog) -> None:
    _revoke_row(ledger, 10)
    late = Observations(mailbox_writes=(MailboxWrite(ACCOUNT, 1, at=_at(20).isoformat()),))
    assert _ids(export_bundle(ledger, observations=late)) == {MAILBOX_WRITE_APPROVED}


def test_an_untimed_write_is_refused_even_with_an_approval(tmp_path: Path) -> None:
    """With no time a write cannot be ordered against a revoke, so it fails invariant 2
    (owner's decision, 2026-09-30) -- even when the account was approved and never
    revoked. Only a caller's ``--observations`` can lack a time."""
    log = AuditLog(db_path=tmp_path / "audit.db")
    _approve_row(log, 0)
    untimed = Observations(mailbox_writes=(MailboxWrite(ACCOUNT, 1),))
    [violation] = verify_bundle(export_bundle(log, observations=untimed))
    assert violation.invariant == MAILBOX_WRITE_APPROVED
    assert "no time" in violation.detail


# Equal timestamps: a row at exactly the write's time counts as before it, and rows that
# share one time are ordered by ledger id -- the last recorded decides.


def test_an_approval_at_the_writes_own_time_authorises_it(tmp_path: Path) -> None:
    log = AuditLog(db_path=tmp_path / "audit.db")
    _approve_row(log, 20)
    _write_row(log, 20)
    assert verify_bundle(export_bundle(log)) == []


def test_a_revoke_at_the_writes_own_time_fails_it(ledger: AuditLog) -> None:
    _revoke_row(ledger, 20)
    _write_row(ledger, 20)
    assert _ids(export_bundle(ledger)) == {MAILBOX_WRITE_APPROVED}


def test_rows_sharing_a_time_are_decided_by_the_last_recorded(tmp_path: Path) -> None:
    approve_last = AuditLog(db_path=tmp_path / "a.db")
    _revoke_row(approve_last, 0)
    _approve_row(approve_last, 0)
    _write_row(approve_last, 20)
    assert verify_bundle(export_bundle(approve_last)) == []

    revoke_last = AuditLog(db_path=tmp_path / "b.db")
    _approve_row(revoke_last, 0)
    _revoke_row(revoke_last, 0)
    _write_row(revoke_last, 20)
    assert _ids(export_bundle(revoke_last)) == {MAILBOX_WRITE_APPROVED}


def test_an_approval_row_with_an_unreadable_time_fails_closed(ledger: AuditLog) -> None:
    _write_row(ledger, 20)
    bundle = export_bundle(ledger)
    approval = next(r for r in bundle["ledger"] if r["hook_point"] == "mailbox_writes")
    approval["ts"] = "yesterday"
    bundle["content_sha256"] = content_sha256(bundle)
    [violation] = verify_bundle(bundle)
    assert violation.invariant == MAILBOX_WRITE_APPROVED and "ISO 8601" in violation.detail


# -- the CLI -------------------------------------------------------------------------------


def test_the_cli_exports_and_verifies(ledger: AuditLog, tmp_path: Path) -> None:
    runner = CliRunner()
    obs = tmp_path / "obs.json"
    timed = {"account": ACCOUNT, "count": 3, "at": _at(20).isoformat()}
    obs.write_text(json.dumps({"mailbox_writes": [timed]}))
    out = tmp_path / "bundle.json"
    export = [
        "proof-bundle", "export", "--db", str(ledger.db_path), "--out", str(out),
        "--session-logs", str(tmp_path / "no-logs"), "--observations", str(obs),
    ]  # fmt: skip
    made = runner.invoke(governance_app, export)
    assert made.exit_code == 0, made.output
    verified = runner.invoke(governance_app, ["proof-bundle", "verify", str(out)])
    assert verified.exit_code == 0, verified.output

    bundle = json.loads(out.read_text())
    bundle["ledger"] = [r for r in bundle["ledger"] if r["hook_point"] != "mailbox_writes"]
    bundle["content_sha256"] = content_sha256(bundle)
    out.write_text(json.dumps(bundle))
    failed = runner.invoke(governance_app, ["proof-bundle", "verify", str(out)])
    assert failed.exit_code == 1 and MAILBOX_WRITE_APPROVED in failed.output

    # A caller's write without a time is refused: it cannot be ordered against a revoke.
    obs.write_text(json.dumps({"mailbox_writes": [{"account": ACCOUNT, "count": 3}]}))
    assert runner.invoke(governance_app, export).exit_code == 0
    failed = runner.invoke(governance_app, ["proof-bundle", "verify", str(out)])
    assert failed.exit_code == 1 and "cannot be ordered" in " ".join(failed.output.split())

    out.write_text("not json")
    assert runner.invoke(governance_app, ["proof-bundle", "verify", str(out)]).exit_code == 2
