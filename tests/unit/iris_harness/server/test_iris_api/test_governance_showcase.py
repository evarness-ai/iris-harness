"""The builder showcase reads (OSS plan R17/R12): who called, how it was answered, the proof.

``GET /governance/audit`` returns the documented payload fields and a masked reason and
nothing else of a row; it filters by caller. ``GET /governance/proof-bundle/check`` is
export + verify over a window, per invariant, with the evidence each was judged on.
``GET /api/traces/{id}`` lists the turn's hook decisions in order, reasons masked.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from iris_harness.cli.governance import governance_app
from iris_harness.foundation.auth import auth_headers
from iris_harness.foundation.observability import session_log
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.server.iris_api.main import create_app

ADDRESS = "jordan.owner@example.com"
NOW = datetime.now(UTC)


def _ts(seconds: float) -> datetime:
    return NOW - timedelta(minutes=5) + timedelta(seconds=seconds)


@pytest.fixture
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AuditLog:
    db = tmp_path / "audit.db"
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(db))
    monkeypatch.setenv("IRIS_SESSION_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(session_log, "LOG_DIR", None)
    return AuditLog(db_path=db)


@pytest.fixture
def client(ledger: AuditLog) -> Iterator[TestClient]:
    with TestClient(create_app(auto_start_runtime=False), headers=auth_headers()) as c:
        yield c


def _record(log: AuditLog, at: float, hook: str, **kw: object) -> int:
    payload = kw.pop("payload", {})
    return log.record(
        run_id=str(kw.pop("run_id", "run-1")),
        step_id=kw.pop("step_id", 0),  # type: ignore[arg-type]
        agent_type="chat",
        hook_point=hook,
        plugin=str(kw.pop("plugin", "egress_gate")),
        decision=str(kw.pop("decision", "allow")),
        severity="info",
        reason=str(kw.pop("reason", "ok")),
        classification=kw.pop("classification", None),  # type: ignore[arg-type]
        tier=kw.pop("tier", None),  # type: ignore[arg-type]
        payload=payload,  # type: ignore[arg-type]
        ts=_ts(at),
    )


# ---------------------------------------------------------------------------- audit


def test_audit_entry_shows_the_documented_fields_and_nothing_else(
    client: TestClient, ledger: AuditLog
) -> None:
    _record(
        ledger,
        1,
        "post_tool_use",
        plugin="mcp_client_egress",
        decision="deny",
        classification="personal",
        tier="tier_3",
        reason=f"withheld from mcp:desktop; the result named {ADDRESS}",
        payload={
            "caller": "mcp:desktop",
            "tool_name": "email_search",
            "digest_alg": "hmac-sha256/v1/k1",
            "args_digest": "abc",
            "result_digest": "def",
            "args": {"q": ADDRESS},
            "account": f"gmail:{ADDRESS}",
            "exception": f"boom {ADDRESS}",
            "session_id": "s-1",
        },
    )
    _record(
        ledger,
        2,
        "pre_response",
        plugin="curator",
        tier="tier_1",
        payload={"deterministic": True, "handler": "dues_intercept", "audience": "owner"},
    )

    body = client.get("/governance/audit").json()
    answer, call = body["entries"]
    assert call["caller"] == "mcp:desktop"
    assert call["tool_name"] == "email_search"
    assert call["digest_alg"] == "hmac-sha256/v1/k1"
    assert call["session_id"] == "s-1"
    assert call["locality"] == "cloud"
    assert answer["deterministic"] is True
    assert answer["handler"] == "dues_intercept"
    assert answer["locality"] == "local"
    # Never the payload, the digests or an address.
    for key in ("payload_json", "args", "account", "exception", "args_digest", "result_digest"):
        assert key not in call
    assert ADDRESS not in json.dumps(body)
    assert "jo***er@example.com" in call["reason"]


def test_audit_filters_by_caller_exactly_and_by_namespace(
    client: TestClient, ledger: AuditLog
) -> None:
    _record(ledger, 1, "pre_tool_use", payload={"caller": "mcp:desktop"})
    _record(ledger, 2, "pre_tool_use", payload={"caller": "mcp:cli-agent"})
    _record(ledger, 3, "pre_tool_use", payload={"caller": "plugin:email"})
    _record(ledger, 4, "pre_llm_call")

    everything = client.get("/governance/audit").json()
    assert everything["count"] == 4
    assert everything["callers"] == ["mcp:cli-agent", "mcp:desktop", "plugin:email"]

    mcp = client.get("/governance/audit", params={"caller": "mcp:"}).json()
    assert [e["caller"] for e in mcp["entries"]] == ["mcp:cli-agent", "mcp:desktop"]
    one = client.get("/governance/audit", params={"caller": "plugin:email"}).json()
    assert [e["caller"] for e in one["entries"]] == ["plugin:email"]
    assert one["total"] == 4  # the ledger's size, not the filter's


# ---------------------------------------------------------------------------- proof bundle


def test_proof_bundle_check_passes_a_clean_window_with_its_evidence(
    client: TestClient, ledger: AuditLog
) -> None:
    _record(ledger, 1, "pre_llm_call", classification="personal", tier="tier_1")
    _record(ledger, 2, "pre_llm_call", classification="public", tier="tier_3", step_id=1)

    body = client.get("/governance/proof-bundle/check", params={"days": 1}).json()
    assert body["ok"] is True
    assert body["ledger_rows"] == 2
    by_id = {i["id"]: i for i in body["invariants"]}
    assert set(by_id) == {
        "no-private-to-cloud",
        "mailbox-write-approved",
        "every-call-and-answer-audited",
    }
    assert all(i["ok"] for i in by_id.values())
    assert by_id["no-private-to-cloud"]["evidence"] == {"model_call_rows": 2}
    assert by_id["mailbox-write-approved"]["evidence"] == {
        "approval_rows": 0,
        "writes_observed": 0,
    }


def test_proof_bundle_check_fails_private_data_to_cloud_and_unaudited_calls(
    client: TestClient, ledger: AuditLog, tmp_path: Path
) -> None:
    _record(ledger, 1, "pre_llm_call", classification="personal", tier="tier_3")
    logs = tmp_path / "logs"
    logs.mkdir()
    events = [
        {"kind": "llm_call", "session_id": "s-9", "ts": _ts(1).isoformat(), "tier": "tier1"},
        {"kind": "agent_response", "session_id": "s-9", "ts": _ts(2).isoformat()},
    ]
    (logs / "session-s-9.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")

    body = client.get("/governance/proof-bundle/check", params={"days": 1}).json()
    assert body["ok"] is False
    by_id = {i["id"]: i for i in body["invariants"]}
    cloud = by_id["no-private-to-cloud"]
    assert cloud["ok"] is False and cloud["violation_count"] == 1
    assert "personal model call was allowed to tier_3" in cloud["violations"][0]
    audited = by_id["every-call-and-answer-audited"]
    assert audited["ok"] is False
    assert audited["evidence"] == {"model_calls_observed": 1, "answers_observed": 1}
    assert by_id["mailbox-write-approved"]["ok"] is True


def test_proof_bundle_check_observes_mailbox_writes_from_the_ledger(
    client: TestClient, ledger: AuditLog
) -> None:
    """Invariant 2 over the web: the email library's write rows are the observations, and
    each needs an earlier approval row for its account. No address in the response."""
    account = f"gmail:{ADDRESS}"
    _record(
        ledger, 1, "mailbox_writes", plugin="email_write_approvals",
        payload={"account": account, "actor": "owner", "ref": "setup"},
    )  # fmt: skip
    _record(
        ledger, 2, "mailbox_write_performed", plugin="email_write_approvals",
        payload={"account": account, "op": "label", "count": 3},
    )  # fmt: skip
    body = client.get("/governance/proof-bundle/check", params={"days": 1}).json()
    by_id = {i["id"]: i for i in body["invariants"]}
    writes = by_id["mailbox-write-approved"]
    assert writes["ok"] is True
    assert writes["evidence"] == {"approval_rows": 1, "writes_observed": 3}

    _record(
        ledger, 3, "mailbox_write_performed", plugin="email_write_approvals",
        payload={"account": "imap:someone.else@example.org", "op": "trash", "count": 1},
    )  # fmt: skip
    response = client.get("/governance/proof-bundle/check", params={"days": 1})
    writes = {i["id"]: i for i in response.json()["invariants"]}["mailbox-write-approved"]
    assert writes["ok"] is False and writes["violation_count"] == 1
    assert ADDRESS not in response.text and "someone.else" not in response.text


def test_proof_bundle_check_rejects_a_bad_window(client: TestClient) -> None:
    assert client.get("/governance/proof-bundle/check", params={"days": 0}).status_code == 422
    assert client.get("/governance/proof-bundle/check", params={"days": 400}).status_code == 422


# ---------------------------------------------------------------------------- call trace


def test_trace_lists_the_turns_hook_decisions_in_order_masked(
    client: TestClient, ledger: AuditLog, tmp_path: Path
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    events = [
        {"kind": "user_message", "ts": _ts(0).isoformat(), "session_id": "s-1", "text": "hi"},
        {
            "kind": "agent_response",
            "ts": _ts(10).isoformat(),
            "session_id": "s-1",
            "response": "hello",
        },
    ]
    (logs / "session-s-1.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    common = {"run_id": "r", "payload": {"session_id": "s-1"}}
    _record(ledger, 1, "pre_turn", plugin="input_safety", **common)  # type: ignore[arg-type]
    _record(
        ledger,
        2,
        "pre_llm_call",
        classification="personal",
        tier="tier_1",
        run_id="r",
        payload={"session_id": "s-1"},
    )
    _record(
        ledger,
        3,
        "pre_tool_use",
        plugin="caller_policy",
        reason=f"approved for gmail:{ADDRESS}",
        run_id="r",
        payload={"session_id": "s-1", "caller": "plugin:email", "args": {"to": ADDRESS}},
    )
    _record(
        ledger,
        4,
        "pre_response",
        plugin="curator",
        run_id="r",
        payload={"session_id": "s-1", "deterministic": True, "handler": "greeting"},
    )

    trace = client.get("/api/traces/s-1~0").json()
    events_out = trace["governance"]
    assert [e["hook_point"] for e in events_out] == [
        "pre_turn",
        "pre_llm_call",
        "pre_tool_use",
        "pre_response",
    ]
    assert events_out[1]["classification"] == "personal"
    assert events_out[1]["locality"] == "local"
    assert events_out[2]["caller"] == "plugin:email"
    assert "args" not in events_out[2]
    assert events_out[3]["deterministic"] is True
    assert ADDRESS not in json.dumps(trace)


# ---------------------------------------------------------------------------- CLI parity


def test_cli_audit_prints_the_same_view(ledger: AuditLog) -> None:
    _record(ledger, 1, "pre_tool_use", reason=f"for {ADDRESS}", payload={"caller": "mcp:a"})
    _record(ledger, 2, "pre_tool_use", payload={"caller": "plugin:email"})
    out = CliRunner().invoke(governance_app, ["audit", "--caller", "mcp:", "--json"])
    assert out.exit_code == 0, out.output
    view = json.loads(out.output)
    assert [e["caller"] for e in view["entries"]] == ["mcp:a"]
    assert ADDRESS not in out.output


def test_cli_proof_bundle_check_exits_non_zero_on_a_violation(ledger: AuditLog) -> None:
    runner = CliRunner()
    _record(ledger, 1, "pre_llm_call", classification="public", tier="tier_3")
    clean = runner.invoke(governance_app, ["proof-bundle", "check", "--days", "1"])
    assert clean.exit_code == 0, clean.output
    assert "ok no-private-to-cloud" in clean.output
    _record(ledger, 2, "pre_llm_call", classification="secret", tier="tier_3", step_id=1)
    failed = runner.invoke(governance_app, ["proof-bundle", "check", "--days", "1"])
    assert failed.exit_code == 1
    assert "FAIL no-private-to-cloud" in failed.output


# ------------------------------------------------- who ran it: model and tool owner (#129)


def _identity_rows(ledger: AuditLog) -> None:
    _record(
        ledger,
        1,
        "pre_llm_call",
        tier="tier_1",
        run_id="r",
        payload={
            "session_id": "s-1",
            "model": "qwen-test",
            "provider": "ollama",
            # Content that must never surface.
            "prompt": f"write to {ADDRESS}",
            "query": f"what is {ADDRESS}",
        },
    )
    _record(
        ledger,
        2,
        "pre_tool_use",
        run_id="r",
        payload={
            "session_id": "s-1",
            "tool_name": "email_search",
            "tool_plugin": "mail",
            "args": {"q": ADDRESS},
            "result": f"found {ADDRESS}",
            # Not identifiers: a non-string model is dropped, not shown.
            "model": {"name": ADDRESS},
        },
    )


def test_audit_and_trace_name_the_model_and_provider_never_the_prompt(
    client: TestClient, ledger: AuditLog, tmp_path: Path
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "session-s-1.jsonl").write_text(
        "\n".join(
            json.dumps(e)
            for e in (
                {
                    "kind": "user_message",
                    "ts": _ts(0).isoformat(),
                    "session_id": "s-1",
                    "text": "hi",
                },
                {
                    "kind": "agent_response",
                    "ts": _ts(10).isoformat(),
                    "session_id": "s-1",
                    "response": "hello",
                },
            )
        )
        + "\n"
    )
    _identity_rows(ledger)

    audit = client.get("/governance/audit").json()
    tool_row, llm_row = audit["entries"]
    assert (llm_row["model"], llm_row["provider"]) == ("qwen-test", "ollama")
    assert tool_row["tool_plugin"] == "mail"
    assert "model" not in tool_row  # a non-string value is not a public identifier
    trace = client.get("/api/traces/s-1~0").json()
    traced = {e["hook_point"]: e for e in trace["governance"]}
    assert traced["pre_llm_call"]["model"] == "qwen-test"
    assert traced["pre_llm_call"]["provider"] == "ollama"
    assert ADDRESS not in json.dumps(audit) + json.dumps(trace)
    for entry in audit["entries"] + trace["governance"]:
        assert not {"prompt", "query", "args", "result"} & set(entry)


def test_cli_audit_table_has_a_by_column_naming_the_owner_or_model(ledger: AuditLog) -> None:
    _identity_rows(ledger)
    out = CliRunner().invoke(governance_app, ["audit"], env={"COLUMNS": "240"})
    assert out.exit_code == 0, out.output
    header = next(line for line in out.output.splitlines() if "hook" in line and "caller" in line)
    assert " by " in header
    rows = {
        hook: line
        for line in out.output.splitlines()
        for hook in ("pre_llm_call", "pre_tool_use")
        if hook in line
    }
    assert "qwen-test" in rows["pre_llm_call"]
    assert "mail" in rows["pre_tool_use"]
    assert ADDRESS not in out.output


# ------------------------------------------------- which call a row is about (#134)

CALL = "01JABCDEFGHJKMNPQRSTVWXYZ0"
HELD = "01JABCDEFGHJKMNPQRSTVWXYZ1"


def _call_rows(ledger: AuditLog) -> None:
    _record(
        ledger,
        1,
        "pre_tool_use",
        run_id="r",
        payload={
            "session_id": "s-1",
            "tool_name": "email_search",
            "call_id": CALL,
            "held_call_id": HELD,
            # Never shown: argument text, and the old alias is not a public field.
            "args": {"q": ADDRESS},
            "tool_call_id": CALL,
        },
    )
    _record(
        ledger,
        2,
        "post_tool_use",
        run_id="r",
        payload={"session_id": "s-1", "tool_name": "email_search", "call_id": {"x": ADDRESS}},
    )


def test_audit_and_trace_show_the_call_id_as_an_identifier_only(
    client: TestClient, ledger: AuditLog, tmp_path: Path
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "session-s-1.jsonl").write_text(
        "\n".join(
            json.dumps(e)
            for e in (
                {
                    "kind": "user_message",
                    "ts": _ts(0).isoformat(),
                    "session_id": "s-1",
                    "text": "x",
                },
                {
                    "kind": "agent_response",
                    "ts": _ts(10).isoformat(),
                    "session_id": "s-1",
                    "response": "y",
                },
            )
        )
        + "\n"
    )
    _call_rows(ledger)

    audit = client.get("/governance/audit").json()
    post_row, pre_row = audit["entries"]
    assert (pre_row["call_id"], pre_row["held_call_id"]) == (CALL, HELD)
    assert "call_id" not in post_row  # a non-string value is not an identifier
    trace = client.get("/api/traces/s-1~0").json()
    traced = {e["hook_point"]: e for e in trace["governance"]}
    assert traced["pre_tool_use"]["call_id"] == CALL
    assert ADDRESS not in json.dumps(audit) + json.dumps(trace)
    for entry in audit["entries"] + trace["governance"]:
        assert not {"args", "result", "tool_call_id"} & set(entry)


def test_cli_audit_table_and_json_show_the_call_id(ledger: AuditLog) -> None:
    _call_rows(ledger)
    out = CliRunner().invoke(governance_app, ["audit"], env={"COLUMNS": "300"})
    assert out.exit_code == 0, out.output
    header = next(line for line in out.output.splitlines() if "hook" in line and "caller" in line)
    assert " call " in header
    assert CALL in out.output and ADDRESS not in out.output
    as_json = CliRunner().invoke(governance_app, ["audit", "--json"])
    assert as_json.exit_code == 0 and CALL in as_json.output and ADDRESS not in as_json.output
