"""Tests for the read-only Governance + Settings endpoints (Phase 6)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.server.iris_api.main import create_app


@pytest.fixture
def audit_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    db = tmp_path / "audit.db"
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(db))
    log = AuditLog(db_path=db)
    log.record(
        run_id="run-1",
        step_id=0,
        agent_type="chat",
        hook_point="pre_llm_call",
        plugin="DataClassifierHook",
        decision="allow",
        severity="info",
        reason="public content",
        classification="public",
        tier="tier_1",
    )
    log.record(
        run_id="run-1",
        step_id=1,
        agent_type="chat",
        hook_point="pre_tool_use",
        plugin="EgressGate",
        decision="deny",
        severity="warn",
        reason="secret would egress",
        classification="secret",
    )
    with TestClient(create_app(auto_start_runtime=False), headers=auth_headers()) as c:
        yield c


def test_governance_state_reports_flags(audit_client: TestClient) -> None:
    body = audit_client.get("/governance/state").json()
    assert body["enabled"] is True  # secure default
    assert body["audit_count"] == 2
    keys = {f["key"]: f["on"] for f in body["flags"]}
    assert keys["IRIS_GOVERNANCE_COMMAND_SANDBOX"] is True  # default-on
    assert keys["IRIS_GOVERNANCE_PROMPT_GUARD"] is False  # default-off (opt-in)
    assert keys["IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER"] is True  # default-on (issue #73)
    assert keys["IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL"] is False  # scope: opt-in


@pytest.mark.parametrize(
    ("ledger", "scope", "want_ledger", "want_all"),
    [
        ("1", None, True, False),  # explicit on is the same as unset
        (None, "1", True, True),
        ("0", None, False, False),
        ("0", "1", False, False),  # no effect while the ledger is off
        ("maybe", "maybe", True, False),  # unrecognised: each setting's default
    ],
)
def test_governance_state_reports_both_side_effect_ledger_settings(
    audit_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    ledger: str | None,
    scope: str | None,
    want_ledger: bool,
    want_all: bool,
) -> None:
    for name, raw in (
        ("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER", ledger),
        ("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL", scope),
    ):
        if raw is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, raw)
    keys = {f["key"]: f["on"] for f in audit_client.get("/governance/state").json()["flags"]}
    assert keys["IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER"] is want_ledger
    assert keys["IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL"] is want_all


def test_governance_audit_returns_recent_decisions_newest_first(audit_client: TestClient) -> None:
    body = audit_client.get("/governance/audit").json()
    assert body["total"] == 2
    assert body["count"] == 2
    assert body["entries"][0]["decision"] == "deny"  # newest first
    assert body["entries"][0]["plugin"] == "EgressGate"


def test_governance_audit_filters_by_decision(audit_client: TestClient) -> None:
    body = audit_client.get("/governance/audit", params={"decision": "deny"}).json()
    assert body["count"] == 1
    assert all(e["decision"] == "deny" for e in body["entries"])


def test_settings_dumps_config_without_secrets(tmp_path: Path) -> None:
    tier_router = SimpleNamespace(
        _tiers={
            "tier1": SimpleNamespace(
                name="tier1",
                provider="ollama",
                model="granite4:latest",
                max_tokens=2048,
                temperature=0.2,
                use_for=("general",),
            )
        },
        intent_tier_map=lambda: {"general": "tier1"},
    )
    runtime = SimpleNamespace(
        tier_router=tier_router,
        config_dir=tmp_path / "config",
        data_dir=tmp_path / "data",
    )
    with TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    ) as c:
        body = c.get("/settings").json()

    assert body["tiers"][0]["model"] == "granite4:latest"
    assert body["intent_tier_map"] == {"general": "tier1"}
    # Providers are presence booleans — never secret values.
    assert set(body["providers"]) == {"anthropic", "openrouter", "github"}
    assert all(isinstance(v, bool) for v in body["providers"].values())
    assert "host" in body and "stores" in body and "flags" in body
    # The core's ADR-0072 feedback flags are surfaced (read-only); defaults are OFF. The
    # plugins' flags (semantic email search, finance dues) are not: the core carries no
    # plugin's vocabulary, and /settings/catalog lists them (ADR-0120).
    flag_keys = {f["key"] for f in body["flags"]}
    assert {"IRIS_FEEDBACK_CAPTURE", "IRIS_FEEDBACK_CLARIFY"} <= flag_keys
    assert not flag_keys & {"IRIS_EMAIL_SEMANTIC_SEARCH", "IRIS_FINANCE_DUES_FROM_EMAIL"}
