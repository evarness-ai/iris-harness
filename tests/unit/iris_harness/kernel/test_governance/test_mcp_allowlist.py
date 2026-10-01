"""Unit tests for MCPAllowlistHook (story 12.gov-4.6).

Mirrors the acceptance criteria in
``docs/stories/12.gov-4.6.mcp-per-persona-allowlist.story.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from iris_harness.kernel.governance import HookContext, HookPoint
from iris_harness.kernel.governance.plugins.mcp_allowlist import (
    MCPAllowlistHook,
    MCPPersonaGrant,
    MCPServerGovernance,
    verify_mcp_signature,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _gov_map(
    *,
    github_developer_tools: tuple[str, ...] = ("*",),
    github_tester_tools: tuple[str, ...] | None = None,
    filesystem_developer_tools: tuple[str, ...] = ("*",),
    filesystem_tester_tools: tuple[str, ...] = ("read_file", "list_directory"),
) -> dict[str, MCPServerGovernance]:
    github_grants: list[MCPPersonaGrant] = [
        MCPPersonaGrant(persona="developer", tools=github_developer_tools),
    ]
    if github_tester_tools is not None:
        github_grants.append(MCPPersonaGrant(persona="tester", tools=github_tester_tools))

    return {
        "github": MCPServerGovernance(allowed_for_personas=tuple(github_grants)),
        "filesystem": MCPServerGovernance(
            allowed_for_personas=(
                MCPPersonaGrant(persona="developer", tools=filesystem_developer_tools),
                MCPPersonaGrant(persona="tester", tools=filesystem_tester_tools),
            )
        ),
    }


def _ctx(
    *,
    mcp_server: str | None = "github",
    mcp_tool: str | None = "create_issue",
    persona: str | None = "developer",
    agent_type: str = "mcp",
    extra_payload: dict[str, object] | None = None,
) -> HookContext:
    payload: dict[str, object] = {}
    if mcp_server is not None:
        payload["mcp_server"] = mcp_server
        payload["tool_name"] = f"mcp/{mcp_server}/{mcp_tool}"
    if mcp_tool is not None:
        payload["mcp_tool"] = mcp_tool
    if extra_payload:
        payload.update(extra_payload)
    return HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="r-1",
        agent_type=agent_type,
        persona=persona,
        route=f"mcp/{mcp_server}/{mcp_tool}" if mcp_server and mcp_tool else "unknown",
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Plugin metadata
# ---------------------------------------------------------------------------


def test_plugin_metadata() -> None:
    hook = MCPAllowlistHook()
    assert hook.name == "mcp_allowlist"
    assert hook.hook_point == HookPoint.PRE_TOOL_USE
    # Must run between PersonaSurface (15) and ToolPolicyHook (20).
    assert hook.priority == 18


# ---------------------------------------------------------------------------
# AC-1: server with no governance block → warn + allow
# ---------------------------------------------------------------------------


async def test_ac1_ungoverned_server_allows_with_warn_audit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    hook = MCPAllowlistHook(governance_map=_gov_map())  # only github + filesystem
    with caplog.at_level("WARNING"):
        decision = await hook(_ctx(mcp_server="unknown-server", mcp_tool="some_tool"))
    assert decision.outcome == "allow"
    assert decision.severity == "warn"
    assert decision.reason == MCPAllowlistHook.ALLOW_REASON_UNGOVERNED
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("no governance block" in r.message for r in warnings)


# ---------------------------------------------------------------------------
# AC-2: developer + tools: ["*"] → allow
# ---------------------------------------------------------------------------


async def test_ac2_developer_wildcard_tool_allowed() -> None:
    hook = MCPAllowlistHook(governance_map=_gov_map())
    decision = await hook(_ctx(mcp_server="github", mcp_tool="create_issue"))
    assert decision.outcome == "allow"
    assert decision.audit_metadata["mcp_server"] == "github"
    assert decision.audit_metadata["mcp_tool"] == "create_issue"
    assert decision.audit_metadata["matched_pattern"] == "*"


async def test_ac2_developer_named_tool_allowed() -> None:
    hook = MCPAllowlistHook(
        governance_map=_gov_map(github_developer_tools=("create_issue", "list_prs"))
    )
    decision = await hook(_ctx(mcp_server="github", mcp_tool="create_issue"))
    assert decision.outcome == "allow"


# ---------------------------------------------------------------------------
# AC-3: tester + tool not in list → deny, reason names (persona, server, tool)
# ---------------------------------------------------------------------------


async def test_ac3_tester_unlisted_tool_denied() -> None:
    hook = MCPAllowlistHook(
        governance_map=_gov_map(github_tester_tools=("read_diagnostic", "list_test_results"))
    )
    decision = await hook(_ctx(mcp_server="github", mcp_tool="delete_repo", persona="tester"))
    assert decision.outcome == "deny"
    assert decision.severity == "error"
    assert "tester" in decision.reason
    assert "github" in decision.reason
    assert "delete_repo" in decision.reason
    assert decision.audit_metadata["deny_reason"] == "tool_not_allowed"


async def test_ac3_tester_listed_tool_allowed() -> None:
    hook = MCPAllowlistHook(
        governance_map=_gov_map(github_tester_tools=("read_diagnostic", "list_test_results"))
    )
    decision = await hook(_ctx(mcp_server="github", mcp_tool="read_diagnostic", persona="tester"))
    assert decision.outcome == "allow"


# ---------------------------------------------------------------------------
# AC-4: persona not in allowed_for_personas → default-deny
# ---------------------------------------------------------------------------


async def test_ac4_persona_not_listed_denied() -> None:
    # analyst is not in the github governance block.
    hook = MCPAllowlistHook(governance_map=_gov_map())
    decision = await hook(_ctx(mcp_server="github", mcp_tool="list_prs", persona="analyst"))
    assert decision.outcome == "deny"
    assert decision.severity == "error"
    assert "analyst" in decision.reason
    assert "github" in decision.reason
    assert decision.audit_metadata["deny_reason"] == "persona_not_allowed"


async def test_ac4_sm_not_in_github_block_denied() -> None:
    hook = MCPAllowlistHook(governance_map=_gov_map())  # sm not in github block
    decision = await hook(_ctx(mcp_server="github", mcp_tool="create_pr", persona="sm"))
    assert decision.outcome == "deny"
    assert "sm" in decision.reason


# ---------------------------------------------------------------------------
# AC-5: verified via mcp_bridge tests — see test_mcp_bridge.py
#        Here we confirm the payload field is what the hook reads.
# ---------------------------------------------------------------------------


async def test_ac5_hook_reads_mcp_server_and_mcp_tool_from_payload() -> None:
    """Hook correctly reads the (mcp_server, mcp_tool) tuple the bridge sets."""
    hook = MCPAllowlistHook(governance_map=_gov_map())
    ctx = HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="r-1",
        agent_type="mcp",
        persona="developer",
        route="mcp/github/create_pr",
        payload={
            "tool_name": "mcp/github/create_pr",
            "mcp_server": "github",
            "mcp_tool": "create_pr",
            "args": {"title": "feat: …"},
        },
    )
    decision = await hook(ctx)
    assert decision.outcome == "allow"
    assert decision.audit_metadata["mcp_server"] == "github"
    assert decision.audit_metadata["mcp_tool"] == "create_pr"


# ---------------------------------------------------------------------------
# AC-6: verify_mcp_signature returns True (v1 stub) + skip-marked Phase 6 test
# ---------------------------------------------------------------------------


def test_ac6_verify_mcp_signature_returns_true_for_all_servers() -> None:
    """v1 stub always returns True — Phase 6 will implement real signing."""
    assert verify_mcp_signature("github", Path("/tmp/manifest.json")) is True
    assert verify_mcp_signature("filesystem", Path("/tmp/manifest.json")) is True
    assert verify_mcp_signature("unknown-server", Path("/tmp/manifest.json")) is True


def test_ac6_verify_mcp_signature_real_signing() -> None:
    """Phase 6 (6b.1/6b.2): real Ed25519 verification now lives in
    ``iris_harness.kernel.governance.mcp_signing`` and is wired into the MCP bridge gate."""
    from iris_harness.kernel.governance.mcp_signing import (
        ServerSpec,
        TrustedKey,
        TrustStore,
        generate_keypair,
        sign,
        verify_server_signature,
    )

    spec = ServerSpec(name="github", transport="stdio", command="/usr/local/bin/gh-mcp")
    private, public = generate_keypair()
    signature = sign(private, spec.canonical_bytes())
    store = TrustStore(keys={"ops": TrustedKey(key_id="ops", public_key=public)})

    good = verify_server_signature(
        spec=spec, signature=signature, signed_by="ops", trust_store=store
    )
    assert good.status == "verified"

    tampered = ServerSpec(name="github", transport="stdio", command="/usr/local/bin/evil")
    bad = verify_server_signature(
        spec=tampered, signature=signature, signed_by="ops", trust_store=store
    )
    assert bad.status == "invalid"


# ---------------------------------------------------------------------------
# Wildcard tool matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool,pattern,expected_allow",
    [
        ("create_issue", "*", True),
        ("list_prs", "*", True),
        ("read_file", "read_*", True),
        ("write_file", "read_*", False),
        ("list_directory", "list_*", True),
        ("delete_repo", "create_*,list_*", False),
    ],
)
async def test_wildcard_tool_patterns(tool: str, pattern: str, expected_allow: bool) -> None:
    tools = tuple(p.strip() for p in pattern.split(","))
    gov_map = {
        "github": MCPServerGovernance(
            allowed_for_personas=(MCPPersonaGrant(persona="developer", tools=tools),)
        )
    }
    hook = MCPAllowlistHook(governance_map=gov_map)
    decision = await hook(_ctx(mcp_server="github", mcp_tool=tool))
    assert decision.outcome == (
        "allow" if expected_allow else "deny"
    ), f"tool={tool!r} pattern={pattern!r}: expected {'allow' if expected_allow else 'deny'}"


# ---------------------------------------------------------------------------
# Short-circuit paths
# ---------------------------------------------------------------------------


async def test_non_mcp_dispatch_short_circuits_allow() -> None:
    """Payloads without ``mcp_server`` are never MCP dispatches."""
    hook = MCPAllowlistHook(governance_map=_gov_map())
    ctx = HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="r-1",
        agent_type="coding",
        persona="developer",
        payload={"tool_name": "run_command", "args": {"command": "pytest"}},
    )
    decision = await hook(ctx)
    assert decision.outcome == "allow"
    assert decision.reason == MCPAllowlistHook.ALLOW_REASON_NOT_MCP


async def test_no_governance_map_degrades_to_allow_with_single_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    hook = MCPAllowlistHook(governance_map=None)
    with caplog.at_level("WARNING"):
        d1 = await hook(_ctx())
        d2 = await hook(_ctx(mcp_tool="delete_repo"))
    assert d1.outcome == "allow"
    assert d1.reason == MCPAllowlistHook.ALLOW_REASON_DEGRADED
    assert d2.outcome == "allow"
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1


async def test_missing_persona_in_mcp_dispatch_denied() -> None:
    hook = MCPAllowlistHook(governance_map=_gov_map())
    decision = await hook(_ctx(persona=None))
    assert decision.outcome == "deny"
    assert "missing persona" in decision.reason


async def test_missing_mcp_tool_in_payload_denied() -> None:
    """mcp_server present but mcp_tool absent → deny."""
    hook = MCPAllowlistHook(governance_map=_gov_map())
    ctx = HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="r-1",
        agent_type="mcp",
        persona="developer",
        payload={"mcp_server": "github"},  # mcp_tool missing
    )
    decision = await hook(ctx)
    assert decision.outcome == "deny"
    assert "mcp_tool" in decision.reason


# ---------------------------------------------------------------------------
# Filesystem server (different persona surface)
# ---------------------------------------------------------------------------


async def test_tester_allowed_for_filesystem_read_tool() -> None:
    hook = MCPAllowlistHook(governance_map=_gov_map())
    decision = await hook(_ctx(mcp_server="filesystem", mcp_tool="read_file", persona="tester"))
    assert decision.outcome == "allow"


async def test_tester_denied_for_filesystem_write_tool() -> None:
    hook = MCPAllowlistHook(governance_map=_gov_map())
    decision = await hook(_ctx(mcp_server="filesystem", mcp_tool="write_file", persona="tester"))
    assert decision.outcome == "deny"
    assert decision.audit_metadata["deny_reason"] == "tool_not_allowed"


# ---------------------------------------------------------------------------
# MCPServerGovernance + MCPPersonaGrant model validation
# ---------------------------------------------------------------------------


def test_mcp_server_governance_parses_correctly() -> None:
    gov = MCPServerGovernance.model_validate(
        {
            "allowed_for_personas": [
                {"persona": "developer", "tools": ["*"]},
                {"persona": "tester", "tools": ["read_file", "list_directory"]},
            ]
        }
    )
    assert len(gov.allowed_for_personas) == 2
    assert gov.allowed_for_personas[0].persona == "developer"
    assert gov.allowed_for_personas[0].tools == ("*",)
    assert gov.allowed_for_personas[1].tools == ("read_file", "list_directory")


def test_mcp_server_governance_empty_allowed_for_personas() -> None:
    gov = MCPServerGovernance()
    assert gov.allowed_for_personas == ()


def test_mcp_persona_grant_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        MCPPersonaGrant.model_validate({"persona": "developer", "tools": ["*"], "unexpected": True})
