"""Unit tests for NetworkEgress (story 12.gov-4.5).

Mirrors the acceptance criteria in
``docs/stories/12.gov-4.5.network-egress.story.md``.
"""

from __future__ import annotations

import pytest

from iris_harness.kernel.governance import HookContext, HookPoint
from iris_harness.kernel.governance.plugins.network_egress import (
    CLOUD_LLM_HOSTNAMES,
    DEFAULT_NETWORK_TOOLS,
    NetworkEgress,
    _parse_url,
    is_network_tool,
)
from iris_harness.kernel.governance.plugins.persona_surface import (
    PersonaPolicy,
    PersonaPolicyDocument,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _policy(
    *,
    developer_domains: tuple[str, ...] = ("api.github.com",),
    sm_domains: tuple[str, ...] = ("*.github.com", "slack.com"),
    analyst_domains: tuple[str, ...] = (),
) -> PersonaPolicyDocument:
    return PersonaPolicyDocument(
        personas={
            "developer": PersonaPolicy(
                name="developer",
                allowed_tools=("github_create_issue", "github_create_pr", "web_fetch"),
                network_egress_domains=developer_domains,
            ),
            "sm": PersonaPolicy(
                name="sm",
                allowed_tools=("github_*", "slack_post"),
                network_egress_domains=sm_domains,
            ),
            "analyst": PersonaPolicy(
                name="analyst",
                allowed_tools=("research",),
                network_egress_domains=analyst_domains,
            ),
        }
    )


def _ctx(
    *,
    tool: str = "github_create_issue",
    url: str | None = None,
    persona: str | None = "developer",
    agent_type: str = "coding",
    nested_url: bool = False,
    extra_payload: dict[str, object] | None = None,
) -> HookContext:
    payload: dict[str, object] = {"tool_name": tool}
    if url is not None:
        if nested_url:
            payload["args"] = {"url": url}
        else:
            payload["url"] = url
    if extra_payload:
        payload.update(extra_payload)
    return HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="r-1",
        agent_type=agent_type,
        persona=persona,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Plugin metadata
# ---------------------------------------------------------------------------


def test_plugin_metadata() -> None:
    egress = NetworkEgress(policy=None)
    assert egress.name == "network_egress"
    assert egress.hook_point == HookPoint.PRE_TOOL_USE
    # Must run after FSJail (30).
    assert egress.priority == 35


# ---------------------------------------------------------------------------
# AC-1: developer + api.github.com → allow
# ---------------------------------------------------------------------------


async def test_ac1_developer_allowed_domain() -> None:
    egress = NetworkEgress(policy=_policy())
    decision = await egress(_ctx(url="https://api.github.com/repos/x/y/issues"))
    assert decision.outcome == "allow"
    assert decision.audit_metadata["hostname"] == "api.github.com"
    # Query-string must be absent in the audit URL.
    assert "?" not in decision.audit_metadata["url"]


# ---------------------------------------------------------------------------
# AC-2: developer + unlisted domain → deny (hostname_not_allowed)
# ---------------------------------------------------------------------------


async def test_ac2_developer_denied_unlisted_domain() -> None:
    egress = NetworkEgress(policy=_policy())
    decision = await egress(_ctx(url="https://evil.example.com/steal"))
    assert decision.outcome == "deny"
    assert decision.severity == "error"
    assert "hostname_not_allowed" in decision.reason
    assert decision.audit_metadata["deny_reason"] == "hostname_not_allowed"
    assert decision.audit_metadata["hostname"] == "evil.example.com"


# ---------------------------------------------------------------------------
# AC-3: sm + *.github.com wildcard → allow
# ---------------------------------------------------------------------------


async def test_ac3_sm_wildcard_github_allowed() -> None:
    egress = NetworkEgress(policy=_policy())
    decision = await egress(
        _ctx(
            tool="github_create_pr",
            url="https://api.github.com/repos/org/repo/pulls",
            persona="sm",
        )
    )
    assert decision.outcome == "allow"
    assert decision.audit_metadata["hostname"] == "api.github.com"


async def test_ac3_sm_wildcard_does_not_match_root_domain() -> None:
    """``*.github.com`` must NOT match ``github.com`` (no subdomain)."""
    egress = NetworkEgress(policy=_policy())
    decision = await egress(
        _ctx(tool="github_create_pr", url="https://github.com/org/repo", persona="sm")
    )
    # github.com is not covered by *.github.com — should deny.
    assert decision.outcome == "deny"
    assert decision.audit_metadata["hostname"] == "github.com"


async def test_ac3_sm_slack_allowed() -> None:
    # Use ``http_request`` (in DEFAULT_NETWORK_TOOLS) to exercise the
    # slack.com entry; ``slack_post`` is a named tool not yet in the
    # default network-tool set.
    egress = NetworkEgress(policy=_policy())
    decision = await egress(
        _ctx(tool="http_request", url="https://slack.com/api/chat.postMessage", persona="sm")
    )
    assert decision.outcome == "allow"
    assert decision.audit_metadata["hostname"] == "slack.com"


# ---------------------------------------------------------------------------
# AC-4: empty network_egress_domains → default-deny
# ---------------------------------------------------------------------------


async def test_ac4_empty_domains_default_deny() -> None:
    egress = NetworkEgress(policy=_policy(analyst_domains=()))
    decision = await egress(
        _ctx(tool="research", url="https://duckduckgo.com/q=iris", persona="analyst")
    )
    assert decision.outcome == "deny"
    assert decision.severity == "error"
    assert "hostname_not_allowed" in decision.reason
    assert decision.audit_metadata["deny_reason"] == "hostname_not_allowed"


async def test_ac4_explicit_empty_list_also_denies() -> None:
    """Explicitly setting an empty list is the same as unset — default-deny."""
    policy = PersonaPolicyDocument(
        personas={
            "analyst": PersonaPolicy(
                name="analyst",
                allowed_tools=("research",),
                network_egress_domains=(),
            )
        }
    )
    egress = NetworkEgress(policy=policy)
    decision = await egress(_ctx(tool="research", url="https://example.com", persona="analyst"))
    assert decision.outcome == "deny"


# ---------------------------------------------------------------------------
# AC-5: unparseable URL → deny (unparseable_url)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "not-a-url",
        "//no-scheme.example.com",
        "ftp://",  # scheme but empty netloc
        "",
    ],
)
async def test_ac5_unparseable_url_denied(url: str) -> None:
    egress = NetworkEgress(policy=_policy())
    decision = await egress(_ctx(url=url))
    assert decision.outcome == "deny"
    assert "unparseable_url" in decision.reason or "missing URL" in decision.reason


async def test_ac5_missing_url_field_denied() -> None:
    egress = NetworkEgress(policy=_policy())
    ctx = _ctx(url=None)  # no url field at all
    decision = await egress(ctx)
    assert decision.outcome == "deny"
    assert "missing URL" in decision.reason


# ---------------------------------------------------------------------------
# AC-6: cloud-LLM hostnames → allow (governed by EgressGate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostname",
    ["api.openai.com", "api.anthropic.com", "openrouter.ai", "models.inference.ai.azure.com"],
)
async def test_ac6_cloud_llm_hostname_skipped(hostname: str) -> None:
    """Cloud-LLM hosts short-circuit with allow regardless of persona allowlist."""
    # developer_domains does NOT include these hosts.
    egress = NetworkEgress(policy=_policy(developer_domains=("api.github.com",)))
    decision = await egress(_ctx(tool="http_request", url=f"https://{hostname}/v1/chat"))
    assert decision.outcome == "allow"
    assert decision.reason == NetworkEgress.ALLOW_REASON_CLOUD_LLM
    assert decision.audit_metadata["hostname"] == hostname


async def test_ac6_cloud_llm_skipped_even_with_empty_allowlist() -> None:
    """Cloud-LLM skip fires before the default-deny gate."""
    egress = NetworkEgress(policy=_policy(developer_domains=()))
    decision = await egress(_ctx(tool="http_request", url="https://api.openai.com/v1/chat"))
    assert decision.outcome == "allow"
    assert decision.reason == NetworkEgress.ALLOW_REASON_CLOUD_LLM


# ---------------------------------------------------------------------------
# AC-7: query params stripped in audit
# ---------------------------------------------------------------------------


async def test_ac7_query_params_stripped_in_audit() -> None:
    egress = NetworkEgress(policy=_policy())
    decision = await egress(_ctx(url="https://api.github.com/repos?token=supersecret&other=val"))
    assert decision.outcome == "allow"
    captured_url = decision.audit_metadata["url"]
    assert "supersecret" not in captured_url
    assert "token=" not in captured_url
    assert "api.github.com" in captured_url


async def test_ac7_fragment_also_stripped() -> None:
    egress = NetworkEgress(policy=_policy())
    decision = await egress(_ctx(url="https://api.github.com/repos#section"))
    assert decision.outcome == "allow"
    assert "#section" not in decision.audit_metadata["url"]


# ---------------------------------------------------------------------------
# Non-network tool / non-coding agent short-circuit paths
# ---------------------------------------------------------------------------


async def test_non_network_tool_short_circuits_allow() -> None:
    egress = NetworkEgress(policy=_policy())
    # read_file has a path that looks like a URL — must not be inspected.
    ctx = _ctx(
        tool="read_file",
        extra_payload={"path": "https://evil.example.com/secret"},
    )
    decision = await egress(ctx)
    assert decision.outcome == "allow"
    assert decision.reason == NetworkEgress.ALLOW_REASON_NOT_NETWORK_TOOL


async def test_non_coding_agent_short_circuits_allow() -> None:
    egress = NetworkEgress(policy=_policy())
    decision = await egress(_ctx(agent_type="chat", persona=None, url="https://evil.example.com"))
    assert decision.outcome == "allow"
    assert decision.reason == NetworkEgress.ALLOW_REASON_NON_CODING


async def test_coding_dispatch_missing_persona_denies() -> None:
    egress = NetworkEgress(policy=_policy())
    decision = await egress(_ctx(persona=None, url="https://api.github.com"))
    assert decision.outcome == "deny"
    assert decision.severity == "error"
    assert "missing persona" in decision.reason


async def test_orchestrator_denied_even_if_policy_present() -> None:
    egress = NetworkEgress(
        policy=PersonaPolicyDocument(
            personas={
                "orchestrator": PersonaPolicy(
                    name="orchestrator",
                    allowed_tools=("delegate_to_persona",),
                    network_egress_domains=("api.github.com",),
                )
            }
        )
    )
    decision = await egress(
        _ctx(tool="web_fetch", url="https://api.github.com", persona="orchestrator")
    )
    assert decision.outcome == "deny"
    assert decision.severity == "error"
    assert "orchestrator" in decision.reason


async def test_unknown_persona_denies() -> None:
    egress = NetworkEgress(policy=_policy())
    decision = await egress(_ctx(persona="unknown-persona", url="https://api.github.com"))
    assert decision.outcome == "deny"
    assert "unknown persona" in decision.reason


async def test_no_policy_degrades_to_allow_with_single_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    egress = NetworkEgress(policy=None)
    with caplog.at_level("WARNING"):
        d1 = await egress(_ctx(url="https://api.github.com"))
        d2 = await egress(_ctx(url="https://evil.example.com"))
    assert d1.outcome == "allow"
    assert d1.reason == NetworkEgress.ALLOW_REASON_DEGRADED
    assert d2.outcome == "allow"
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# Tool-name matching matrix (fnmatch wildcards)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name,expected_match",
    [
        ("web_fetch", True),
        ("research", True),
        ("http_request", True),
        ("github_create_issue", True),
        ("github_create_pr", True),
        ("github_list_prs", True),
        ("mcp_github", True),
        ("mcp_anything", True),
        ("read_file", False),
        ("edit_file", False),
        ("run_command", False),
        ("git_commit", False),  # git_* is local, not github_*
    ],
)
def testis_network_tool_matching(tool_name: str, expected_match: bool) -> None:
    assert is_network_tool(tool_name, DEFAULT_NETWORK_TOOLS) == expected_match


# ---------------------------------------------------------------------------
# Nested args payload shape
# ---------------------------------------------------------------------------


async def test_nested_args_url_shape_accepted() -> None:
    """Runtime coding-agent payload: ``payload['args']['url']``."""
    egress = NetworkEgress(policy=_policy())
    decision = await egress(
        _ctx(
            tool="github_create_issue",
            url="https://api.github.com/repos/x/y/issues",
            nested_url=True,
        )
    )
    assert decision.outcome == "allow"


async def test_endpoint_key_also_extracted() -> None:
    """Tools that use ``endpoint`` instead of ``url`` are also guarded."""
    egress = NetworkEgress(policy=_policy())
    ctx = HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="r-1",
        agent_type="coding",
        persona="developer",
        payload={
            "tool_name": "http_request",
            "endpoint": "https://api.github.com/v3/repos",
        },
    )
    decision = await egress(ctx)
    assert decision.outcome == "allow"
    assert decision.audit_metadata["hostname"] == "api.github.com"


async def test_endpoint_in_args_also_extracted() -> None:
    egress = NetworkEgress(policy=_policy())
    ctx = HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="r-1",
        agent_type="coding",
        persona="developer",
        payload={
            "tool_name": "http_request",
            "args": {"endpoint": "https://api.github.com/v3/repos"},
        },
    )
    decision = await egress(ctx)
    assert decision.outcome == "allow"


# ---------------------------------------------------------------------------
# URL parser unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected_hostname",
    [
        ("https://api.github.com/repos", "api.github.com"),
        ("https://api.github.com:443/repos", "api.github.com"),
        ("http://example.com/path?q=1#frag", "example.com"),
        ("https://API.GITHUB.COM/", "api.github.com"),  # lowercased
    ],
)
def test_parse_url_extracts_hostname(url: str, expected_hostname: str) -> None:
    result = _parse_url(url)
    assert result is not None
    hostname, _ = result
    assert hostname == expected_hostname


@pytest.mark.parametrize(
    "url",
    [
        "not-a-url",
        "//missing-scheme.com",
        "",
    ],
)
def test_parse_url_returns_none_for_invalid(url: str) -> None:
    assert _parse_url(url) is None


def test_parse_url_strips_query_and_fragment() -> None:
    result = _parse_url("https://api.github.com/repos?token=secret#anchor")
    assert result is not None
    _, clean_url = result
    assert "secret" not in clean_url
    assert "token=" not in clean_url
    assert "#anchor" not in clean_url
    assert "api.github.com" in clean_url


# ---------------------------------------------------------------------------
# CLOUD_LLM_HOSTNAMES coverage
# ---------------------------------------------------------------------------


def test_cloud_llm_hostnames_cover_major_providers() -> None:
    for required in ("api.openai.com", "api.anthropic.com", "openrouter.ai"):
        assert required in CLOUD_LLM_HOSTNAMES


def test_default_network_tools_cover_story_spec() -> None:
    assert {"web_fetch", "research", "http_request"} <= DEFAULT_NETWORK_TOOLS
    # Wildcard patterns — is_network_tool is the right way to check these.
    assert is_network_tool("github_create_issue", DEFAULT_NETWORK_TOOLS)
    assert is_network_tool("mcp_server_call", DEFAULT_NETWORK_TOOLS)
