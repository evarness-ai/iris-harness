"""Unit tests for route matching and approval logic in the IRIS governor."""

from __future__ import annotations

from pathlib import Path

from iris_harness.kernel.governor.models import GovernorGuardRequest
from iris_harness.kernel.governor.policy import (
    GovernorPolicyEngine,
    load_governor_policy,
    route_matches,
)


def write_policy(repo_root: Path) -> None:
    policy_dir = repo_root / "config" / "governor"
    policy_dir.mkdir(parents=True)
    (policy_dir / "policy.yaml").write_text(
        "version: '1'\n"
        "routes:\n"
        "  - route: coding/git\n"
        "    allowed_actions:\n"
        "      - branch_commit_push\n"
        "    rate_limit:\n"
        "      requests: 20\n"
        "      window_seconds: 3600\n"
        "  - route: coding/tool/*\n"
        "    rate_limit:\n"
        "      requests: 60\n"
        "      window_seconds: 3600\n"
        "  - route: coding/mcp\n"
        "    allowed_actions:\n"
        "      - call_tool\n"
        "    requires_approval: true\n"
        "    rate_limit:\n"
        "      requests: 5\n"
        "      window_seconds: 60\n",
        encoding="utf-8",
    )


def test_route_matches_supports_exact_and_wildcard_patterns() -> None:
    assert route_matches("coding/git", "coding/git") is True
    assert route_matches("coding/tool/*", "coding/tool/read_file") is True
    assert route_matches("coding/tool/*", "coding/skill/read_file") is False


def test_policy_engine_matches_exact_then_wildcard_routes(tmp_path: Path) -> None:
    write_policy(tmp_path)
    policy = load_governor_policy(tmp_path)
    engine = GovernorPolicyEngine(policy)

    assert engine.match_route("coding/git").route == "coding/git"
    assert engine.match_route("coding/tool/read_file").route == "coding/tool/*"
    assert engine.match_route("unknown/route") is None


def test_policy_engine_enforces_action_and_approval_flags(tmp_path: Path) -> None:
    write_policy(tmp_path)
    engine = GovernorPolicyEngine(load_governor_policy(tmp_path))

    denied_without_approval = engine.evaluate(
        GovernorGuardRequest(
            route="coding/mcp",
            action="call_tool",
            metadata={"server": "filesystem"},
        )
    )
    denied_wrong_action = engine.evaluate(
        GovernorGuardRequest(
            route="coding/mcp",
            action="session_open",
            metadata={"approval_granted": True},
        )
    )
    allowed = engine.evaluate(
        GovernorGuardRequest(
            route="coding/mcp",
            action="call_tool",
            metadata={"approval_granted": True, "server": "filesystem"},
        )
    )

    assert denied_without_approval.allowed is False
    assert "approval_granted=true" in denied_without_approval.reason
    assert denied_wrong_action.allowed is False
    assert "not allowed" in denied_wrong_action.reason
    assert allowed.allowed is True
    assert allowed.matched_policy == "coding/mcp"
