"""Unit tests for PersonaSurface + PersonaPolicyDocument (story 12.gov-4.2).

Mirrors the acceptance criteria in
``docs/stories/12.gov-4.2.persona-policy-and-surface.story.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.kernel.governance import HookContext, HookPoint
from iris_harness.kernel.governance.plugins.persona_surface import (
    CANONICAL_PERSONAS,
    PersonaPolicy,
    PersonaPolicyDocument,
    PersonaSurface,
)


def _packaged_policy_path() -> Path:
    """The packaged persona-policy.yaml ships with the coding agent, which is not
    part of the harness release (OSS plan decision 2): skip where it is absent."""
    resource_paths = pytest.importorskip("iris_code.resource_paths")
    return Path(resource_paths.default_config_path("persona_policy"))


# ---------------------------------------------------------------------------
# PersonaPolicyDocument: packaged YAML parses + roster check
# ---------------------------------------------------------------------------


def test_packaged_persona_policy_parses_and_covers_canonical_roster() -> None:
    """AC-1: the packaged persona-policy.yaml parses, names match the
    canonical roster, and every record has a non-empty ``allowed_tools``
    (orchestrator's policy is informational — the surface is hard-coded —
    but the YAML still declares the list for documentation parity)."""
    doc = PersonaPolicyDocument.from_yaml(_packaged_policy_path())
    assert set(doc.personas) == set(CANONICAL_PERSONAS)
    for name, policy in doc.personas.items():
        assert isinstance(policy, PersonaPolicy)
        assert policy.allowed_tools, f"persona {name!r} has no allowed_tools"


def test_missing_persona_in_yaml_fails_at_load(tmp_path: Path) -> None:
    """Missing canonical persona → load fails loudly, not silently."""
    yaml_path = tmp_path / "persona-policy.yaml"
    yaml_path.write_text(
        "personas:\n  developer:\n    allowed_tools: [read_file]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="roster mismatch"):
        PersonaPolicyDocument.from_yaml(yaml_path)


def test_unknown_persona_in_yaml_fails_at_load(tmp_path: Path) -> None:
    yaml_path = tmp_path / "persona-policy.yaml"
    rows = "\n".join(f"  {name}:\n    allowed_tools: [read_file]" for name in CANONICAL_PERSONAS)
    yaml_path.write_text(
        f"personas:\n{rows}\n  fake_persona:\n    allowed_tools: [read_file]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="roster mismatch"):
        PersonaPolicyDocument.from_yaml(yaml_path)


# ---------------------------------------------------------------------------
# PersonaSurface enforcement — fixtures + acceptance criteria
# ---------------------------------------------------------------------------


@pytest.fixture()
def policy() -> PersonaPolicyDocument:
    """A small policy doc covering every canonical persona for tests."""
    return PersonaPolicyDocument(
        personas={
            "analyst": PersonaPolicy(
                name="analyst",
                allowed_tools=("read_file", "grep_search"),
                fs_read_only=True,
            ),
            "architect": PersonaPolicy(
                name="architect", allowed_tools=("read_file", "draw_diagram")
            ),
            "developer": PersonaPolicy(
                name="developer",
                allowed_tools=("read_file", "edit_file", "run_command", "git_*"),
            ),
            "orchestrator": PersonaPolicy(
                name="orchestrator", allowed_tools=("delegate_to_persona", "read_file")
            ),
            "sm": PersonaPolicy(name="sm", allowed_tools=("read_file", "github_*", "slack_post")),
            "tester": PersonaPolicy(name="tester", allowed_tools=("read_file", "run_command")),
            "ux-designer": PersonaPolicy(
                name="ux-designer", allowed_tools=("read_file", "draw_diagram")
            ),
        }
    )


def _ctx(
    *,
    agent_type: str = "coding",
    persona: str | None,
    tool: str | None,
) -> HookContext:
    payload = {"tool_name": tool} if tool is not None else {}
    return HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="r-1",
        agent_type=agent_type,
        persona=persona,
        payload=payload,
    )


async def test_developer_can_invoke_run_command(policy: PersonaPolicyDocument) -> None:
    """AC-2 allow side."""
    surface = PersonaSurface(policy=policy)
    decision = await surface(_ctx(persona="developer", tool="run_command"))
    assert decision.outcome == "allow"
    assert "developer" in decision.reason


async def test_analyst_cannot_invoke_run_command(policy: PersonaPolicyDocument) -> None:
    """AC-2 deny side."""
    surface = PersonaSurface(policy=policy)
    decision = await surface(_ctx(persona="analyst", tool="run_command"))
    assert decision.outcome == "deny"
    assert "analyst" in decision.reason and "run_command" in decision.reason


async def test_orchestrator_cannot_invoke_run_command_even_if_policy_listed(
    policy: PersonaPolicyDocument,
) -> None:
    """AC-3: orchestrator delegation-only rule is hard-coded; it stays
    enforced even if a policy author mistakenly puts other tools on the
    orchestrator's allowed_tools list."""
    # Mutate the policy so the orchestrator *appears* to allow run_command:
    forced_policy = PersonaPolicyDocument(
        personas={
            **policy.personas,
            "orchestrator": PersonaPolicy(
                name="orchestrator",
                allowed_tools=("delegate_to_persona", "read_file", "run_command"),
            ),
        }
    )
    surface = PersonaSurface(policy=forced_policy)
    decision = await surface(_ctx(persona="orchestrator", tool="run_command"))
    assert decision.outcome == "deny"
    assert decision.severity == "error"
    assert "delegation-only" in decision.reason


async def test_orchestrator_can_delegate(policy: PersonaPolicyDocument) -> None:
    surface = PersonaSurface(policy=policy)
    decision = await surface(_ctx(persona="orchestrator", tool="delegate_to_persona"))
    assert decision.outcome == "allow"


async def test_orchestrator_can_read_file(policy: PersonaPolicyDocument) -> None:
    """Reading is the one non-delegation exception so the orchestrator can
    audit work before dispatching it."""
    surface = PersonaSurface(policy=policy)
    decision = await surface(_ctx(persona="orchestrator", tool="read_file"))
    assert decision.outcome == "allow"


async def test_coding_dispatch_missing_persona_denies(
    policy: PersonaPolicyDocument,
) -> None:
    """AC-4: a coding-agent dispatch without ``persona`` is a
    misconfiguration, not a permissive case."""
    surface = PersonaSurface(policy=policy)
    decision = await surface(_ctx(persona=None, tool="read_file"))
    assert decision.outcome == "deny"
    assert decision.severity == "error"
    assert "missing persona" in decision.reason


async def test_non_coding_agent_dispatch_short_circuits_allow(
    policy: PersonaPolicyDocument,
) -> None:
    """AC-5: chat / voice / etc. pass through unchanged. PersonaSurface is
    the coding-pipeline guard only."""
    surface = PersonaSurface(policy=policy)
    decision = await surface(_ctx(agent_type="chat", persona=None, tool="search"))
    assert decision.outcome == "allow"
    assert "not a coding-agent dispatch" in decision.reason


async def test_wildcard_git_star_matches_git_commit(
    policy: PersonaPolicyDocument,
) -> None:
    """AC-7 allow side."""
    surface = PersonaSurface(policy=policy)
    decision = await surface(_ctx(persona="developer", tool="git_commit"))
    assert decision.outcome == "allow"


async def test_wildcard_git_star_matches_git_push(policy: PersonaPolicyDocument) -> None:
    """AC-7 allow side."""
    surface = PersonaSurface(policy=policy)
    decision = await surface(_ctx(persona="developer", tool="git_push"))
    assert decision.outcome == "allow"


async def test_wildcard_git_star_does_not_match_github_issue_create(
    policy: PersonaPolicyDocument,
) -> None:
    """AC-7 deny side — the underscore in ``git_*`` is literal; it must
    not bleed across the ``github_`` prefix."""
    surface = PersonaSurface(policy=policy)
    decision = await surface(_ctx(persona="developer", tool="github_issue_create"))
    assert decision.outcome == "deny"


async def test_unknown_persona_in_policy_denies(
    policy: PersonaPolicyDocument,
) -> None:
    """A persona name that's in the canonical roster but absent from the
    runtime policy (impossible via the YAML loader's roster check but
    possible when callers construct an in-memory partial policy) denies
    with a clear reason."""
    partial = PersonaPolicyDocument(personas={"developer": policy.personas["developer"]})
    surface = PersonaSurface(policy=partial)
    decision = await surface(_ctx(persona="analyst", tool="read_file"))
    assert decision.outcome == "deny"
    assert "unknown persona" in decision.reason


# ---------------------------------------------------------------------------
# Degraded mode — no policy loaded
# ---------------------------------------------------------------------------


async def test_no_policy_allows_with_degraded_reason() -> None:
    """When constructed with ``policy=None``, the plugin runs in degraded
    mode (chat-only installs, tests). Returns allow + warns once.

    Production deployments always load the packaged YAML in
    ``build_default_kernel`` so this path is for off-the-beaten-track
    callers only.
    """
    surface = PersonaSurface(policy=None)
    decision = await surface(_ctx(persona="developer", tool="run_command"))
    assert decision.outcome == "allow"
    assert "degraded" in decision.reason


# ---------------------------------------------------------------------------
# Plugin metadata
# ---------------------------------------------------------------------------


def test_plugin_metadata() -> None:
    """Hook framework needs ``name``, ``hook_point``, ``priority`` per §5.2."""
    surface = PersonaSurface(policy=None)
    assert surface.name == "persona_surface"
    assert surface.hook_point == HookPoint.PRE_TOOL_USE
    # Must run before ToolPolicyHook (priority 20).
    assert surface.priority == 15
