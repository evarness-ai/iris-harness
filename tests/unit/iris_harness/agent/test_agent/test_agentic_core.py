from __future__ import annotations

from iris_harness.agent.agentic_core import AgenticComponent, AgenticCore, AgenticCoreConfig


def test_agentic_core_uses_domain_shaped_pipeline_names() -> None:
    core = AgenticCore()

    assert core.describe_pipeline() == (
        "intent_router",
        "task_planner",
        "react_loop",
        "agent_executor",
        "response_curator",
    )


def test_agentic_core_initializes_and_validates_components() -> None:
    core = AgenticCore(AgenticCoreConfig(max_iterations=3, timeout_seconds=30))

    before_initialization = core.validate_components()
    initialized_components = core.initialize_components()
    report = core.generate_validation_report()

    assert before_initialization == {
        "intent_router": False,
        "task_planner": False,
        "react_loop": False,
        "agent_executor": False,
        "response_curator": False,
    }
    assert initialized_components["react_loop"].name is AgenticComponent.REACT_LOOP
    assert report["checks"] == {
        "validation_results": {
            "intent_router": True,
            "task_planner": True,
            "react_loop": True,
            "agent_executor": True,
            "response_curator": True,
        },
        "component_checks": {
            "intent_router": True,
            "task_planner": True,
            "react_loop": True,
            "agent_executor": True,
            "response_curator": True,
        },
        "all_checks_passed": True,
    }


def test_agentic_core_public_api_does_not_leak_prd_section_names() -> None:
    public_members = [name for name in dir(AgenticCore) if not name.startswith("_")]

    assert all(not name.startswith("prd_") for name in public_members)


def test_resolve_tool_alias_redirects_hallucinated_names() -> None:
    """A near-miss tool name resolves to the obvious real tool, so the model's clear
    intent succeeds instead of stalling (e.g. 'portfolio_summary' -> 'portfolio')."""
    from iris_harness.agent.agentic_core import _resolve_tool_alias

    index = {n: object() for n in ("portfolio", "daily_plan", "finance_lookup", "research")}
    assert _resolve_tool_alias("portfolio_summary", index) == "portfolio"
    assert _resolve_tool_alias("get_portfolio", index) == "portfolio"
    assert _resolve_tool_alias("daily_plan_tool", index) == "daily_plan"
    assert _resolve_tool_alias("financelookup", index) == "finance_lookup"  # fuzzy
    assert _resolve_tool_alias("portfolio", index) == "portfolio"  # exact passes through


def test_resolve_tool_alias_returns_none_when_unclear() -> None:
    from iris_harness.agent.agentic_core import _resolve_tool_alias

    index = {n: object() for n in ("portfolio", "research")}
    assert _resolve_tool_alias("totally_unknown_xyz", index) is None
    assert _resolve_tool_alias("", index) is None
    # "search" is contained in neither and not a confident fuzzy match → no redirect.
    assert _resolve_tool_alias("zzz", index) is None


def test_last_good_observation_strips_context_header() -> None:
    """A recovered observation must never ship the model-facing context header
    (the '[Current date: …; provider: …]' research prefix leaked verbatim in
    the 2026-07-05 campaign when a fallback surfaced the observation)."""
    from iris_harness.agent.agentic_core import ReactStep, _last_good_observation

    obs = "[Current date: 2026-07-05; provider: searxng; cached]\n1. **A result** (https://x)"
    steps = [ReactStep(thought="t", action="research", observation=obs)]
    recovered = _last_good_observation(steps)
    assert recovered == "1. **A result** (https://x)"


def test_last_good_observation_keeps_inline_bracketed_content() -> None:
    """Only the header LINE is internal markup — bracketed ids inside real
    content (task lists, dues digests) must survive recovery untouched."""
    from iris_harness.agent.agentic_core import ReactStep, _last_good_observation

    obs = "[f0e06982] Prep: pool session\n[a1b2c3d4] Pay water bill"
    steps = [ReactStep(thought="t", action="daily_plan", observation=obs)]
    assert _last_good_observation(steps) == obs


def test_last_good_observation_still_skips_errors() -> None:
    from iris_harness.agent.agentic_core import ReactStep, _last_good_observation

    steps = [
        ReactStep(thought="t", action="finance_lookup", observation="3 dues found"),
        ReactStep(thought="t", action="research", observation="Error: provider down"),
    ]
    assert _last_good_observation(steps) == "3 dues found"
