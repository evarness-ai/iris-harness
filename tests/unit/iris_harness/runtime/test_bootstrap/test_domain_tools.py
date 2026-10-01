"""ADR-0077: domain capabilities exposed as universal tools on the unified loop.

The core builds none of them any more. The finance tools left with their domain at
M4, the daily plan with the planner at M5.7 track A, and `calendar_lookup` plus the
email tools with their libraries at M6.1b (OSS plan M6, decision 2). Each registers
through `PluginAPI.register_tool` and is tested beside its plugin.

What stays here (ADR-0110 follow-up): the core names no plugin tool. A skill may not
shadow a name already in the pool — decided per turn from the pool — and a tool stays
on the menu past shortlisting because its manifest says ``pinned``, not because the
core listed it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from iris_harness.agent.agentic_core import ToolSpec
from iris_harness.runtime.handlers.react import (
    _REACT_CORE_TOOL_NAMES,
    _shortlist_react_tools,
    _skills_to_react_tools,
)


class _Tool:
    def invoke(self, args: dict[str, Any]) -> str:
        return "skill ran"


def _skill_registry(*tool_names: str) -> Any:
    package = SimpleNamespace(
        manifest=SimpleNamespace(
            name="shadowy",
            description="a skill",
            kind="tool",
            brief=None,
            tools=[SimpleNamespace(name=n, description=f"skill {n}") for n in tool_names],
        ),
        tool_classes=[_Tool for _ in tool_names],
    )
    return SimpleNamespace(
        discover=lambda: None, list_packages=lambda only_loadable=True: [package]
    )


def test_a_skill_cannot_shadow_a_name_already_in_the_pool() -> None:
    registry = _skill_registry("finance_lookup", "skill_only")
    specs = _skills_to_react_tools(
        registry, query="", taken=frozenset({"finance_lookup", "search_inbox"})
    )
    assert [t.name for t in specs] == ["skill_only"]


def test_with_nothing_taken_the_skill_tool_is_offered() -> None:
    registry = _skill_registry("finance_lookup")
    assert [t.name for t in _skills_to_react_tools(registry, query="")] == ["finance_lookup"]


def test_the_core_keeps_only_its_own_two_names() -> None:
    assert _REACT_CORE_TOOL_NAMES == frozenset({"memory_search", "ask_user"})


def test_a_pinned_plugin_tool_survives_shortlisting_without_the_core_naming_it() -> None:
    tools = [ToolSpec(name=f"t{i}", description="x", call=lambda a: "") for i in range(8)] + [
        ToolSpec(name="search_inbox", description="find mail", call=lambda a: "", pinned=True)
    ]
    kept, dropped = _shortlist_react_tools(tools, "totally unrelated words", cap=3, router=None)
    assert "search_inbox" in {t.name for t in kept}
    assert "search_inbox" not in dropped
