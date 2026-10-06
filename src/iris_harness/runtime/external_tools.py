"""Which mounted tools and capabilities return text a third party wrote (issue #136).

One list for Health: the same three pools the ReAct loop assembles each turn (the core's
own tools, plugin tools, skill tools) plus the capabilities a mounted provider serves. A tool
is in it when it declares ``content: external``; nothing is read from a tool's output.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from iris_harness.foundation import capabilities as catalogue

if TYPE_CHECKING:
    from iris_harness.runtime.facade import IrisRuntime

_CORE = "core"


def mounted_external_tools(runtime: IrisRuntime) -> list[tuple[str, str]]:
    """``(owner, tool)`` for each mounted tool or capability declared ``content: external``.

    ``owner`` is the plugin, ``skill:<name>`` or ``core``; a capability method is listed under
    its provider as ``capability:<name>.<method>``. Sorted, so a row reads the same each pass.
    """
    from iris_harness.runtime.handlers.react import _skills_to_react_tools
    from iris_harness.runtime.react_tools import builtin_react_tools

    found: set[tuple[str, str]] = set()
    builtin = builtin_react_tools(
        semantic_index=None, wiki=None, repo_root=Path("."), memory_store=None
    )
    found.update((_CORE, t.name) for t in builtin if t.content == "external")
    plugin_tools = runtime.plugin_registry.tools()
    found.update((t.plugin or _CORE, t.name) for t in plugin_tools if t.content == "external")
    taken = frozenset(t.name for t in (*builtin, *plugin_tools))
    skill_tools = _skills_to_react_tools(runtime.skill_registry, taken=taken)
    found.update((t.plugin or _CORE, t.name) for t in skill_tools if t.content == "external")
    for name, spec in catalogue.CAPABILITIES.items():
        external = [m for m, declared in spec.methods.items() if declared.content == "external"]
        if not external:
            continue
        for provider in runtime.plugin_registry.capability_providers(name):
            found.update((provider, catalogue.capability_tool_name(name, m)) for m in external)
    return sorted(found)


__all__ = ["mounted_external_tools"]
