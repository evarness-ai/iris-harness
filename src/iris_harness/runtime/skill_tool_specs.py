"""The ``ToolSpec`` of a skill tool, built in one place.

A skill's tool class is exposed three ways: to the ReAct loop (``_skills_to_react_tools``), to
the legacy general lane's tool pool (the same builder, #155) and to the lane's deterministic
direct answer (``local_skills``). Each needs the same declaration -- the name and description
from the manifest, the manifest's ``content`` (what makes the runner's external-content floor
apply) and the owner of the tool (``skill:<name>``) -- so it is built here and nowhere else.
Only what the tool's ``call`` returns differs: the loop wants a string, the direct answer wants
the structured value back (:func:`json_skill_call`).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from iris_harness.agent.agentic_core import ToolSpec


def text_skill_call(tool_class: type[Any]) -> Callable[[dict[str, Any]], str]:
    """The tool run in-process, its result as text: what the loop and the lane's pool use."""

    def _call(args: dict[str, Any]) -> str:
        return str(tool_class().invoke(args))

    return _call


def json_skill_call(tool_class: type[Any]) -> Callable[[dict[str, Any]], str]:
    """The tool run in-process, its result as JSON text.

    A governed call hands back text, and the direct answer's formatter (and the pending-action
    recorder) read the STRUCTURED result, so the call serialises it (``default=str``, as the
    formatter does for what JSON cannot hold) and the caller parses the governed text back.
    """

    def _call(args: dict[str, Any]) -> str:
        return json.dumps(tool_class().invoke(args), default=str)

    return _call


def skill_tool_spec(
    package: Any,
    tool_manifest: Any,
    tool_class: type[Any],
    *,
    call: Callable[[dict[str, Any]], str] | None = None,
) -> ToolSpec:
    """The ``ToolSpec`` of ``tool_manifest`` (a tool of ``package``), declared as the loop
    declares it: ``content`` from the manifest, ``plugin`` ``skill:<name>``."""
    return ToolSpec(
        name=tool_manifest.name,
        description=tool_manifest.description,
        call=call if call is not None else text_skill_call(tool_class),
        content=tool_manifest.content,
        plugin=f"skill:{package.manifest.name}",
    )


__all__ = ["json_skill_call", "skill_tool_spec", "text_skill_call"]
