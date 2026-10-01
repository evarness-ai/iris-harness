"""Rendering a brief skill as a tool the general handler can bind.

Gate-1 extraction (OSS plan M5.7). ``make_brief_runner`` closes a skill package's brief
over its registry so it can be called with no arguments; ``make_brief_render_tool`` wraps
that runner in the LangChain tool triple the handler binds. Both are called from inside
``_make_general_handler`` and from ``_skills_to_react_tools`` outside it, which is why
they were module-level in bootstrap already and why they move as-is.

Extracted ahead of the local-skills cluster that needs them: the measurement for that
slice found these two (and the skill-matching helpers beside them) were prerequisites, not
successors, so the plan's ordering was inverted here rather than worked around with
imports back into bootstrap.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from iris_harness.tools.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


def make_brief_runner(skill_registry: SkillRegistry, skill_name: str) -> Callable[[], str]:
    """Return a closure that renders the named brief via *skill_registry*.

    Shared between the general-handler tool path and the ReAct tool path so
    both expose the same synthetic ``render_<skill>`` tool to the LLM.
    """

    def _run() -> str:
        package = next(
            (
                p
                for p in skill_registry.list_packages(only_loadable=True)
                if p.manifest.name == skill_name and p.manifest.kind == "brief"
            ),
            None,
        )
        if package is None:
            return f"Brief skill {skill_name!r} is not loaded."
        from iris_harness.runtime.handlers.skill_brief import (
            render_brief_package,
        )

        try:
            return render_brief_package(package, skill_registry)
        except Exception as exc:
            logger.exception("synthetic brief tool failed: %s", skill_name)
            return f"Failed to render {skill_name!r} brief: {exc}"

    return _run


def make_brief_render_tool(
    skill_name: str,
    skill_description: str,
    runner: Callable[[], str],
) -> tuple[str, type[Any], str]:
    """Build a synthetic ``BaseTool`` that renders a brief skill on demand.

    Briefs (``kind: brief`` skills) are declarative slot+layout specs and
    expose zero callable tools by themselves. Auto-wrapping them as
    ``render_<skill_name>`` tools lets the LLM select them via normal
    tool-use rather than depending on the deterministic short-circuit.
    """
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel

    safe = re.sub(r"[^a-z0-9]+", "_", skill_name.lower()).strip("_") or "brief"
    tool_name = f"render_{safe}"
    tool_description = (
        f"Render the '{skill_name}' brief — a multi-section summary. "
        f"Takes no arguments. {skill_description}"
    ).strip()

    class _NoArgs(BaseModel):
        pass

    class _BriefTool(BaseTool):
        name: str = tool_name
        description: str = tool_description
        args_schema: type[BaseModel] = _NoArgs

        def _run(self) -> str:
            return runner()

    _BriefTool.__name__ = f"Render{''.join(part.title() for part in safe.split('_'))}Tool"
    return tool_name, _BriefTool, tool_description
