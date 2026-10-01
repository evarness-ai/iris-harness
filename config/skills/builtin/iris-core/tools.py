"""Core IRIS read-only tools: active items, approved routines, the learned-yesterday line."""

from __future__ import annotations

import os
from pathlib import Path

from iris_harness.memory.identity.loader import list_active_items
from iris_harness.services.routines.store import RoutineStore
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class _NoArgs(BaseModel):
    """Empty input schema for parameterless tools."""


def _routines_db_path() -> Path:
    """Resolve the routines DB path from env or default project layout."""
    override = os.getenv("IRIS_ROUTINES_DB")
    if override:
        return Path(override)
    return Path("data/routines.db")


class ListActiveItemsTool(BaseTool):
    """Return undone active.md checkbox entries."""

    name: str = "list_active_items"
    description: str = "List undone checkbox items from active.md."
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> list[dict[str, object]]:
        return [{"text": item.text} for item in list_active_items() if not item.done]

    async def _arun(self) -> list[dict[str, object]]:
        return self._run()


class ListApprovedRoutinesTool(BaseTool):
    """Return approved or scheduled routines with title and schedule."""

    name: str = "list_approved_routines"
    description: str = "List approved or scheduled user routines."
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> list[dict[str, object]]:
        store = RoutineStore(_routines_db_path())
        return [
            {"title": routine.title, "schedule": routine.schedule}
            for routine in store.list_executable()
        ]

    async def _arun(self) -> list[dict[str, object]]:
        return self._run()


class LearnedYesterdayTool(BaseTool):
    """The digest footer: what the owner's signals of the previous local day changed.

    Loop-proof D17 / graph §9: one line, never omitted — ``learned yesterday: nothing``
    on a quiet day. Each signal store registers its own phrases in
    ``iris_harness.services.digest.learned``; this tool only asks the registry.
    """

    name: str = "learned_yesterday"
    description: str = (
        "The morning digest's footer line naming what IRIS learned from the owner's "
        "signals yesterday (not-useful marks, settings changes)."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        from iris_harness.sdk.digest import learned_yesterday_line  # noqa: PLC0415
        from iris_harness.services.digest.footer import footer_lines  # noqa: PLC0415

        # The D13 lines (which jobs ran yesterday) follow, each its own paragraph so the
        # markdown renderers keep them on separate lines (footer.py).
        return "\n\n".join([learned_yesterday_line(), *footer_lines()])

    async def _arun(self) -> str:
        return self._run()


__all__ = ["LearnedYesterdayTool", "ListActiveItemsTool", "ListApprovedRoutinesTool"]

SKILL_TOOLS = [ListActiveItemsTool, ListApprovedRoutinesTool, LearnedYesterdayTool]
