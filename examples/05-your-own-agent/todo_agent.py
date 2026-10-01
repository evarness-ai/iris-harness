"""Your own agent: a to-do domain the harness's governed loop answers, over your tools.

``api.register_loop_intent("planner", fallback=...)`` hands every turn the router
classifies as ``planner`` to the one loop the harness builds and governs
(``PRE_LLM_CALL``, the tool runner, approvals, the curator), over the tools every plugin
registered -- here ``list_todos`` and ``add_todo``. ``fallback`` is the floor under the
loop: a deterministic answer for a turn the loop cannot finish (the model is down,
answers nothing, or answers without reading).

The router picks from a fixed set of intents (``planner`` is the to-do lane); a plugin
cannot add a new one yet. See this example's README.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.types import AgentTask

INTENT = "planner"


@dataclass
class TodoList:
    """The owner's to-dos. A real plugin keeps them in its own store."""

    items: list[str] = field(default_factory=list)

    def render(self) -> str:
        if not self.items:
            return "Your to-do list is empty."
        return "\n".join(f"{n}. {item}" for n, item in enumerate(self.items, start=1))


def setup(api: PluginAPI, todos: TodoList | None = None) -> None:
    todos = todos if todos is not None else TodoList(["Renew the library card", "Call Petra"])

    def list_todos(args: dict[str, Any]) -> str:
        return todos.render()

    def add_todo(args: dict[str, Any]) -> str:
        item = str(args.get("item", "")).strip()
        todos.items.append(item)
        return f"Added: {item}"

    def needs_item(args: dict[str, Any]) -> str | None:
        return None if str(args.get("item", "")).strip() else "add_todo needs a non-empty item"

    def fallback(task: AgentTask) -> str:
        """The deterministic floor: the list as it stands, no model involved."""
        return "Here is your to-do list:\n" + todos.render()

    api.register_tool("list_todos", "List the owner's to-dos, numbered.", list_todos)
    api.register_tool(
        "add_todo",
        'Add one to-do to the end of the list. Args: {"item": str}.',
        add_todo,
        validate=needs_item,
    )
    api.register_loop_intent(INTENT, fallback=fallback)
