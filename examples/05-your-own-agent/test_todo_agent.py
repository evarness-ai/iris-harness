"""The to-do agent: planner turns go to the governed loop over its tools, offline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from todo_agent import INTENT, TodoList, setup

from iris_harness.sdk import PluginAPI
from iris_harness.testing import harness, plugin

MANIFEST = Path(__file__).with_name("manifest.yaml")

# The router's call picks the intent; each loop step is one more scripted call.
ROUTE = {
    "name": "route to the to-do lane",
    "match": {"system": "request router"},
    "reply": {"json": {"intent": INTENT}},
}
SCRIPT: dict[str, Any] = {
    "rules": [
        ROUTE,
        {
            "name": "confirm the addition from the list",
            "match": {"user": r"(?s)Observation: Added: Buy oat milk.*Observation: .*Buy oat milk"},
            "reply": {"content": 'Thought: Done.\nFinal Answer: Added "Buy oat milk".'},
        },
        {
            # A planner answer must come from a read (`read_first_intents`), so the
            # loop checks the list after writing to it.
            "name": "check the list after adding",
            "match": {"user": r"(?s)Observation: Added: Buy oat milk"},
            "reply": {"content": "Thought: Check it.\nAction: list_todos\nAction Input: {}"},
        },
        {
            "name": "answer from the list",
            "match": {"user": r"(?s)Observation: 1\. Renew the library card"},
            "reply": {
                "content": "Thought: I have the list.\n"
                "Final Answer: Two things: renew the library card, then call Petra."
            },
        },
        {
            "name": "read the list",
            "match": {"user": r"User: What is on my to-do list"},
            "reply": {"content": "Thought: Read it.\nAction: list_todos\nAction Input: {}"},
        },
        {
            "name": "add to the list",
            "match": {"user": r"User: Add buy oat milk"},
            "reply": {
                "content": "Thought: Add it.\nAction: add_todo\n"
                'Action Input: {"item": "Buy oat milk"}'
            },
        },
    ]
}
# A model that answers without reading the list, every time.
GUESSING: dict[str, Any] = {
    "rules": [
        ROUTE,
        {
            "name": "guess",
            "match": {"user": r"to-do list"},
            "reply": {"content": "Thought: I know this.\nFinal Answer: You have nothing to do."},
        },
    ]
}


def _plugin(todos: TodoList) -> Any:
    def setup_with(api: PluginAPI) -> None:
        setup(api, todos)

    return plugin(setup_with, manifest=MANIFEST)


def test_a_planner_turn_is_answered_by_the_loop_over_the_plugins_tools() -> None:
    todos = TodoList(["Renew the library card", "Call Petra"])
    with harness(plugins=[_plugin(todos)], fake_model=SCRIPT) as h:
        result = h.chat("What is on my to-do list?")

        assert (result.intent, result.agent) == (INTENT, INTENT)
        assert result.text == "Two things: renew the library card, then call Petra."
        rows = h.audit_rows(hook_point="pre_tool_use", session_id=result.session_id)
        assert {row.tool for row in rows} == {"list_todos"}
        assert h.audit_gaps() == []


def test_the_loop_writes_through_the_plugins_write_tool() -> None:
    todos = TodoList(["Renew the library card"])
    with harness(plugins=[_plugin(todos)], fake_model=SCRIPT) as h:
        result = h.chat_stream("Add buy oat milk to my to-do list.")

        assert result.answered, result.error
        assert result.text == 'Added "Buy oat milk".'
        assert todos.items == ["Renew the library card", "Buy oat milk"]
        assert h.audit_gaps() == []


def test_an_answer_that_never_read_the_list_falls_back_to_the_deterministic_floor() -> None:
    todos = TodoList(["Renew the library card", "Call Petra"])
    with harness(plugins=[_plugin(todos)], fake_model=GUESSING) as h:
        result = h.chat("What is on my to-do list?")

        # `read_first_intents: [planner]` refused the guess; the fallback answered.
        assert "You have nothing to do" not in result.text
        assert result.text == "Here is your to-do list:\n" + todos.render()
        assert h.audit_gaps() == []
