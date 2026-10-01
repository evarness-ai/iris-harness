"""The shortlist trims the menu, not the pool (ADR-0110 follow-up).

A tool the shortlist dropped is still callable: when the model names it — because a
plugin's guidance or an observation told it to — the loop admits it on first call
instead of answering "unknown tool". On the acceptance query gpt-4o called read_email,
which the dues trailer names and the shortlist had dropped, and looped on the error.
"""

from __future__ import annotations

from typing import Any

from iris_harness.agent.agentic_core import AgenticCore, AgenticCoreConfig, ToolSpec

_CALL_RESERVE = 'Thought: read it\nAction: read_email\nAction Input: {"id": "m1"}'
_CALL_GHOST = "Thought: try\nAction: teleport\nAction Input: {}"
_FINAL = "Thought: done\nFinal Answer: ok"


class _LLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0) if self._responses else _FINAL


def _tools() -> tuple[list[ToolSpec], list[ToolSpec], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def _read(args: dict[str, Any]) -> str:
        calls.append(dict(args))
        return "the email body"

    menu = [ToolSpec(name="search_inbox", description="find mail", call=lambda a: "hits")]
    reserve = [ToolSpec(name="read_email", description="read one", call=_read)]
    return menu, reserve, calls


def test_a_reserve_tool_is_admitted_on_first_call_sync() -> None:
    menu, reserve, calls = _tools()
    llm = _LLM([_CALL_RESERVE, _FINAL])
    core = AgenticCore(
        AgenticCoreConfig(max_iterations=4), llm_call=llm, tools=menu, reserve_tools=reserve
    )

    trace = core.run("what does it say?")

    assert calls == [{"id": "m1"}]
    assert trace.steps[0].observation == "the email body"
    assert trace.final_answer == "ok"
    # Once admitted it is on the menu for the rest of the run.
    assert "- read_email: read one" in llm.prompts[1]
    assert "- read_email" not in llm.prompts[0]


def test_a_reserve_tool_is_admitted_on_first_call_streaming() -> None:
    menu, reserve, calls = _tools()
    llm = _LLM([_CALL_RESERVE, _FINAL])
    core = AgenticCore(
        AgenticCoreConfig(max_iterations=4, streaming=True),
        llm_call=llm,
        tools=menu,
        reserve_tools=reserve,
    )
    list(core.run_stream("what does it say?"))
    assert calls == [{"id": "m1"}]


def test_a_tool_in_neither_pool_is_still_an_honest_error() -> None:
    menu, reserve, calls = _tools()
    llm = _LLM([_CALL_GHOST, _FINAL])
    core = AgenticCore(
        AgenticCoreConfig(max_iterations=4), llm_call=llm, tools=menu, reserve_tools=reserve
    )
    trace = core.run("go")
    assert (trace.steps[0].observation or "").startswith("Error: unknown tool 'teleport'")
    assert calls == []
