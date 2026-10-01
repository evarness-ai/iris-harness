"""Issue 0002 — the clarify agent handler asks one grounded question instead of
guessing or stalling when the router can't route a turn."""

from __future__ import annotations

from iris_harness.agent.agent_executor import AgentTask
from iris_harness.runtime.handlers.general import _make_clarify_handler


def _task(query: str) -> AgentTask:
    return AgentTask(query=query, agent_type="clarify", memory_context=None)


def test_clarify_handler_returns_llm_question_with_metadata() -> None:
    handler, _ = _make_clarify_handler(lambda _p: "Which email did you mean — the CourseHub one?")
    text, meta = handler(_task("what is the summary?"))
    assert text == "Which email did you mean — the CourseHub one?"
    assert meta["agent_type"] == "clarify"
    assert meta["clarify"] is True


def test_clarify_handler_defaults_without_llm() -> None:
    handler, _ = _make_clarify_handler(None)
    text, _ = handler(_task("hmm"))
    assert "could you" in text.lower()


def test_clarify_handler_survives_llm_error() -> None:
    def _boom(_p: str) -> str:
        raise RuntimeError("ollama down")

    handler, _ = _make_clarify_handler(_boom)
    text, _ = handler(_task("hmm"))
    assert text  # falls back to the default, never raises


def test_clarify_stream_yields_question_then_metadata() -> None:
    _, stream = _make_clarify_handler(lambda _p: "What would you like me to do?")
    items = list(stream(_task("?")))
    assert items[0] == "What would you like me to do?"
    assert isinstance(items[1], dict) and items[1]["clarify"] is True
