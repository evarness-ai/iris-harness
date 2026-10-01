"""A finished loop run is handed to the installed reviewer, once, with its route (§9.2)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from iris_harness.agent.agentic_core import AgenticCore, AgenticCoreConfig, ToolSpec
from iris_harness.agent.run_review import (
    CompletedRun,
    install_run_reviewer,
    review_completed_run,
)

_TOOL_THEN_ANSWER = [
    'Thought: look it up\nAction: echo\nAction Input: {"q": "dues"}',
    "Thought: done\nFinal Answer: you owe 40",
]
_ASK = 'Thought: need their call\nAction: ask_user\nAction Input: {"question": "Which card?"}'


class _Scripted:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def __call__(self, _prompt: str) -> str:
        return self._responses.pop(0) if self._responses else "Thought: x\nFinal Answer: done"


def _core(responses: list[str], *, route: str | None = "finance") -> AgenticCore:
    return AgenticCore(
        config=AgenticCoreConfig(max_iterations=4, allow_ask_user=True),
        llm_call=_Scripted(responses),
        tools=[ToolSpec(name="echo", description="echo", call=lambda a: f"got {a.get('q')}")],
        agent_type="system",
        review_route=route,
    )


@pytest.fixture
def reviewed() -> Iterator[list[tuple[CompletedRun, str, str]]]:
    seen: list[tuple[CompletedRun, str, str]] = []
    install_run_reviewer(lambda run, route, agent_type: seen.append((run, route, agent_type)))
    yield seen
    install_run_reviewer(None)


def test_a_finished_run_is_reviewed_once_with_its_route(
    reviewed: list[tuple[CompletedRun, str, str]],
) -> None:
    trace = _core(_TOOL_THEN_ANSWER).run("what do I owe")

    assert len(reviewed) == 1
    run, route, agent_type = reviewed[0]
    assert (route, agent_type) == ("finance", "system")
    assert run.run_id == trace.run_id
    assert run.query == "what do I owe"
    assert run.final_answer == "you owe 40"
    assert run.used_tools
    assert run.steps[0]["action"] == "echo"


def test_the_streamed_run_is_reviewed_too(
    reviewed: list[tuple[CompletedRun, str, str]],
) -> None:
    items = list(_core(_TOOL_THEN_ANSWER).run_stream("what do I owe"))

    meta = [i for i in items if isinstance(i, dict)][-1]
    assert len(reviewed) == 1
    run, route, _ = reviewed[0]
    assert route == "finance"
    assert run.run_id == meta["run_id"]
    assert run.final_answer == "you owe 40"


def test_a_run_paused_for_the_owner_is_not_reviewed(
    reviewed: list[tuple[CompletedRun, str, str]],
) -> None:
    _core([_ASK]).run("pay my card")
    list(_core([_ASK]).run_stream("pay my card"))

    assert reviewed == []


def test_a_loop_without_a_route_opts_out(
    reviewed: list[tuple[CompletedRun, str, str]],
) -> None:
    _core(_TOOL_THEN_ANSWER, route=None).run("what do I owe")
    assert reviewed == []


def test_a_plain_answer_is_reported_but_says_it_used_no_tools(
    reviewed: list[tuple[CompletedRun, str, str]],
) -> None:
    _core(["Thought: easy\nFinal Answer: hello"]).run("hi")
    assert len(reviewed) == 1
    assert reviewed[0][0].used_tools is False


def test_a_failing_reviewer_never_breaks_the_run() -> None:
    def boom(run: CompletedRun, route: str, agent_type: str) -> None:
        raise RuntimeError("reviewer down")

    install_run_reviewer(boom)
    try:
        trace = _core(_TOOL_THEN_ANSWER).run("what do I owe")
        streamed = list(_core(_TOOL_THEN_ANSWER).run_stream("what do I owe"))
    finally:
        install_run_reviewer(None)

    assert trace.final_answer == "you owe 40"
    assert "you owe 40" in streamed


def test_with_no_reviewer_a_report_is_a_no_op() -> None:
    install_run_reviewer(None)
    review_completed_run(
        CompletedRun(run_id="r", query="q", steps=(), final_answer="a", success=True),
        route="general",
        agent_type="system",
    )


def test_every_loop_in_the_code_names_its_review_route() -> None:
    """A loop built without ``review_route`` is never judged, silently. Every production
    ``AgenticCore(...)`` — the core's and the plugins' — must name the route its model
    runs on, so the governance judge reviews it on that same route."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[5] / "src"
    missing = []
    for path in src.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None)) == "AgenticCore"
                and not any(kw.arg == "review_route" for kw in node.keywords)
            ):
                missing.append(f"{path.relative_to(src)}:{node.lineno}")
    assert missing == [], f"AgenticCore built without review_route: {missing}"
