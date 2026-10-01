"""Guard: a local model that emits its raw ReAct scaffolding instead of a reply
must never leak that scratchpad to the user (recover the embedded answer, else
suppress). Observed live as `{"thought": "...check the calendar..."}` surfacing
verbatim before the calendar agent was wired."""

from __future__ import annotations

from iris_harness.agent.agent_executor import AgentResult
from iris_harness.agent.response_curator import (
    ResponseCurator,
    _looks_like_react_scaffolding,
    _recover_react_answer,
)


def _result(output: str) -> AgentResult:
    return AgentResult(agent_type="system", output=output, success=True, latency_ms=1.0)


# ── curate() behaviour ───────────────────────────────────────────────────────


def test_suppresses_raw_thought_json() -> None:
    curator = ResponseCurator()
    leak = '{"thought": "The user is asking about their schedule. I need to check the calendar."}'
    out = curator.curate([_result(leak)], query="any meeting tomorrow?")
    assert "thought" not in out.text.lower()
    assert "check the calendar" not in out.text
    assert out.metadata.get("react_leak_suppressed") is True


def test_recovers_final_answer_from_react_json() -> None:
    curator = ResponseCurator()
    blob = (
        '{"thought": "I have the info", "action": "Final Answer", '
        '"action_input": "You have 2 meetings tomorrow."}'
    )
    out = curator.curate([_result(blob)], query="meetings?")
    assert out.text == "You have 2 meetings tomorrow."
    assert out.metadata.get("react_answer_recovered") is True


def test_suppresses_plaintext_react_step() -> None:
    curator = ResponseCurator()
    leak = "Thought: I should check.\nAction: calendar_lookup\nAction Input: tomorrow"
    out = curator.curate([_result(leak)], query="x")
    assert out.metadata.get("react_leak_suppressed") is True
    assert "Action:" not in out.text


def test_passes_normal_answer_through() -> None:
    curator = ResponseCurator()
    normal = "You have 3 events tomorrow:\n- 09:00 Dentist"
    out = curator.curate([_result(normal)], query="x")
    assert out.text == normal
    assert "react_leak_suppressed" not in out.metadata
    assert "react_answer_recovered" not in out.metadata


# ── detector units ───────────────────────────────────────────────────────────


def test_detector_precision() -> None:
    assert _looks_like_react_scaffolding('{"thought": "x"}') is True
    assert _looks_like_react_scaffolding('  {"action": "search", "action_input": "ai"}') is True
    # ordinary prose that merely mentions "thought" is NOT scaffolding
    assert _looks_like_react_scaffolding("My thoughts on this: it's a good plan.") is False
    assert _looks_like_react_scaffolding("You have 3 events tomorrow.") is False


def test_recover_only_on_final_answer_json() -> None:
    assert _recover_react_answer('{"action": "Final Answer", "action_input": "hi"}') == "hi"
    assert _recover_react_answer('{"thought": "still thinking"}') is None
    assert _recover_react_answer("just a normal sentence") is None
