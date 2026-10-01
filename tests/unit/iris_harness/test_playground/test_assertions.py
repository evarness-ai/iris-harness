"""Unit tests for the pure expectation evaluator."""

from __future__ import annotations

from iris_harness.playground.assertions import ObservedTurn, evaluate
from iris_harness.playground.models import ScenarioExpectation


def _turn(**kw: object) -> ObservedTurn:
    base: dict[str, object] = {
        "response": "",
        "intent": "general",
        "agent_type": "general",
        "handler": None,
        "sources": (),
        "metadata": {},
        "has_errors": False,
    }
    base.update(kw)
    return ObservedTurn(**base)  # type: ignore[arg-type]


def test_empty_expectation_produces_no_assertions() -> None:
    assert evaluate(ScenarioExpectation(), _turn()) == []


def test_intent_and_agent_equality() -> None:
    exp = ScenarioExpectation(intent="finance", agent_type="finance")
    res = evaluate(exp, _turn(intent="finance", agent_type="finance"))
    assert [a.ok for a in res] == [True, True]

    res = evaluate(exp, _turn(intent="general", agent_type="finance"))
    assert [a.field for a in res if not a.ok] == ["intent"]


def test_handler_empty_means_no_intercept() -> None:
    # expect.handler == "" asserts the turn reached the agent loop (handler None)
    exp = ScenarioExpectation(handler="")
    assert evaluate(exp, _turn(handler=None))[0].ok is True
    assert evaluate(exp, _turn(handler="dues_request"))[0].ok is False


def test_handler_named_intercept() -> None:
    exp = ScenarioExpectation(handler="dues_request")
    assert evaluate(exp, _turn(handler="dues_request"))[0].ok is True
    assert evaluate(exp, _turn(handler=None))[0].ok is False


def test_sources_include_exclude() -> None:
    exp = ScenarioExpectation(sources_include=("portfolio",), sources_exclude=("research",))
    ok = evaluate(exp, _turn(sources=("portfolio", "stock_quote")))
    assert all(a.ok for a in ok)
    bad = evaluate(exp, _turn(sources=("research",)))
    assert [a.field for a in bad if not a.ok] == ["sources_include", "sources_exclude"]


def test_response_contains_and_not_contains_case_insensitive() -> None:
    exp = ScenarioExpectation(
        response_contains=("Portfolio",), response_not_contains=("web search",)
    )
    res = evaluate(exp, _turn(response="Your portfolio is up 2%."))
    assert all(a.ok for a in res)
    res = evaluate(exp, _turn(response="Let me do a Web Search for that."))
    assert [a.field for a in res if not a.ok] == ["response_contains", "response_not_contains"]


def test_response_regex() -> None:
    exp = ScenarioExpectation(response_regex=r"\b\d{1,2}:\d{2}\b")
    assert evaluate(exp, _turn(response="It is 14:30 now."))[0].ok is True
    assert evaluate(exp, _turn(response="no time here"))[0].ok is False


def test_metadata_subset_match() -> None:
    exp = ScenarioExpectation(metadata={"deterministic_time_date": True})
    assert evaluate(exp, _turn(metadata={"deterministic_time_date": True, "x": 1}))[0].ok is True
    assert evaluate(exp, _turn(metadata={}))[0].ok is False


def test_no_pii_leak_catches_email_and_markers() -> None:
    exp = ScenarioExpectation(no_pii_leak=True)
    assert evaluate(exp, _turn(response="all clear"))[0].ok is True
    assert evaluate(exp, _turn(response="contact bob@example.com"))[0].ok is False
    assert evaluate(exp, _turn(response="Thought: I should search"))[0].ok is False


def test_has_errors() -> None:
    exp = ScenarioExpectation(has_errors=False)
    assert evaluate(exp, _turn(has_errors=False))[0].ok is True
    assert evaluate(exp, _turn(has_errors=True))[0].ok is False
