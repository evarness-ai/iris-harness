"""Tests that ``code_exec`` queries reach the right agent type."""

from __future__ import annotations

import pytest

from iris_harness.agent.intent_router import KeywordClassifier
from iris_harness.plugins_builtin.code_exec.handler import _extract_skill_request


@pytest.mark.parametrize(
    "query",
    [
        "create a pdf summary of my notes",
        "generate an excel spreadsheet from this data",
        "convert this json to csv",
        "plot a chart of the cashflows",
        "render the markdown to html",
        "calculate the IRR of these cashflows",
        "produce a png of the histogram",
        "use the sandbox tool",
        "please run_shell this command in sandbox",
        "execute in the sandbox and give me the output",
        "hello can you get the AI news today and send me updated pdf, this time give me a summary of each news (3-4 lines)",
        "the pdf looks clumpsy, can add it like article with spaces and formatting?",
        "reformat the pdf with better spacing and layout",
        "improve the report styling and regenerate the document",
        # Runnable artifacts — small general-tier models tend to inline a code
        # block instead of calling `code_exec`, so the router pre-empts them.
        "can you make a small tic-tac-toe game?",
        "build a snake game",
        "create a sudoku puzzle",
        "make a simulation of a random walk",
    ],
)
def test_artifact_requests_route_to_code_exec(query: str) -> None:
    classifier = KeywordClassifier()
    result = classifier.classify(query)
    assert result.agent_type == "code_exec"
    assert result.intent == "code_exec"


@pytest.mark.parametrize(
    "query",
    [
        "write a function to parse JSON",
        "refactor this class to use dataclasses",
        "debug my python script",
        "implement a binary tree in rust",
    ],
)
def test_pure_coding_requests_still_route_to_coding_agent(query: str) -> None:
    classifier = KeywordClassifier()
    result = classifier.classify(query)
    assert result.agent_type == "coding_agent"


@pytest.mark.parametrize(
    "query",
    [
        "what time is it",
        "tell me a joke",
        "who is the president of the US",
    ],
)
def test_general_chat_still_routes_to_system(query: str) -> None:
    classifier = KeywordClassifier()
    result = classifier.classify(query)
    assert result.agent_type == "system"


# ---------------------------------------------------------------------------
# Tests for _extract_skill_request (skill-proposal hook detection)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected_wants,expected_slug",
    [
        (
            "fetch top 10 github repos and save the script as draft skill - call it fetch-top-repos",
            True,
            "fetch-top-repos",
        ),
        (
            "what are the top stars repos and save as a draft skill named my-skill",
            True,
            "my-skill",
        ),
        (
            "run this and save as skill",
            True,
            None,
        ),
        (
            "store this script as a reusable skill",
            True,
            None,
        ),
        (
            "propose this as a skill call it weather-checker",
            True,
            "weather-checker",
        ),
        (
            "what are the top 10 github repositories with most stars today",
            False,
            None,
        ),
        (
            "create a pdf report",
            False,
            None,
        ),
        (
            "calculate the IRR",
            False,
            None,
        ),
    ],
)
def test_extract_skill_request(query: str, expected_wants: bool, expected_slug: str | None) -> None:
    wants, slug = _extract_skill_request(query)
    assert wants is expected_wants
    assert slug == expected_slug
