from __future__ import annotations

from iris_harness.agent.intent_router import (
    KeywordClassifier,
    LLMRouterClassifier,
    agent_for_intent,
)


def test_agent_for_intent_default_search_system(monkeypatch) -> None:
    monkeypatch.delenv("IRIS_RESEARCH_AGENT", raising=False)
    assert agent_for_intent("search") == "system"


def test_agent_for_intent_search_remap(monkeypatch) -> None:
    monkeypatch.setenv("IRIS_RESEARCH_AGENT", "1")
    assert agent_for_intent("search") == "research"


def test_keyword_classifier_search_respects_research_flag(monkeypatch) -> None:
    monkeypatch.setenv("IRIS_RESEARCH_AGENT", "true")
    result = KeywordClassifier().classify("search for latest AI news")
    assert result.intent == "search"
    # KeywordClassifier is static; remap happens via agent_for_intent at call sites.
    assert result.agent_type == "system"


def test_llm_router_uses_helper_for_agent_mapping(monkeypatch) -> None:
    monkeypatch.setenv("IRIS_RESEARCH_AGENT", "yes")

    def invoke(_system: str, _user: str) -> str:
        return '{"intent": "search"}'

    result = LLMRouterClassifier(invoke=invoke).classify("what is new with llama")
    assert result.intent == "search"
    assert result.agent_type == "research"
