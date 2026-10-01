"""ProvenanceLedger — Phase 5 grounding prerequisite (P1)."""

from __future__ import annotations

from iris_harness.agent.provenance import RETRIEVAL_TOOLS, ProvenanceLedger


def test_records_retrieval_tool_output() -> None:
    led = ProvenanceLedger()
    led.record("research", "Paris is the capital of France.")
    assert not led.is_empty
    assert led.sources() == ("research",)
    assert "Paris is the capital" in led.render()
    assert "[research]" in led.render()


def test_ignores_non_retrieval_tools() -> None:
    led = ProvenanceLedger()
    led.record("code_exec", "print(2+2)")
    led.record("propose_skill_from_sandbox", "{...}")
    assert led.is_empty


def test_ignores_empty_content() -> None:
    led = ProvenanceLedger()
    led.record("wiki_search", "   ")
    assert led.is_empty


def test_dedupes_identical_entries() -> None:
    led = ProvenanceLedger()
    led.record("memory_search", "same hit")
    led.record("memory_search", "same hit")
    assert len(led.sources()) == 1
    assert led.render().count("same hit") == 1


def test_distinct_sources_in_first_seen_order() -> None:
    led = ProvenanceLedger()
    led.record("wiki_search", "w")
    led.record("research", "x")
    led.record("wiki_search", "y")  # second wiki entry, source already seen
    assert led.sources() == ("wiki_search", "research")


def test_max_entries_cap() -> None:
    led = ProvenanceLedger(max_entries=2)
    led.record("research", "alpha")
    led.record("research", "beta")
    led.record("research", "gamma")  # dropped
    assert led.render().count("[research]") == 2
    assert "gamma" not in led.render()


def test_render_truncates_to_max_chars() -> None:
    led = ProvenanceLedger(max_chars=40)
    led.record("research", "x" * 100)
    rendered = led.render()
    assert "...[truncated]" in rendered
    assert len(rendered) <= 40 + len("\n...[truncated]") + len("[research]\n")


def test_retrieval_tools_registry_is_frozen() -> None:
    assert "research" in RETRIEVAL_TOOLS
    assert "memory_search" in RETRIEVAL_TOOLS
    assert "code_exec" not in RETRIEVAL_TOOLS
    assert isinstance(RETRIEVAL_TOOLS, frozenset)
