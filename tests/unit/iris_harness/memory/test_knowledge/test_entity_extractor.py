"""Behavioral tests for the entity extractor."""

from __future__ import annotations

from iris_harness.memory.knowledge.entity_extractor import EntityExtractor


def test_extracts_person_name() -> None:
    extractor = EntityExtractor()
    entities = extractor.extract("I received an email from John Smith about the contract.")

    names = {e.name for e in entities}
    assert "John Smith" in names


def test_extracts_institution_name() -> None:
    extractor = EntityExtractor()
    entities = extractor.extract("My statement from Chase Bank arrived today.")

    names = {e.name for e in entities}
    assert any("Chase Bank" in n for n in names)


def test_hints_take_priority() -> None:
    extractor = EntityExtractor()
    entities = extractor.extract("Some text about things.", hints=["ACME Corp"])

    names = {e.name for e in entities}
    assert "ACME Corp" in names


def test_short_text_returns_empty() -> None:
    extractor = EntityExtractor()
    entities = extractor.extract("Hi")

    assert entities == []


def test_deduplication_across_patterns() -> None:
    extractor = EntityExtractor()
    entities = extractor.extract(
        "Chase Bank is great. I always use Chase Bank for payments.", hints=["Chase Bank"]
    )
    names = [e.name for e in entities]
    assert names.count("Chase Bank") == 1


def test_llm_entities_supplement_regex() -> None:
    def llm(prompt: str) -> str:
        return "NAME: OpenAI | TYPE: institution"

    extractor = EntityExtractor(llm_call=llm)
    entities = extractor.extract("We discussed OpenAI's latest models at the conference today.")

    names = {e.name for e in entities}
    assert "OpenAI" in names


def test_llm_failure_falls_back_gracefully() -> None:
    def bad_llm(prompt: str) -> str:
        raise RuntimeError("LLM offline")

    extractor = EntityExtractor(llm_call=bad_llm)
    entities = extractor.extract("John Smith met with Mary Johnson at the Apple Inc headquarters.")

    assert len(entities) >= 0  # No crash; may have regex results


# ---------------------------------------------------------------------------
# ADR-0010 — precision filters: _PERSON_RE bigram denylist + hint validation
# ---------------------------------------------------------------------------


def test_bigram_stopword_rejects_breaking_news() -> None:
    """A 'Breaking News' bigram must NOT be extracted as a person."""
    extractor = EntityExtractor()
    entities = extractor.extract("Breaking News: market crash today.")

    names = {e.name for e in entities}
    assert "Breaking News" not in names


def test_bigram_stopword_rejects_today_programme() -> None:
    """A 'Today Programme' bigram must NOT be extracted as a person."""
    extractor = EntityExtractor()
    entities = extractor.extract("Today Programme covered the markets this morning.")

    names = {e.name for e in entities}
    assert "Today Programme" not in names


def test_allowlist_accepts_new_york() -> None:
    """A bigram with a stopword in the allowlist must still be extracted."""
    extractor = EntityExtractor()
    entities = extractor.extract("I went to New York yesterday.")

    names = {e.name for e in entities}
    assert "New York" in names


def test_hint_snake_case_rejected() -> None:
    """Snake_case hints (tool names) must not become entities."""
    extractor = EntityExtractor()
    entities = extractor.extract("Some context.", hints=["web_fetch", "data_storage"])

    names = {e.name for e in entities}
    assert "web_fetch" not in names
    assert "data_storage" not in names


def test_hint_too_short_rejected() -> None:
    """Hints shorter than 3 chars must be dropped."""
    extractor = EntityExtractor()
    entities = extractor.extract("Some context.", hints=["hi", "X"])

    names = {e.name for e in entities}
    assert "hi" not in names
    assert "X" not in names


def test_hint_too_long_rejected() -> None:
    """Hints longer than 60 chars must be dropped."""
    runaway = "x" * 80
    extractor = EntityExtractor()
    entities = extractor.extract("Some context.", hints=[runaway])

    names = {e.name for e in entities}
    assert runaway not in names


def test_hint_bigram_stopword_rejected() -> None:
    """Two-word hints where either word is a stopword must be dropped."""
    extractor = EntityExtractor()
    entities = extractor.extract("Some context.", hints=["hello world", "the bank"])

    names = {e.name for e in entities}
    assert "hello world" not in names
    assert "the bank" not in names


def test_hint_real_name_accepted() -> None:
    """Real-name bigram hints must still produce entity pages."""
    extractor = EntityExtractor()
    entities = extractor.extract("Random text.", hints=["Bob Smith"])

    names = {e.name for e in entities}
    assert "Bob Smith" in names
