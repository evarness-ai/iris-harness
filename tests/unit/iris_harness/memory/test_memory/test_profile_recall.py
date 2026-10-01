"""Tests for deterministic profile recall (issue 0023)."""

from __future__ import annotations

from dataclasses import dataclass

from iris_harness.memory.profile_recall import build_profile_recall


@dataclass
class _F:
    key: str
    value: str


FACTS = [_F("blog", "www.web3notes.example"), _F("name", "Robin"), _F("email", "s@x.com")]


def test_recall_matches_blog_with_trailing_noun() -> None:
    assert build_profile_recall("Do you know my blog site?", FACTS) == (
        "Yes — your blog is www.web3notes.example."
    )


def test_recall_matches_name_and_email() -> None:
    assert "Robin" in build_profile_recall("what's my name?", FACTS)
    assert "s@x.com" in build_profile_recall("what is my email address?", FACTS)


def test_recall_unknown_attribute_is_honest_not_a_guess() -> None:
    out = build_profile_recall("do you know my zodiac sign?", FACTS)
    assert "don't have your zodiac sign" in out


def test_non_recall_queries_defer_to_llm() -> None:
    # broad profile asks + non-profile questions return None (LLM handles them)
    assert build_profile_recall("what do you know about me?", FACTS) is None
    assert build_profile_recall("what's the weather today?", FACTS) is None


def test_no_facts_still_defers_when_not_a_recall() -> None:
    assert build_profile_recall("hello there", []) is None
