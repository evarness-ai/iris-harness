"""The two filters in front of the review queue: allowed keys, first-person messages."""

from __future__ import annotations

import pytest

from iris_harness.memory import fact_keys

# Fact keys such as `bank` and `credit_card` come from the test vocabulary fragment
# (tests/fixtures/test_vocabulary, installed by the `test_vocabulary` fixture), not
# from whichever domain plugin the tree happens to carry.
pytestmark = pytest.mark.usefixtures("test_vocabulary")


@pytest.fixture(autouse=True)
def _fresh_config() -> None:
    fact_keys.reset_cache()


class TestKeyAllowlist:
    @pytest.mark.parametrize("key", ["name", "city", "employer", "bank", "timezone", "interest"])
    def test_profile_keys_are_allowed(self, key: str) -> None:
        assert fact_keys.canonical_key(key) == key

    @pytest.mark.parametrize(
        "key",
        [
            "topic",  # topic=murder plot
            "source",  # source=CBS News
            "number",  # number=4
            "error",  # error=it shows error
            "folder",  # folder=one folder
            "file_extension",  # file_extension=png
            "alleged wrongdoing",
            "provider",  # provider=ollama
            "environment_variable",
        ],
    )
    def test_conversation_subject_keys_are_not(self, key: str) -> None:
        """Every one of these is a real key from the store this replaces."""
        assert fact_keys.canonical_key(key) is None

    def test_aliases_map_to_the_canonical_key(self) -> None:
        assert fact_keys.canonical_key("organization") == "employer"
        assert fact_keys.canonical_key("home_town") == "hometown"
        assert fact_keys.canonical_key("citizenship") == "nationality"
        assert fact_keys.canonical_key("job") == "profession"

    def test_case_and_separators_do_not_matter(self) -> None:
        assert fact_keys.canonical_key("  Home-Town ") == "hometown"

    def test_origin_is_never_folded_into_residence(self) -> None:
        """Someone from India living in the UK holds both facts; neither overwrites."""
        for origin in ("hometown", "home_town", "nationality", "citizenship"):
            assert fact_keys.canonical_key(origin) not in {"city", "country", "location"}

    def test_an_unreadable_ontology_allows_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A broken allowlist must stop proposals, not wave everything through.

        Since memris PR 3 the allowlist IS the ontology's fact mappings.
        """

        def broken() -> None:
            raise ValueError("ontology.yaml does not compile")

        monkeypatch.setattr(fact_keys, "memory_ontology", broken)
        fact_keys.reset_cache()
        try:
            assert fact_keys.canonical_key("name") is None
            assert fact_keys.allowed_keys() == frozenset()
        finally:
            monkeypatch.undo()
            fact_keys.reset_cache()


class TestFirstPersonGate:
    @pytest.mark.parametrize(
        "message",
        [
            "my name is Robin",
            "I work as a solutions architect",
            "i'm based in Springfield",
            "we bank with Northwind",
        ],
    )
    def test_self_statements_pass(self, message: str) -> None:
        assert fact_keys.is_self_statement(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "the Department of Justice said the charges were filed",  # employer=DoJ came from this
            "Remy Kumar works at Acme Bank",
            "what's happening in India today?",
            "",
        ],
    )
    def test_third_party_content_does_not(self, message: str) -> None:
        assert fact_keys.is_self_statement(message) is False

    def test_quoted_text_is_not_the_user_speaking(self) -> None:
        assert fact_keys.is_self_statement("> I am the CEO of Acme\n\nwhat do you think?") is False

    def test_a_fenced_block_is_not_either(self) -> None:
        assert fact_keys.is_self_statement("```\nI am a config file\n```") is False
