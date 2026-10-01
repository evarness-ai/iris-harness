"""Unit tests for the LLM-driven user fact extractor (Slice 5)."""

from __future__ import annotations

from dataclasses import dataclass

from iris_harness.memory.fact_extractor import (
    ExtractedFact,
    _parse_facts_json,
    extract_facts_with_llm,
)


@dataclass
class _FakeClient:
    """Minimal stand-in for CodingLLMClient.invoke."""

    response: str
    raises: BaseException | None = None
    last_user_prompt: str = ""

    def invoke(self, *, system_prompt: str, user_prompt: str) -> str:
        self.last_user_prompt = user_prompt
        if self.raises is not None:
            raise self.raises
        return self.response


class TestParseFactsJson:
    def test_clean_json_array(self) -> None:
        raw = '[{"key":"name","value":"Robin","confidence":0.9}]'
        facts = _parse_facts_json(raw)
        assert facts == [ExtractedFact(key="name", value="Robin", confidence=0.9)]

    def test_strips_code_fence(self) -> None:
        raw = '```json\n[{"key":"city","value":"Springfield","confidence":0.8}]\n```'
        facts = _parse_facts_json(raw)
        assert len(facts) == 1
        assert facts[0].value == "Springfield"

    def test_extracts_array_from_preamble(self) -> None:
        raw = (
            "Sure, here are the facts:\n"
            '[{"key":"name","value":"Ada","confidence":0.95}]\n'
            "Hope that helps!"
        )
        facts = _parse_facts_json(raw)
        assert facts[0].key == "name"
        assert facts[0].value == "Ada"

    def test_empty_array_returns_empty_list(self) -> None:
        assert _parse_facts_json("[]") == []

    def test_invalid_json_returns_empty(self) -> None:
        assert _parse_facts_json("not json at all") == []

    def test_drops_items_missing_key_or_value(self) -> None:
        raw = (
            '[{"key":"name","value":"Robin","confidence":0.9},'
            '{"key":"","value":"x","confidence":0.5},'
            '{"key":"x","value":"","confidence":0.5}]'
        )
        facts = _parse_facts_json(raw)
        assert [f.key for f in facts] == ["name"]

    def test_clamps_confidence_into_range(self) -> None:
        raw = (
            '[{"key":"a","value":"1","confidence":2.5},'
            '{"key":"b","value":"2","confidence":-0.4}]'
        )
        facts = _parse_facts_json(raw)
        confidences = [f.confidence for f in facts]
        assert confidences == [1.0, 0.0]

    def test_missing_confidence_defaults_to_half(self) -> None:
        raw = '[{"key":"a","value":"1"}]'
        facts = _parse_facts_json(raw)
        assert facts[0].confidence == 0.5

    def test_value_punctuation_trimmed(self) -> None:
        raw = '[{"key":"name","value":"Robin.","confidence":0.9}]'
        facts = _parse_facts_json(raw)
        assert facts[0].value == "Robin"

    def test_keys_lowercased(self) -> None:
        raw = '[{"key":"NAME","value":"Robin","confidence":0.9}]'
        facts = _parse_facts_json(raw)
        assert facts[0].key == "name"

    def test_top_level_object_returns_empty(self) -> None:
        raw = '{"key":"name","value":"x","confidence":0.9}'
        # Not a list — extractor treats it as malformed.
        assert _parse_facts_json(raw) == []


class TestExtractFactsWithLlm:
    def test_routes_message_through_client(self) -> None:
        client = _FakeClient(response='[{"key":"name","value":"Robin","confidence":0.9}]')
        facts = extract_facts_with_llm("My name is Robin.", client=client)
        assert len(facts) == 1
        assert facts[0].key == "name"
        # The user prompt should embed the message verbatim.
        assert "My name is Robin." in client.last_user_prompt

    def test_empty_message_skips_call(self) -> None:
        client = _FakeClient(response="should not be returned")
        assert extract_facts_with_llm("   ", client=client) == []
        # Client.invoke was never called → last_user_prompt stays empty.
        assert client.last_user_prompt == ""

    def test_client_exception_returns_empty(self) -> None:
        client = _FakeClient(response="", raises=RuntimeError("boom"))
        assert extract_facts_with_llm("hello", client=client) == []


# ── Deterministic declarative capture (issue 0020) ─────────────────────────

from iris_harness.memory.fact_extractor import extract_declarative_facts  # noqa: E402
from iris_harness.memory.fact_validation import is_fact_grounded, is_plausible_fact  # noqa: E402


def test_declarative_captures_blog_url_and_passes_gates() -> None:
    msg = (
        "I write blogs in www.web3notes.example about my experiences on agentic "
        "harness and Local AI experiments"
    )
    facts = extract_declarative_facts(msg)
    assert len(facts) == 1
    f = facts[0]
    assert f.key == "blog" and f.value == "www.web3notes.example"
    # must clear the persistence gates (the long sentence used to fail them)
    assert is_plausible_fact(f.key, f.value)[0]
    assert is_fact_grounded(f.value, msg)


def test_declarative_captures_my_noun_is_url() -> None:
    facts = {f.key: f.value for f in extract_declarative_facts("my website is https://example.org")}
    assert facts["website"] == "https://example.org"


def test_declarative_no_false_positive_without_url() -> None:
    assert extract_declarative_facts("I love agentic harness and Local AI experiments") == []
    # "blog" mentioned but no URL -> nothing to capture
    assert extract_declarative_facts("I should start a blog someday") == []


# --- closed extraction: the prompt is generated from the ontology (memris PR 3) -------


def test_the_prompt_offers_every_fact_key_and_nothing_else() -> None:
    import re

    from iris_harness.memory import fact_keys
    from iris_harness.memory.fact_extractor import keys_block, system_prompt

    fact_keys.reset_cache()
    offered = set(re.findall(r"^- ([a-z_]+)", keys_block(), re.MULTILINE))
    assert offered == set(fact_keys.allowed_keys())
    prompt = system_prompt()
    assert keys_block() in prompt and "{keys}" not in prompt
    # the key the hand-written prompt used to teach does not exist
    assert "dietary_restriction" not in prompt


def test_a_relation_key_says_what_it_points_at() -> None:
    from iris_harness.memory.fact_extractor import keys_block

    lines = dict(line[2:].split(":", 1) for line in keys_block().splitlines() if ":" in line)
    assert "→" in lines["employer"] and "→" in lines["hometown"]
    assert lines["hometown"] != lines["city"]  # origin and residence stay apart


def test_the_user_prompt_carries_the_message() -> None:
    from iris_harness.memory.fact_extractor import user_prompt

    assert "I live in Leeds" in user_prompt("I live in Leeds")


def test_a_fact_about_someone_one_hop_away_carries_their_name() -> None:
    raw = (
        '[{"key":"spouse","value":"Petra","confidence":0.9},'
        ' {"key":"employer","value":"Infosys","confidence":0.9,"about":" Petra. "}]'
    )

    facts = _parse_facts_json(raw)

    assert [(f.key, f.about) for f in facts] == [("spouse", None), ("employer", "Petra")]


def test_the_prompt_teaches_about() -> None:
    from iris_harness.memory.fact_extractor import system_prompt

    assert '"about"' in system_prompt()
