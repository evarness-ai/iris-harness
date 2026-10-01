"""Tests for teach-a-preference standing-instruction capture (ADR-0089)."""

from __future__ import annotations

from iris_harness.memory.standing_instructions import (
    StandingInstruction,
    extract_standing_instruction,
    looks_like_standing_instruction,
)

# The exact utterance from the live session (web-3dedfff1) that never stuck.
_TAUGHT = (
    "when I ask you how is my day looks like, just tell me top 10 emails and any "
    "dues, to-do's and my calendar schedules or reminders"
)


class TestLooksLikeStandingInstruction:
    def test_detects_when_i_ask_rule(self) -> None:
        assert looks_like_standing_instruction(_TAUGHT)

    def test_detects_from_now_on(self) -> None:
        assert looks_like_standing_instruction(
            "from now on, send my brief with a greeting at the top"
        )

    def test_detects_whenever_i_say(self) -> None:
        assert looks_like_standing_instruction("whenever I say good morning, read my inbox")

    def test_ignores_plain_question(self) -> None:
        assert not looks_like_standing_instruction("how is my day today?")

    def test_ignores_short_and_empty(self) -> None:
        assert not looks_like_standing_instruction("when?")
        assert not looks_like_standing_instruction("")

    def test_ignores_when_without_an_ask_clause(self) -> None:
        # "when" + a verb but NOT bound to the user asking/saying — that's a habit
        # for the miner, not a triggered rule.
        assert not looks_like_standing_instruction("I usually do email when I wake up")


class TestExtractViaLLM:
    def test_parses_clean_json(self) -> None:
        def caller(_prompt: str) -> str:
            return (
                '{"name": "day-overview", '
                '"trigger_keywords": ["how is my day", "how is my day today"], '
                '"instruction": "Show top 10 emails from both accounts plus today\'s '
                'dues, to-dos and calendar events or reminders."}'
            )

        out = extract_standing_instruction(_TAUGHT, caller)
        assert out is not None
        assert out.name == "day-overview"
        assert "how is my day" in out.trigger_keywords
        assert "top 10 emails" in out.instruction

    def test_tolerates_fenced_json(self) -> None:
        def caller(_prompt: str) -> str:
            return '```json\n{"name":"x","trigger_keywords":["hi"],"instruction":"Wave."}\n```'

        out = extract_standing_instruction("whenever I say hi, wave", caller)
        assert out is not None and out.instruction == "Wave."

    def test_llm_declines_falls_back_to_regex(self) -> None:
        # LLM returns the decline sentinel, but the utterance *is* a rule — the
        # regex floor still recovers a matchable trigger + instruction.
        out = extract_standing_instruction(_TAUGHT, lambda _p: '{"name": null}')
        assert out is not None
        assert "how is my day" in out.trigger_keywords[0]
        assert "emails" in out.instruction.lower()

    def test_llm_exception_falls_back_to_regex(self) -> None:
        def boom(_prompt: str) -> str:
            raise RuntimeError("ollama down")

        out = extract_standing_instruction("when I ask my status, show open tasks", boom)
        assert out is not None
        assert out.trigger_keywords[0] == "my status"

    def test_returns_none_for_non_instruction(self) -> None:
        assert extract_standing_instruction("what's my net worth?", lambda _p: "{}") is None


class TestSummary:
    def test_summary_is_one_line_and_truncates(self) -> None:
        si = StandingInstruction(
            name="x", trigger_keywords=("how is my day",), instruction="A" * 200
        )
        s = si.summary()
        assert s.startswith("when you ask")
        assert "how is my day" in s
        assert len(s) < 180
