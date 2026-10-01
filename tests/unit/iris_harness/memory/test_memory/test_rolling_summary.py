"""The rolling session summary: incremental, sectioned, and never extractive.

What this replaces: no production caller ever passed the compactor an LLM, so every
summary was the first 3 turns cut to 80 characters — one real stored example began
"assistant: 1. **Business News | Today's International Headlines | Reuters** (https://www.re".
Each compaction also overwrote the previous summary without reading it, so anything
older than the last compaction was gone.
"""

from __future__ import annotations

import pytest

from iris_harness.memory.compactor import (
    ConversationCompactor,
    ConversationTurn,
    reset_summary_config_cache,
)


@pytest.fixture(autouse=True)
def _fresh_config() -> None:
    reset_summary_config_cache()


def _turns(n: int, *, text: str = "message") -> list[ConversationTurn]:
    out: list[ConversationTurn] = []
    for i in range(n):
        out.append(ConversationTurn(role="user", content=f"{text} {i} " + "filler " * 20))
        out.append(ConversationTurn(role="assistant", content=f"reply {i} " + "filler " * 20))
    return out


class TestRollingForward:
    def test_the_previous_summary_is_an_input(self) -> None:
        seen: list[str] = []

        def _llm(prompt: str) -> str:
            seen.append(prompt)
            return "Goal: ship the memory work."

        compactor = ConversationCompactor(compaction_threshold=2, keep_recent=2, llm_call=_llm)
        compactor.compact(_turns(4), previous_summary="Goal: an earlier goal. Open items: none.")

        assert "Goal: an earlier goal." in seen[0]
        assert "EARLIER SUMMARY" in seen[0]

    def test_the_prompt_names_every_section(self) -> None:
        seen: list[str] = []
        compactor = ConversationCompactor(
            compaction_threshold=2, keep_recent=2, llm_call=lambda p: seen.append(p) or "ok"
        )

        compactor.compact(_turns(4))

        for label in ("Goal", "Decisions & outcomes", "Open items", "Referenced"):
            assert label in seen[0]

    def test_the_summary_is_capped(self) -> None:
        compactor = ConversationCompactor(
            compaction_threshold=2, keep_recent=2, llm_call=lambda p: "word " * 5000
        )

        result = compactor.compact(_turns(4))

        assert len(result.summary) <= 320 * 4 + 1


class TestFailureKeepsWhatWeHad:
    def test_no_summarizer_keeps_the_previous_summary(self) -> None:
        compactor = ConversationCompactor(compaction_threshold=2, keep_recent=2, llm_call=None)

        result = compactor.compact(_turns(4), previous_summary="Goal: keep me.")

        assert result.summary == "Goal: keep me."
        assert result.trigger == "count"  # it still compacts the window

    def test_a_raising_summarizer_keeps_the_previous_summary(self) -> None:
        def _boom(prompt: str) -> str:
            raise RuntimeError("model down")

        compactor = ConversationCompactor(compaction_threshold=2, keep_recent=2, llm_call=_boom)

        result = compactor.compact(_turns(4), previous_summary="Goal: keep me.")

        assert result.summary == "Goal: keep me."

    def test_an_empty_answer_keeps_the_previous_summary(self) -> None:
        compactor = ConversationCompactor(
            compaction_threshold=2, keep_recent=2, llm_call=lambda p: "   "
        )

        result = compactor.compact(_turns(4), previous_summary="Goal: keep me.")

        assert result.summary == "Goal: keep me."

    def test_there_is_no_extractive_fallback_any_more(self) -> None:
        """The old fallback produced summaries that looked real and were not."""
        import iris_harness.memory.compactor as module

        assert not hasattr(module, "_extractive_summary")

        compactor = ConversationCompactor(compaction_threshold=2, keep_recent=2, llm_call=None)
        result = compactor.compact(
            [
                ConversationTurn(role="assistant", content="1. **Business News | Reuters** " * 20),
                ConversationTurn(role="user", content="what's happening in India today?"),
                ConversationTurn(role="assistant", content="Top things happening " * 20),
                ConversationTurn(role="user", content="and tomorrow?"),
            ]
        )

        assert result.summary == ""  # nothing invented, nothing pasted


class TestTokenTrigger:
    def test_a_token_heavy_session_rolls_before_the_count_threshold(self) -> None:
        compactor = ConversationCompactor(
            compaction_threshold=100,  # count floor far away
            keep_recent=10,
            llm_call=lambda p: "Goal: rolled.",
            token_budget=400,
            compaction_ratio=0.8,
        )
        history = _turns(6)

        assert compactor.needs_compaction(history) is True
        result = compactor.compact(history)

        assert result.trigger == "tokens"
        assert result.summary == "Goal: rolled."
        assert result.tokens_after < result.tokens_before
