"""Behavioral tests for the conversation compactor."""

from __future__ import annotations

from iris_harness.memory.compactor import ConversationCompactor, ConversationTurn


def _turns(n: int) -> list[ConversationTurn]:
    return [
        ConversationTurn(role="user" if i % 2 == 0 else "assistant", content=f"turn {i}")
        for i in range(n)
    ]


def test_no_compaction_needed_below_threshold() -> None:
    compactor = ConversationCompactor(compaction_threshold=20, keep_recent=10)
    result = compactor.compact(_turns(5))

    assert result.archived_count == 0
    assert len(result.recent_turns) == 5
    assert result.summary == ""


def test_compaction_archives_older_turns() -> None:
    compactor = ConversationCompactor(compaction_threshold=10, keep_recent=4)
    result = compactor.compact(_turns(15))

    assert result.archived_count == 11
    assert len(result.recent_turns) == 4


def test_no_summarizer_means_no_summary() -> None:
    """Without an LLM the window still compacts, but nothing is invented.

    This used to assert a non-empty summary, which the extractive fallback produced by
    pasting the first 3 turns cut to 80 chars. That ran in production for months and
    filled the store with "summaries" like
    "assistant: 1. **Business News | Today's International Headlines | Reuters** (https://www.re".
    """
    compactor = ConversationCompactor(compaction_threshold=5, keep_recent=2)
    result = compactor.compact(_turns(10))

    assert result.summary == ""
    assert len(result.recent_turns) < 10  # the window is still trimmed


def test_llm_summarizer_is_called_when_provided() -> None:
    called = [False]

    def llm(prompt: str) -> str:
        called[0] = True
        return "A short summary."

    compactor = ConversationCompactor(compaction_threshold=5, keep_recent=2, llm_call=llm)
    result = compactor.compact(_turns(10))

    assert called[0]
    assert result.summary == "A short summary."


def test_llm_error_keeps_the_previous_summary() -> None:
    def bad_llm(prompt: str) -> str:
        raise RuntimeError("llm unavailable")

    compactor = ConversationCompactor(compaction_threshold=5, keep_recent=2, llm_call=bad_llm)
    result = compactor.compact(_turns(10), previous_summary="Goal: what we already knew.")

    assert result.summary == "Goal: what we already knew."


def test_legacy_compact_conversation_api() -> None:
    compactor = ConversationCompactor(
        compaction_threshold=5, keep_recent=2, llm_call=lambda p: "Goal: legacy."
    )
    turns = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
    result = compactor.compact_conversation(turns)

    assert any(t.get("type") == "summary" for t in result)


def test_validate_compaction_passes_for_valid_output() -> None:
    compactor = ConversationCompactor(
        compaction_threshold=5, keep_recent=2, llm_call=lambda p: "Goal: legacy."
    )
    turns = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
    compacted = compactor.compact_conversation(turns)

    assert compactor.validate_compaction(compacted)


# ---------------------------------------------------------------------------
# ADR-0079 — window-aware (token-budget) auto-compaction
# ---------------------------------------------------------------------------


def _big_turns(n: int, chars: int) -> list[ConversationTurn]:
    return [
        ConversationTurn(role="user" if i % 2 == 0 else "assistant", content="x" * chars)
        for i in range(n)
    ]


def test_count_trigger_labels_telemetry() -> None:
    compactor = ConversationCompactor(compaction_threshold=5, keep_recent=2)
    result = compactor.compact(_turns(10))

    assert result.trigger == "count"
    assert result.tokens_after <= result.tokens_before


def test_token_trigger_fires_before_count_threshold() -> None:
    # A few huge turns are well under the 20-turn count floor but blow the window.
    # 6 turns * 400 chars ~= 600 tokens; budget 500, ratio 0.8 -> fires at >=400 tokens.
    compactor = ConversationCompactor(
        compaction_threshold=20, token_budget=500, compaction_ratio=0.8
    )
    turns = _big_turns(6, 400)

    assert compactor.needs_compaction(turns)
    result = compactor.compact(turns)
    assert result.trigger == "tokens"
    assert result.archived_count > 0


def test_token_keep_window_bounds_recent_by_tokens() -> None:
    # keep budget = 500 * 0.8 * 0.5 = 200 tokens; each turn ~100 tokens -> ~2 kept.
    compactor = ConversationCompactor(
        compaction_threshold=20, token_budget=500, compaction_ratio=0.8
    )
    result = compactor.compact(_big_turns(8, 400))

    assert 1 <= len(result.recent_turns) <= 3
    assert result.tokens_after < result.tokens_before


def test_single_huge_turn_keeps_at_least_one() -> None:
    compactor = ConversationCompactor(token_budget=100, compaction_ratio=0.8)
    result = compactor.compact(_big_turns(3, 2000))

    assert len(result.recent_turns) >= 1


def test_empty_turns_dropped_from_summary() -> None:
    seen = {}

    def llm(prompt: str) -> str:
        seen["prompt"] = prompt
        return "summary"

    compactor = ConversationCompactor(compaction_threshold=3, keep_recent=1, llm_call=llm)
    turns = [
        ConversationTurn(role="user", content="real question about taxes"),
        ConversationTurn(role="assistant", content="   "),
        ConversationTurn(role="user", content=""),
        ConversationTurn(role="assistant", content="real answer about taxes"),
        ConversationTurn(role="user", content="latest"),
    ]
    compactor.compact(turns)

    assert "taxes" in seen["prompt"]
    # Whitespace-only / empty archived turns never reach the summarizer.
    assert "user: \nassistant:    " not in seen["prompt"]


def test_no_token_budget_is_pure_count_behaviour() -> None:
    # Identical to the legacy path: huge turns under the count floor do NOT compact.
    compactor = ConversationCompactor(compaction_threshold=20, keep_recent=10)
    result = compactor.compact(_big_turns(5, 5000))

    assert result.trigger == "none"
    assert result.archived_count == 0


def test_archived_turns_exposed_for_learning() -> None:
    # ADR-0082: the summarized-away turns are exposed so a caller can mine them.
    compactor = ConversationCompactor(compaction_threshold=10, keep_recent=4)
    turns = _turns(15)
    result = compactor.compact(turns)

    assert len(result.archived_turns) == result.archived_count == 11
    assert result.archived_turns == tuple(turns[:11])  # the oldest 11, in order
    # The kept and archived spans partition the input.
    assert result.archived_turns + result.recent_turns == tuple(turns)


def test_archived_turns_empty_without_compaction() -> None:
    compactor = ConversationCompactor(compaction_threshold=20, keep_recent=10)
    result = compactor.compact(_turns(5))
    assert result.archived_turns == ()


# ---------------------------------------------------------------------------
# ADR-0084 — on-demand (forced) compaction
# ---------------------------------------------------------------------------


def test_force_compacts_below_threshold() -> None:
    # 12 turns is under the count floor (20) and there's no token budget -> auto trigger
    # is "none"; force compacts anyway, labelling the trigger "manual".
    compactor = ConversationCompactor(compaction_threshold=20, keep_recent=4)
    result = compactor.compact(_turns(12), force=True)

    assert result.trigger == "manual"
    assert result.archived_count == 8
    assert len(result.recent_turns) == 4


def test_force_is_noop_when_nothing_to_archive() -> None:
    # 5 turns, keep_recent 10 -> the whole conversation is kept, nothing to summarize.
    compactor = ConversationCompactor(compaction_threshold=20, keep_recent=10)
    result = compactor.compact(_turns(5), force=True)

    assert result.trigger == "none"
    assert result.archived_count == 0
    assert result.recent_turns == tuple(_turns(5))
