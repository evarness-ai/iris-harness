"""Unit tests for the prompt-size budget helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from iris_harness.llm.budget import budget_for, estimate_tokens, trim_messages, trim_text


@dataclass
class _Msg:
    role: str
    content: str


def test_estimate_tokens_returns_zero_for_empty() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_uses_char_quarter_with_floor_of_one() -> None:
    assert estimate_tokens("hi") == 1
    assert estimate_tokens("a" * 400) == 100


def test_budget_for_default_fraction_reserves_response_room() -> None:
    # 75% of 4096 leaves room for num_predict on the model side.
    assert budget_for(4096) == 3072


def test_budget_for_zero_or_negative_returns_zero() -> None:
    assert budget_for(0) == 0
    assert budget_for(-1) == 0


def test_budget_for_rejects_out_of_range_fraction() -> None:
    with pytest.raises(ValueError):
        budget_for(2048, fraction=0.0)
    with pytest.raises(ValueError):
        budget_for(2048, fraction=1.0)


def test_trim_text_preserves_short_input() -> None:
    assert trim_text("short", max_chars=100) == "short"


def test_trim_text_truncates_with_ellipsis() -> None:
    out = trim_text("a" * 500, max_chars=10)
    assert len(out) == 10
    assert out.endswith("…")


def test_trim_text_returns_empty_when_budget_is_zero() -> None:
    assert trim_text("anything", max_chars=0) == ""


def test_trim_messages_preserves_first_and_last() -> None:
    # System + 3 history turns + final user — middle turns should be evicted first.
    msgs = [
        _Msg("system", "S" * 400),  # ~100 tokens
        _Msg("user", "U1" * 200),
        _Msg("assistant", "A1" * 200),
        _Msg("user", "U2" * 200),
        _Msg("user", "current question"),
    ]
    trimmed = trim_messages(msgs, budget_tokens=120)
    # First (system) and last (current question) always kept.
    assert trimmed[0].role == "system"
    assert trimmed[-1].content == "current question"
    # Middle turns reduced to fit budget.
    assert len(trimmed) < len(msgs)


def test_trim_messages_skips_when_two_or_fewer() -> None:
    msgs = [_Msg("system", "S"), _Msg("user", "U")]
    assert trim_messages(msgs, budget_tokens=1) == msgs


def test_trim_messages_returns_empty_for_empty_input() -> None:
    assert trim_messages([], budget_tokens=1000) == []
