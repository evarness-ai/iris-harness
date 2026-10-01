"""Scope-qualifier detection for conversation-local statements (Phase 3 fix)."""

from __future__ import annotations

import pytest

from iris_harness.runtime.turn_capture import _is_conversation_scoped


@pytest.mark.parametrize(
    "message",
    [
        "for this conversation, my favorite color is teal",
        "in this chat, call me Alex",
        "just for now, set my timezone to UTC",
        "only for today, use metric units",
        "temporarily, my role is reviewer",
        "don't remember this, but I'm testing something",
    ],
)
def test_scoped_messages_detected(message: str) -> None:
    assert _is_conversation_scoped(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "my name is Robin",
        "I work at Acme as a teacher",
        "what is my favorite color?",
        "schedule a meeting for now",  # "for now" must not over-match
        "I live in Springfield",
    ],
)
def test_durable_messages_not_scoped(message: str) -> None:
    assert _is_conversation_scoped(message) is False
