"""Deterministic clock/calendar replies cover natural trailing qualifiers.

"what time is it today?" must answer from the system clock, not fall through to
the LLM (which has no reliable clock and hallucinates the date/timezone).
"""

from __future__ import annotations

import pytest

from iris_harness.runtime.nlu_parsing import _deterministic_time_date_reply


@pytest.mark.parametrize(
    "message",
    [
        "what time is it today?",
        "what time is it?",
        "what time is it right now?",
        "what's the time now",
        "what is the time today",
        "whats the time now?",
        "could you tell me what time it is",
        "can you tell me the time?",
        "tell me what time it is",
        "please tell me the current time",
    ],
)
def test_time_queries_are_deterministic(message: str) -> None:
    reply = _deterministic_time_date_reply(message)
    assert reply is not None and reply.startswith("Current local time:")


@pytest.mark.parametrize(
    "message",
    [
        "what is the date today?",
        "what's today's date?",
        "whats the date today",
        "could you tell me today's date?",
        "tell me what the date is",
    ],
)
def test_date_queries_are_deterministic(message: str) -> None:
    reply = _deterministic_time_date_reply(message)
    assert reply is not None and reply.startswith("Today's date:")


def test_day_query_with_today() -> None:
    assert (_deterministic_time_date_reply("what day is it today?") or "").startswith("Today is ")


@pytest.mark.parametrize(
    "message",
    [
        "tell me a joke",
        "what is the weather today?",
        "what time does the store open?",
        "could you tell me what time the meeting is?",
        "can you tell me a story about time",
    ],
)
def test_non_clock_queries_fall_through(message: str) -> None:
    assert _deterministic_time_date_reply(message) is None
