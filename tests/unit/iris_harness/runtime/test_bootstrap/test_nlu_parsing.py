"""Smoke tests for the core NLU parsing module (Phase 2, trimmed at M5.7 track A).

The confirmation decision and the time/date reply are the two things the core still
reads off a raw message. The reminder and meeting parsers moved to the calendar
plugin with their vocabulary (``tests/unit/test_calendar/test_nlu.py``);
this file pins that they are gone from here, so no domain trigger word sits in core.
"""

from __future__ import annotations

import inspect

import pytest

from iris_harness.runtime import nlu_parsing


def test_direct_imports_resolve() -> None:
    # The names the core's own mechanisms rely on must be importable here.
    for name in (
        "_parse_confirmation_decision",
        "_deterministic_time_date_reply",
        "_current_local_datetime",
    ):
        assert hasattr(nlu_parsing, name), name


def test_the_calendar_parsers_and_their_vocabulary_are_gone() -> None:
    """The owner's rule: the core carries no plugin's rules or intent keywords."""
    for name in (
        "_parse_meeting_request",
        "_parse_reminder_request",
        "_format_reminder_confirmation",
        "_format_event_confirmation",
        "_MEETING_VERB_RE",
        "_REMINDER_REQUEST_RE",
    ):
        assert not hasattr(nlu_parsing, name), name
    source = inspect.getsource(nlu_parsing)
    for word in ("remind me", "schedule", "appointment", "_MEETING", "_REMINDER"):
        assert word not in source, f"calendar vocabulary {word!r} is back in the core"


@pytest.mark.parametrize(
    "message,prefix",
    [
        ("what time is it?", "Current local time:"),
        ("what's today's date?", "Today's date:"),
        ("what day is it today?", "Today is "),
        ("tell me a joke", None),
    ],
)
def test_deterministic_time_date_reply(message: str, prefix: str | None) -> None:
    reply = nlu_parsing._deterministic_time_date_reply(message)
    if prefix is None:
        assert reply is None
    else:
        assert reply is not None and reply.startswith(prefix)


def test_confirmation_decision() -> None:
    assert nlu_parsing._parse_confirmation_decision("approve") == "approve"
    assert nlu_parsing._parse_confirmation_decision("no, cancel") == "reject"
    assert nlu_parsing._parse_confirmation_decision("what's on tomorrow?") is None


@pytest.mark.parametrize(
    "message",
    [
        # The incident's actual reply, from session web-48547cb8: IRIS asked
        # "Would you like to proceed with this script?" and could not read this
        # as the yes it plainly was.
        "proceed with this script",
        "proceed",
        "continue",
        "carry on",
        "go for it",
        "please do",
        "let's do it",
        "lets do it",
    ],
)
def test_answers_that_echo_our_own_offer_wording_read_as_approval(message: str) -> None:
    assert nlu_parsing._parse_confirmation_decision(message) == "approve"


@pytest.mark.parametrize(
    "message",
    [
        # A word boundary keeps the new verbs from swallowing unrelated turns.
        "proceeds from the sale were reinvested",
        "continuous integration is failing",
        "can you show me the script first",
        "what does the script do?",
    ],
)
def test_new_approval_verbs_do_not_swallow_unrelated_turns(message: str) -> None:
    assert nlu_parsing._parse_confirmation_decision(message) is None


def test_negated_proceed_still_reads_as_rejection() -> None:
    assert nlu_parsing._parse_confirmation_decision("don't proceed") == "reject"
    assert nlu_parsing._parse_confirmation_decision("no, not yet") == "reject"
