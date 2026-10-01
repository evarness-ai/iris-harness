"""Tests for the durable-fact plausibility gate (``memory.fact_validation``).

The gate is the backstop that prevents an eval/loose phrase from being written
as an identity fact.  The two ``REJECT`` cases at the top are the literal values
that corrupted a real user's profile — they must never validate again.
"""

from __future__ import annotations

import pytest

from iris_harness.memory.fact_validation import (
    is_durable_fact,
    is_fact_grounded,
    is_plausible_fact,
)

# --- Durability gate: ephemeral tokens must never become durable facts ------
# These are the exact pollution entries found in a real memory.db audit.
DURABILITY_REJECTS = [
    ("greeting", "Hello"),
    ("day", "tomorrow"),
    ("task", "plan"),
    ("reminder_time", "6 pm tomorrow"),
    ("date", "today"),
    ("mood", "happy"),
    ("time", "6pm"),
    ("status", "tomorrow"),
    # System / conversational metadata mis-captured as user facts (issue 0032).
    ("action", "fetch market indexes and compare"),  # a query echo
    ("model", "gpt-5-mini"),  # the assistant's model, not a user fact
    ("version", "5.4"),
    ("command", "show portfolio"),
    ("intent", "finance"),
    ("query", "how is my day"),
]

DURABILITY_KEEPS = [
    ("name", "Robin"),
    ("country", "India"),
    ("email", "user@example.com"),
    ("blog", "www.web3notes.example"),
    ("location", "Springfield"),
    ("employer", "Acme"),
    ("timezone", "America/Los_Angeles"),
    ("profession", "teacher"),
    ("age", "12"),  # bare number must NOT be mistaken for a clock time
]


@pytest.mark.parametrize(("key", "value"), DURABILITY_REJECTS)
def test_durability_gate_rejects_ephemeral(key: str, value: str) -> None:
    ok, reason = is_durable_fact(key, value)
    assert ok is False, f"{key}={value!r} should be rejected as ephemeral"
    assert reason


@pytest.mark.parametrize(("key", "value"), DURABILITY_KEEPS)
def test_durability_gate_keeps_real_facts(key: str, value: str) -> None:
    ok, _ = is_durable_fact(key, value)
    assert ok is True, f"{key}={value!r} is a durable fact and must pass"


# --- The exact regression that motivated this gate -------------------------

REGRESSION_REJECTS = [
    ("name", "working on a project called Aur"),
    ("location", "Berlin and I prefer very concise answers"),
]

# --- Other fragments / mis-captures that must be rejected ------------------

OTHER_REJECTS = [
    ("name", "working"),  # bare verb
    ("name", ""),  # empty
    ("name", "   "),  # whitespace only
    ("name", "going to the store later"),  # >3 words + verbs
    ("name", "Robin and I live in Berlin"),  # clause + first-person
    ("location", "Berlin and I work remotely"),  # clause
    ("location", "the place where I currently happen to reside today"),  # too long/wordy
    (
        "profession",
        "a software engineer who really loves building agentic systems daily",
    ),  # >80 chars
    ("bio", "I am a senior engineer"),  # first-person pronoun
]

# --- Genuine atomic facts that MUST still pass -----------------------------

VALID = [
    ("name", "Robin"),
    ("name", "Mary Jane"),
    ("name", "Jean-Luc"),
    ("name", "O'Brien"),
    ("location", "Springfield, Illinois"),
    ("location", "Berlin"),
    ("location", "San Francisco, California, USA"),
    ("employer", "Acme"),
    ("profession", "teacher"),
    ("preferred_language", "Python"),
    ("languages", "Spanish, French"),
    ("timezone", "America/Chicago"),
]


@pytest.mark.parametrize(("key", "value"), REGRESSION_REJECTS)
def test_regression_corruption_cases_are_rejected(key: str, value: str) -> None:
    ok, reason = is_plausible_fact(key, value)
    assert ok is False, f"the {key!r} corruption value should be rejected"
    assert reason, "a rejection must carry an auditable reason"


@pytest.mark.parametrize(("key", "value"), OTHER_REJECTS)
def test_sentence_fragments_are_rejected(key: str, value: str) -> None:
    ok, reason = is_plausible_fact(key, value)
    assert ok is False, f"{key}={value!r} should be rejected"
    assert reason


@pytest.mark.parametrize(("key", "value"), VALID)
def test_genuine_facts_pass(key: str, value: str) -> None:
    ok, reason = is_plausible_fact(key, value)
    assert ok is True, f"{key}={value!r} should pass but was rejected: {reason}"
    assert reason == ""


# --- Grounding: a fact value must be supported by the user's message ---------

# (value, message) pairs the extractor must REJECT as ungrounded. The headline
# case is the LLM bleeding "teacher" from its own few-shot example into a
# turn that never mentioned it.
UNGROUNDED = [
    ("teacher", "can you show me a sample of my daily brief today"),
    ("Acme", "what's on my calendar tomorrow?"),
    ("Springfield", "summarize my unread email"),
    ("New York City", "i live in NYC"),  # normalised away from the wording — rejected by design
]

# (value, message) pairs that ARE grounded and must pass.
GROUNDED = [
    ("Robin", "my name is Robin"),
    ("Berlin", "I live in Berlin"),
    ("Springfield, Illinois", "I stay in Springfield, Illinois"),
    ("teacher", "I work as a teacher at a startup"),
    ("Python", "I mostly write Python these days"),
]


@pytest.mark.parametrize(("value", "message"), UNGROUNDED)
def test_ungrounded_facts_are_rejected(value: str, message: str) -> None:
    assert is_fact_grounded(value, message) is False


@pytest.mark.parametrize(("value", "message"), GROUNDED)
def test_grounded_facts_pass(value: str, message: str) -> None:
    assert is_fact_grounded(value, message) is True
