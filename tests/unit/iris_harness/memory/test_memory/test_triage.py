"""Tests for deterministic memory triage rails."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from iris_harness.memory.triage import (
    MemoryActionability,
    MemoryDestination,
    MemoryDurability,
    MemoryTriageKind,
    triage_memory_item,
)


def test_typed_user_fact_routes_to_user_profile() -> None:
    result = triage_memory_item("preferred_language: Python", signal_type="extracted_fact")

    assert result.schema_version == "memory-triage/v1"
    assert result.kind == MemoryTriageKind.USER_FACT
    assert result.destination == MemoryDestination.USER_MD
    assert result.durability == MemoryDurability.LONG_TERM
    assert not result.requires_review


def test_active_context_routes_open_loops_to_active_md() -> None:
    result = triage_memory_item("We need to finish the memory framework slice next.")

    assert result.kind == MemoryTriageKind.ACTIVE_CONTEXT
    assert result.destination == MemoryDestination.ACTIVE_MD
    assert result.durability == MemoryDurability.SHORT_TERM


def test_routine_requires_review() -> None:
    result = triage_memory_item("Every morning create a repo briefing and send a digest.")

    assert result.kind == MemoryTriageKind.ROUTINE
    assert result.destination == MemoryDestination.ROUTINE_STORE
    assert result.actionability == MemoryActionability.WORKFLOW
    assert result.requires_review


def test_behavior_requires_review() -> None:
    result = triage_memory_item("When tests fail, explain the likely root cause before editing.")

    assert result.kind == MemoryTriageKind.BEHAVIOR
    assert result.destination == MemoryDestination.BEHAVIOR
    assert result.requires_review


def test_automation_candidate_routes_to_skill_queue() -> None:
    result = triage_memory_item("Make this a reusable skill after it works twice.")

    assert result.kind == MemoryTriageKind.AUTOMATION_CANDIDATE
    assert result.destination == MemoryDestination.SKILL_QUEUE
    assert result.actionability == MemoryActionability.AUTOMATION_CANDIDATE
    assert result.requires_review


def test_source_preference_routes_to_episodic_index() -> None:
    result = triage_memory_item("Prefer official docs when researching Python APIs.")

    assert result.kind == MemoryTriageKind.SOURCE_PREFERENCE
    assert result.destination == MemoryDestination.EPISODIC_MD


def test_tool_outcome_routes_to_learning_db() -> None:
    result = triage_memory_item("calendar sync", signal_type="tool_result", tool_outcome="failed")

    assert result.kind == MemoryTriageKind.LEARNING_SIGNAL
    assert result.destination == MemoryDestination.LEARNING_DB


def test_result_is_frozen_and_json_serializable() -> None:
    result = triage_memory_item("Remember that I prefer concise answers.")

    with pytest.raises(ValidationError):
        result.confidence = 0.1  # type: ignore[misc]

    payload = result.model_dump(mode="json")
    assert payload["kind"] == "user_fact"
    assert payload["destination"] == "user_md"
    assert payload["schema_version"] == "memory-triage/v1"
