from __future__ import annotations

import pytest
from pydantic import ValidationError

from iris_harness.memory.state import ChatCheckpointPayload
from iris_harness.memory.state.chat import CheckpointStep


def test_roundtrip_preserves_steps_and_memory() -> None:
    payload = ChatCheckpointPayload(
        query="what's the weather?",
        steps=(
            CheckpointStep(thought="check tool", action="research", action_input={"q": "weather"}),
            CheckpointStep(thought="parse result", observation="sunny, 72F"),
        ),
        iteration=2,
        halt_reason="step_cap reached",
        memory_context={"facts": [{"k": "location", "v": "Austin"}]},
    )
    serialized = payload.to_payload()
    rehydrated = ChatCheckpointPayload.from_payload(serialized)
    assert rehydrated.query == "what's the weather?"
    assert rehydrated.iteration == 2
    assert len(rehydrated.steps) == 2
    assert rehydrated.steps[0].action == "research"
    assert rehydrated.steps[0].action_input == {"q": "weather"}
    assert rehydrated.memory_context == {"facts": [{"k": "location", "v": "Austin"}]}
    assert rehydrated.halt_reason == "step_cap reached"


def test_empty_query_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatCheckpointPayload(query="", iteration=0)


def test_negative_iteration_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatCheckpointPayload(query="x", iteration=-1)


def test_minimal_payload_round_trips() -> None:
    payload = ChatCheckpointPayload(query="hi", iteration=0)
    out = ChatCheckpointPayload.from_payload(payload.to_payload())
    assert out.steps == ()
    assert out.memory_context is None
    assert out.halt_reason is None
