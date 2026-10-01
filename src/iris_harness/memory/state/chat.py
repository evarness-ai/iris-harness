"""Chat-shape checkpoint payload.

Phase 3 chat resume requires the original query, the steps so far, and
whatever memory/retrieval context the run was using. This module owns
that serialization. AgenticCore's ``ReactTrace`` is the source of
truth at runtime; ``ChatCheckpointPayload`` is the on-disk shape.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CheckpointStep(BaseModel):
    """One ReAct step's serialized form."""

    model_config = ConfigDict(frozen=True)

    thought: str = ""
    action: str | None = None
    action_input: dict[str, Any] = Field(default_factory=dict)
    observation: str | None = None
    final_answer: str | None = None
    is_terminal: bool = False


class ChatCheckpointPayload(BaseModel):
    """Resume payload for a halted chat run.

    Encodes what AgenticCore needs to continue from step N+1:

    - ``query``: original user prompt
    - ``steps``: the trace through the halted step
    - ``memory_context``: retrieved memory snippets (raw, opaque to
      this module — caller knows the shape)
    - ``iteration``: the next iteration index to attempt
    - ``halt_reason``: the evaluator/kernel reason string at halt
    """

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., min_length=1)
    steps: tuple[CheckpointStep, ...] = Field(default_factory=tuple)
    iteration: int = Field(..., ge=0)
    halt_reason: str | None = None
    memory_context: dict[str, Any] | None = None
    # ADR-0118: the approval a destructive call is waiting on. The resumed run settles
    # it — executing the pinned calls if it was approved — before the loop continues.
    pending_approval_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, raw: dict[str, Any]) -> ChatCheckpointPayload:
        return cls.model_validate(raw)
