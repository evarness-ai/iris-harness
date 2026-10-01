"""Phase 3 checkpoint store.

When the evaluator decides ``halt`` or ``require_approval`` at
``PostStep``, the caller (AgenticCore / coding pipeline) writes a
checkpoint here so the run can be inspected or resumed. v1 ships the
store + a chat-shape serializer (~10–50 KB per row, design §10.2).
Cross-process resume for the coding agent + the side-effect ledger
land in Phase 4.

Default TTL is 7 days. Rows with ``pinned=1`` survive purge — pin via
``iris checkpoint pin <run_id>``.
"""

from iris_harness.memory.state.chat import ChatCheckpointPayload
from iris_harness.memory.state.continuations import (
    Continuation,
    ContinuationConflictError,
    ContinuationStore,
)
from iris_harness.memory.state.store import (
    Checkpoint,
    CheckpointNotFoundError,
    CheckpointStore,
    CheckpointTooLargeError,
)

__all__ = [
    "ChatCheckpointPayload",
    "Checkpoint",
    "CheckpointNotFoundError",
    "CheckpointStore",
    "CheckpointTooLargeError",
    "Continuation",
    "ContinuationConflictError",
    "ContinuationStore",
]
