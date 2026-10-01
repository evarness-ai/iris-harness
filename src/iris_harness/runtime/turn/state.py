"""The request and mutable state that flow through one chat turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from iris_harness.kernel.governance.hooks.response_payload import Audience

if TYPE_CHECKING:
    from iris_harness.agent.agent_executor import AgentResult, AgentTask
    from iris_harness.agent.intent_router import IntentResult
    from iris_harness.agent.task_planner import TaskPlan
    from iris_harness.memory.retriever import MemoryContext
    from iris_harness.memory.state.continuations import Continuation
    from iris_harness.runtime.types import ChatResult

OnError = Literal["event", "raise"]


@dataclass(frozen=True)
class TurnRequest:
    """Everything the caller supplies for one turn (the ``chat`` keyword arguments)."""

    message: str
    session_id: str = "default"
    channel: str = "console"
    # Who reads the answer (ADR-0125): ``other`` when people besides the owner do (a
    # Telegram group). The pipeline publishes it for the turn (``audience_scope``).
    audience: Audience = "owner"
    preferred_model: str | None = None
    provider_profile: str | None = None
    router_model: str | None = None
    strict: bool = False
    # An approved governance halt continuing itself (ADR-0106 Tier B machinery, reused).
    # Unlike a Tier B continuation there is no new user message behind this turn — the
    # human answered an approval, not a question — so nothing is injected into the loop
    # and the pipeline runs without its `intercept` stage. See `pipeline.RESUME_STAGES`.
    resume_run_id: str | None = None
    resume_step_id: int | None = None
    # A turn the system opened with no user message (ADR-0127): the name of the opener
    # (``config/intercepts.yaml`` ``openers:``) that answers it. ``message`` is empty and
    # the pipeline runs ``OPENER_STAGES``: open → guard → record.
    opener: str | None = None

    @property
    def resume_point(self) -> tuple[str, int] | None:
        """The run and step this turn continues, when it continues one."""
        if self.resume_run_id is None or self.resume_step_id is None:
            return None
        return (self.resume_run_id, self.resume_step_id)


@dataclass
class TurnState:
    """What the stages produce, in order. A stage reads what earlier stages wrote.

    ``result`` set by any stage ends the turn (an intercept hit, or the curated
    answer). ``stage_spans`` is true on the synchronous path, where stage spans can
    be context-attached; the streaming path leaves it false (generator boundary).
    """

    request: TurnRequest
    span: Any = None
    stage_spans: bool = False
    intent_result: IntentResult | None = None
    memory_ctx: MemoryContext | None = None
    plan: TaskPlan | None = None
    tasks: list[AgentTask] = field(default_factory=list)
    results: list[AgentResult] = field(default_factory=list)
    result: ChatResult | None = None
    intercepted: bool = False
    # The deterministic handler that answered, when one did (the intercept spec's name).
    handler: str | None = None
    # That handler declared its answer repeats third-party text (``InterceptSpec.guard_output``).
    guard_output: bool = False
    # Set by ``screen`` (stage 0): the data classification the PRE_TURN screens
    # inferred for the user's message, and whether they refused the turn.
    classification: str | None = None
    screened_out: bool = False
    # ADR-0106: the question this turn answered, and how ("approve"/"reject"), when
    # the classify stage routed it back to whoever asked. None on an ordinary turn.
    continuation: Continuation | None = None
    continuation_decision: str | None = None
    # The option picked when the answered continuation was a ``choice`` — the dict the
    # owner stored for that option, handed to it on ``AgentTask.selected_choice``.
    continuation_choice: dict[str, Any] | None = None
    # When the turn began. A continuation created at or after this was opened BY this
    # turn, which is how `record` tells an owner's explicit question from one it would
    # only infer, and how a stale choice is told from a fresh one.
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def message(self) -> str:
        return self.request.message

    @property
    def session_id(self) -> str:
        return self.request.session_id


__all__ = ["OnError", "TurnRequest", "TurnState"]
