"""Hook framework types — modeled on the Claude Agent SDK hook taxonomy.

Six hook points fire across the agent lifecycle. Each hook returns a
``HookDecision`` that either allows, denies, transforms, or requests
approval. Plugins register against the kernel at startup; runtime
registration is forbidden (see ``GovernanceKernel.init_lock`` and §5.3
of the design doc).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class HookPoint(str, Enum):
    """Lifecycle points at which the kernel fires hooks."""

    # Once per turn, on the raw user message, before anything answers it. A turn can
    # be answered by a deterministic handler with no model call, and PRE_CLASSIFY
    # fires only where text enters a model, so without this point such a turn was
    # never screened (docs/architecture/deterministic-path-parity.md).
    PRE_TURN = "pre_turn"
    PRE_CLASSIFY = "pre_classify"
    PRE_LLM_CALL = "pre_llm_call"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    POST_STEP = "post_step"
    PRE_RESPONSE = "pre_response"
    # A request a plugin makes through the SDK's governed HTTP client (issue #103): before
    # it is sent (is the host declared, may this run's data go there) and after it ends
    # (status, bytes, duration). Never the path, query, headers or body.
    PRE_EGRESS = "pre_egress"
    POST_EGRESS = "post_egress"


HookOutcome = Literal["allow", "deny", "transform", "require_approval"]
HookSeverity = Literal["info", "warn", "error", "critical"]
DataClassification = Literal["public", "internal", "personal", "secret"]
LLMTier = Literal["tier_1", "tier_2", "tier_3"]


class HookContext(BaseModel):
    """Snapshot passed to every hook invocation.

    The kernel builds this from the caller's state. Hooks may produce a
    transformed copy via ``HookDecision.transformed_payload``; the kernel
    re-binds ``payload`` for subsequent hooks at the same hook point.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    hook_point: HookPoint
    run_id: str = Field(..., min_length=1)
    agent_type: str = Field(..., min_length=1)
    step_id: int | None = None
    persona: str | None = None
    route: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    classification: DataClassification | None = None
    tier: LLMTier | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HookDecision(BaseModel):
    """The verdict a hook returns to the kernel.

    ``transformed_payload`` is applied only when ``outcome == "transform"``.
    ``set_classification`` and ``set_tier`` are orthogonal annotations:
    the kernel applies them to the running context regardless of outcome,
    so a hook can deny *and* still record the classification it inferred.
    """

    model_config = ConfigDict(frozen=True)

    outcome: HookOutcome
    reason: str = Field(..., min_length=1)
    transformed_payload: dict[str, Any] | None = None
    set_classification: DataClassification | None = None
    set_tier: LLMTier | None = None
    approval_request_id: UUID | None = None
    severity: HookSeverity = "info"
    audit_metadata: dict[str, Any] = Field(default_factory=dict)
    # The hook that decided a denial or an approval request, stamped by the kernel on the
    # decision ``fire`` returns (a hook never sets it). Lets a caller tell the egress
    # gate's refusal from, say, a prompt guard's, without parsing the reason.
    decided_by: str | None = None


@runtime_checkable
class Hook(Protocol):
    """Protocol every plugin hook must satisfy.

    ``priority`` is dispatch order ascending: lower number = earlier.
    The kernel short-circuits on the first ``deny`` or ``require_approval``.
    """

    name: str
    hook_point: HookPoint
    priority: int

    async def __call__(self, ctx: HookContext) -> HookDecision: ...
