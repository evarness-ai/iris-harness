"""Evaluator plane types.

A ``Signal`` is a pure function of a ``StepRecord`` (plus any
signal-private state managed via the registry's per-run context). It
returns a ``SignalResult`` carrying a verdict, a reason, severity, and
audit metadata. The ``EvaluatorRegistry`` aggregates verdicts via a
worst-wins policy and the ``EvaluatorHook`` translates the aggregate
into a ``HookDecision`` at ``PostStep``.

Signals do **not** see prompt text, retrieval context, or working
memory. They only see the structured step record. This keeps them
amenable to running out-of-process in 12.gov-3.9 — the IPC payload is
just ``StepRecord`` JSON.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from iris_harness.kernel.governance.hooks.types import (
    DataClassification,
    HookSeverity,
    LLMTier,
)

SignalVerdict = Literal["ok", "warn", "require_approval", "halt"]


class StepRecord(BaseModel):
    """One ReAct step's worth of structured input to every signal.

    Built by the kernel at ``PostStep`` from the running ``HookContext``
    plus per-step bookkeeping. Stable across in-process and
    out-of-process evaluator modes — keep JSON-serializable.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., min_length=1)
    step_id: int = Field(..., ge=0)
    agent_type: str = Field(..., min_length=1)
    # Optional context the four cheap signals consume in 12.gov-3.2.
    thought: str | None = None
    tool_name: str | None = None
    tool_args_hash: str | None = None
    # A bounded, sorted-key JSON rendering of the tool args. The hash above is for
    # exact-repeat counting; this text is for the semantic signals, so a thought that
    # repeats over *different* items (a fan-out) embeds differently from a thought
    # that repeats over the same call (a loop).
    tool_args_text: str | None = None
    tool_error: str | None = None
    tier: LLMTier | None = None
    classification: DataClassification | None = None
    cost_usd: float | None = None
    original_task: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SignalResult(BaseModel):
    """The verdict one signal returns for one step."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1)
    verdict: SignalVerdict
    reason: str = Field(..., min_length=1)
    severity: HookSeverity = "info"
    audit_metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Signal(Protocol):
    """Protocol every evaluator signal must satisfy.

    ``priority`` orders signals within a step (lower = earlier). The
    registry runs every signal regardless of order — priority just
    governs the order of audit entries when reasons need to be read
    sequentially. The aggregate verdict is worst-wins (halt >
    require_approval > warn > ok), not first-wins.
    """

    name: str
    priority: int

    def __call__(self, step: StepRecord, *, state: dict[str, Any]) -> SignalResult: ...
