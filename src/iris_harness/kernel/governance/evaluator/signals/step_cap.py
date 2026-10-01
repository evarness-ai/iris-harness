"""step_cap signal — halt after configurable iteration count.

Design §9.1: default 20 (chat) / 50 (coding). Per-agent-type override
via ctor; ``StepRecord.agent_type`` drives the lookup at evaluation
time so a single signal instance covers both surfaces.
"""

from __future__ import annotations

from typing import Any

from iris_harness.kernel.governance.evaluator.types import SignalResult, StepRecord

_DEFAULT_THRESHOLDS = {"chat": 20, "coding": 50}


class StepCapSignal:
    """Halt the run when ``step_id`` exceeds the agent-type threshold."""

    name: str = "step_cap"
    priority: int = 10

    def __init__(
        self,
        *,
        thresholds: dict[str, int] | None = None,
        default_threshold: int = 20,
    ) -> None:
        self._thresholds = dict(thresholds) if thresholds else dict(_DEFAULT_THRESHOLDS)
        self._default = default_threshold

    def threshold_for(self, agent_type: str) -> int:
        return self._thresholds.get(agent_type, self._default)

    def __call__(self, step: StepRecord, *, state: dict[str, Any]) -> SignalResult:
        threshold = self.threshold_for(step.agent_type)
        if step.step_id < threshold:
            return SignalResult(
                name=self.name,
                verdict="ok",
                reason=f"step {step.step_id}/{threshold}",
                audit_metadata={"step_id": step.step_id, "threshold": threshold},
            )

        return SignalResult(
            name=self.name,
            verdict="halt",
            reason=(
                f"step_cap: step {step.step_id} reached threshold "
                f"{threshold} for agent_type={step.agent_type}"
            ),
            severity="warn",
            audit_metadata={
                "step_id": step.step_id,
                "threshold": threshold,
                "agent_type": step.agent_type,
            },
        )
