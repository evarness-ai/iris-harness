"""classification_violation signal — cloud tier saw personal/secret data.

Design §9.1: "Cloud LLM called despite personal/secret classification
→ halt + critical audit". The EgressGate at PreLLMCall already denies
the in-flight call; this signal is the *post-hoc* witness so the
evaluator trace records the violation as a first-class event even if
EgressGate succeeded at blocking it.

For now we treat any tier_3 step that carries a personal/secret
classification as a violation regardless of whether the call was
actually allowed — if the kernel let the step proceed despite the
classification (a wiring bug), this signal catches it. If EgressGate
denied, the signal still fires on the recorded ``HookContext``,
giving the operator a stronger audit trail.
"""

from __future__ import annotations

from typing import Any

from iris_harness.kernel.governance.evaluator.types import SignalResult, StepRecord

_FORBIDDEN_ON_CLOUD: frozenset[str] = frozenset({"personal", "secret"})


class ClassificationViolationSignal:
    """Halt when cloud (tier_3) sees personal/secret classification."""

    name: str = "classification_violation"
    priority: int = 5  # earliest — most important

    def __call__(self, step: StepRecord, *, state: dict[str, Any]) -> SignalResult:
        tier = step.tier
        classification = step.classification

        if tier != "tier_3" or classification not in _FORBIDDEN_ON_CLOUD:
            return SignalResult(
                name=self.name,
                verdict="ok",
                reason=(f"classification_violation: tier={tier} class={classification} ok"),
                audit_metadata={"tier": tier, "classification": classification},
            )

        return SignalResult(
            name=self.name,
            verdict="halt",
            reason=(
                f"classification_violation: classification={classification!r} "
                f"reached tier_3 (cloud) — should never happen if EgressGate is wired"
            ),
            severity="critical",
            audit_metadata={
                "tier": tier,
                "classification": classification,
                "policy": "personal_or_secret_to_cloud",
            },
        )
