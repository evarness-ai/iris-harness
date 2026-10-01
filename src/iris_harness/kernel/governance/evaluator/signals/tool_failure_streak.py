"""tool_failure_streak signal — halt after N consecutive tool errors.

Design §9.1: 3 consecutive tool failures → halt. A successful tool
call (``tool_name`` present, ``tool_error`` absent) resets the streak.
Steps with no tool call neither increment nor reset — they're outside
the streak's domain.

State shape (per-run-id, per-signal):

    {
        "streak": int,
    }
"""

from __future__ import annotations

from typing import Any

from iris_harness.kernel.governance.evaluator.types import SignalResult, StepRecord


class ToolFailureStreakSignal:
    """Halt when consecutive tool errors reach the threshold."""

    name: str = "tool_failure_streak"
    priority: int = 30

    def __init__(self, *, threshold: int = 3) -> None:
        if threshold < 1:
            raise ValueError("tool_failure_streak threshold must be >= 1")
        self._threshold = threshold

    def __call__(self, step: StepRecord, *, state: dict[str, Any]) -> SignalResult:
        # No tool was invoked this step — leave the streak untouched.
        if not step.tool_name:
            current = state.get("streak", 0)
            return SignalResult(
                name=self.name,
                verdict="ok",
                reason=f"tool_failure_streak: no tool call (streak={current})",
                audit_metadata={"streak": current, "threshold": self._threshold},
            )

        if step.tool_error:
            state["streak"] = state.get("streak", 0) + 1
        else:
            state["streak"] = 0

        streak = state["streak"]
        audit = {"streak": streak, "threshold": self._threshold, "tool_name": step.tool_name}

        if streak >= self._threshold:
            return SignalResult(
                name=self.name,
                verdict="halt",
                reason=(
                    f"tool_failure_streak: {streak} consecutive tool errors "
                    f"(threshold={self._threshold})"
                ),
                severity="error",
                audit_metadata=audit,
            )

        return SignalResult(
            name=self.name,
            verdict="ok",
            reason=f"tool_failure_streak: {streak}/{self._threshold}",
            audit_metadata=audit,
        )
