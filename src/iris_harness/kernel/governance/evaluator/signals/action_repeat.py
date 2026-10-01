"""action_repeat signal — halt after N identical (tool, args) calls.

Design §9.1: 3 identical calls → inject-correction-then-halt. v1 ships
the halt half; the correction-injection layer lands when the ReAct
loop grows a mid-stream correction surface (separate story).

State shape (per-run-id, per-signal):

    {
        "counts": {(tool_name, tool_args_hash): int, ...},
    }
"""

from __future__ import annotations

from typing import Any

from iris_harness.kernel.governance.evaluator.types import SignalResult, StepRecord


class ActionRepeatSignal:
    """Halt when the same (tool_name, args-hash) repeats too often."""

    name: str = "action_repeat"
    priority: int = 20

    def __init__(self, *, threshold: int = 3) -> None:
        if threshold < 2:
            raise ValueError("action_repeat threshold must be >= 2")
        self._threshold = threshold

    def __call__(self, step: StepRecord, *, state: dict[str, Any]) -> SignalResult:
        if not step.tool_name or not step.tool_args_hash:
            return SignalResult(
                name=self.name,
                verdict="ok",
                reason="action_repeat: step had no tool call",
            )

        key = (step.tool_name, step.tool_args_hash)
        counts: dict[tuple[str, str], int] = state.setdefault("counts", {})
        counts[key] = counts.get(key, 0) + 1
        count = counts[key]

        audit = {
            "tool_name": step.tool_name,
            "tool_args_hash": step.tool_args_hash,
            "count": count,
            "threshold": self._threshold,
        }

        if count >= self._threshold:
            return SignalResult(
                name=self.name,
                verdict="halt",
                reason=(
                    f"action_repeat: tool {step.tool_name!r} called {count} times "
                    f"with the same args (threshold={self._threshold})"
                ),
                severity="warn",
                audit_metadata=audit,
            )

        return SignalResult(
            name=self.name,
            verdict="ok",
            reason=f"action_repeat: {count}/{self._threshold}",
            audit_metadata=audit,
        )
