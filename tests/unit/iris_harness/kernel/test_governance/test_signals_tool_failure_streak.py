from __future__ import annotations

import pytest

from iris_harness.kernel.governance.evaluator import StepRecord
from iris_harness.kernel.governance.evaluator.signals import ToolFailureStreakSignal


def _step(
    step_id: int,
    *,
    tool_name: str | None = "research",
    tool_error: str | None = None,
) -> StepRecord:
    return StepRecord(
        run_id="r",
        step_id=step_id,
        agent_type="chat",
        tool_name=tool_name,
        tool_error=tool_error,
    )


def test_three_consecutive_errors_halt() -> None:
    signal = ToolFailureStreakSignal(threshold=3)
    state: dict = {}
    assert signal(_step(1, tool_error="boom"), state=state).verdict == "ok"
    assert signal(_step(2, tool_error="boom"), state=state).verdict == "ok"
    halted = signal(_step(3, tool_error="boom"), state=state)
    assert halted.verdict == "halt"
    assert halted.severity == "error"
    assert halted.audit_metadata["streak"] == 3


def test_success_resets_streak() -> None:
    signal = ToolFailureStreakSignal(threshold=3)
    state: dict = {}
    signal(_step(1, tool_error="boom"), state=state)
    signal(_step(2, tool_error="boom"), state=state)
    assert signal(_step(3, tool_error=None), state=state).verdict == "ok"
    # Streak reset to 0 — would need 3 more to halt.
    assert signal(_step(4, tool_error="boom"), state=state).verdict == "ok"
    assert signal(_step(5, tool_error="boom"), state=state).verdict == "ok"
    assert signal(_step(6, tool_error="boom"), state=state).verdict == "halt"


def test_step_without_tool_does_not_touch_streak() -> None:
    signal = ToolFailureStreakSignal(threshold=2)
    state: dict = {}
    signal(_step(1, tool_error="boom"), state=state)
    # No-tool step in the middle — must not reset, must not advance.
    no_tool = signal(_step(2, tool_name=None), state=state)
    assert no_tool.verdict == "ok"
    assert state["streak"] == 1
    assert signal(_step(3, tool_error="boom"), state=state).verdict == "halt"


def test_threshold_below_one_rejected() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        ToolFailureStreakSignal(threshold=0)
