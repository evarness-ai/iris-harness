from __future__ import annotations

import pytest

from iris_harness.kernel.governance.evaluator import StepRecord
from iris_harness.kernel.governance.evaluator.signals import ActionRepeatSignal


def _step(
    step_id: int,
    *,
    tool_name: str | None = "research",
    tool_args_hash: str | None = "abc",
) -> StepRecord:
    return StepRecord(
        run_id="r",
        step_id=step_id,
        agent_type="chat",
        tool_name=tool_name,
        tool_args_hash=tool_args_hash,
    )


def test_three_identical_calls_halt() -> None:
    signal = ActionRepeatSignal(threshold=3)
    state: dict = {}
    assert signal(_step(1), state=state).verdict == "ok"
    assert signal(_step(2), state=state).verdict == "ok"
    halted = signal(_step(3), state=state)
    assert halted.verdict == "halt"
    assert halted.severity == "warn"
    assert halted.audit_metadata == {
        "tool_name": "research",
        "tool_args_hash": "abc",
        "count": 3,
        "threshold": 3,
    }


def test_distinct_args_do_not_count_together() -> None:
    signal = ActionRepeatSignal(threshold=3)
    state: dict = {}
    assert signal(_step(1, tool_args_hash="a"), state=state).verdict == "ok"
    assert signal(_step(2, tool_args_hash="b"), state=state).verdict == "ok"
    # Same tool, different args — should not halt yet.
    assert signal(_step(3, tool_args_hash="c"), state=state).verdict == "ok"


def test_distinct_tools_do_not_share_state() -> None:
    signal = ActionRepeatSignal(threshold=2)
    state: dict = {}
    assert signal(_step(1, tool_name="t1"), state=state).verdict == "ok"
    assert signal(_step(2, tool_name="t2"), state=state).verdict == "ok"
    # Each tool only saw one call at this point.
    assert signal(_step(3, tool_name="t1"), state=state).verdict == "halt"


def test_step_without_tool_is_noop() -> None:
    signal = ActionRepeatSignal(threshold=2)
    state: dict = {}
    result = signal(_step(1, tool_name=None, tool_args_hash=None), state=state)
    assert result.verdict == "ok"
    assert "no tool call" in result.reason
    assert "counts" not in state


def test_threshold_below_two_rejected() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        ActionRepeatSignal(threshold=1)
