from __future__ import annotations

import pytest

from iris_harness.kernel.governance.evaluator import StepRecord
from iris_harness.kernel.governance.evaluator.signals import StepCapSignal


def _step(agent_type: str, step_id: int) -> StepRecord:
    return StepRecord(run_id="r", step_id=step_id, agent_type=agent_type)


def test_chat_default_threshold_is_20() -> None:
    signal = StepCapSignal()
    assert signal.threshold_for("chat") == 20
    result = signal(_step("chat", 19), state={})
    assert result.verdict == "ok"
    halted = signal(_step("chat", 20), state={})
    assert halted.verdict == "halt"
    assert halted.severity == "warn"
    assert halted.audit_metadata["threshold"] == 20


def test_coding_default_threshold_is_50() -> None:
    signal = StepCapSignal()
    assert signal.threshold_for("coding") == 50
    assert signal(_step("coding", 49), state={}).verdict == "ok"
    assert signal(_step("coding", 50), state={}).verdict == "halt"


def test_unknown_agent_uses_default_threshold() -> None:
    signal = StepCapSignal(default_threshold=5)
    assert signal.threshold_for("voice") == 5
    assert signal(_step("voice", 4), state={}).verdict == "ok"
    assert signal(_step("voice", 5), state={}).verdict == "halt"


def test_thresholds_override() -> None:
    signal = StepCapSignal(thresholds={"chat": 3})
    assert signal.threshold_for("chat") == 3
    assert signal(_step("chat", 2), state={}).verdict == "ok"
    assert signal(_step("chat", 3), state={}).verdict == "halt"


def test_negative_step_id_rejected_at_model_level() -> None:
    with pytest.raises(ValueError):
        StepRecord(run_id="r", step_id=-1, agent_type="chat")
