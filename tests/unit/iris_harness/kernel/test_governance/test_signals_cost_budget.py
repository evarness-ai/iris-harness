"""Tests for the cost_budget evaluator signal (story 12.gov-3.7 / AC-3, AC-6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.kernel.governance.cost import CostStore
from iris_harness.kernel.governance.evaluator import StepRecord
from iris_harness.kernel.governance.evaluator.signals import CostBudgetSignal


@pytest.fixture()
def store(tmp_path: Path) -> CostStore:
    return CostStore(db_path=tmp_path / "cost.db")


def _step() -> StepRecord:
    return StepRecord(run_id="r", step_id=0, agent_type="chat")


def _populate(store: CostStore, *, amount: float, user_id: str = "local") -> None:
    store.record(
        run_id="x",
        agent_type="chat",
        tier="tier_3",
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=amount,
        user_id=user_id,
    )


def test_far_under_warn_is_ok(store: CostStore) -> None:
    _populate(store, amount=0.10)
    signal = CostBudgetSignal(store=store, daily_cap_usd=5.00)
    result = signal(_step(), state={})
    assert result.verdict == "ok"
    assert result.audit_metadata["ratio"] == pytest.approx(0.02)


def test_at_eighty_percent_warns_no_halt(store: CostStore) -> None:
    """AC-3: 80% returns warn, not halt."""
    _populate(store, amount=4.00)  # 80% of $5
    signal = CostBudgetSignal(store=store, daily_cap_usd=5.00, warn_ratio=0.80)
    result = signal(_step(), state={})
    assert result.verdict == "warn"
    assert result.severity == "warn"
    assert "approaching" in result.reason


def test_just_under_warn_threshold_is_ok(store: CostStore) -> None:
    _populate(store, amount=3.99)  # 79.8% of $5
    signal = CostBudgetSignal(store=store, daily_cap_usd=5.00, warn_ratio=0.80)
    result = signal(_step(), state={})
    assert result.verdict == "ok"


def test_at_one_hundred_percent_halts_critical(store: CostStore) -> None:
    _populate(store, amount=5.00)
    signal = CostBudgetSignal(store=store, daily_cap_usd=5.00)
    result = signal(_step(), state={})
    assert result.verdict == "halt"
    assert result.severity == "critical"


def test_above_one_hundred_percent_still_halts(store: CostStore) -> None:
    _populate(store, amount=7.50)
    signal = CostBudgetSignal(store=store, daily_cap_usd=5.00)
    result = signal(_step(), state={})
    assert result.verdict == "halt"
    assert result.audit_metadata["ratio"] == pytest.approx(1.5)


def test_zero_cap_is_disabled(store: CostStore) -> None:
    """A zero cap means 'no limit configured' — signal is a no-op."""
    _populate(store, amount=99.0)
    signal = CostBudgetSignal(store=store, daily_cap_usd=0.0)
    result = signal(_step(), state={})
    assert result.verdict == "ok"
    assert "disabled" in result.reason


def test_other_user_spend_does_not_trip(store: CostStore) -> None:
    _populate(store, amount=99.0, user_id="someone_else")
    signal = CostBudgetSignal(store=store, daily_cap_usd=5.00, user_id="local")
    result = signal(_step(), state={})
    assert result.verdict == "ok"


def test_invalid_cap_rejected(store: CostStore) -> None:
    with pytest.raises(ValueError):
        CostBudgetSignal(store=store, daily_cap_usd=-1.0)


def test_invalid_warn_ratio_rejected(store: CostStore) -> None:
    with pytest.raises(ValueError):
        CostBudgetSignal(store=store, daily_cap_usd=5.0, warn_ratio=1.5)
    with pytest.raises(ValueError):
        CostBudgetSignal(store=store, daily_cap_usd=5.0, warn_ratio=0.0)


def test_signal_protocol_compliance(store: CostStore) -> None:
    from iris_harness.kernel.governance.evaluator.types import Signal

    sig = CostBudgetSignal(store=store, daily_cap_usd=5.0)
    assert isinstance(sig, Signal)


def test_signal_via_registry_end_to_end(store: CostStore) -> None:
    from iris_harness.kernel.governance.evaluator import EvaluatorRegistry

    _populate(store, amount=5.00)
    reg = EvaluatorRegistry()
    reg.register(CostBudgetSignal(store=store, daily_cap_usd=5.00))
    reg.init_lock()

    results = reg.evaluate(_step())
    worst = reg.worst(results)
    assert worst is not None
    assert worst.verdict == "halt"
    assert worst.name == "cost_budget"
