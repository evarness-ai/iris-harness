from __future__ import annotations

from typing import Any

import pytest

from iris_harness.kernel.governance.evaluator import (
    EvaluatorRegistry,
    SignalResult,
    StepRecord,
)
from iris_harness.kernel.governance.evaluator.registry import EvaluatorRegistryLockedError


class _FakeSignal:
    def __init__(self, name: str, verdict: str, priority: int = 50) -> None:
        self.name = name
        self.priority = priority
        self._verdict = verdict
        self.call_count = 0

    def __call__(self, step: StepRecord, *, state: dict[str, Any]) -> SignalResult:
        self.call_count += 1
        return SignalResult(name=self.name, verdict=self._verdict, reason="stub")


class _StatefulCounter:
    """A signal that uses ``state`` to count its own per-run invocations."""

    name = "stateful_counter"
    priority = 10

    def __call__(self, step: StepRecord, *, state: dict[str, Any]) -> SignalResult:
        state["count"] = state.get("count", 0) + 1
        return SignalResult(name=self.name, verdict="ok", reason=f"count={state['count']}")


def _step(run_id: str = "run-1", step_id: int = 1) -> StepRecord:
    return StepRecord(run_id=run_id, step_id=step_id, agent_type="chat")


def test_register_after_lock_raises() -> None:
    registry = EvaluatorRegistry()
    registry.init_lock()
    with pytest.raises(EvaluatorRegistryLockedError):
        registry.register(_FakeSignal("late", "ok"))


def test_empty_registry_evaluate_returns_no_results() -> None:
    registry = EvaluatorRegistry()
    registry.init_lock()
    assert registry.evaluate(_step()) == ()
    assert registry.worst(()) is None


def test_worst_wins_precedence_across_signals() -> None:
    registry = EvaluatorRegistry()
    registry.register(_FakeSignal("a", "ok"))
    registry.register(_FakeSignal("b", "warn"))
    registry.register(_FakeSignal("c", "halt"))
    registry.register(_FakeSignal("d", "require_approval"))
    registry.init_lock()

    results = registry.evaluate(_step())
    worst = registry.worst(results)
    assert worst is not None
    assert worst.name == "c"
    assert worst.verdict == "halt"


def test_state_is_scoped_per_run() -> None:
    registry = EvaluatorRegistry()
    counter = _StatefulCounter()
    registry.register(counter)
    registry.init_lock()

    # Two steps within run-1 → counter accumulates.
    r1 = registry.evaluate(_step("run-1", 1))
    r2 = registry.evaluate(_step("run-1", 2))
    assert r1[0].reason == "count=1"
    assert r2[0].reason == "count=2"

    # A different run has its own counter.
    r3 = registry.evaluate(_step("run-2", 1))
    assert r3[0].reason == "count=1"

    # reset_run_state drops the per-run counter for run-1 only.
    registry.reset_run_state("run-1")
    r4 = registry.evaluate(_step("run-1", 1))
    assert r4[0].reason == "count=1"
    r5 = registry.evaluate(_step("run-2", 2))
    assert r5[0].reason == "count=2"


def test_signal_exception_fails_closed() -> None:
    class _Boom:
        name = "boom"
        priority = 10

        def __call__(self, step: StepRecord, *, state: dict[str, Any]) -> SignalResult:
            raise RuntimeError("kaboom")

    registry = EvaluatorRegistry()
    registry.register(_Boom())
    registry.register(_FakeSignal("calm", "ok"))
    registry.init_lock()

    results = registry.evaluate(_step())
    by_name = {r.name: r for r in results}
    assert by_name["boom"].verdict == "halt"
    assert by_name["boom"].severity == "error"
    assert by_name["calm"].verdict == "ok"


def test_signals_ordered_by_priority_in_results() -> None:
    registry = EvaluatorRegistry()
    registry.register(_FakeSignal("late", "ok", priority=99))
    registry.register(_FakeSignal("early", "ok", priority=1))
    registry.init_lock()

    results = registry.evaluate(_step())
    assert [r.name for r in results] == ["early", "late"]
