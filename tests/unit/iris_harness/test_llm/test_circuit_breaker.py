"""State-machine tests for the Ollama circuit breaker (deterministic, injected clock)."""

from __future__ import annotations

import pytest

from iris_harness.llm.arbiter import CircuitBreakerOpenError, OllamaCircuitBreaker

KEY = "http://localhost:11434/v1"


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_opens_after_consecutive_failures_then_fast_fails() -> None:
    clk = _Clock()
    b = OllamaCircuitBreaker(failure_threshold=2, cooldown_seconds=30, clock=clk)
    b.before_call(KEY)  # closed
    b.record_failure(KEY)  # 1/2
    b.before_call(KEY)  # still closed
    b.record_failure(KEY)  # 2/2 -> OPEN
    assert b.is_open(KEY)
    with pytest.raises(CircuitBreakerOpenError):
        b.before_call(KEY)


def test_half_open_probe_success_closes() -> None:
    clk = _Clock()
    b = OllamaCircuitBreaker(failure_threshold=1, cooldown_seconds=30, clock=clk)
    b.record_failure(KEY)  # OPEN (threshold 1)
    with pytest.raises(CircuitBreakerOpenError):
        b.before_call(KEY)
    clk.t = 31  # cooldown elapsed
    b.before_call(KEY)  # half-open probe allowed
    b.record_success(KEY)  # probe ok -> CLOSED
    assert not b.is_open(KEY)
    b.before_call(KEY)  # closed, no raise


def test_half_open_probe_failure_reopens() -> None:
    clk = _Clock()
    b = OllamaCircuitBreaker(failure_threshold=1, cooldown_seconds=30, clock=clk)
    b.record_failure(KEY)
    clk.t = 31
    b.before_call(KEY)  # probe allowed
    b.record_failure(KEY)  # probe failed -> re-OPEN at t=31
    clk.t = 40  # within the new cooldown window
    with pytest.raises(CircuitBreakerOpenError):
        b.before_call(KEY)


def test_success_resets_consecutive_failure_count() -> None:
    b = OllamaCircuitBreaker(failure_threshold=2)
    b.record_failure(KEY)  # 1
    b.record_success(KEY)  # reset
    b.record_failure(KEY)  # 1 again, NOT 2
    b.before_call(KEY)  # still closed
    assert not b.is_open(KEY)


def test_endpoints_are_independent() -> None:
    b = OllamaCircuitBreaker(failure_threshold=1)
    b.record_failure("a")
    with pytest.raises(CircuitBreakerOpenError):
        b.before_call("a")
    b.before_call("b")  # different endpoint unaffected
