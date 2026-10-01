"""CodingLLMClient ↔ Ollama circuit-breaker integration."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from iris_harness.llm.arbiter import CircuitBreakerOpenError, OllamaCircuitBreaker
from iris_harness.llm.client import CodingLLMClient, CodingLLMConfig

OLLAMA_URL = "http://localhost:11434/v1"


def _client(
    breaker: OllamaCircuitBreaker, factory: Any, *, provider: str = "ollama"
) -> CodingLLMClient:
    cfg = CodingLLMConfig(provider=provider, model="m", base_url=OLLAMA_URL, timeout_seconds=1)
    return CodingLLMClient(
        cfg,
        model_factory=factory,
        circuit_breaker=breaker,
        governance_handled_upstream=True,
    )


def _down_factory(**_: object) -> Any:
    class _M:
        def invoke(self, messages: object, **_k: object) -> object:
            raise httpx.ConnectError("connection refused")

    return _M()


def _ok_factory(**_: object) -> Any:
    class _M:
        def invoke(self, messages: object, **_k: object) -> object:
            return type("R", (), {"content": "hi"})()

    return _M()


def _invoke(client: CodingLLMClient) -> str:
    return client.invoke(system_prompt="s", user_prompt="u")


def test_breaker_opens_then_fast_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_OLLAMA_BREAKER", raising=False)
    b = OllamaCircuitBreaker(failure_threshold=2)
    c = _client(b, _down_factory)
    # Two real connection failures trip the breaker...
    for _ in range(2):
        with pytest.raises(httpx.ConnectError):
            _invoke(c)
    # ...then the next call fast-fails WITHOUT touching the network.
    with pytest.raises(CircuitBreakerOpenError):
        _invoke(c)
    assert b.is_open(OLLAMA_URL)


def test_success_does_not_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_OLLAMA_BREAKER", raising=False)
    b = OllamaCircuitBreaker(failure_threshold=1)
    c = _client(b, _ok_factory)
    assert _invoke(c) == "hi"
    assert not b.is_open(OLLAMA_URL)


def test_cloud_provider_bypasses_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_OLLAMA_BREAKER", raising=False)
    b = OllamaCircuitBreaker(failure_threshold=1)
    c = _client(b, _down_factory, provider="openrouter")
    # Cloud failures NEVER trip the breaker — every call surfaces the raw error.
    with pytest.raises(httpx.ConnectError):
        _invoke(c)
    with pytest.raises(httpx.ConnectError):  # not CircuitBreakerOpenError
        _invoke(c)
    assert not b.is_open(OLLAMA_URL)


def test_env_kill_switch_disables_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_OLLAMA_BREAKER", "0")
    b = OllamaCircuitBreaker(failure_threshold=1)
    c = _client(b, _down_factory)
    # With the breaker disabled, the raw error always surfaces (never fast-fail).
    with pytest.raises(httpx.ConnectError):
        _invoke(c)
    with pytest.raises(httpx.ConnectError):
        _invoke(c)
    assert not b.is_open(OLLAMA_URL)


def test_non_connection_error_does_not_trip_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_OLLAMA_BREAKER", raising=False)

    def _value_error_factory(**_: object) -> Any:
        class _M:
            def invoke(self, messages: object, **_k: object) -> object:
                raise ValueError("bad model output")

        return _M()

    b = OllamaCircuitBreaker(failure_threshold=1)
    c = _client(b, _value_error_factory)
    with pytest.raises(ValueError):
        _invoke(c)
    assert not b.is_open(OLLAMA_URL)  # only connection faults count
