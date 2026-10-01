"""Per-call usage events are bounded, and a usage mark still counts exactly."""

from __future__ import annotations

import pytest

from iris_harness.llm import client as client_module
from iris_harness.llm.client import CodingLLMClient, CodingLLMConfig, LLMTokenUsage


def _usage(n: int) -> LLMTokenUsage:
    return LLMTokenUsage(prompt_tokens=n, completion_tokens=1, total_tokens=n + 1)


def test_marks_stay_exact_after_old_events_fall_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "_USAGE_EVENTS_MAX", 3)
    client = CodingLLMClient(CodingLLMConfig(provider="ollama"))
    for n in range(5):
        client._record_usage(_usage(n))

    assert len(client._usage_events) == 3
    mark = client.get_usage_mark()
    assert mark == 5  # absolute: counts every call, kept or not
    client._record_usage(_usage(10))
    client._record_usage(_usage(20))

    assert client.get_token_usage_since(mark) == {
        "prompt_tokens": 30,
        "completion_tokens": 2,
        "total_tokens": 32,
        "call_count": 2,
    }
    assert client.get_token_usage_since(client.get_usage_mark())["call_count"] == 0


def test_a_mark_older_than_what_is_kept_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "_USAGE_EVENTS_MAX", 2)
    client = CodingLLMClient(CodingLLMConfig(provider="ollama"))
    mark = client.get_usage_mark()
    for n in range(3):
        client._record_usage(_usage(n))
    with pytest.raises(ValueError, match="older than"):
        client.get_token_usage_since(mark)
    with pytest.raises(ValueError, match="out of range"):
        client.get_token_usage_since(client.get_usage_mark() + 1)
