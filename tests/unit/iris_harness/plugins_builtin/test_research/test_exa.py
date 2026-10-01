"""Unit tests for the Exa search provider and its place in the selection chain.

Hermetic: no real network. Exa's urllib call is monkeypatched.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from iris_harness.plugins_builtin.research.providers import ExaProvider, select_providers

_KEY_VARS = ("IRIS_SEARXNG_URL", "TAVILY_API_KEY", "EXA_API_KEY", "BRAVE_API_KEY")


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a clean slate: no provider keys/URLs set."""
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- is_available


def test_exa_is_available_reflects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ExaProvider().is_available() is False
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    assert ExaProvider().is_available() is True


# --------------------------------------------------------------------------- selection


def test_select_providers_full_priority_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_SEARXNG_URL", "http://searx.local")
    monkeypatch.setenv("TAVILY_API_KEY", "tav-key")
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    monkeypatch.setenv("BRAVE_API_KEY", "brave-key")
    names = [p.name for p in select_providers()]
    assert names == ["searxng", "tavily", "exa", "brave", "ddg"]


def test_select_providers_exa_before_ddg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    names = [p.name for p in select_providers()]
    assert names == ["exa", "ddg"]


# --------------------------------------------------------------------------- search


class _FakeResponse:
    """Minimal context manager mimicking ``urllib.request.urlopen``'s return."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_exa_search_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    payload = json.dumps(
        {"results": [{"title": "T", "url": "http://x", "text": "snip", "score": 0.9}]}
    ).encode()

    def fake_urlopen(*args: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    results = ExaProvider().search("q", max_results=5)
    assert len(results) == 1
    assert results[0].title == "T"
    assert results[0].url == "http://x"
    assert results[0].snippet == "snip"
    assert results[0].source == "exa"
    # Exa's own score is not carried: the engine scores every provider's hits alike.
    assert not hasattr(results[0], "score")


def test_exa_search_returns_empty_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-key")

    def boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert ExaProvider().search("q", max_results=5) == []
