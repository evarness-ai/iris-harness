"""Unit tests for the Exa search provider and its place in the selection chain.

Hermetic: no real network. The governed client Exa asks for is a stand-in.
"""

from __future__ import annotations

import pytest

from iris_harness.plugins_builtin.research.providers import ExaProvider, select_providers
from iris_harness.sdk.http import EgressDenied

from .conftest import FakeClient

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


def test_exa_search_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    client = FakeClient(
        body={"results": [{"title": "T", "url": "http://x", "text": "snip", "score": 0.9}]}
    )
    provider = ExaProvider()
    provider._http = client  # type: ignore[assignment]

    results = provider.search("q", max_results=5)
    assert len(results) == 1
    assert results[0].title == "T"
    assert results[0].url == "http://x"
    assert results[0].snippet == "snip"
    assert results[0].source == "exa"
    # Exa's own score is not carried: the engine scores every provider's hits alike.
    assert not hasattr(results[0], "score")
    [(method, url, kwargs)] = client.calls
    assert (method, url) == ("POST", "https://api.exa.ai/search")
    assert kwargs["headers"] == {"x-api-key": "exa-key"} and kwargs["json"]["query"] == "q"


def test_exa_search_returns_empty_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    provider = ExaProvider()
    provider._http = FakeClient(error=OSError("network down"))  # type: ignore[assignment]
    assert provider.search("q", max_results=5) == []


def test_exa_search_returns_empty_on_an_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    provider = ExaProvider()
    provider._http = FakeClient(status=429)  # type: ignore[assignment]
    assert provider.search("q", max_results=5) == []


def test_a_request_the_governed_client_refuses_returns_no_hits(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    provider = ExaProvider()
    provider._http = FakeClient(error=EgressDenied("host not declared", host="api.exa.ai"))  # type: ignore[assignment]
    assert provider.search("q", max_results=5) == []
    assert "exa search not sent" in caplog.text
