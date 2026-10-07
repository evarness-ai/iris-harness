"""Unit tests for the research search providers and selection ordering.

Hermetic: no real network. SearXNG's urllib call and the ddgs client are monkeypatched.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from iris_harness.plugins_builtin.research.providers import (
    BraveProvider,
    DuckDuckGoProvider,
    SearxngProvider,
    TavilyProvider,
    select_providers,
)

from .conftest import FakeClient

# Every env var that gates a provider in select_providers(). Must list ALL of them so
# the autouse fixture below makes these tests hermetic regardless of ambient env or any
# key a real local environment happens to have set (e.g. EXA_API_KEY).
_KEY_VARS = ("IRIS_SEARXNG_URL", "TAVILY_API_KEY", "EXA_API_KEY", "BRAVE_API_KEY")


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a clean slate: no provider keys/URLs set."""
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- selection


def test_select_providers_only_ddg_without_keys() -> None:
    providers = select_providers()
    assert [p.name for p in providers] == ["ddg"]


def test_select_providers_searxng_first_ddg_last(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_SEARXNG_URL", "http://searx.local")
    names = [p.name for p in select_providers()]
    assert names[0] == "searxng"
    assert names[-1] == "ddg"
    assert names == ["searxng", "ddg"]


def test_select_providers_full_priority_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_SEARXNG_URL", "http://searx.local")
    monkeypatch.setenv("TAVILY_API_KEY", "tav-key")
    monkeypatch.setenv("BRAVE_API_KEY", "brave-key")
    names = [p.name for p in select_providers()]
    assert names == ["searxng", "tavily", "brave", "ddg"]


def test_select_providers_keyed_before_ddg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tav-key")
    names = [p.name for p in select_providers()]
    assert names == ["tavily", "ddg"]


# --------------------------------------------------------------------------- is_available


def test_searxng_is_available_reflects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert SearxngProvider().is_available() is False
    monkeypatch.setenv("IRIS_SEARXNG_URL", "http://searx.local")
    assert SearxngProvider().is_available() is True


def test_searxng_strips_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_SEARXNG_URL", "  http://searx.local/  ")
    assert SearxngProvider().base_url == "http://searx.local"


def test_tavily_is_available_reflects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert TavilyProvider().is_available() is False
    monkeypatch.setenv("TAVILY_API_KEY", "tav-key")
    assert TavilyProvider().is_available() is True


def test_brave_is_available_reflects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert BraveProvider().is_available() is False
    monkeypatch.setenv("BRAVE_API_KEY", "brave-key")
    assert BraveProvider().is_available() is True


def test_ddg_always_available() -> None:
    assert DuckDuckGoProvider().is_available() is True


# --------------------------------------------------------------------------- searxng search


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


def test_searxng_search_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_SEARXNG_URL", "http://searx.local")
    payload = json.dumps(
        {"results": [{"title": "T", "url": "http://x", "content": "snip"}]}
    ).encode()

    def fake_urlopen(*args: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    results = SearxngProvider().search("q", max_results=5)
    assert len(results) == 1
    assert results[0].title == "T"
    assert results[0].url == "http://x"
    assert results[0].snippet == "snip"
    assert results[0].source == "searxng"


def test_searxng_search_returns_empty_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_SEARXNG_URL", "http://searx.local")

    def boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert SearxngProvider().search("q", max_results=5) == []


def test_searxng_search_caps_to_max_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_SEARXNG_URL", "http://searx.local")
    hits = [{"title": f"T{i}", "url": f"http://x/{i}", "content": "c"} for i in range(10)]
    payload = json.dumps({"results": hits}).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResponse(payload))
    results = SearxngProvider().search("q", max_results=3)
    assert len(results) == 3


# --------------------------------------------------------------------------- ddg search


class _FakeDDGS:
    def __init__(self, rows: list[dict[str, Any]] | None = None, raises: bool = False) -> None:
        self._rows = rows or []
        self._raises = raises

    def text(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        if self._raises:
            raise RuntimeError("rate limited")
        return self._rows

    def news(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        if self._raises:
            raise RuntimeError("rate limited")
        return self._rows


def test_ddg_search_parses_text_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"title": "T", "href": "http://x", "body": "b"}]
    monkeypatch.setattr(
        "iris_harness.plugins_builtin.research.providers.duckduckgo.DDGS",
        lambda *a, **k: _FakeDDGS(rows),
    )
    results = DuckDuckGoProvider().search("q", max_results=5)
    assert len(results) == 1
    assert results[0].title == "T"
    assert results[0].url == "http://x"
    assert results[0].snippet == "b"
    assert results[0].source == "ddg"


def test_ddg_search_returns_empty_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "iris_harness.plugins_builtin.research.providers.duckduckgo.DDGS",
        lambda *a, **k: _FakeDDGS(raises=True),
    )
    assert DuckDuckGoProvider().search("q", max_results=5) == []


def test_ddg_news_uses_news_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "title": "N",
            "url": "http://news",
            "body": "story",
            "date": "2026-06-25T00:00:00",
        }
    ]
    monkeypatch.setattr(
        "iris_harness.plugins_builtin.research.providers.duckduckgo.DDGS",
        lambda *a, **k: _FakeDDGS(rows),
    )
    results = DuckDuckGoProvider().search("q", max_results=5, search_type="news")
    assert len(results) == 1
    assert results[0].url == "http://news"
    assert results[0].published is not None


# --------------------------------------------------------------------------- language hint


class _RecordingDDGS(_FakeDDGS):
    def __init__(self, seen: list[dict[str, Any]]) -> None:
        super().__init__([{"title": "T", "url": "http://x", "body": "b"}])
        self._seen = seen

    def news(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self._seen.append(kwargs)
        return super().news(*args, **kwargs)


@pytest.mark.parametrize(
    ("language", "region"), [("en", "us-en"), ("ja", "wt-wt"), (None, "<library default>")]
)
def test_ddg_turns_the_language_into_a_region(
    monkeypatch: pytest.MonkeyPatch, language: str | None, region: str
) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "iris_harness.plugins_builtin.research.providers.duckduckgo.DDGS",
        lambda *a, **k: _RecordingDDGS(seen),
    )
    DuckDuckGoProvider().search("q", max_results=5, search_type="news", language=language)
    assert seen[0].get("region", "<library default>") == region


def test_searxng_and_brave_pass_the_language(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_SEARXNG_URL", "http://searx.local")
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    urls: list[str] = []

    def fake_urlopen(request: Any, *args: Any, **kwargs: Any) -> _FakeResponse:
        urls.append(request.full_url)
        return _FakeResponse(b'{"results": []}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    brave = BraveProvider()
    client = FakeClient(body={"results": []})
    brave._http = client  # type: ignore[assignment]
    SearxngProvider().search("q", max_results=5, language="en")
    brave.search("q", max_results=5, search_type="news", language="en")
    SearxngProvider().search("q", max_results=5)
    assert "language=en" in urls[0]
    assert "language=" not in urls[1]
    [(method, url, kwargs)] = client.calls
    assert (method, url) == ("GET", "https://api.search.brave.com/res/v1/news/search")
    assert kwargs["params"]["search_lang"] == "en"
