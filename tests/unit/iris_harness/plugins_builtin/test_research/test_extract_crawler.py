"""Tests for the extractor-selection seam (trafilatura vs crawl4ai).

Hermetic: crawl4ai is not installed, so the crawl4ai path is exercised only for its
graceful-degradation behavior; the selector and ``extract_into`` wiring are tested
with monkeypatched env / fakes.
"""

from __future__ import annotations

import iris_harness.plugins_builtin.research.extract as extract_mod
from iris_harness.plugins_builtin.research.extract import (
    _crawl4ai_extract,
    _fetch_and_extract,
    _select_extractor,
    extract_into,
)
from iris_harness.plugins_builtin.research.models import SearchResult


def test_select_extractor_default_is_trafilatura(monkeypatch) -> None:
    monkeypatch.delenv("IRIS_RESEARCH_CRAWLER", raising=False)
    assert _select_extractor() is _fetch_and_extract


def test_select_extractor_trafilatura_explicit(monkeypatch) -> None:
    monkeypatch.setenv("IRIS_RESEARCH_CRAWLER", "trafilatura")
    assert _select_extractor() is _fetch_and_extract


def test_select_extractor_crawl4ai(monkeypatch) -> None:
    monkeypatch.setenv("IRIS_RESEARCH_CRAWLER", "crawl4ai")
    assert _select_extractor() is _crawl4ai_extract


def test_crawl4ai_extract_graceful_when_not_installed() -> None:
    # crawl4ai is not a project dependency: must return None, never raise.
    assert _crawl4ai_extract("https://example.com", timeout=8.0) is None


async def test_extract_into_uses_selected_extractor(monkeypatch) -> None:
    def fake_extractor(url: str, timeout: float) -> str:
        return f"CRAWLED {url}"

    monkeypatch.setattr(extract_mod, "_select_extractor", lambda: fake_extractor)

    results = [SearchResult(title="One", url="https://example.com/1")]
    await extract_into(results, max_pages=1)

    assert results[0].content == "CRAWLED https://example.com/1"
