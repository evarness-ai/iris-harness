"""Tests for the research content extractor.

Hermetic: no real network. ``_fetch_and_extract`` (or, for the truncation test, the
underlying urllib + trafilatura calls) is patched.
"""

from __future__ import annotations

import iris_harness.plugins_builtin.research.extract as extract_mod
from iris_harness.plugins_builtin.research.extract import _fetch_and_extract, extract_into
from iris_harness.plugins_builtin.research.models import SearchResult


def _make_results() -> list[SearchResult]:
    return [
        SearchResult(title="One", url="https://example.com/1"),
        SearchResult(title="Two", url="https://example.com/2"),
        SearchResult(title="Three", url="https://example.com/3"),
    ]


async def test_extract_into_respects_max_pages(monkeypatch) -> None:
    def fake_fetch(url: str, timeout: float) -> str:
        return f"# content for {url}"

    monkeypatch.setattr(extract_mod, "_fetch_and_extract", fake_fetch)

    results = _make_results()
    await extract_into(results, max_pages=2)

    assert results[0].content == "# content for https://example.com/1"
    assert results[1].content == "# content for https://example.com/2"
    # Third result is past the max_pages cap and must stay untouched.
    assert results[2].content is None


async def test_extract_into_isolates_failures(monkeypatch) -> None:
    def boom(url: str, timeout: float) -> str:
        raise RuntimeError("extraction exploded")

    monkeypatch.setattr(extract_mod, "_fetch_and_extract", boom)

    results = _make_results()
    # Must complete without propagating the exception.
    await extract_into(results, max_pages=3)

    assert all(r.content is None for r in results)


async def test_extract_into_empty_list() -> None:
    # Should be a no-op and never raise.
    await extract_into([], max_pages=5)


def test_fetch_and_extract_truncates(monkeypatch) -> None:
    long_markdown = "x" * 10_000
    monkeypatch.setattr(extract_mod, "_fetch", lambda url, timeout: b"<html>ignored</html>")
    monkeypatch.setattr(extract_mod.trafilatura, "extract", lambda html, **kwargs: long_markdown)

    out = _fetch_and_extract("https://example.com/long", timeout=8.0)

    assert out is not None
    assert out.endswith("… [truncated]")
    assert len(out) <= extract_mod._MAX_CONTENT_CHARS


def test_fetch_and_extract_returns_none_on_empty(monkeypatch) -> None:
    monkeypatch.setattr(extract_mod, "_fetch", lambda url, timeout: b"<html></html>")
    monkeypatch.setattr(extract_mod.trafilatura, "extract", lambda html, **kwargs: None)

    assert _fetch_and_extract("https://example.com/empty", timeout=8.0) is None


def test_fetch_and_extract_returns_none_on_an_error_status(monkeypatch) -> None:
    monkeypatch.setattr(extract_mod, "_fetch", lambda url, timeout: None)

    assert _fetch_and_extract("https://example.com/missing", timeout=8.0) is None


def test_fetch_and_extract_returns_none_on_exception(monkeypatch) -> None:
    def boom(url, timeout):  # type: ignore[no-untyped-def]
        raise OSError("network down")

    monkeypatch.setattr(extract_mod, "_fetch", boom)

    assert _fetch_and_extract("https://example.com/err", timeout=8.0) is None
