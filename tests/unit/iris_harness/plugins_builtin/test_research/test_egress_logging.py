"""Egress-audit logging tests for the research engine's outbound network calls.

Hermetic: no real network. The urllib / trafilatura calls are monkeypatched, and the
``iris.egress`` logger is captured to assert exactly one egress line is emitted right
before each outbound request. Logging must never change provider behavior.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

import iris_harness.plugins_builtin.research.extract as extract_mod
from iris_harness.plugins_builtin.research.extract import _fetch_and_extract
from iris_harness.plugins_builtin.research.providers import SearxngProvider

# Mirror test_providers.py: clear every provider-gating env var so these tests are
# hermetic regardless of ambient env.
_KEY_VARS = ("IRIS_SEARXNG_URL", "TAVILY_API_KEY", "EXA_API_KEY", "BRAVE_API_KEY")


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)


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


def test_searxng_emits_egress_line(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("IRIS_SEARXNG_URL", "http://searx.local")
    payload = json.dumps(
        {"results": [{"title": "T", "url": "http://x", "content": "snip"}]}
    ).encode()

    def fake_urlopen(*args: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with caplog.at_level(logging.INFO, logger="iris.egress"):
        results = SearxngProvider().search("q", max_results=5)

    # Behavior is unchanged.
    assert len(results) == 1

    egress_lines = [r.getMessage() for r in caplog.records if r.name == "iris.egress"]
    assert any(
        "EGRESS search" in line and "searxng" in line and "searx.local" in line
        for line in egress_lines
    ), egress_lines
    # Never leak the query value.
    assert all("q=q" not in line for line in egress_lines)


def test_fetch_and_extract_emits_crawl_egress(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _CrawlResponse:
        def read(self) -> bytes:
            return b"<html><body>ignored</body></html>"

        def __enter__(self) -> _CrawlResponse:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def fake_urlopen(request: Any, timeout: Any) -> _CrawlResponse:
        return _CrawlResponse()

    monkeypatch.setattr(extract_mod, "_open_url", fake_urlopen)
    monkeypatch.setattr(extract_mod.trafilatura, "extract", lambda html, **kwargs: "# md")

    with caplog.at_level(logging.INFO, logger="iris.egress"):
        out = _fetch_and_extract("https://example.com/page", timeout=8.0)

    assert out == "# md"

    egress_lines = [r.getMessage() for r in caplog.records if r.name == "iris.egress"]
    assert any(
        "EGRESS crawl" in line and "trafilatura" in line and "example.com" in line
        for line in egress_lines
    ), egress_lines
