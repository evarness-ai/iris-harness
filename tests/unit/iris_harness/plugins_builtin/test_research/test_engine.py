"""Tests for the research engine orchestration (cache → provider chain → rank → extract)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from iris_harness.plugins_builtin.research import engine as engine_mod
from iris_harness.plugins_builtin.research.cache import ResearchCache
from iris_harness.plugins_builtin.research.engine import ResearchEngine
from iris_harness.plugins_builtin.research.models import (
    Freshness,
    ResearchInput,
    SearchHit,
    SearchResult,
    SearchType,
)
from iris_harness.plugins_builtin.research.providers.base import SearchProvider
from iris_harness.plugins_builtin.research.rank import score_results


class _FakeProvider(SearchProvider):
    def __init__(self, name: str, hits: list[SearchHit], available: bool = True) -> None:
        self.name = name
        self._hits = hits
        self._available = available
        self.calls = 0

    def is_available(self) -> bool:
        return self._available

    def search(
        self,
        query: str,
        *,
        max_results: int,
        search_type: SearchType = "web",
        freshness: Freshness = "any",
        safe_search: bool = True,
    ) -> list[SearchHit]:
        self.calls += 1
        return list(self._hits)


def _r(title: str, url: str, snippet: str = "") -> SearchHit:
    return SearchHit(title=title, url=url, snippet=snippet)


def test_first_nonempty_provider_wins() -> None:
    empty = _FakeProvider("p1", [])
    full = _FakeProvider("p2", [_r("A", "http://a.com")])
    never = _FakeProvider("p3", [_r("B", "http://b.com")])
    eng = ResearchEngine(providers=[empty, full, never])
    out = eng.research(ResearchInput(query="x", fetch_content=False))
    assert out.provider == "p2"
    assert [r.url for r in out.results] == ["http://a.com"]
    assert empty.calls == 1 and full.calls == 1 and never.calls == 0  # short-circuits


def test_dedupe_and_top_n() -> None:
    hits = [
        _r("A", "http://a.com/x"),
        _r("A dup", "http://a.com/x/"),  # same after normalization
        _r("B", "http://b.com"),
        _r("C", "http://c.com"),
    ]
    eng = ResearchEngine(providers=[_FakeProvider("p", hits)])
    out = eng.research(ResearchInput(query="x", max_results=2, fetch_content=False))
    assert len(out.results) == 2  # deduped to 3, capped to 2
    urls = {r.url for r in out.results}
    assert len(urls) == 2


def test_no_results_sets_error() -> None:
    eng = ResearchEngine(providers=[_FakeProvider("p", [])])
    out = eng.research(ResearchInput(query="x", fetch_content=False))
    assert out.results == [] and out.error and out.provider == "none"


def test_fetch_content_invokes_extractor(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    called: dict[str, int] = {"n": 0}

    def _fake_extract(
        results: list[SearchResult], *, max_pages: int = 5, timeout: float = 8.0
    ) -> None:
        called["n"] = len(results)
        for r in results:
            r.content = "extracted"

    monkeypatch.setattr(engine_mod, "extract_into_sync", _fake_extract)
    eng = ResearchEngine(providers=[_FakeProvider("p", [_r("A", "http://a.com")])])
    out = eng.research(ResearchInput(query="x", fetch_content=True))
    assert called["n"] == 1 and out.results[0].content == "extracted"


def test_cache_hit_skips_providers(tmp_path: Path) -> None:
    cache = ResearchCache(db_path=tmp_path / "rc.db", ttl_seconds=3600)
    provider = _FakeProvider("p", [_r("A", "http://a.com", "snip")])
    eng = ResearchEngine(providers=[provider], cache=cache)
    first = eng.research(ResearchInput(query="x", fetch_content=False))
    assert first.cached is False and provider.calls == 1
    second = eng.research(ResearchInput(query="x", fetch_content=False))
    assert second.cached is True and provider.calls == 1  # served from cache
    assert [r.url for r in second.results] == ["http://a.com"]


def test_reranker_is_applied_when_wired() -> None:
    """An injected reranker re-orders results after the blended score (Phase 2)."""

    class _Reranker:
        def __init__(self) -> None:
            self.called = 0

        def rerank(
            self, query: str, results: list[SearchResult], *, top_k: int | None = None
        ) -> None:
            self.called += 1
            results.reverse()  # deterministic reorder we can assert

    rr = _Reranker()
    hits = [_r("A", "http://a.com"), _r("B", "http://b.com"), _r("C", "http://c.com")]
    eng = ResearchEngine(providers=[_FakeProvider("p", hits)], reranker=rr)
    out = eng.research(ResearchInput(query="x", max_results=3, fetch_content=False, rerank=True))
    assert rr.called == 1
    # rerank=False must skip the reranker entirely
    rr2 = _Reranker()
    eng2 = ResearchEngine(providers=[_FakeProvider("p", list(hits))], reranker=rr2)
    eng2.research(ResearchInput(query="x", max_results=3, fetch_content=False, rerank=False))
    assert rr2.called == 0
    assert len(out.results) == 3


def test_provider_exception_falls_through() -> None:
    class _Boom(_FakeProvider):
        def search(self, *a: object, **k: object) -> list[SearchHit]:
            raise RuntimeError("down")

    boom = _Boom("boom", [])
    full = _FakeProvider("p2", [_r("A", "http://a.com")])
    eng = ResearchEngine(providers=[boom, full])
    out = eng.research(ResearchInput(query="x", fetch_content=False))
    assert out.provider == "p2"


# --------------------------------------------------------------------------- language


class _LangProvider(_FakeProvider):
    """Records the ``language`` hint it was given (None when the engine sent none)."""

    def __init__(self, name: str, hits: list[SearchHit]) -> None:
        super().__init__(name, hits)
        self.languages: list[str | None] = []

    def search(
        self,
        query: str,
        *,
        max_results: int,
        search_type: SearchType = "web",
        freshness: Freshness = "any",
        safe_search: bool = True,
        language: str | None = None,
    ) -> list[SearchHit]:
        self.languages.append(language)
        return super().search(query, max_results=max_results)


def test_language_is_a_provider_hint_and_a_filter() -> None:
    provider = _LangProvider(
        "p", [_r("生成AIの新モデル", "http://jp.example/1"), _r("New AI model", "http://a.com")]
    )
    eng = ResearchEngine(providers=[provider])
    out = eng.research(ResearchInput(query="AI news", language="en", fetch_content=False))
    assert provider.languages == ["en"]
    assert [r.url for r in out.results] == ["http://a.com"]
    assert out.dropped_by_language == 1 and out.error is None


def test_no_language_sends_no_hint_so_older_providers_still_work() -> None:
    old = _FakeProvider("old", [_r("生成AIの新モデル", "http://jp.example/1")])  # no kwarg
    out = ResearchEngine(providers=[old]).research(ResearchInput(query="x", fetch_content=False))
    assert [r.title for r in out.results] == ["生成AIの新モデル"]
    assert out.dropped_by_language == 0


def test_a_provider_left_with_nothing_in_the_language_falls_through() -> None:
    japanese = _LangProvider("searxng", [_r("生成AIの新モデル", "http://jp.example/1")])
    english = _LangProvider("ddg", [_r("New AI model", "http://a.com")])
    out = ResearchEngine(providers=[japanese, english]).research(
        ResearchInput(query="AI news", language="en", fetch_content=False)
    )
    assert out.provider == "ddg"
    assert [r.url for r in out.results] == ["http://a.com"]
    assert out.dropped_by_language == 1


def test_every_result_in_another_language_is_an_error_not_an_empty_answer() -> None:
    provider = _LangProvider(
        "p", [_r("生成AIの新モデル", "http://jp.example/1"), _r("中国新闻", "http://cn.example/2")]
    )
    out = ResearchEngine(providers=[provider]).research(
        ResearchInput(query="AI news", language="en", fetch_content=False)
    )
    assert out.results == []
    assert out.dropped_by_language == 2
    assert out.error is not None and "'en'" in out.error


def test_language_is_part_of_the_cache_key(tmp_path: Path) -> None:
    cache = ResearchCache(db_path=tmp_path / "c.db")
    provider = _LangProvider("p", [_r("New AI model", "http://a.com")])
    eng = ResearchEngine(providers=[provider], cache=cache)
    eng.research(ResearchInput(query="AI news", fetch_content=False))
    eng.research(ResearchInput(query="AI news", language="en", fetch_content=False))
    assert provider.languages == [None, "en"]  # the second was not served from the first


def test_research_input_rejects_a_language_that_is_not_a_code() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ResearchInput(query="x", language="english")


# --------------------------------------------------------------------------- SearchHit


def test_a_hit_becomes_the_engines_result_field_for_field() -> None:
    when = datetime(2026, 9, 1, tzinfo=UTC)
    hit = SearchHit(
        url="https://docs.python.org/3/",
        title="Python docs",
        snippet="The reference.",
        published=when,
        extra={"engine": "bing"},
    )
    out = ResearchEngine(providers=[_FakeProvider("chain-name", [hit])]).research(
        ResearchInput(query="python", fetch_content=False)
    )

    (result,) = out.results
    assert (result.url, result.title, result.snippet, result.published) == (
        "https://docs.python.org/3/",
        "Python docs",
        "The reference.",
        when,
    )
    # No ``source`` from the provider: the chain's name for it stands in.
    assert result.source == "chain-name"
    assert result.metadata["engine"] == "bing"
    # The engine fills its own copy; the provider's hit is untouched.
    assert dict(hit.extra) == {"engine": "bing"}
    assert result.score > 0 and result.trust_score > 0


def test_a_providers_own_source_name_is_kept() -> None:
    hit = SearchHit(url="https://a.com", title="A", source="upstream")
    out = ResearchEngine(providers=[_FakeProvider("p", [hit])]).research(
        ResearchInput(query="a", fetch_content=False)
    )
    assert out.results[0].source == "upstream"


def test_the_engine_scores_a_hit_as_it_scores_its_own_result() -> None:
    """Converting changes nothing the ranker sees: same score as a hand-built result."""
    hits = [
        SearchHit(url="https://docs.python.org/3/", title="Python docs", snippet="reference"),
        SearchHit(url="https://spam.example/x", title="Python", snippet="buy now"),
    ]
    out = ResearchEngine(providers=[_FakeProvider("p", hits)], feedback_store=False).research(
        ResearchInput(query="python docs", fetch_content=False, rerank=False)
    )
    expected = [SearchResult(title=h.title, url=h.url, snippet=h.snippet) for h in hits]
    score_results("python docs", expected, embed=None, search_type="web")

    got = {r.url: (r.score, r.trust_score) for r in out.results}
    want = {r.url: (r.score, r.trust_score) for r in expected}
    assert got == want
    assert [r.url for r in out.results] == [r.url for r in expected]
