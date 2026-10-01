"""Unit tests for the web-fetch skill's fetch_web_content tool."""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import ValidationError

from iris_harness.tools.skills.loader import load_skill_manifest, load_skill_tool_classes

SKILL_DIR = Path("config/skills/builtin/web-fetch")


def _load_tools_module() -> Any:
    # Trigger the loader so the dynamically-named module is imported and cached,
    # then return it from sys.modules.
    load_skill_tool_classes(SKILL_DIR)
    for name, module in sys.modules.items():
        if (
            name.startswith("iris_dynamic_skill_")
            and name.endswith("_tools")
            and "web_fetch" in name
        ):
            return module
    raise RuntimeError("web-fetch tools module not found in sys.modules")


def _make_response(*, text: str = "", json_payload: Any = None) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.text = text
    response.raise_for_status = MagicMock()
    if json_payload is not None:
        response.json = MagicMock(return_value=json_payload)
    else:
        response.json = MagicMock(side_effect=ValueError("no json"))
    return response


def test_web_fetch_manifest_is_loader_compatible() -> None:
    manifest = load_skill_manifest(SKILL_DIR)
    tool_classes = load_skill_tool_classes(SKILL_DIR)
    assert manifest.name == "web-fetch"
    assert [t.name for t in manifest.tools] == ["fetch_web_content"]
    assert [cls().name for cls in tool_classes] == ["fetch_web_content"]


def test_input_clamps_limit_above_50() -> None:
    module = _load_tools_module()
    with pytest.raises(ValidationError):
        module.FetchWebContentInput(type="git", category="git-repositories", limit=999)


def test_git_repositories_returns_expected_shape() -> None:
    module = _load_tools_module()
    html = """
    <article class="Box-row">
      <h2><a href="/octocat/Hello-World"> octocat / Hello-World </a></h2>
      <p class="col-9">A friendly repo</p>
      <span itemprop="programmingLanguage">Python</span>
      <span class="d-inline-block float-sm-right">123 stars today</span>
    </article>
    """
    with patch.object(module.requests, "get", return_value=_make_response(text=html)):
        items = module.FetchWebContentTool()._run(type="git", category="git-repositories", limit=5)
    assert len(items) == 1
    item = items[0]
    assert item["title"] == "octocat/Hello-World"
    assert item["url"] == "https://github.com/octocat/Hello-World"
    assert item["description"] == "A friendly repo"
    assert item["language"] == "Python"
    assert item["stars_today"] == 123


def _fake_engine_returning(results: list[Any]) -> Any:
    """A stand-in research engine whose .research() returns the given SearchResults."""
    from iris_harness.plugins_builtin.research.models import ResearchResult

    class _Eng:
        def research(self, _inp: Any) -> ResearchResult:
            return ResearchResult(query="q", results=results, provider="ddg")

    return _Eng()


@pytest.mark.parametrize("category", ["ai-news", "global-news", "usa-news"])
def test_news_categories_route_through_research(category: str) -> None:
    """News no longer scrapes HN/RSS — it goes through the research engine, and the
    result is mapped to the brief's {title, url, source, published_at} shape."""
    from datetime import UTC, datetime

    from iris_harness.plugins_builtin.research.models import SearchResult

    module = _load_tools_module()
    results = [
        SearchResult(
            title="Big AI release",
            url="https://www.techcrunch.com/ai-story",
            snippet="...",
            published=datetime(2026, 6, 25, tzinfo=UTC),
        ),
        SearchResult(title="Second story", url="https://example.org/news/2"),
    ]
    with patch(
        "iris_harness.plugins_builtin.research.tool.get_engine",
        return_value=_fake_engine_returning(results),
    ):
        items = module.FetchWebContentTool()._run(type="news", category=category, limit=10)

    assert [it["title"] for it in items] == ["Big AI release", "Second story"]
    # source is the registrable domain (www. stripped), not the provider name
    assert items[0]["source"] == "techcrunch.com"
    assert items[0]["published_at"] == "2026-06-25"
    assert items[1]["source"] == "example.org" and items[1]["published_at"] == ""
    assert all(set(it.keys()) >= {"title", "url", "source", "published_at"} for it in items)


def test_news_returns_empty_on_research_failure() -> None:
    module = _load_tools_module()

    class _Boom:
        def research(self, _inp: Any) -> Any:
            raise RuntimeError("down")

    with patch("iris_harness.plugins_builtin.research.tool.get_engine", return_value=_Boom()):
        items = module.FetchWebContentTool()._run(type="news", category="ai-news", limit=10)
    assert items == []


def _yahoo_chart(symbol: str, *, price: float, prev: float, name: str) -> dict:
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "symbol": symbol,
                        "shortName": name,
                        "regularMarketPrice": price,
                        "chartPreviousClose": prev,
                        "currency": "USD",
                    }
                }
            ]
        }
    }


def _yahoo_screener(*rows: dict) -> dict:
    return {"finance": {"result": [{"quotes": list(rows)}]}}


def test_stocks_trending_uses_most_active_screener() -> None:
    module = _load_tools_module()
    payload = _yahoo_screener(
        {
            "symbol": "AAPL",
            "shortName": "Apple Inc.",
            "regularMarketPrice": 202.46,
            "regularMarketChangePercent": 1.23,
            "currency": "USD",
        },
        {
            "symbol": "TSLA",
            "shortName": "Tesla, Inc.",
            "regularMarketPrice": 292.50,
            "regularMarketChangePercent": -2.5,
            "currency": "USD",
        },
    )
    with patch.object(module.requests, "get", return_value=_make_response(json_payload=payload)):
        items = module.FetchWebContentTool()._run(
            type="stocks", category="stocks-trending", limit=2
        )
    assert items[0] == {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "price": 202.46,
        "change_pct": 1.23,
        "currency": "USD",
    }
    assert items[1]["symbol"] == "TSLA"
    assert items[1]["change_pct"] == -2.5


def test_stocks_quotes_explicit_symbols_us_and_india() -> None:
    """stocks-quotes quotes the passed symbols (the portfolio seam), with the
    per-symbol currency preserved for Indian (INR) tickers."""
    module = _load_tools_module()

    def _fake_get(url, *a, **k):
        symbol = url.rstrip("/").rsplit("/", 1)[-1]
        currency = "INR" if symbol.endswith((".NS", ".BO")) else "USD"
        payload = {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "symbol": symbol,
                            "shortName": symbol,
                            "regularMarketPrice": 100.0,
                            "chartPreviousClose": 100.0,
                            "currency": currency,
                        }
                    }
                ]
            }
        }
        return _make_response(json_payload=payload)

    with patch.object(module.requests, "get", side_effect=_fake_get):
        items = module.FetchWebContentTool()._run(
            type="stocks",
            category="stocks-quotes",
            symbols=("aapl", "reliance.ns"),
            limit=10,
        )
    assert [it["symbol"] for it in items] == ["AAPL", "RELIANCE.NS"]
    assert items[1]["currency"] == "INR"


def test_indexes_quotes_major_benchmarks() -> None:
    module = _load_tools_module()

    def _fake_get(url, *a, **k):
        sym = url.rsplit("/", 1)[-1]
        return _make_response(
            json_payload=_yahoo_chart(sym, price=24056.0, prev=24000.0, name="Index")
        )

    with patch.object(module.requests, "get", side_effect=_fake_get):
        items = module.FetchWebContentTool()._run(
            type="stocks", category="indexes", region="ALL", limit=10
        )
    # India + USA majors: NIFTY 50, SENSEX, NIFTY Bank, S&P 500, Dow, NASDAQ
    assert len(items) == 6
    assert all("change_pct" in it for it in items)


def test_indexes_region_filter() -> None:
    module = _load_tools_module()
    captured: list[str] = []

    def _fake_get(url, *a, **k):
        captured.append(url)
        return _make_response(json_payload=_yahoo_chart("idx", price=100.0, prev=100.0, name="idx"))

    with patch.object(module.requests, "get", side_effect=_fake_get):
        module.FetchWebContentTool()._run(type="stocks", category="indexes", region="IN")
    # India region -> only the 3 Indian index symbols (^NSEI, ^BSESN, ^NSEBANK)
    assert len(captured) == 3
    assert all(any(s in u for s in ("NSEI", "BSESN", "NSEBANK")) for u in captured)


def test_stocks_quotes_watchlist_is_env_overridable(monkeypatch) -> None:
    module = _load_tools_module()
    monkeypatch.setenv("IRIS_STOCKS_SYMBOLS", "nflx, dis")
    captured: list[str] = []

    def _fake_get(url, *a, **k):
        captured.append(url)
        symbol = url.rstrip("/").rsplit("/", 1)[-1]
        return _make_response(
            json_payload=_yahoo_chart(symbol, price=100.0, prev=100.0, name=symbol)
        )

    with patch.object(module.requests, "get", side_effect=_fake_get):
        # no explicit symbols -> falls back to IRIS_STOCKS_SYMBOLS
        items = module.FetchWebContentTool()._run(type="stocks", category="stocks-quotes", limit=10)
    assert [it["symbol"] for it in items] == ["NFLX", "DIS"]
    assert all(url.endswith(("NFLX", "DIS")) for url in captured)


def test_returns_empty_list_when_http_fails() -> None:
    module = _load_tools_module()
    with patch.object(module.requests, "get", side_effect=requests.RequestException("boom")):
        items = module.FetchWebContentTool()._run(type="git", category="git-repositories", limit=5)
    assert items == []


def test_unsupported_combo_returns_empty() -> None:
    module = _load_tools_module()
    # type/category mismatch — dispatcher logs warning and returns []
    items = module.FetchWebContentTool()._run(
        type="stocks", category="ai-news", limit=5  # type: ignore[arg-type]
    )
    assert items == []


# ─── digest-topics: the morning digest's news (loop-proof D4) ────────────────


class _TopicEngine:
    """Answers each topic query with its own results; records the queries asked."""

    def __init__(
        self,
        by_query: dict[str, list[Any]],
        fail: set[str] | None = None,
        foreign: set[str] | None = None,
    ) -> None:
        self.by_query = by_query
        self.fail = fail or set()
        self.foreign = foreign or set()  # queries whose every hit was in another language
        self.asked: list[str] = []
        self.languages: list[str | None] = []

    def research(self, inp: Any) -> Any:
        from iris_harness.plugins_builtin.research.models import ResearchResult

        self.asked.append(inp.query)
        self.languages.append(inp.language)
        assert inp.search_type == "news"
        if inp.query in self.fail:
            raise TimeoutError("research engine timeout after 20 s")
        if inp.query in self.foreign:
            return ResearchResult(
                query=inp.query,
                results=[],
                provider="ddg",
                error="none of the 4 results was in language 'en'",
                dropped_by_language=4,
            )
        return ResearchResult(
            query=inp.query, results=self.by_query.get(inp.query, []), provider="ddg"
        )


def _digest_settings(monkeypatch: pytest.MonkeyPatch, **fields: Any) -> None:
    from iris_harness.sdk import digest as digest_settings

    monkeypatch.setattr(
        digest_settings,
        "load_digest_settings",
        lambda _data_dir, _config_dir=None: digest_settings.DigestSettings(**fields),
    )


def _hit(title: str, url: str) -> Any:
    from iris_harness.plugins_builtin.research.models import SearchResult

    return SearchResult(title=title, url=url)


def test_digest_topics_each_topic_is_a_query_and_topics_interleave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _digest_settings(monkeypatch, news_topics=("AI", "world"), news_sources=())
    engine = _TopicEngine(
        {
            "AI news": [_hit("ai-1", "https://a.example/1"), _hit("ai-2", "https://a.example/2")],
            "world news": [_hit("w-1", "https://w.example/1")],
        }
    )
    module = _load_tools_module()
    with patch("iris_harness.plugins_builtin.research.tool.get_engine", return_value=engine):
        items = module.FetchWebContentTool()._run(type="news", category="digest-topics", limit=10)

    assert engine.asked == ["AI news", "world news"]
    assert [it["title"] for it in items] == ["ai-1", "w-1", "ai-2"]
    assert [it["topic"] for it in items] == ["AI", "world", "AI"]
    assert all(it["preferred"] == "" for it in items)


def test_digest_topics_preferred_sources_rank_first_and_nothing_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _digest_settings(
        monkeypatch, news_topics=("world",), news_sources=("https://www.CBSNews.com/",)
    )
    engine = _TopicEngine(
        {
            "world news": [
                _hit("bbc", "https://www.bbc.com/a"),
                _hit("cbs", "https://www.cbsnews.com/b"),
                _hit("reuters", "https://reuters.com/c"),
                _hit("cbs-sub", "https://live.cbsnews.com/d"),
            ]
        }
    )
    module = _load_tools_module()
    with patch("iris_harness.plugins_builtin.research.tool.get_engine", return_value=engine):
        items = module.FetchWebContentTool()._run(type="news", category="digest-topics", limit=10)

    assert [it["title"] for it in items] == ["cbs", "cbs-sub", "bbc", "reuters"]
    assert [it["preferred"] for it in items] == [" ★", " ★", "", ""]


def test_digest_topics_dedupes_and_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    _digest_settings(monkeypatch, news_topics=("AI", "tech"))
    same = "https://a.example/same"
    engine = _TopicEngine(
        {
            "AI news": [_hit("s", same), _hit("a2", "https://a.example/2")],
            "tech news": [_hit("s", same), _hit("t2", "https://t.example/2")],
        }
    )
    module = _load_tools_module()
    with patch("iris_harness.plugins_builtin.research.tool.get_engine", return_value=engine):
        items = module.FetchWebContentTool()._run(type="news", category="digest-topics", limit=2)
    assert [it["url"] for it in items] == [same, "https://a.example/2"]


def test_digest_topics_one_failed_topic_thins_the_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _digest_settings(monkeypatch, news_topics=("AI", "world"))
    engine = _TopicEngine({"AI news": [_hit("ai-1", "https://a.example/1")]}, fail={"world news"})
    module = _load_tools_module()
    with patch("iris_harness.plugins_builtin.research.tool.get_engine", return_value=engine):
        items = module.FetchWebContentTool()._run(type="news", category="digest-topics", limit=10)
    assert [it["title"] for it in items] == ["ai-1"]


def test_digest_topics_every_topic_failing_raises_so_the_digest_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loop-proof D5: a failed section is named in the digest, never an empty heading."""
    _digest_settings(monkeypatch, news_topics=("AI", "world"))
    engine = _TopicEngine({}, fail={"AI news", "world news"})
    module = _load_tools_module()
    with (
        patch("iris_harness.plugins_builtin.research.tool.get_engine", return_value=engine),
        pytest.raises(RuntimeError, match="every topic"),
    ):
        module.FetchWebContentTool()._run(type="news", category="digest-topics", limit=10)


def test_digest_topics_without_topics_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _digest_settings(monkeypatch, news_topics=())
    engine = _TopicEngine({})
    module = _load_tools_module()
    with patch("iris_harness.plugins_builtin.research.tool.get_engine", return_value=engine):
        assert module.FetchWebContentTool()._run(type="news", category="digest-topics") == []
    assert engine.asked == []


def test_digest_topics_ask_for_the_news_language(monkeypatch: pytest.MonkeyPatch) -> None:
    _digest_settings(monkeypatch, news_topics=("AI",), news_language="en")
    engine = _TopicEngine({"AI news": [_hit("ai-1", "https://a.example/1")]})
    module = _load_tools_module()
    with patch("iris_harness.plugins_builtin.research.tool.get_engine", return_value=engine):
        module.FetchWebContentTool()._run(type="news", category="digest-topics", limit=10)
    assert engine.languages == ["en"]


def test_digest_topics_any_language_asks_for_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _digest_settings(monkeypatch, news_topics=("AI",), news_language="any")
    engine = _TopicEngine({"AI news": [_hit("ai-1", "https://a.example/1")]})
    module = _load_tools_module()
    with patch("iris_harness.plugins_builtin.research.tool.get_engine", return_value=engine):
        module.FetchWebContentTool()._run(type="news", category="digest-topics", limit=10)
    assert engine.languages == [None]


def test_digest_topics_a_topic_all_in_another_language_thins_the_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _digest_settings(monkeypatch, news_topics=("AI", "world"))
    engine = _TopicEngine(
        {"AI news": [_hit("ai-1", "https://a.example/1")]}, foreign={"world news"}
    )
    module = _load_tools_module()
    with patch("iris_harness.plugins_builtin.research.tool.get_engine", return_value=engine):
        items = module.FetchWebContentTool()._run(type="news", category="digest-topics", limit=10)
    assert [it["title"] for it in items] == ["ai-1"]


def test_digest_topics_everything_in_another_language_is_named_never_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filtering must not turn the section silently empty (loop-proof D5)."""
    _digest_settings(monkeypatch, news_topics=("AI", "world"))
    engine = _TopicEngine({}, foreign={"AI news", "world news"})
    module = _load_tools_module()
    with (
        patch("iris_harness.plugins_builtin.research.tool.get_engine", return_value=engine),
        pytest.raises(RuntimeError, match=r"every topic.*language 'en'"),
    ):
        module.FetchWebContentTool()._run(type="news", category="digest-topics", limit=10)


def test_ad_hoc_news_categories_stay_best_effort_without_a_language() -> None:
    """Only the digest asks for a language; the fixed categories keep their behaviour."""
    engine = _TopicEngine({"latest artificial intelligence news": []})
    module = _load_tools_module()
    with patch("iris_harness.plugins_builtin.research.tool.get_engine", return_value=engine):
        assert module.FetchWebContentTool()._run(type="news", category="ai-news") == []
    assert engine.languages == [None]


# ─── digest v5: one news section per group (news_ai, news_global, news_local) ──

_GROUPS = {
    "news_ai": {"title": "AI / Tech", "topics": ("AI", "technology")},
    "news_local": {"title": "Local — {news_local_area}", "topics": ("{news_local_area}",)},
}


def test_a_news_group_asks_its_own_topics_with_the_local_area_filled_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _digest_settings(
        monkeypatch,
        news_topics=("ignored",),
        news_groups=_GROUPS,
        news_local_area="St. Louis",
        news_sources=("ksdk.com",),
    )
    engine = _TopicEngine(
        {
            "St. Louis news": [
                _hit("metro", "https://www.stltoday.com/metro"),
                _hit("cards", "https://www.ksdk.com/cards"),
            ]
        }
    )
    module = _load_tools_module()
    with patch("iris_harness.plugins_builtin.research.tool.get_engine", return_value=engine):
        items = module.FetchWebContentTool().invoke(
            {"type": "news", "category": "digest-topics", "news_group": "news_local", "limit": 10}
        )

    assert engine.asked == ["St. Louis news"]
    # Preferred sources and the language apply to every group.
    assert [it["title"] for it in items] == ["cards", "metro"]
    assert engine.languages == ["en"]


def test_the_ai_group_asks_each_of_its_topics(monkeypatch: pytest.MonkeyPatch) -> None:
    _digest_settings(monkeypatch, news_groups=_GROUPS)
    engine = _TopicEngine({"AI news": [_hit("ai-1", "https://a.example/1")]})
    module = _load_tools_module()
    with patch("iris_harness.plugins_builtin.research.tool.get_engine", return_value=engine):
        items = module.FetchWebContentTool()._run(
            type="news", category="digest-topics", news_group="news_ai"
        )
    assert engine.asked == ["AI news", "technology news"]
    assert [it["title"] for it in items] == ["ai-1"]


def test_an_unknown_news_group_is_named_as_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    _digest_settings(monkeypatch, news_groups=_GROUPS)
    engine = _TopicEngine({})
    module = _load_tools_module()
    with (
        patch("iris_harness.plugins_builtin.research.tool.get_engine", return_value=engine),
        pytest.raises(RuntimeError, match="news_sports"),
    ):
        module.FetchWebContentTool()._run(
            type="news", category="digest-topics", news_group="news_sports"
        )
    assert engine.asked == []
