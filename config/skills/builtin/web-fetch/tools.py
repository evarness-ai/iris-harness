"""Web content fetcher — dispatches on (type, category).

News categories route through the research engine (`src/iris/research/`); git-trending
and stocks remain structured-source fetchers (GitHub Trending HTML, Yahoo Finance quotes)
that a generic web search can't replicate (star counts, live quotes). The stocks source
base URL + watchlist are env-overridable (IRIS_STOCKS_SOURCE_URL / IRIS_STOCKS_SYMBOLS)."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Literal

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; iris-web-fetch/0.1)"}
TIMEOUT = 20.0

GITHUB_TRENDING_URL = "https://github.com/trending"

# Stocks source — Yahoo Finance (free, no auth). Two paths:
#   * ``stocks-trending`` → the predefined "most actives" screener (real
#     most-traded equities), one request, quotes inline.
#   * ``stocks-quotes`` → explicit per-symbol quotes via the v8 chart endpoint,
#     for an arbitrary watchlist / portfolio. Works for US (``AAPL``) and Indian
#     (``RELIANCE.NS``, ``INFY.BO``) symbols alike; currency is reported per quote.
# Symbols can be passed as a tool parameter (``symbols=[...]``) — the seam for a
# future portfolio extracted from demat accounts / broker emails. Both base URLs
# and the default watchlist are env-overridable so the source can be retargeted
# without code changes.
YAHOO_CHART_URL = os.environ.get(
    "IRIS_STOCKS_SOURCE_URL", "https://query1.finance.yahoo.com/v8/finance/chart/"
)
YAHOO_SCREENER_URL = os.environ.get(
    "IRIS_STOCKS_SCREENER_URL",
    "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved",
)
DEFAULT_STOCK_SYMBOLS = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "SPY",
    "QQQ",
    "BTC-USD",
)


def _stock_symbols(explicit: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Resolve the watchlist for an explicit-symbols stock fetch.

    Priority: ``explicit`` (tool param) → ``IRIS_STOCKS_SYMBOLS`` (CSV env) →
    ``DEFAULT_STOCK_SYMBOLS``.
    """
    if explicit:
        symbols = tuple(s.strip().upper() for s in explicit if s and s.strip())
        if symbols:
            return symbols
    raw = os.environ.get("IRIS_STOCKS_SYMBOLS", "").strip()
    if raw:
        symbols = tuple(item.strip().upper() for item in raw.split(",") if item.strip())
        if symbols:
            return symbols
    return DEFAULT_STOCK_SYMBOLS


# Market index symbols (Yahoo uses a ^ prefix). region "IN" / "US" / "ALL".
INDEX_SYMBOLS: dict[str, tuple[str, ...]] = {
    "IN": ("^NSEI", "^BSESN", "^NSEBANK"),  # NIFTY 50, SENSEX, NIFTY Bank
    "US": ("^GSPC", "^DJI", "^IXIC"),  # S&P 500, Dow Jones, NASDAQ
}


def _index_symbols(region: str = "ALL") -> tuple[str, ...]:
    """Resolve index symbols. IRIS_INDEX_SYMBOLS (CSV) overrides; else by region."""
    raw = os.environ.get("IRIS_INDEX_SYMBOLS", "").strip()
    if raw:
        symbols = tuple(item.strip().upper() for item in raw.split(",") if item.strip())
        if symbols:
            return symbols
    key = (region or "ALL").upper()
    if key in INDEX_SYMBOLS:
        return INDEX_SYMBOLS[key]
    return INDEX_SYMBOLS["IN"] + INDEX_SYMBOLS["US"]


# News categories are no longer hardcoded HN/RSS scrapes — they go through the research
# engine (provider chain: SearXNG → … → DuckDuckGo). Each maps to a news-search query.
# These fixed queries serve ad-hoc fetches; the morning digest uses DIGEST_TOPICS below.
_NEWS_CATEGORY_QUERIES: dict[str, str] = {
    "ai-news": "latest artificial intelligence news",
    "global-news": "top world news headlines today",
    "usa-news": "top United States news headlines today",
}

# The morning digest's news (loop-proof D4, digest v5): with ``news_group`` (a news
# section, e.g. ``news_local``), that group's topics from Settings → Digest
# (``news_groups``, ``{news_local_area}`` filled in); without, ``news_topics``. Each
# topic is one news query, and ``news_sources`` rank first — preferred, never a strict
# allowlist, so a non-preferred result is reordered, never dropped. Results are in
# ``news_language`` (the research engine's provider hint + script filter).
DIGEST_TOPICS = "digest-topics"


ContentType = Literal["git", "news", "stocks"]
ContentCategory = Literal[
    "git-repositories",
    "ai-news",
    "usa-news",
    "global-news",
    "digest-topics",
    "stocks-trending",
    "stocks-quotes",
    "indexes",
]


class FetchWebContentInput(BaseModel):
    """Input schema for the generic web content fetcher."""

    type: ContentType = Field(default="news", description="Content family.")
    category: ContentCategory = Field(
        default="ai-news",
        description="Concrete source/category within the type.",
    )
    limit: int = Field(default=10, ge=1, le=50, description="Max items to return.")
    symbols: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "For category=stocks-quotes: explicit ticker symbols to quote "
            "(US e.g. 'AAPL'; India e.g. 'RELIANCE.NS', 'INFY.BO'). "
            "Falls back to the configured watchlist when omitted."
        ),
    )
    region: str = Field(
        default="US",
        description="Region for category=stocks-trending (most-actives screener).",
    )
    news_group: str | None = Field(
        default=None,
        description=(
            "For category=digest-topics: the digest news section whose topics to fetch "
            "(e.g. 'news_ai', 'news_local'); omitted = the ungrouped news_topics."
        ),
    )


def _http_get(url: str, *, params: dict[str, Any] | None = None) -> requests.Response | None:
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        logger.warning("web-fetch GET failed url=%s err=%s", url, exc)
        return None


# ---------------------------------------------------------------------------
# git-repositories — GitHub Trending HTML scrape
# ---------------------------------------------------------------------------


def _parse_int(text: str) -> int:
    if not text:
        return 0
    match = re.search(r"([0-9,]+)", text)
    return int(match.group(1).replace(",", "")) if match else 0


def _fetch_github_trending(limit: int) -> list[dict[str, object]]:
    response = _http_get(GITHUB_TRENDING_URL, params={"since": "daily"})
    if response is None:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    items: list[dict[str, object]] = []
    for element in soup.select("article.Box-row")[:limit]:
        link = element.select_one("h1 a") or element.select_one("h2 a")
        if not link or not link.get("href"):
            continue
        repo_full = " ".join(link.text.split()).replace(" / ", "/").replace(" ", "")
        repo_url = "https://github.com" + str(link["href"]).strip()
        description_tag = element.select_one("p.col-9") or element.select_one("p")
        description = description_tag.text.strip() if description_tag else ""
        language_tag = element.select_one("[itemprop=programmingLanguage]")
        language = language_tag.text.strip() if language_tag else ""
        stars_today_tag = element.select_one(
            "span.d-inline-block.float-sm-right"
        ) or element.select_one("span.float-sm-right")
        stars_today = _parse_int(stars_today_tag.text) if stars_today_tag else 0
        if stars_today == 0:
            match = re.search(r"([0-9,]+)\s+stars?\s+today", element.text)
            if match:
                stars_today = int(match.group(1).replace(",", ""))

        items.append(
            {
                "title": repo_full,
                "url": repo_url,
                "description": description,
                "language": language,
                "stars_today": stars_today,
            }
        )
    return items


# ---------------------------------------------------------------------------
# news (ai / global / usa) — via the research engine (no hardcoded sources)
# ---------------------------------------------------------------------------


def _domain(url: str) -> str:
    """Registrable host for display as the news `source` (e.g. 'techcrunch.com')."""
    try:
        from urllib.parse import urlparse  # noqa: PLC0415

        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:  # noqa: BLE001
        return ""


def _fetch_research_news(category: str, limit: int) -> list[dict[str, object]]:
    """Fetch news headlines for a category through the research engine.

    Replaces the old HackerNews + Google-RSS scrapes: the query goes through the
    provider chain (SearXNG → Tavily → Brave → DuckDuckGo) with ``search_type=news``.
    Maps each result to the brief's expected shape ({title, url, source, published_at}).
    Snippets only — no page crawl — to keep briefs fast.
    """
    query = _NEWS_CATEGORY_QUERIES.get(category)
    if not query:
        return []
    return _research_news(query, limit, label=category)


def _is_preferred(source: str, preferred: tuple[str, ...]) -> bool:
    """``source`` is one of the preferred domains or a subdomain of one."""
    host = source.lower()
    return any(host == p or host.endswith("." + p) for p in preferred)


def _normalise_sources(sources: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """'https://www.CBSNews.com/' → 'cbsnews.com'."""
    out: list[str] = []
    for raw in sources:
        text = str(raw).strip().lower()
        if not text:
            continue
        host = _domain(text if "://" in text else "https://" + text)
        if host and host not in out:
            out.append(host)
    return tuple(out)


def prefer_sources(
    items: list[dict[str, object]], preferred: tuple[str, ...]
) -> list[dict[str, object]]:
    """Preferred-source items first, each group in its original order; nothing dropped.

    Marks every item with ``preferred`` (" ★" or "") for the brief's item template.
    """
    marked = [
        {**item, "preferred": " ★" if _is_preferred(str(item.get("source", "")), preferred) else ""}
        for item in items
    ]
    return [i for i in marked if i["preferred"]] + [i for i in marked if not i["preferred"]]


def _fetch_digest_topics(limit: int, news_group: str | None = None) -> list[dict[str, object]]:
    """The digest's news: one query per owner topic, preferred sources ranked first.

    ``news_group`` names the digest news section whose topics to use; a group the
    digest does not have raises (the section is named as failed, never silently
    empty). Topics and sources come from Settings → Digest at call time, so an edit changes
    tomorrow's digest with no re-approval. Each topic's results are re-ranked locally
    (the research engine has no domain boost), then the topics are interleaved so
    every topic is represented, de-duplicated by URL and capped at ``limit``.

    One topic failing only thins the section. **Every** topic failing raises, so the
    digest names the section in its failure line instead of printing an empty one
    (loop-proof D5: a failure is named, never silent). A topic whose every result was
    in another language than ``news_language`` counts as failed, for the same reason.
    """
    from pathlib import Path  # noqa: PLC0415

    from iris_harness.sdk.digest import (  # noqa: PLC0415
        load_digest_settings,
        news_group_topics,
    )

    settings = load_digest_settings(Path(os.environ.get("IRIS_DATA_DIR") or "data"))
    if news_group:
        if news_group not in settings.news_groups:
            raise RuntimeError(f"no digest news group {news_group!r}")
        wanted: tuple[str, ...] = news_group_topics(settings, news_group)
    else:
        wanted = settings.news_topics
    topics = [t.strip() for t in wanted if t and t.strip()]
    preferred = _normalise_sources(settings.news_sources)
    language = None if settings.news_language in ("", "any") else settings.news_language
    per_topic: list[list[dict[str, object]]] = []
    errors: list[str] = []
    for topic in dict.fromkeys(topics):
        try:
            found = _research_news(
                f"{topic} news", limit, label=topic, raise_errors=True, language=language
            )
        except Exception as exc:  # noqa: BLE001 - one topic failing thins the section
            logger.warning("web-fetch digest news failed topic=%s err=%s", topic, exc)
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        per_topic.append(prefer_sources([{**it, "topic": topic} for it in found], preferred))
    if errors and not per_topic:
        raise RuntimeError(f"news research failed for every topic ({errors[0]})")
    merged: list[dict[str, object]] = []
    seen: set[str] = set()
    for rank in range(max((len(group) for group in per_topic), default=0)):
        for group in per_topic:
            if rank >= len(group):
                continue
            item = group[rank]
            url = str(item.get("url", ""))
            if url in seen:
                continue
            seen.add(url)
            merged.append(item)
    return merged[:limit]


def _research_news(
    query: str,
    limit: int,
    *,
    label: str,
    raise_errors: bool = False,
    language: str | None = None,
) -> list[dict[str, object]]:
    """One news query through the research engine, mapped to the brief's item shape.

    Best-effort by default (a failure is an empty list); ``raise_errors`` lets the
    caller tell a failed query from a query with no results. ``language`` (ISO 639-1)
    asks for results in it only; with ``raise_errors``, a query whose every result was
    in another language raises too (the engine's ``error``), rather than looking empty.
    """
    try:
        from iris_harness.plugins_builtin.research.models import ResearchInput  # noqa: PLC0415
        from iris_harness.plugins_builtin.research.tool import get_engine  # noqa: PLC0415

        result = get_engine().research(
            ResearchInput(
                query=query,
                search_type="news",
                max_results=limit,
                fetch_content=False,
                language=language,
            )
        )
        if raise_errors and result.dropped_by_language and not result.results:
            raise RuntimeError(result.error or f"no results in language {language!r}")
    except Exception as exc:  # noqa: BLE001 - news is best-effort
        if raise_errors:
            raise
        logger.warning("web-fetch research news failed query=%s err=%s", label, exc)
        return []
    items: list[dict[str, object]] = []
    for r in result.results[:limit]:
        items.append(
            {
                "title": r.title,
                "url": r.url,
                "source": _domain(r.url) or r.source,
                "published_at": r.published.date().isoformat() if r.published else "",
            }
        )
    return items


# ---------------------------------------------------------------------------
# stocks — Yahoo Finance (most-actives screener + per-symbol quotes, no auth)
# ---------------------------------------------------------------------------


def _parse_yahoo_chart(payload: dict[str, Any]) -> dict[str, object] | None:
    """Parse one symbol's Yahoo v8 chart response into a quote dict.

    Change percent is computed against the previous close so the briefing shows
    day-over-day movement.
    """
    try:
        meta = payload["chart"]["result"][0]["meta"]
    except (KeyError, IndexError, TypeError):
        return None
    price_raw = meta.get("regularMarketPrice")
    prev_raw = meta.get("chartPreviousClose") or meta.get("previousClose")
    if price_raw is None:
        return None
    try:
        price = float(price_raw)
        prev = float(prev_raw) if prev_raw else 0.0
    except (TypeError, ValueError):
        return None
    change_pct = round(((price - prev) / prev) * 100, 2) if prev else 0.0
    symbol = str(meta.get("symbol") or "").upper()
    name = str(meta.get("shortName") or meta.get("longName") or symbol)
    return {
        "symbol": symbol,
        "name": name,
        "price": round(price, 2),
        "change_pct": change_pct,
        "currency": str(meta.get("currency") or ""),
    }


def _quote_from_screener(quote: dict[str, Any]) -> dict[str, object] | None:
    """Map one Yahoo screener row to a quote dict (price + change inline)."""
    symbol = str(quote.get("symbol") or "").upper()
    price_raw = quote.get("regularMarketPrice")
    if not symbol or price_raw is None:
        return None
    try:
        price = round(float(price_raw), 2)
    except (TypeError, ValueError):
        return None
    change_raw = quote.get("regularMarketChangePercent")
    try:
        change_pct = round(float(change_raw), 2) if change_raw is not None else 0.0
    except (TypeError, ValueError):
        change_pct = 0.0
    name = str(quote.get("shortName") or quote.get("longName") or symbol)
    return {
        "symbol": symbol,
        "name": name,
        "price": price,
        "change_pct": change_pct,
        "currency": str(quote.get("currency") or ""),
    }


def _fetch_yahoo_most_active(limit: int, region: str) -> list[dict[str, object]]:
    """Fetch the real most-traded equities via Yahoo's predefined screener."""
    response = _http_get(
        YAHOO_SCREENER_URL,
        params={
            "scrIds": "most_actives",
            "count": max(1, min(limit, 50)),
            "region": (region or "US").upper(),
        },
    )
    if response is None:
        return []
    try:
        payload = response.json()
    except ValueError:
        logger.warning("web-fetch stocks: non-JSON screener response")
        return []
    try:
        quotes = payload["finance"]["result"][0]["quotes"]
    except (KeyError, IndexError, TypeError):
        return []
    items: list[dict[str, object]] = []
    for quote in quotes[:limit]:
        parsed = _quote_from_screener(quote)
        if parsed is not None:
            items.append(parsed)
    return items


def _quote_each(symbols: tuple[str, ...]) -> list[dict[str, object]]:
    """Quote a fixed list of symbols, one v8 chart request each (graceful)."""
    items: list[dict[str, object]] = []
    for symbol in symbols:
        response = _http_get(
            f"{YAHOO_CHART_URL}{symbol}",
            params={"interval": "1d", "range": "1d"},
        )
        if response is None:
            continue
        try:
            payload = response.json()
        except ValueError:
            logger.warning("web-fetch: non-JSON quote response for %s", symbol)
            continue
        parsed = _parse_yahoo_chart(payload)
        if parsed is not None:
            items.append(parsed)
    return items


def _fetch_yahoo_quotes(
    limit: int, symbols: tuple[str, ...] | None = None
) -> list[dict[str, object]]:
    """Quote explicit symbols (watchlist / portfolio).

    Resolves the symbol list from ``symbols`` → ``IRIS_STOCKS_SYMBOLS`` →
    default. Handles US and Indian (``.NS`` / ``.BO``) tickers identically.
    """
    return _quote_each(_stock_symbols(symbols)[:limit])


def _fetch_yahoo_indexes(limit: int, region: str = "ALL") -> list[dict[str, object]]:
    """Quote the major market indexes (NIFTY/SENSEX/BankNifty + S&P/Dow/NASDAQ).

    Region "IN" / "US" / "ALL"; IRIS_INDEX_SYMBOLS overrides. Yahoo returns the
    index level as ``regularMarketPrice``, so the quote shape matches stocks.
    """
    return _quote_each(_index_symbols(region)[:limit])


# ---------------------------------------------------------------------------
# Tool entrypoint
# ---------------------------------------------------------------------------


class FetchWebContentTool(BaseTool):
    """Generic content fetcher with category-based dispatch."""

    name: str = "fetch_web_content"
    description: str = (
        "Fetch ranked content from a configured upstream. "
        "type ∈ {git,news,stocks}; category selects the concrete source."
    )
    args_schema: type[BaseModel] = FetchWebContentInput

    def _run(
        self,
        type: ContentType = "news",
        category: ContentCategory = "ai-news",
        limit: int = 10,
        symbols: tuple[str, ...] | None = None,
        region: str = "US",
        news_group: str | None = None,
    ) -> list[dict[str, object]]:
        if type == "git" and category == "git-repositories":
            return _fetch_github_trending(limit)
        if type == "news" and category == DIGEST_TOPICS:
            return _fetch_digest_topics(limit, news_group)
        if type == "news" and category in _NEWS_CATEGORY_QUERIES:
            return _fetch_research_news(category, limit)
        if type == "stocks" and category == "stocks-trending":
            return _fetch_yahoo_most_active(limit, region)
        if type == "stocks" and category == "stocks-quotes":
            return _fetch_yahoo_quotes(limit, symbols)
        if type == "stocks" and category == "indexes":
            return _fetch_yahoo_indexes(limit, region)
        logger.warning("web-fetch unsupported combo type=%s category=%s", type, category)
        return []

    async def _arun(
        self,
        type: ContentType = "news",
        category: ContentCategory = "ai-news",
        limit: int = 10,
        symbols: tuple[str, ...] | None = None,
        region: str = "US",
        news_group: str | None = None,
    ) -> list[dict[str, object]]:
        return self._run(
            type=type,
            category=category,
            limit=limit,
            symbols=symbols,
            region=region,
            news_group=news_group,
        )


SKILL_TOOLS = [FetchWebContentTool]

__all__ = ["FetchWebContentTool", "FetchWebContentInput", "SKILL_TOOLS"]
