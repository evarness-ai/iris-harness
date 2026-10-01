"""DuckDuckGo search provider — the keyless floor.

Always available, tried last, so the engine can always serve *something* without keys or a
self-hosted instance. Backed by the ``ddgs`` package. Rate-limit / network errors are
swallowed (one retry, no sleeps) and yield ``[]`` per the provider contract.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from ddgs import DDGS

from iris_harness.plugins_builtin.research.models import Freshness, SearchHit, SearchType
from iris_harness.plugins_builtin.research.providers.base import SearchProvider
from iris_harness.sdk.logging import log_egress

logger = logging.getLogger(__name__)

# ``language`` -> ddgs region: English asks for the US-English index; any other code
# gets the no-region index (ddgs regions are country-language pairs, and a language
# alone does not name a country). No language -> the library default.
_ENGLISH_REGION = "us-en"
_ANY_REGION = "wt-wt"


def _region(language: str | None) -> str | None:
    if not language:
        return None
    return _ENGLISH_REGION if language == "en" else _ANY_REGION


# freshness -> ddgs timelimit ("any" -> None)
_TIMELIMIT: dict[Freshness, str] = {
    "day": "d",
    "week": "w",
    "month": "m",
    "year": "y",
}


class DuckDuckGoProvider(SearchProvider):
    """Keyless fallback backed by the ``ddgs`` library."""

    name = "ddg"

    def is_available(self) -> bool:
        return True

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
        timelimit = _TIMELIMIT.get(freshness)
        safesearch = "moderate" if safe_search else "off"
        is_news = search_type == "news"
        region = _region(language)

        rows = self._query(query, max_results, safesearch, timelimit, is_news, region)
        if not rows:
            # One retry on empty/failure — no sleep (ddgs may transiently rate-limit).
            rows = self._query(query, max_results, safesearch, timelimit, is_news, region)

        results: list[SearchHit] = []
        for row in rows:
            parsed = self._to_result(row, is_news)
            if parsed is not None:
                results.append(parsed)
            if len(results) >= max_results:
                break
        return results

    def _query(
        self,
        query: str,
        max_results: int,
        safesearch: str,
        timelimit: str | None,
        is_news: bool,
        region: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            client = DDGS()
            extra: dict[str, Any] = {"region": region} if region else {}
            log_egress(
                destination="duckduckgo.com",
                method="GET",
                kind="search",
                purpose="ddg",
            )
            if is_news:
                return client.news(
                    query,
                    max_results=max_results,
                    safesearch=safesearch,
                    timelimit=timelimit,
                    **extra,
                )
            return client.text(
                query,
                max_results=max_results,
                safesearch=safesearch,
                timelimit=timelimit,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001 - provider must never raise
            logger.debug("ddg search failed: %s", exc)
            return []

    @staticmethod
    def _to_result(row: dict[str, Any], is_news: bool) -> SearchHit | None:
        url = str(row.get("href") or row.get("url") or "")
        if not url:
            return None
        title = str(row.get("title", "") or "")
        snippet = str(row.get("body", "") or "")
        published = _parse_date(row.get("date")) if is_news else None
        return SearchHit(
            title=title,
            url=url,
            snippet=snippet,
            source="ddg",
            published=published,
        )


def _parse_date(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


__all__ = ["DuckDuckGoProvider"]
