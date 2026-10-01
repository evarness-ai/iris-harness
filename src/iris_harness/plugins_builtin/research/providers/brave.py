"""Brave Search provider — keyed, optional.

Enabled when ``BRAVE_API_KEY`` is set. Uses the web or news endpoint depending on the
search type, with Brave's ``freshness`` shorthand (pd/pw/pm/py) for recency.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime

from iris_harness.plugins_builtin.research.models import Freshness, SearchHit, SearchType
from iris_harness.plugins_builtin.research.providers.base import SearchProvider
from iris_harness.sdk.logging import log_egress

logger = logging.getLogger(__name__)

_WEB_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_NEWS_ENDPOINT = "https://api.search.brave.com/res/v1/news/search"
_TIMEOUT_S = 10

# freshness -> Brave "freshness" shorthand ("any" omits the param)
_FRESHNESS: dict[Freshness, str] = {
    "day": "pd",
    "week": "pw",
    "month": "pm",
    "year": "py",
}


class BraveProvider(SearchProvider):
    """Keyed provider backed by the Brave Search API."""

    name = "brave"

    def is_available(self) -> bool:
        return bool(os.environ.get("BRAVE_API_KEY"))

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
        api_key = os.environ.get("BRAVE_API_KEY")
        if not api_key:
            return []
        try:
            is_news = search_type == "news"
            params: dict[str, str] = {
                "q": query,
                "count": str(max_results),
                "safesearch": "strict" if safe_search else "off",
            }
            if language:
                params["search_lang"] = language
            fresh = _FRESHNESS.get(freshness)
            if fresh is not None:
                params["freshness"] = fresh

            endpoint = _NEWS_ENDPOINT if is_news else _WEB_ENDPOINT
            url = f"{endpoint}?{urllib.parse.urlencode(params)}"
            request = urllib.request.Request(  # noqa: S310 - fixed HTTPS endpoint
                url,
                headers={
                    "X-Subscription-Token": api_key,
                    "Accept": "application/json",
                },
            )
            log_egress(
                destination="api.search.brave.com",
                method="GET",
                kind="search",
                purpose="brave",
            )
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:  # noqa: S310
                payload = response.read()
            data = json.loads(payload)

            if is_news:
                rows = data.get("results", [])
            else:
                web = data.get("web") or {}
                rows = web.get("results", []) if isinstance(web, dict) else []

            results: list[SearchHit] = []
            for hit in rows:
                hit_url = str(hit.get("url", "") or "")
                if not hit_url:
                    continue
                published = _parse_iso(hit.get("age") or hit.get("page_age"))
                results.append(
                    SearchHit(
                        title=str(hit.get("title", "") or ""),
                        url=hit_url,
                        snippet=str(hit.get("description", "") or ""),
                        source="brave",
                        published=published,
                    )
                )
                if len(results) >= max_results:
                    break
            return results
        except Exception as exc:  # noqa: BLE001 - provider must never raise
            logger.debug("brave search failed: %s", exc)
            return []


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


__all__ = ["BraveProvider"]
