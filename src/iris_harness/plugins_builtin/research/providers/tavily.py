"""Tavily search provider — keyed, optional.

Enabled when ``TAVILY_API_KEY`` is set. Tavily returns pre-scored results; their order
is the hits' order (the engine does its own scoring). POSTs JSON to the search endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime

from iris_harness.plugins_builtin.research.models import Freshness, SearchHit, SearchType
from iris_harness.plugins_builtin.research.providers.base import SearchProvider
from iris_harness.sdk.logging import log_egress

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.tavily.com/search"
_TIMEOUT_S = 12

# freshness -> Tavily "days" lookback (news topic only; "any" omits it)
_DAYS: dict[Freshness, int] = {
    "day": 1,
    "week": 7,
    "month": 30,
    "year": 365,
}


class TavilyProvider(SearchProvider):
    """Keyed provider backed by the Tavily search API."""

    name = "tavily"

    def is_available(self) -> bool:
        return bool(os.environ.get("TAVILY_API_KEY"))

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
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return []
        try:
            topic = "news" if search_type == "news" else "general"
            body: dict[str, object] = {
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "topic": topic,
            }
            if topic == "news":
                days = _DAYS.get(freshness)
                if days is not None:
                    body["days"] = days

            request = urllib.request.Request(  # fixed HTTPS endpoint
                _ENDPOINT,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            log_egress(
                destination="api.tavily.com",
                method="POST",
                kind="search",
                purpose="tavily",
            )
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:  # noqa: S310
                payload = response.read()
            data = json.loads(payload)

            results: list[SearchHit] = []
            for hit in data.get("results", []):
                hit_url = str(hit.get("url", "") or "")
                if not hit_url:
                    continue
                results.append(
                    SearchHit(
                        title=str(hit.get("title", "") or ""),
                        url=hit_url,
                        snippet=str(hit.get("content", "") or ""),
                        source="tavily",
                        published=_parse_iso(hit.get("published_date")),
                    )
                )
                if len(results) >= max_results:
                    break
            return results
        except Exception as exc:  # noqa: BLE001 - provider must never raise
            logger.debug("tavily search failed: %s", exc)
            return []


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


__all__ = ["TavilyProvider"]
