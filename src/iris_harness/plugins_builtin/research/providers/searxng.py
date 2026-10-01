"""SearXNG search provider — the primary, self-hosted, keyless-but-URL-gated backend.

Talks to a SearXNG instance's JSON API (``GET {base_url}/search?format=json``). Selected
first when ``IRIS_SEARXNG_URL`` is set; otherwise the engine falls through to the keyed
providers and finally DuckDuckGo.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime
from urllib.parse import urlparse

from iris_harness.plugins_builtin.research.models import Freshness, SearchHit, SearchType
from iris_harness.plugins_builtin.research.providers.base import SearchProvider
from iris_harness.sdk.logging import log_egress

logger = logging.getLogger(__name__)

_TIMEOUT_S = 8
_USER_AGENT = "iris-research/1.0 (+https://github.com/iris)"

# search_type -> SearXNG category
_CATEGORY: dict[SearchType, str] = {
    "web": "general",
    "news": "news",
    "github": "it",
    "reddit": "social media",
    "docs": "general",
}

# freshness -> SearXNG time_range ("any" omits the param)
_TIME_RANGE: dict[Freshness, str] = {
    "day": "day",
    "week": "week",
    "month": "month",
    "year": "year",
}


class SearxngProvider(SearchProvider):
    """Primary provider: a self-hosted SearXNG instance addressed via ``IRIS_SEARXNG_URL``."""

    name = "searxng"

    @property
    def base_url(self) -> str:
        # Read per call (IRIS_SEARXNG_URL applies now): the provider is registered once,
        # at setup.
        return os.environ.get("IRIS_SEARXNG_URL", "").strip().rstrip("/")

    def is_available(self) -> bool:
        return bool(self.base_url)

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
        if not self.base_url:
            return []
        try:
            params: dict[str, str] = {
                "q": query,
                "format": "json",
                "safesearch": "1" if safe_search else "0",
                "categories": _CATEGORY.get(search_type, "general"),
            }
            if language:
                params["language"] = language
            time_range = _TIME_RANGE.get(freshness)
            if time_range is not None:
                params["time_range"] = time_range

            url = f"{self.base_url}/search?{urllib.parse.urlencode(params)}"
            request = urllib.request.Request(  # noqa: S310 - URL is operator-configured
                url, headers={"User-Agent": _USER_AGENT}
            )
            log_egress(
                destination=urlparse(self.base_url).netloc,
                method="GET",
                kind="search",
                purpose="searxng",
            )
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:  # noqa: S310
                payload = response.read()
            data = json.loads(payload)

            results: list[SearchHit] = []
            for hit in data.get("results", []):
                title = str(hit.get("title", "") or "")
                hit_url = str(hit.get("url", "") or "")
                if not hit_url:
                    continue
                snippet = str(hit.get("content", "") or "")
                published = _parse_iso(hit.get("publishedDate"))
                engine = hit.get("engine") or hit.get("engines")
                if isinstance(engine, list):
                    engine = ", ".join(str(e) for e in engine)
                extra = {"engine": str(engine)} if engine else {}
                results.append(
                    SearchHit(
                        title=title,
                        url=hit_url,
                        snippet=snippet,
                        source="searxng",
                        published=published,
                        extra=extra,
                    )
                )
                if len(results) >= max_results:
                    break
            return results
        except Exception as exc:  # noqa: BLE001 - provider must never raise
            logger.debug("searxng search failed: %s", exc)
            return []


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


__all__ = ["SearxngProvider"]
