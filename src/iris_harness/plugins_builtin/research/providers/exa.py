"""Exa search provider — keyed, optional.

Enabled when ``EXA_API_KEY`` is set. POSTs JSON to Exa's search endpoint; Exa returns
neural-ranked results; their order is the hits' order (the engine does its own scoring).
The search lens shapes the request (news category, GitHub/Reddit domain filters) and the
freshness window maps to ``startPublishedDate``.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import UTC, datetime, timedelta

from iris_harness.plugins_builtin.research.models import Freshness, SearchHit, SearchType
from iris_harness.plugins_builtin.research.providers.base import SearchProvider
from iris_harness.sdk.logging import log_egress

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.exa.ai/search"
_TIMEOUT_S = 12

# freshness -> lookback window for startPublishedDate ("any" omits the param)
_LOOKBACK_DAYS: dict[Freshness, int] = {
    "day": 1,
    "week": 7,
    "month": 30,
    "year": 365,
}


class ExaProvider(SearchProvider):
    """Keyed provider backed by the Exa search API."""

    name = "exa"

    @property
    def _api_key(self) -> str:
        # Read per call: the provider is registered once, at setup, and a key set
        # afterwards must still count.
        return os.environ.get("EXA_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self._api_key)

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
        if not self._api_key:
            return []
        try:
            body: dict[str, object] = {
                "query": query,
                "numResults": max_results,
                "contents": {"text": False},
            }
            if search_type == "news":
                body["category"] = "news"
            elif search_type == "github":
                body["includeDomains"] = ["github.com"]
            elif search_type == "reddit":
                body["includeDomains"] = ["reddit.com"]

            days = _LOOKBACK_DAYS.get(freshness)
            if days is not None:
                start = datetime.now(UTC) - timedelta(days=days)
                body["startPublishedDate"] = start.isoformat()

            request = urllib.request.Request(  # fixed HTTPS endpoint
                _ENDPOINT,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self._api_key,
                },
                method="POST",
            )
            log_egress(
                destination="api.exa.ai",
                method="POST",
                kind="search",
                purpose="exa",
            )
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:  # noqa: S310
                payload = response.read()
            data = json.loads(payload)

            results: list[SearchHit] = []
            for hit in data.get("results", []):
                hit_url = str(hit.get("url", "") or "")
                if not hit_url:
                    continue
                snippet = hit.get("text") or hit.get("snippet") or ""
                results.append(
                    SearchHit(
                        title=str(hit.get("title", "") or ""),
                        url=hit_url,
                        snippet=str(snippet),
                        source="exa",
                        published=_parse_iso(hit.get("publishedDate")),
                    )
                )
                if len(results) >= max_results:
                    break
            return results
        except Exception as exc:  # noqa: BLE001 - provider must never raise
            logger.debug("exa search failed: %s", exc)
            return []


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


__all__ = ["ExaProvider"]
