"""Typed contract for the research engine.

Every research module (extraction, ranking, cache, engine) speaks in these types. A
provider returns the SDK's frozen ``SearchHit`` rows; the engine copies each into its own
``SearchResult`` (:meth:`SearchResult.from_hit`, in one place) and enriches, ranks and
returns them in a ``ResearchResult``. The LLM-facing tool serializes the latter.

``SearchResult`` is the engine's and nobody else's: it carries what the engine fills
(``score``, ``trust_score``, ``content``, the rerank signals in ``metadata``), so it is not
a provider's to build. ``SearchHit``, ``SearchType`` and ``Freshness`` are the SDK's (every
provider in the chain speaks them); they are re-exported here so the engine's modules keep
one import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, Field

# What a provider returns and the lens/window it is asked for are the SDK's: every
# provider in the chain, a plugin's too, speaks them (iris_harness.sdk.research).
from iris_harness.sdk.research import Freshness, SearchHit, SearchType


@dataclass
class SearchResult:
    """One hit as the engine works on it.

    Built from a provider's :class:`SearchHit` by :meth:`from_hit` (or from the cache);
    the engine then fills ``score`` and ``trust_score`` (the ranker), ``content`` (page
    extraction) and its rerank signals in ``metadata``.
    """

    title: str
    url: str
    snippet: str = ""
    source: str = ""
    score: float = 0.0
    published: datetime | None = None
    content: str | None = None
    trust_score: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_hit(cls, hit: SearchHit, *, provider: str) -> SearchResult:
        """The engine's copy of ``hit``; ``source`` falls back to the provider's chain name.

        The provider's ``extra`` labels seed ``metadata`` (a copy: the engine adds to it)."""
        return cls(
            title=hit.title,
            url=hit.url,
            snippet=hit.snippet,
            source=hit.source or provider,
            published=hit.published,
            metadata=dict(hit.extra),
        )

    def to_public_dict(self) -> dict[str, object]:
        """Serialize for the LLM-facing tool. Omits Nones to keep the payload lean."""
        out: dict[str, object] = {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "score": round(self.score, 4),
            "trust_score": round(self.trust_score, 2),
        }
        if self.published is not None:
            out["published"] = self.published.date().isoformat()
        if self.content:
            out["content"] = self.content
        if self.metadata:
            out["metadata"] = self.metadata
        return out


class ResearchInput(BaseModel):
    """LLM-facing tool input. One question in; a ranked, citation-ready set out."""

    query: str = Field(..., min_length=1, description="What to research.")
    search_type: SearchType = Field("web", description="Search lens.")
    max_results: int = Field(5, ge=1, le=20, description="Max ranked results to return.")
    fetch_content: bool = Field(
        True, description="Crawl + extract clean markdown for the top results."
    )
    rerank: bool = Field(True, description="Semantically rerank against the query.")
    freshness: Freshness = Field("any", description="Recency window hint.")
    safe_search: bool = Field(True, description="Enable provider safe-search where supported.")
    language: str | None = Field(
        None,
        pattern=r"^[a-z]{2}$",
        description=(
            "Two-letter ISO 639-1 code the results should be in (e.g. 'en'); passed to "
            "providers that support it, and results in another script are dropped."
        ),
    )


@dataclass
class ResearchResult:
    """Engine output: the query, the ranked results, and which backend served them."""

    query: str
    results: list[SearchResult]
    provider: str  # the provider that produced the hits ("searxng", "ddg", "cache", ...)
    cached: bool = False
    error: str | None = None  # set when all providers failed; results is then []
    # Hits dropped for not being in ``ResearchInput.language``. With ``results`` empty
    # and ``error`` set, every hit was in another language (not a provider failure).
    dropped_by_language: int = 0

    def to_public_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "query": self.query,
            "provider": self.provider,
            "cached": self.cached,
            "results": [r.to_public_dict() for r in self.results],
        }
        if self.error:
            out["error"] = self.error
        return out


__all__ = [
    "Freshness",
    "ResearchInput",
    "ResearchResult",
    "SearchHit",
    "SearchResult",
    "SearchType",
]
