"""Research engine — orchestrates the cache → provider → rank → extract pipeline.

One entry point, ``ResearchEngine.research(ResearchInput) -> ResearchResult``:

1. cache lookup (SQLite TTL) — return the stored result on a hit;
2. try the search-provider chain in order (``config/search_providers.yaml``: SearXNG →
   Tavily → Exa → Brave → any plugin's provider → DuckDuckGo), first non-empty wins;
3. dedupe by normalized URL (with ``language`` set, step 2 already dropped hits in
   another script — ``language.py`` — and skipped a provider left with none; every
   hit dropped is an ``error``, never a silent empty answer);
4. score + rerank (semantic if an ``embed`` callable is wired, else lexical) and keep
   the top ``max_results``;
5. optionally crawl + extract clean markdown for those results;
6. cache and return.

Never raises: any failure degrades to an empty ``ResearchResult`` with ``error`` set, so
the LLM-facing tool always gets a serializable answer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Protocol
from urllib.parse import urlparse, urlunparse

from iris_harness.plugins_builtin.research.cache import ResearchCache
from iris_harness.plugins_builtin.research.extract import extract_into_sync
from iris_harness.plugins_builtin.research.language import in_language
from iris_harness.plugins_builtin.research.models import (
    Freshness,
    ResearchInput,
    ResearchResult,
    SearchHit,
    SearchResult,
    SearchType,
)
from iris_harness.plugins_builtin.research.providers import select_providers
from iris_harness.plugins_builtin.research.rank import score_results

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]

# How many top candidates a (potentially expensive) cross-encoder reranker re-scores.
_RERANK_TOP_K = 10


class ChainProvider(Protocol):
    """A provider as the engine tries it: a name for the result and the logs, and search.

    What ``select_providers()`` returns (the SDK's ``ChainLink``); a test injects its own."""

    @property
    def name(self) -> str: ...

    def search(
        self,
        query: str,
        *,
        max_results: int,
        search_type: SearchType = ...,
        freshness: Freshness = ...,
        safe_search: bool = ...,
        language: str | None = ...,
    ) -> list[SearchHit]: ...


class Reranker(Protocol):
    """A second-stage reranker (Phase 2: BGE cross-encoder). Re-scores in place."""

    def rerank(
        self, query: str, results: list[SearchResult], *, top_k: int | None = ...
    ) -> None: ...


class CacheBackend(Protocol):
    """Either ``ResearchCache`` (SQLite) or ``RedisCache`` — the engine only needs these."""

    def get(self, key: str) -> dict[str, object] | None: ...
    def put(self, key: str, payload: dict[str, object]) -> None: ...


def _normalize_url(url: str) -> str:
    """Canonicalize a URL for dedupe: drop fragment + trailing slash, lowercase host."""
    try:
        p = urlparse(url)
        path = p.path.rstrip("/") or "/"
        return urlunparse((p.scheme, p.netloc.lower(), path, "", p.query, ""))
    except Exception:  # noqa: BLE001
        return url


def _dedupe(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in results:
        key = _normalize_url(r.url)
        if not r.url or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


class ResearchEngine:
    """Pluggable research pipeline. All collaborators are injectable for testing."""

    def __init__(
        self,
        *,
        providers: Sequence[ChainProvider] | None = None,
        cache: CacheBackend | None = None,
        embed: EmbedFn | None = None,
        reranker: Reranker | None = None,
        feedback_store: object | None = None,
    ) -> None:
        # Providers are resolved lazily per call when not injected, so a key/URL set
        # after construction (or in a test monkeypatch) is still honored.
        self._providers = providers
        self._cache = cache
        self._embed = embed
        self._reranker = reranker
        # Surface-feedback spine: downranks sources the user marked "not useful"
        # (issue 0028). Default-built lazily so suppression is honored by default;
        # injectable for tests.
        self._feedback_store = feedback_store

    def _resolve_providers(self) -> Sequence[ChainProvider]:
        return self._providers if self._providers is not None else select_providers()

    def _feedback_store_or_default(self) -> object | None:
        """The surface-feedback store, lazily default-built so suppression is on by
        default. Best-effort — a build failure simply disables suppression."""
        if self._feedback_store is None:
            try:
                from iris_harness.sdk.learning import SurfaceFeedbackStore

                store = SurfaceFeedbackStore()
                store.ensure_schema()
                self._feedback_store = store
            except Exception:  # suppression is optional polish
                logger.debug("research: surface-feedback store unavailable", exc_info=True)
                self._feedback_store = False  # sentinel: tried, unavailable
        return self._feedback_store or None

    def research(self, inp: ResearchInput) -> ResearchResult:
        # A language is part of the request: English-only hits must not answer (or be
        # answered by) the same query without one.
        lens = f"{inp.search_type}:{inp.language}" if inp.language else str(inp.search_type)
        cache_key = ResearchCache.make_key(inp.query, lens, inp.max_results, inp.fetch_content)
        # Passed only when set, so a provider written before ``language`` still works.
        hints: dict[str, Any] = {"language": inp.language} if inp.language else {}
        if self._cache is not None:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return _result_from_cache(inp.query, hit)

        # 1. provider chain — first provider returning hits (in ``language``, when one
        # is set) wins; a provider whose every hit is in another language is skipped.
        found: list[SearchHit] = []
        used = "none"
        last_error: str | None = None
        dropped = 0
        for provider in self._resolve_providers():
            try:
                found = provider.search(
                    inp.query,
                    max_results=max(inp.max_results * 2, inp.max_results),
                    search_type=inp.search_type,
                    freshness=inp.freshness,
                    safe_search=inp.safe_search,
                    **hints,
                )
            except Exception as exc:  # noqa: BLE001 - providers shouldn't raise, but guard
                last_error = f"{provider.name}: {type(exc).__name__}"
                logger.debug("research provider %s raised: %s", provider.name, exc)
                continue
            if found and inp.language:
                kept = [h for h in found if in_language(h.title, inp.language)]
                if len(kept) < len(found):
                    dropped += len(found) - len(kept)
                    logger.info(
                        "research: dropped %d of %d %s results not in language %s",
                        len(found) - len(kept),
                        len(found),
                        provider.name,
                        inp.language,
                    )
                found = kept
            if found:
                used = provider.name
                break

        if not found:
            if dropped:
                return ResearchResult(
                    query=inp.query,
                    results=[],
                    provider=used,
                    error=f"none of the {dropped} results was in language {inp.language!r}",
                    dropped_by_language=dropped,
                )
            return ResearchResult(
                query=inp.query,
                results=[],
                provider=used,
                error=last_error or "no results from any search provider",
            )

        # The engine's own copies from here: score, trust and content are filled on them.
        hits = [SearchResult.from_hit(hit, provider=used) for hit in found]

        # 2. dedupe → 3. score/rerank → 4. top-N.
        hits = _dedupe(hits)
        embed = self._embed if inp.rerank else None
        try:
            score_results(
                inp.query,
                hits,
                embed=embed,
                search_type=str(inp.search_type),
                feedback_store=self._feedback_store_or_default(),
            )
        except Exception as exc:  # noqa: BLE001 - ranking is best-effort
            logger.debug("research ranking failed: %s", exc)
        # Optional second-stage cross-encoder rerank over the top candidates (Phase 2,
        # opt-in). Best-effort; a missing model is a no-op inside the reranker.
        if inp.rerank and self._reranker is not None:
            try:
                self._reranker.rerank(inp.query, hits, top_k=min(len(hits), _RERANK_TOP_K))
            except Exception as exc:  # noqa: BLE001
                logger.debug("research cross-encoder rerank failed: %s", exc)
        top = hits[: inp.max_results]

        # 5. content extraction for the survivors only.
        if inp.fetch_content and top:
            extract_into_sync(top, max_pages=len(top))

        result = ResearchResult(
            query=inp.query,
            results=top,
            provider=used,
            dropped_by_language=dropped,
        )

        # 6. cache the public payload (post-extraction).
        if self._cache is not None:
            self._cache.put(cache_key, result.to_public_dict())
        return result


def _result_from_cache(query: str, payload: dict[str, object]) -> ResearchResult:
    """Rehydrate a ResearchResult from a cached public payload (best-effort)."""
    raw = payload.get("results")
    rows: list[SearchResult] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            rows.append(
                SearchResult(
                    title=str(item.get("title") or ""),
                    url=str(item.get("url") or ""),
                    snippet=str(item.get("snippet") or ""),
                    source=str(item.get("source") or ""),
                    score=float(item.get("score") or 0.0),
                    trust_score=float(item.get("trust_score") or 0.0),
                    content=(str(item["content"]) if item.get("content") else None),
                )
            )
    provider = str(payload.get("provider") or "cache")
    return ResearchResult(query=query, results=rows, provider=provider, cached=True)


__all__ = ["EmbedFn", "ResearchEngine"]
