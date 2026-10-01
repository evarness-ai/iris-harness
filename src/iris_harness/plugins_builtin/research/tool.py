"""LLM-facing surfaces for the research engine.

Two shapes over one engine:
- ``run_research(args, *, embed=None) -> str`` — the ReAct-loop tool callable; takes the
  model's action-input dict and returns a formatted, citation-ready observation string.
- ``ResearchTool`` — a LangChain ``BaseTool`` for skills that want the structured payload.

The engine is built once per process (with a SQLite cache) and reused.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel

from iris_harness.plugins_builtin.research.cache import build_cache
from iris_harness.plugins_builtin.research.engine import EmbedFn, ResearchEngine
from iris_harness.plugins_builtin.research.models import ResearchInput, ResearchResult
from iris_harness.plugins_builtin.research.rerank import build_reranker
from iris_harness.sdk.process_state import track_globals
from iris_harness.sdk.time import local_today

logger = logging.getLogger(__name__)

_ENGINE: ResearchEngine | None = None


# A question that asks for news. Deterministic and model-free, like every other pre-call
# decision in the harness — the model is not asked to remember a flag it was never told
# about, and the same words always pick the same lens.
#
# Why this exists: ``search_type`` defaults to "web", which calls the provider's plain
# keyword endpoint. Asked for "breaking news today" on 2026-09-16, that returned a hockey
# analysis piece, a celebrity item, an Emmys recap and a how-to on weed-eater line —
# every one a literal match on "breaking". Nothing in the tool description tells the model
# that a news question needs ``search_type="news"``, so it never passed it.
#
# Word boundaries throughout: "newspaper" and "renews" must not match.
_NEWS_QUERY_RE = re.compile(
    r"\bnews\b|\bheadlines?\b|\bbreaking\b|"
    r"\bwhat(?:'s|s| is| has|ever)? ?happen(?:ed|ing|s)?\b|"
    r"\blatest (?:on|from|about|in)\b",
    re.IGNORECASE,
)
# A news ask that names a moment wants today's window; anything else gets the week, so a
# slower-moving story ("latest on the trade talks") still has something to return.
_TODAY_RE = re.compile(
    r"\btoday\b|\btonight\b|\bbreaking\b|\bright now\b|\bthis morning\b|\bjust now\b",
    re.IGNORECASE,
)


def _derive_news_lens(query: str, raw: dict[str, Any]) -> None:
    """Fill in the news lens and its window when the caller named neither.

    Mutates ``raw`` in place. Only ever ADDS keys: a caller that passed ``search_type``
    keeps it, even when it passed "web" for a news-shaped query. ``fetch_web_content``
    builds its ``ResearchInput`` directly and never reaches here, so its explicit
    ``search_type="news"`` is untouched.
    """
    if "search_type" in raw or not _NEWS_QUERY_RE.search(query):
        return
    raw["search_type"] = "news"
    if "freshness" not in raw:
        raw["freshness"] = "day" if _TODAY_RE.search(query) else "week"


def get_engine(*, embed: EmbedFn | None = None) -> ResearchEngine:
    """Process-wide engine (lazy). ``embed`` wires semantic rerank when available."""
    global _ENGINE
    if _ENGINE is None:
        # Backends are selected by env: IRIS_RESEARCH_CACHE (sqlite|redis|off) and
        # IRIS_RESEARCH_RERANKER (none|bge). Both degrade gracefully to the default.
        try:
            cache = build_cache()
        except Exception as exc:  # noqa: BLE001 - cache is optional
            logger.debug("research cache unavailable: %s", exc)
            cache = None
        _ENGINE = ResearchEngine(cache=cache, embed=embed, reranker=build_reranker())
    return _ENGINE


def _coerce_input(args: dict[str, Any]) -> ResearchInput | None:
    """Parse a model action-input dict into a ResearchInput, tolerating loose shapes."""
    query = str(args.get("query") or args.get("input") or args.get("q") or "").strip()
    if not query:
        return None
    raw: dict[str, Any] = {"query": query}
    for key in (
        "search_type",
        "max_results",
        "fetch_content",
        "rerank",
        "freshness",
        "safe_search",
    ):
        if args.get(key) is not None:
            raw[key] = args[key]
    # Tolerate aliases the model might invent.
    if "type" in args and "search_type" not in raw:
        raw["search_type"] = args["type"]
    if "limit" in args and "max_results" not in raw:
        raw["max_results"] = args["limit"]
    _derive_news_lens(query, raw)
    try:
        return ResearchInput(**raw)
    except Exception as exc:  # noqa: BLE001 - fall back to the query plus the lens
        logger.debug("research input coercion fell back to defaults: %s", exc)
        # The lens is kept through the fallback: a model that invents `max_results: "ten"`
        # would otherwise send a news question down the keyword path, which is the bug.
        fallback: dict[str, Any] = {"query": query}
        _derive_news_lens(query, fallback)
        try:
            return ResearchInput(**fallback)
        except Exception:  # noqa: BLE001 - a bare query always validates
            return ResearchInput(query=query)


def format_result(result: ResearchResult) -> str:
    """Render a ResearchResult as a readable, citation-bearing observation string."""
    header = f"[Current date: {local_today().isoformat()}; provider: {result.provider}"
    if result.cached:
        header += "; cached"
    header += "]\n"
    if not result.results:
        return header + (f"No results ({result.error})." if result.error else "No results found.")
    lines: list[str] = []
    for i, r in enumerate(result.results, 1):
        parts = [f"{i}. **{r.title or r.url}**"]
        if r.url:
            parts.append(f"({r.url})")
        if r.published:
            parts.append(f"[{r.published.date().isoformat()}]")
        line = " ".join(parts)
        if r.snippet:
            line += f"\n   {r.snippet.strip()}"
        if r.content:
            excerpt = r.content.strip()
            line += f"\n   ---\n   {excerpt}"
        lines.append(line)
    return header + "\n".join(lines)


def run_research(args: dict[str, Any], *, embed: EmbedFn | None = None) -> str:
    """ReAct-loop entry point. Never raises — returns a string the model can read."""
    inp = _coerce_input(args)
    if inp is None:
        return "Error: research requires a 'query' argument."
    try:
        result = get_engine(embed=embed).research(inp)
    except Exception as exc:  # noqa: BLE001 - the tool must always return a string
        logger.warning("research failed for %r: %s", inp.query, exc)
        return f"Research unavailable: {exc}"
    return format_result(result)


class ResearchTool:
    """LangChain-style tool for skills wanting the structured payload (lazy base class).

    Implemented as a thin factory rather than subclassing ``BaseTool`` at import time so
    the research package has no hard LangChain dependency for the ReAct path.
    """

    name = "research"
    description = (
        "Search the web, crawl the most relevant pages, rerank them, and return "
        "extracted markdown with citations. One query in; a ranked, citation-ready "
        "result set out. A news question already uses the news lens and a recent "
        "window - do NOT pass search_type for one. Args: query (str), search_type "
        "(web|news|github|reddit|docs), max_results (int), fetch_content (bool), "
        "freshness (day|week|month|year|any)."
    )

    def __init__(self, *, embed: EmbedFn | None = None) -> None:
        self._embed = embed

    def run(self, args: dict[str, Any]) -> list[dict[str, object]]:
        inp = _coerce_input(args)
        if inp is None:
            return []
        result = get_engine(embed=self._embed).research(inp)
        return [r.to_public_dict() for r in result.results]


def build_langchain_tool(*, embed: EmbedFn | None = None) -> Any:
    """Build a real LangChain BaseTool wrapper (imported lazily). Used by skills."""
    from langchain_core.tools import BaseTool

    class _ResearchInputSchema(BaseModel):
        query: str
        # `search_type` and `freshness` default to None, not to "web"/"any", so "unset"
        # stays distinguishable all the way down: `_coerce_input` skips None keys, and the
        # news derivation only fires when the caller named no lens. Measured on the
        # pinned LangChain: `invoke` filters the schema's defaults back out and they never
        # reach `_run`, so this changes nothing today. It is here because that filtering
        # is LangChain's implementation detail, and a literal default would read as "the
        # caller chose web" the day it changes. ResearchInput still applies the real
        # defaults.
        search_type: str | None = None
        max_results: int = 5
        fetch_content: bool = True
        freshness: str | None = None

    tool = ResearchTool(embed=embed)

    class _LangChainResearchTool(BaseTool):
        name: str = "research"
        description: str = ResearchTool.description
        args_schema: type[BaseModel] = _ResearchInputSchema

        def _run(self, **kwargs: Any) -> list[dict[str, object]]:
            return tool.run(kwargs)

        async def _arun(self, **kwargs: Any) -> list[dict[str, object]]:
            return tool.run(kwargs)

    return _LangChainResearchTool()


__all__ = [
    "ResearchTool",
    "build_langchain_tool",
    "format_result",
    "get_engine",
    "run_research",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_ENGINE")
