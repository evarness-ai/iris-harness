"""Research engine — provider-based web search + crawl + rerank for IRIS.

One tool (``research``) over a pluggable provider chain (SearXNG → Tavily → Brave →
DuckDuckGo), SQLite TTL caching, Trafilatura content extraction, and embedding/lexical
reranking. See ``docs/architecture/research-engine.md``.
"""

from __future__ import annotations

from iris_harness.plugins_builtin.research.engine import ResearchEngine
from iris_harness.plugins_builtin.research.models import ResearchInput, ResearchResult, SearchResult
from iris_harness.plugins_builtin.research.tool import ResearchTool, run_research

__all__ = [
    "ResearchEngine",
    "ResearchInput",
    "ResearchResult",
    "ResearchTool",
    "SearchResult",
    "run_research",
]
