"""Embedding-based semantic skill router.

Replaces the legacy keyword-overlap scorer (``_score_skill_package`` in
``iris_harness.runtime.bootstrap``) with a cosine-similarity lookup over manifest
text. Reuses the ChromaDB ONNX MiniLM-L6 embedder from
``iris_harness.kernel.governance.evaluator.embeddings`` so no new model dependency is
introduced — the memory subsystem already loads it.

The router is intentionally minimal:

- ``rank(query, packages)``   → packages sorted by similarity (descending).
- ``best_match(query, packages)`` → top package if it clears ``threshold``.
- ``score(query, package)``   → cosine similarity for a single package.

Skill-document embeddings are cached by their text so manifest hot-reloads
naturally invalidate (different text → new embedding) without bookkeeping.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from iris_harness.kernel.governance.evaluator.embeddings import Embedder, cosine_similarity

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.45


class SemanticSkillRouter:
    """Cosine-similarity ranker over loaded skill packages."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._cache: dict[str, list[float]] = {}

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def available(self) -> bool:
        """Whether the embedder can embed at all. An embedder without the notion (a
        test stub) always can."""
        return bool(getattr(self._embedder, "available", True))

    def _skill_doc(self, package: Any) -> str:
        manifest = package.manifest
        parts: list[str] = [
            manifest.name.replace("-", " ").replace("_", " "),
            manifest.description,
        ]
        parts.extend(tool.description for tool in manifest.tools)
        if getattr(package, "agent_context", None):
            parts.append(package.agent_context)
        return " ".join(p for p in parts if p).strip()

    def _embed(self, text: str) -> list[float] | None:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        try:
            vec = list(self._embedder(text))
        except Exception:
            logger.debug("semantic router embedding failed", exc_info=True)
            return None
        if not vec:
            return None
        self._cache[text] = vec
        return vec

    def rank(
        self,
        query: str,
        packages: Sequence[Any],
    ) -> list[tuple[Any, float]]:
        if not query.strip() or not packages:
            return []
        q_vec = self._embed(query)
        if q_vec is None:
            return []
        out: list[tuple[Any, float]] = []
        for pkg in packages:
            if not getattr(pkg, "is_loadable", True):
                continue
            doc = self._skill_doc(pkg)
            if not doc:
                continue
            p_vec = self._embed(doc)
            if p_vec is None:
                continue
            out.append((pkg, cosine_similarity(q_vec, p_vec)))
        out.sort(key=lambda item: item[1], reverse=True)
        return out

    def rank_texts(self, query: str, items: Sequence[tuple[Any, str]]) -> list[tuple[Any, float]]:
        """Rank arbitrary ``(key, text)`` pairs by cosine similarity to *query*.

        Generic counterpart to :meth:`rank` (which is skill-package specific). Used by
        the unified ReAct loop to shortlist individual tools by relevance to the turn
        (ADR-0077 P2), reusing the same cached MiniLM embedder. Keys whose text fails
        to embed are dropped; the rest come back highest-similarity first.
        """
        if not query.strip() or not items:
            return []
        q_vec = self._embed(query)
        if q_vec is None:
            return []
        out: list[tuple[Any, float]] = []
        for key, text in items:
            if not text:
                continue
            v = self._embed(text)
            if v is None:
                continue
            out.append((key, cosine_similarity(q_vec, v)))
        out.sort(key=lambda item: item[1], reverse=True)
        return out

    def best_match(self, query: str, packages: Sequence[Any]) -> Any | None:
        ranked = self.rank(query, packages)
        if not ranked:
            return None
        pkg, sim = ranked[0]
        return pkg if sim >= self._threshold else None

    def score(self, query: str, package: Any) -> float:
        if not query.strip():
            return 0.0
        q_vec = self._embed(query)
        if q_vec is None:
            return 0.0
        doc = self._skill_doc(package)
        if not doc:
            return 0.0
        p_vec = self._embed(doc)
        if p_vec is None:
            return 0.0
        return cosine_similarity(q_vec, p_vec)
