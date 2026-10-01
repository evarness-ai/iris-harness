"""Cross-encoder reranking for the research engine (Phase-2, opt-in).

A BGE cross-encoder re-scores ``(query, result)`` pairs jointly, which is more
accurate than the bi-encoder cosine in :mod:`iris_harness.plugins_builtin.research.rank` but heavier. It is
strictly additive: :func:`build_reranker` returns ``None`` unless the operator opts in
via ``IRIS_RESEARCH_RERANKER``, and the reranker degrades to a no-op whenever the
model can't be loaded. Nothing here ever raises into the engine.

The blend preserves the existing trust/freshness signal: the final score is
``0.7 * sigmoid(cross_score) + 0.3 * old_score`` so a strong cross-encoder match can
reorder without discarding domain authority.
"""

from __future__ import annotations

import logging
import math
import os

from iris_harness.plugins_builtin.research.models import SearchResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-reranker-base"


def _sigmoid(x: float) -> float:
    """Numerically stable logistic squashing, mapping any real into (0, 1)."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class CrossEncoderReranker:
    """Re-score research results in place with a sentence-transformers CrossEncoder.

    The model is loaded lazily on first :meth:`rerank` so importing this module (and
    constructing the reranker) is cheap and side-effect free. If the model can't be
    imported or downloaded, the reranker silently becomes a no-op.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or os.environ.get(
            "IRIS_RESEARCH_RERANKER_MODEL", _DEFAULT_MODEL
        )
        self._model: object | None = None
        self._load_failed = False

    def _ensure_model(self) -> object | None:
        """Lazily construct and cache the CrossEncoder, or ``None`` on any failure."""
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        except Exception:  # missing model/download failure -> no-op
            logger.warning(
                "cross-encoder reranker unavailable (model=%r); leaving order unchanged",
                self._model_name,
                exc_info=True,
            )
            self._load_failed = True
            return None
        return self._model

    def rerank(self, query: str, results: list[SearchResult], *, top_k: int | None = None) -> None:
        """Re-score ``results`` in place against ``query`` and sort descending.

        When ``top_k`` is given, only the first ``top_k`` results (by current score)
        are sent to the cross-encoder, bounding inference cost; the remainder keep
        their existing score. If the model is unavailable, this is a no-op. Never raises.
        """
        if not results:
            return

        model = self._ensure_model()
        if model is None:
            return

        try:
            results.sort(key=lambda r: r.score, reverse=True)
            targets = results[:top_k] if top_k is not None else results
            pairs = [(query, f"{r.title} {r.snippet}") for r in targets]
            scores = model.predict(pairs)  # type: ignore[attr-defined]
            for r, cross_score in zip(targets, scores, strict=True):
                cs = float(cross_score)
                r.metadata["cross_score"] = cs
                r.score = 0.7 * _sigmoid(cs) + 0.3 * r.score
            results.sort(key=lambda r: r.score, reverse=True)
        except Exception:  # rerank must never abort a research turn
            logger.warning("cross-encoder rerank failed; leaving order unchanged", exc_info=True)


def build_reranker() -> CrossEncoderReranker | None:
    """Construct a reranker from ``IRIS_RESEARCH_RERANKER``, or ``None`` if disabled.

    ``""``/``"none"`` -> ``None``; ``"bge"``/``"cross-encoder"`` -> a lazy
    :class:`CrossEncoderReranker`. Any other value is treated as disabled. Never raises.
    """
    mode = os.environ.get("IRIS_RESEARCH_RERANKER", "").strip().lower()
    if mode in {"bge", "cross-encoder"}:
        return CrossEncoderReranker()
    return None


__all__ = ["CrossEncoderReranker", "build_reranker"]
