"""Generic semantic near-duplicate filtering for learning proposals.

Both the behavior miner (digital-twin layer 1) and the intention rollup (layer 3) must
drop candidates that paraphrase each other or something already known. The exact-id PK
dedup in the store only catches identical *normalized* text; this catches rewordings
("daily weather in London" vs "checks London weather each morning") and, crucially,
re-proposals of an item the user already **dismissed** under a slightly different phrasing
— otherwise a rejected proposal returns to the review queue every cycle.

Embedding-based and best-effort: if embeddings are unavailable (the embed call returns an
unusable shape) the candidates pass through unchanged — semantic dedup is never a gate that
silently drops everything.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")

DEFAULT_THRESHOLD = 0.70


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def dedupe_by_text(
    items: Sequence[T],
    *,
    key: Callable[[T], str],
    existing_texts: Sequence[str] = (),
    embed: Callable[[list[str]], list[list[float]]],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[T]:
    """Drop items whose ``key(item)`` is a semantic near-duplicate of a kept item or of an
    existing text (cosine >= ``threshold``).

    ``embed(list[str]) -> list[list[float]]`` is the embedding function (the runtime passes
    ``SemanticIndex.embed``). If it returns nothing usable, items pass through unchanged.
    """
    if not items:
        return []
    cand_texts = [key(it) for it in items]
    existing = [t for t in existing_texts if t.strip()]
    vectors = embed(cand_texts + existing)
    if len(vectors) != len(cand_texts) + len(existing):
        return list(items)  # embeddings unavailable — caller still has exact-id dedup
    cand_vecs = vectors[: len(cand_texts)]
    exist_vecs = vectors[len(cand_texts) :]

    kept: list[T] = []
    kept_vecs: list[list[float]] = []
    for item, vec in zip(items, cand_vecs, strict=True):
        if any(cosine(vec, ref) >= threshold for ref in (*kept_vecs, *exist_vecs)):
            continue
        kept.append(item)
        kept_vecs.append(vec)
    return kept


__all__ = ["DEFAULT_THRESHOLD", "cosine", "dedupe_by_text"]
