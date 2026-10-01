"""Cited document retrieval (RAG R0/R3).

Vector search when the index is ready, deterministic keyword fallback over
the store otherwise. Every hit is a ``RetrievedChunk`` with a ``Citation`` to
the originating file. R3 adds an optional ``tags`` filter — a lightweight
"notebook" scope that restricts retrieval to sources carrying those tags.
"""

from __future__ import annotations

from collections.abc import Callable

from iris_harness.services.rag.index import DocumentIndex
from iris_harness.services.rag.models import Citation, DocumentChunk, RetrievedChunk
from iris_harness.services.rag.store import DocumentStore


def _cite(chunk: DocumentChunk) -> Citation:
    return Citation(
        source_path=chunk.source_path,
        title=chunk.title,
        chunk_index=chunk.chunk_index,
        page=chunk.page,
    )


def _tag_filter(store: DocumentStore, tags: tuple[str, ...]) -> Callable[[str], bool]:
    """Build a predicate: does a chunk's source carry any of ``tags``?"""
    wanted = {t.lower() for t in tags}
    cache: dict[str, set[str]] = {}

    def _ok(source_id: str) -> bool:
        if not wanted:
            return True
        if source_id not in cache:
            src = store.get_source(source_id)
            cache[source_id] = {t.lower() for t in (src.tags if src else ())}
        return bool(wanted & cache[source_id])

    return _ok


def search_documents(
    query: str,
    *,
    store: DocumentStore,
    index: DocumentIndex | None = None,
    limit: int = 5,
    tags: tuple[str, ...] = (),
) -> list[RetrievedChunk]:
    """Return the top chunks for ``query`` with citations, optionally scoped to
    sources tagged with any of ``tags``. Empty if nothing matches."""
    if not query.strip():
        return []

    ok = _tag_filter(store, tags)
    # Over-fetch when filtering so the post-filter still has `limit` to return.
    fetch = limit if not tags else max(limit * 4, 20)

    hits: list[RetrievedChunk] = []
    if index is not None and index.is_ready:
        for chunk_id, score in index.query(query, n=fetch):
            chunk = store.get_chunk(chunk_id)
            if chunk is not None and ok(chunk.source_id):
                hits.append(
                    RetrievedChunk(
                        chunk_id=chunk.id,
                        text=chunk.text,
                        score=score,
                        citation=_cite(chunk),
                        classification=chunk.classification,
                    )
                )
            if len(hits) >= limit:
                break
        if hits:
            return hits

    # Fallback (no vector index, or it returned nothing): keyword search.
    for chunk, score in store.keyword_search(query, limit=fetch):
        if ok(chunk.source_id):
            hits.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    text=chunk.text,
                    score=score,
                    citation=_cite(chunk),
                    classification=chunk.classification,
                )
            )
        if len(hits) >= limit:
            break
    return hits


__all__ = ["search_documents"]
