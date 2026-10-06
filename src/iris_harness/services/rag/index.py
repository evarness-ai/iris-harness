"""Vector index for document chunks (RAG R0).

A dedicated ChromaDB collection (``iris_documents``) on its own persist dir
(``data/chroma_docs``) so the document layer never contends with the memory
``SemanticIndex`` client. Degrades gracefully: when ChromaDB is unavailable
(or ``IRIS_TEST_NULL_EMBEDDINGS`` is set) ``is_ready`` is False and callers
fall back to the store's keyword search.

The index is a projection, not a source of truth: the canonical chunk text lives in
``DocumentStore`` (``data/rag.db``) and the documents themselves stay where the user
keeps them. Losing or corrupting ``data/chroma_docs`` loses nothing: ``rebuild`` (driven
by ``iris_harness.services.rag.ingest.reindex_all``, ``iris docs reindex``) recreates
its contents from the store's chunks. If the persisted collection cannot be opened at all
(the embedding model changed, so Chroma refuses it), ``reset_collection`` deletes and
re-creates it (``iris docs reindex --reset-collection``). Re-running ``sync_all`` is NOT a rebuild: it skips
every file whose mtime/hash is unchanged, so it never re-populates an empty index.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from iris_harness.foundation.persistence import data_path
from iris_harness.services.rag.models import DocumentChunk

logger = logging.getLogger(__name__)

_COLLECTION = "iris_documents"
_REBUILD_BATCH = 500  # chunks per upsert; well under Chroma's per-call maximum


@dataclass
class DocumentIndex:
    persist_dir: Path = field(default_factory=lambda: data_path("chroma_docs"))

    def __post_init__(self) -> None:
        self._ok = False
        self._client: Any = None
        self._col: Any = None
        # Why the collection could not be opened (None when it opened or Chroma is not in
        # use), and whether the cause is the embedding-model conflict `reset_collection` fixes.
        self.open_error: str | None = None
        self.embedder_conflict = False
        if os.environ.get("IRIS_TEST_NULL_EMBEDDINGS"):
            return
        try:
            import chromadb

            from iris_harness.foundation.persistence.embedding import (
                collection_kwargs,
            )

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            # Cosine space → similarity scores land in a sane ~[0,1] range for
            # citations (Chroma's default L2 yields unbounded, sometimes negative
            # "1 - distance"). Only applied when the collection is first created.
            # Shared embedding function (Phase 3): one ONNX model per process.
            self._col = self._client.get_or_create_collection(
                _COLLECTION, metadata={"hnsw:space": "cosine"}, **collection_kwargs()
            )
            self._ok = True
            logger.info("document index ready (chunks=%d)", self._col.count())
        except Exception as exc:
            self.open_error = str(exc)
            self.embedder_conflict = (
                self._client is not None
                and isinstance(exc, ValueError)
                and "embedding function" in str(exc).lower()
            )
            logger.warning("document index unavailable — keyword fallback", exc_info=True)

    @property
    def is_ready(self) -> bool:
        return self._ok

    def _upsert(self, chunks: Sequence[DocumentChunk]) -> None:
        self._col.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source_id": c.source_id,
                    "source_path": c.source_path,
                    "title": c.title,
                    "chunk_index": c.chunk_index,
                    # Chroma metadata values must be scalars (no None) —
                    # only present when the chunk carries a classification.
                    **(
                        {"classification": c.classification} if c.classification is not None else {}
                    ),
                }
                for c in chunks
            ],
        )

    def count(self) -> int:
        """Number of entries in the collection (0 when the index is unavailable)."""
        if not self._ok:
            return 0
        return int(self._col.count())

    def index_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        if not self._ok or not chunks:
            return
        try:
            self._upsert(chunks)
        except Exception:
            logger.debug("document index: upsert failed", exc_info=True)

    def rebuild(self, chunks: Iterable[DocumentChunk]) -> int:
        """Make the collection hold exactly ``chunks``; return how many were indexed.

        Entries whose id is not among ``chunks`` are removed, the rest are upserted. The
        collection itself is kept (not dropped and re-created), so another process holding
        the same collection, such as a running server, stays valid across a rebuild.

        Unlike ``index_chunks`` this raises on a Chroma failure: a rebuild is an explicit
        repair, and reporting success over a half-built index would hide the problem.
        Returns 0 without touching anything when the index is unavailable (check
        ``is_ready`` first to tell that apart from an empty store).
        """
        if not self._ok:
            return 0
        wanted: set[str] = set()
        total = 0
        batch: list[DocumentChunk] = []
        for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= _REBUILD_BATCH:
                self._upsert(batch)
                wanted.update(c.id for c in batch)
                total += len(batch)
                batch = []
        if batch:
            self._upsert(batch)
            wanted.update(c.id for c in batch)
            total += len(batch)
        stale = [cid for cid in self._col.get(include=[])["ids"] if cid not in wanted]
        for i in range(0, len(stale), _REBUILD_BATCH):
            self._col.delete(ids=stale[i : i + _REBUILD_BATCH])
        return total

    def reset_collection(self) -> None:
        """Delete the docs collection and create it again, empty, under the current embedder.

        The repair for a collection Chroma will not open (a persisted one made under a
        different embedding model). Touches only ``iris_documents``: no other collection of
        the client, not ``rag.db``, no source file. The caller refills it with ``rebuild``.

        A reset invalidates any other handle on the old collection (a running server's
        ``app.state.rag_handles`` index): that handle keeps pointing at the deleted
        collection until its process restarts, so queries fall back to keyword search and
        new chunks are not indexed. ``rebuild`` keeps handles valid; this does not.

        Raises ``RuntimeError`` when Chroma itself is not in use (nothing to reset).
        """
        if self._client is None:
            raise RuntimeError(
                "the vector index cannot be reset: ChromaDB is not available "
                "(or embeddings are disabled)"
            )
        from iris_harness.foundation.persistence.embedding import collection_kwargs

        existing = {getattr(c, "name", c) for c in self._client.list_collections()}
        if _COLLECTION in existing:
            self._client.delete_collection(_COLLECTION)
        self._col = self._client.get_or_create_collection(
            _COLLECTION, metadata={"hnsw:space": "cosine"}, **collection_kwargs()
        )
        self._ok = True
        self.open_error = None
        self.embedder_conflict = False

    def delete_source(self, source_id: str) -> None:
        if not self._ok:
            return
        try:
            self._col.delete(where={"source_id": source_id})
        except Exception:
            logger.debug("document index: delete failed for %s", source_id, exc_info=True)

    def query(self, text: str, *, n: int = 5) -> list[tuple[str, float]]:
        """Return (chunk_id, score) pairs ranked by similarity (score = 1 - distance)."""
        if not self._ok or not text.strip():
            return []
        try:
            count = self._col.count()
            if count == 0:
                return []
            res = self._col.query(
                query_texts=[text],
                n_results=min(n, count),
                include=["distances"],
            )
            ids: list[str] = res.get("ids", [[]])[0]
            dists: list[float] = res.get("distances", [[]])[0]
            return [(cid, 1.0 - float(d)) for cid, d in zip(ids, dists, strict=False)]
        except Exception:
            logger.warning("document index: query failed", exc_info=True)
            return []


__all__ = ["DocumentIndex"]
