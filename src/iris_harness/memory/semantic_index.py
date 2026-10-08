"""ChromaDB-backed semantic index for user facts, learning signals, conversation turns, and wiki pages.

Five collections:
  iris_user_facts        — one doc per UserFact (id = fact key)
  iris_learning_signals  — one doc per LearningSignal (id = signal.id)
  iris_conversation_turns— one doc per stored turn (id = SQLite row id as str)
  iris_wiki_pages        — one doc per WikiPage (id = slug)
  iris_episodic_patterns — one doc per pattern parsed from ``~/.iris/memory/episodic.md``

All collections use ChromaDB's default ONNX MiniLM-L6 embedding function — fully
local, no Ollama dependency, ~80 MB one-time download on first use.

On startup call ``sync_from_store(store)`` to index any rows added since the last
run (tracked via ``data/chroma_watermark``).  During a live session call the
individual ``index_*`` methods after every SQLite write so the two stores stay in
sync without a full rescan.

All public methods are safe to call when ChromaDB failed to initialize (``is_ready``
will be ``False``) — they return empty results silently so the rest of the system
degrades to keyword ranking.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from iris_harness.foundation.logsafe import log_safe
from iris_harness.foundation.persistence import data_path

if TYPE_CHECKING:
    from .store import LearningSignal, MemoryStore, UserFact

logger = logging.getLogger(__name__)

_FACTS_COLLECTION = "iris_user_facts"
_SIGNALS_COLLECTION = "iris_learning_signals"
_TURNS_COLLECTION = "iris_conversation_turns"
_WIKI_COLLECTION = "iris_wiki_pages"
_EPISODIC_COLLECTION = "iris_episodic_patterns"
_WATERMARK_FILENAME = "chroma_watermark"


@dataclass(frozen=True)
class RetrievedTurn:
    """A past conversation turn surfaced by semantic recall, with its provenance.

    ``turn_id`` is the originating turn's correlation id (None for turns indexed
    before turn-id provenance shipped); ``row_id`` is the stable Chroma/SQLite id.
    Lets the learning loop attribute a ``downstream_reuse`` outcome back to the
    turn whose content resurfaced (learning-observability.md §4.2).
    """

    row_id: str
    session_id: str
    role: str
    content: str
    turn_id: str | None = None


@dataclass
class SemanticIndex:
    """Semantic retrieval layer backed by a local ChromaDB persistent store."""

    persist_dir: Path = field(default_factory=lambda: data_path("chroma"))

    def __post_init__(self) -> None:
        self._ok = False
        self._client: Any = None
        self._facts: Any = None
        self._signals: Any = None
        self._turns: Any = None
        self._wiki: Any = None
        self._episodic: Any = None
        self._embedder: Any = None  # lazy DefaultEmbeddingFunction for standalone embed()
        self._watermark_path = self.persist_dir.parent / _WATERMARK_FILENAME
        # Tests set IRIS_TEST_NULL_EMBEDDINGS=1 to skip the ~80 MB ONNX model load.
        # Public methods all guard on is_ready, so callers degrade to keyword ranking.
        if os.environ.get("IRIS_TEST_NULL_EMBEDDINGS"):
            return
        try:
            import chromadb

            from iris_harness.foundation.persistence.embedding import (
                collection_kwargs,
            )

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            # Pin the shared embedding function so all 5 collections reuse one
            # ONNX model instead of loading ~80 MB each (Phase 3 footprint).
            ef = collection_kwargs()
            self._facts = self._client.get_or_create_collection(_FACTS_COLLECTION, **ef)
            self._signals = self._client.get_or_create_collection(_SIGNALS_COLLECTION, **ef)
            self._turns = self._client.get_or_create_collection(_TURNS_COLLECTION, **ef)
            self._wiki = self._client.get_or_create_collection(_WIKI_COLLECTION, **ef)
            self._episodic = self._client.get_or_create_collection(_EPISODIC_COLLECTION, **ef)
            self._ok = True
            logger.info(
                "semantic index ready (facts=%d signals=%d turns=%d wiki=%d episodic=%d)",
                self._facts.count(),
                self._signals.count(),
                self._turns.count(),
                self._wiki.count(),
                self._episodic.count(),
            )
        except Exception:
            logger.warning(
                "semantic index unavailable — falling back to keyword ranking", exc_info=True
            )

    @property
    def is_ready(self) -> bool:
        return self._ok

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_fact(self, fact: UserFact) -> None:
        """Upsert a user fact into the semantic index."""
        if not self._ok:
            return
        try:
            self._facts.upsert(
                ids=[fact.key],
                documents=[f"{fact.key}: {fact.value}"],
                metadatas=[
                    {
                        "confidence": fact.confidence,
                        "source": fact.source,
                        "last_confirmed": fact.last_confirmed.isoformat(),
                    }
                ],
            )
        except Exception:
            logger.debug("semantic index: failed to index fact %r", fact.key, exc_info=True)

    def index_signal(self, signal: LearningSignal) -> None:
        """Upsert a learning signal into the semantic index."""
        if not self._ok:
            return
        try:
            doc = f"{signal.query} {signal.domain} {signal.outcome}"
            self._signals.upsert(
                ids=[signal.id],
                documents=[doc],
                metadatas=[{"domain": signal.domain, "agent_type": signal.agent_type}],
            )
        except Exception:
            logger.debug("semantic index: failed to index signal %r", signal.id, exc_info=True)

    def index_turn(
        self,
        row_id: int,
        session_id: str,
        role: str,
        content: str,
        *,
        turn_id: str | None = None,
    ) -> None:
        """Upsert a conversation turn. row_id is the SQLite primary key.

        ``turn_id`` (the correlation id from the live turn) is stored in metadata
        when provided so a later ``downstream_reuse`` signal can be attributed to
        the originating turn. Backfilled rows omit it (None).
        """
        if not self._ok or not content.strip():
            return
        metadata: dict[str, Any] = {"session_id": session_id, "role": role}
        if turn_id:
            metadata["turn_id"] = turn_id
        try:
            self._turns.upsert(
                ids=[str(row_id)],
                documents=[content],
                metadatas=[metadata],
            )
        except Exception:
            logger.debug("semantic index: failed to index turn %d", row_id, exc_info=True)

    def drop_fact(self, key: str) -> None:
        """Remove a fact from the semantic index."""
        if not self._ok:
            return
        try:
            self._facts.delete(ids=[key])
        except Exception:
            logger.debug("semantic index: failed to drop fact %r", log_safe(key), exc_info=True)

    def drop_turns(self, row_ids: list[int]) -> int:
        """Remove conversation turns from the index by SQLite row id.

        The other half of a delete: a row removed from SQLite whose vector stayed
        behind kept coming back through cross-session recall.
        """
        if not self._ok or not row_ids:
            return 0
        try:
            self._turns.delete(ids=[str(i) for i in row_ids])
        except Exception:
            logger.debug("semantic index: failed to drop %d turns", len(row_ids), exc_info=True)
            return 0
        return len(row_ids)

    def turn_ids(self) -> set[str]:
        """Row ids currently held in the turn collection (orphan sweep)."""
        if not self._ok:
            return set()
        try:
            got = self._turns.get(include=[])
        except Exception:
            logger.debug("semantic index: failed to list turn ids", exc_info=True)
            return set()
        return {str(i) for i in (got.get("ids") or [])}

    def fact_keys(self) -> set[str]:
        """Return the set of fact keys currently held in the semantic index.

        Lets the coherence checker diff the derived index against the SQLite
        truth without pulling documents/embeddings.
        """
        if not self._ok:
            return set()
        try:
            return set(self._facts.get(include=[]).get("ids") or [])
        except Exception:
            logger.debug("semantic index: failed to list fact keys", exc_info=True)
            return set()

    def reconcile_facts(self, store: MemoryStore) -> dict[str, int]:
        """Make the facts collection exactly match the SQLite truth.

        Unlike ``sync_from_store`` (additive — it only upserts facts that exist),
        this also DROPS orphan index entries whose key no longer exists in the
        store. That orphan-drop is the leg a ``forget`` needs but the additive
        sync never had, so a forgotten fact stops resurfacing in semantic recall.
        Returns ``{"indexed", "dropped"}``.
        """
        if not self._ok:
            return {"indexed": 0, "dropped": 0}
        # Only confirmed facts belong in the recall index: an unconfirmed one is a
        # review-queue item, and an indexed key reaches the model through
        # memory_search even when the retriever's gate would have dropped it.
        # One entry per key: a key with several values (two cards) lists them all.
        store_facts = store.fetch_fact_projections(confirmed_only=True)
        store_keys = {f.key for f in store_facts}
        for fact in store_facts:
            self.index_fact(fact)
        orphans = self.fact_keys() - store_keys
        for key in orphans:
            self.drop_fact(key)
        return {"indexed": len(store_facts), "dropped": len(orphans)}

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_facts(self, query: str, *, n: int = 10) -> list[str]:
        """Return fact keys ranked by semantic similarity to query."""
        if not self._ok or not query.strip():
            return []
        try:
            count = self._facts.count()
            if count == 0:
                return []
            results = self._facts.query(
                query_texts=[query],
                n_results=min(n, count),
                include=["metadatas"],
            )
            ids: list[str] = results.get("ids", [[]])[0]
            return ids
        except Exception:
            logger.warning("semantic index: fact query failed", exc_info=True)
            return []

    def query_facts_text(
        self, query: str, *, n: int = 10, max_distance: float | None = None
    ) -> list[str]:
        """Return fact DOCUMENTS ("key: value") ranked by similarity — for surfaces
        that show the fact to a human/agent (the retriever uses ``query_facts`` for
        keys). Issue 0021.

        ``max_distance`` (issue 0032) drops results that aren't actually relevant: a
        query with no good match ("number of holdings") otherwise returns the nearest
        facts regardless of distance, dumping unrelated PII (the user's email) to chat.
        With the cutoff, an off-topic query returns nothing instead of junk; relevant
        matches (distance well below the cutoff) still come through."""
        if not self._ok or not query.strip():
            return []
        try:
            count = self._facts.count()
            if count == 0:
                return []
            include = ["documents"] + (["distances"] if max_distance is not None else [])
            results = self._facts.query(
                query_texts=[query],
                n_results=min(n, count),
                include=include,
            )
            docs: list[str] = results.get("documents", [[]])[0]
            if max_distance is None:
                return [d for d in docs if d]
            distances: list[float] = results.get("distances", [[]])[0]
            return [
                d for d, dist in zip(docs, distances, strict=False) if d and dist <= max_distance
            ]
        except Exception:
            logger.warning("semantic index: fact-text query failed", exc_info=True)
            return []

    def query_signals(self, query: str, *, n: int = 5) -> list[str]:
        """Return signal IDs ranked by semantic similarity to query."""
        if not self._ok or not query.strip():
            return []
        try:
            count = self._signals.count()
            if count == 0:
                return []
            results = self._signals.query(
                query_texts=[query],
                n_results=min(n, count),
                include=["metadatas"],
            )
            ids: list[str] = results.get("ids", [[]])[0]
            return ids
        except Exception:
            logger.warning("semantic index: signal query failed", exc_info=True)
            return []

    def query_turns(
        self,
        query: str,
        *,
        exclude_session: str | None = None,
        n: int = 4,
    ) -> list[tuple[str, str]]:
        """Return (role, content) pairs from past turns semantically relevant to query.

        Turns from ``exclude_session`` are filtered out — the current session's
        recent turns are already injected by the caller; this is for cross-session recall.
        """
        return [
            (t.role, t.content)
            for t in self.query_turns_detailed(query, exclude_session=exclude_session, n=n)
        ]

    def query_turns_detailed(
        self,
        query: str,
        *,
        exclude_session: str | None = None,
        only_session: str | None = None,
        n: int = 4,
        max_distance: float | None = None,
    ) -> list[RetrievedTurn]:
        """Like :meth:`query_turns` but carrying each turn's provenance.

        Returns :class:`RetrievedTurn` (row_id + turn_id + session_id), so a
        caller can record that an earlier turn's content was reused in a later
        turn's context (the ``downstream_reuse`` observable).
        """
        if not self._ok or not query.strip():
            return []
        try:
            count = self._turns.count()
            if count == 0:
                return []
            # Fetch more candidates so we have room to filter by session
            fetch_n = min(n * 3, count)
            where: dict[str, Any] | None = None
            if only_session:
                where = {"session_id": only_session}
            elif exclude_session:
                where = {"session_id": {"$ne": exclude_session}}
            include = ["documents", "metadatas"]
            if max_distance is not None:
                include.append("distances")
            kwargs: dict[str, Any] = {
                "query_texts": [query],
                "n_results": fetch_n,
                "include": include,
            }
            if where:
                kwargs["where"] = where
            results = self._turns.query(**kwargs)
            ids: list[str] = results.get("ids", [[]])[0]
            docs: list[str] = results.get("documents", [[]])[0]
            metas: list[dict[str, Any]] = results.get("metadatas", [[]])[0]
            # Without a cutoff the nearest turns come back however far away they are —
            # the same trap issue 0032 fixed for facts.
            dists: list[float] = (
                results.get("distances", [[]])[0] if max_distance is not None else []
            )
            out: list[RetrievedTurn] = []
            for idx, (row_id, doc, meta) in enumerate(zip(ids, docs, metas, strict=False)):
                if max_distance is not None:
                    if idx >= len(dists) or dists[idx] > max_distance:
                        continue
                turn_id = meta.get("turn_id")
                out.append(
                    RetrievedTurn(
                        row_id=str(row_id),
                        session_id=str(meta.get("session_id") or ""),
                        role=str(meta.get("role") or "assistant"),
                        content=doc,
                        turn_id=str(turn_id) if turn_id else None,
                    )
                )
                if len(out) >= n:
                    break
            return out
        except Exception:
            logger.warning("semantic index: turn query failed", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Wiki page indexing
    # ------------------------------------------------------------------

    def index_wiki_page(
        self,
        slug: str,
        page_type: str,
        title: str,
        body: str,
        last_updated: str = "",
    ) -> None:
        """Upsert a wiki page. Uses slug as the stable ChromaDB ID.

        The document is ``title + first 800 chars of body`` so the embedding
        captures the semantic essence without ballooning on large activity logs.
        """
        if not self._ok or not slug.strip():
            return
        try:
            document = f"{title}\n\n{body[:800]}"
            self._wiki.upsert(
                ids=[slug],
                documents=[document],
                metadatas=[{"slug": slug, "page_type": page_type, "last_updated": last_updated}],
            )
        except Exception:
            logger.debug("semantic index: failed to index wiki page %r", slug, exc_info=True)

    def drop_wiki_page(self, slug: str) -> None:
        """Remove a wiki page from the semantic index."""
        if not self._ok:
            return
        try:
            self._wiki.delete(ids=[slug])
        except Exception:
            logger.debug("semantic index: failed to drop wiki page %r", slug, exc_info=True)

    def query_wiki(self, query: str, *, n: int = 5) -> list[tuple[str, str]]:
        """Return (slug, page_type) pairs ranked by semantic similarity to query."""
        if not self._ok or not query.strip():
            return []
        try:
            count = self._wiki.count()
            if count == 0:
                return []
            results = self._wiki.query(
                query_texts=[query],
                n_results=min(n, count),
                include=["metadatas"],
            )
            metas: list[dict[str, Any]] = results.get("metadatas", [[]])[0]
            return [(m["slug"], m["page_type"]) for m in metas]
        except Exception:
            logger.warning("semantic index: wiki query failed", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Episodic patterns (Layer 3)
    # ------------------------------------------------------------------

    def index_episodic(self, pattern_id: str, text: str) -> None:
        """Upsert an episodic pattern. ``pattern_id`` is the stable hash from the loader."""
        if not self._ok or not text.strip():
            return
        try:
            self._episodic.upsert(ids=[pattern_id], documents=[text])
        except Exception:
            logger.debug("semantic index: failed to index episodic %r", pattern_id, exc_info=True)

    def drop_episodic(self, pattern_id: str) -> None:
        """Remove an episodic pattern from the index."""
        if not self._ok:
            return
        try:
            self._episodic.delete(ids=[pattern_id])
        except Exception:
            logger.debug("semantic index: failed to drop episodic %r", pattern_id, exc_info=True)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` with the same model the collections use (MiniLM).

        Returns one vector per input, or ``[]`` on failure / when embeddings are stubbed
        (``IRIS_TEST_NULL_EMBEDDINGS``). Standalone of the persistent client, so callers
        (e.g. behavior-pattern semantic dedup) can compare texts without indexing them.
        """
        if not texts or os.environ.get("IRIS_TEST_NULL_EMBEDDINGS"):
            return []
        try:
            if self._embedder is None:
                from iris_harness.foundation.persistence.embedding import (
                    default_embedding_function,
                )

                self._embedder = default_embedding_function()
            if self._embedder is None:
                return []
            return [list(v) for v in self._embedder(texts)]
        except Exception:
            logger.warning("semantic index: embed failed", exc_info=True)
            return []

    def query_episodic(self, query: str, *, n: int = 5) -> list[str]:
        """Return episodic pattern texts ranked by semantic similarity to ``query``."""
        if not self._ok or not query.strip():
            return []
        try:
            count = self._episodic.count()
            if count == 0:
                return []
            results = self._episodic.query(
                query_texts=[query],
                n_results=min(n, count),
                include=["documents"],
            )
            docs: list[str] = results.get("documents", [[]])[0]
            return [d for d in docs if d]
        except Exception:
            logger.warning("semantic index: episodic query failed", exc_info=True)
            return []

    def sync_episodic_patterns(self, patterns: Sequence[tuple[str, str]]) -> int:
        """Bulk-index episodic patterns at startup. Each tuple: (pattern_id, text).

        Reconciles the index with the file by deleting patterns that are no
        longer present, so the index can never grow stale relative to the
        user-edited markdown.
        """
        if not self._ok:
            return 0
        try:
            existing_ids: set[str] = set(self._episodic.get(include=[]).get("ids", []))
        except Exception:  # index what we have; stale ids stay
            logger.warning(
                "semantic index: could not list episodic ids; stale patterns are not dropped",
                exc_info=True,
            )
            existing_ids = set()
        wanted_ids = {pid for pid, _ in patterns}
        stale = existing_ids - wanted_ids
        if stale:
            try:
                self._episodic.delete(ids=list(stale))
            except Exception:
                logger.debug("semantic index: failed to drop stale episodic ids", exc_info=True)
        count = 0
        for pid, text in patterns:
            self.index_episodic(pid, text)
            count += 1
        if count or stale:
            logger.info(
                "semantic index: synced %d episodic patterns (dropped %d stale)",
                count,
                len(stale),
            )
        return count

    # ------------------------------------------------------------------
    # Wiki bulk sync
    # ------------------------------------------------------------------

    def sync_wiki_pages(self, pages: Sequence[tuple[str, str, str, str, str]]) -> int:
        """Bulk-index wiki pages at startup. Each tuple: (slug, page_type, title, body, last_updated).

        Returns the count of pages indexed.  Idempotent — safe to call on every restart.
        """
        if not self._ok:
            return 0
        count = 0
        for slug, page_type, title, body, last_updated in pages:
            self.index_wiki_page(slug, page_type, title, body, last_updated)
            count += 1
        if count:
            logger.info("semantic index: synced %d wiki pages", count)
        return count

    # ------------------------------------------------------------------
    # Bulk sync from SQLite (called at startup)
    # ------------------------------------------------------------------

    def sync_from_store(self, store: MemoryStore) -> int:
        """Index all facts, signals, and new turns from the SQLite store.

        Returns the number of new turns indexed.  Uses a watermark file to
        avoid re-scanning turns already indexed in a previous run.
        """
        if not self._ok:
            return 0

        # Facts — always re-sync (small set, ensures freshness after manual edits).
        # Confirmed only, and unconfirmed keys are dropped, so revoking a fact's
        # confirmation actually removes it from recall.
        confirmed = store.fetch_fact_projections(confirmed_only=True)
        for fact in confirmed:
            self.index_fact(fact)
        stale = self.fact_keys() - {f.key for f in confirmed}
        for key in stale:
            self.drop_fact(key)

        # Signals — always re-sync
        for signal in store.fetch_learning_signals():
            self.index_signal(signal)

        # Turns — incremental via watermark
        watermark = self._load_watermark()
        new_turns = store.load_turns_since(min_id=watermark)
        for row_id, session_id, role, content in new_turns:
            self.index_turn(row_id, session_id, role, content)
        if new_turns:
            max_id = max(row_id for row_id, *_ in new_turns)
            self._save_watermark(max_id)
            logger.info(
                "semantic index: synced %d new turns (watermark → %d)", len(new_turns), max_id
            )

        return len(new_turns)

    def _load_watermark(self) -> int:
        try:
            return int(self._watermark_path.read_text().strip())
        except (FileNotFoundError, ValueError):
            return 0

    def _save_watermark(self, value: int) -> None:
        try:
            self._watermark_path.write_text(str(value))
        except Exception:
            logger.debug("semantic index: failed to save watermark", exc_info=True)
