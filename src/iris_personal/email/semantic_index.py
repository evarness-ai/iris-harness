"""Semantic email retrieval — a persisted MiniLM/Chroma index over message
snippets (ADR-0071 slice 1).

Stores ONE vector per message — `embed_corpus` (the same sentence-transformers
MiniLM-L6-v2 the triage/kNN pipeline uses) over the same ``sender + subject +
snippet`` text, so query and document vectors share a space — keyed by message id,
plus minimal metadata (``account_id`` + a received-at epoch for filtering). The
index holds **no snippet or body text** (privacy, ADR-0026 / ADR-0071): results are
ids that resolve back to the ``EmailStore``. Local-only; never egresses.

On by default (ADR-0121 PR 4); ``IRIS_EMAIL_SEMANTIC_SEARCH=0`` turns it off. The
``email_semantic_index`` heartbeat keeps it filled in bounded batches
(:func:`refresh_semantic_index`), so a new install and a small VM catch up gradually.
Each vector carries its sender's domain and that domain's parents, so a search can be
scoped to one institution's mail (``search(domains=...)``). ``embed_fn`` is
injectable so tests can pass deterministic vectors without loading the model.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from iris_harness.sdk.activity import chat_in_progress
from iris_harness.sdk.persistence import data_path
from iris_harness.sdk.process_state import track_globals
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore

logger = logging.getLogger(__name__)


def _log_degraded(op: str, exc: BaseException) -> None:
    """Log a search-path failure the caller degrades past (#668 follow-up).

    The agent sees "no results" or an error line; without this, a broken index, store
    or token is invisible to operators. Operation + exception type only, never the
    query or any email content.
    """
    logger.warning("email %s failed (%s); degrading", op, type(exc).__name__, exc_info=True)


_COLLECTION = "email_snippets"
#: Bump when the stored metadata changes; the refresh re-indexes older vectors.
#: 2 = sender domain + parents (ADR-0121 PR 4).
INDEX_VERSION = 2


def semantic_search_enabled() -> bool:
    """On unless ``IRIS_EMAIL_SEMANTIC_SEARCH`` says 0/false/no (ADR-0121 PR 4)."""
    return os.getenv("IRIS_EMAIL_SEMANTIC_SEARCH", "1").strip().lower() not in {"0", "false", "no"}


def _domain_metadata(message: EmailMessage) -> dict[str, str]:
    """The sender's domain and its two- and three-label parents, so a search can ask
    for ``discover.com`` and match mail from ``services.discover.com``."""
    domain = (message.from_domain or "").strip().lower()
    labels = domain.split(".") if domain else []
    return {
        "domain": domain,
        "parent2": ".".join(labels[-2:]) if len(labels) >= 2 else domain,
        "parent3": ".".join(labels[-3:]) if len(labels) >= 3 else domain,
    }


# list[str] -> sequence of vectors (np.ndarray rows or list[list[float]]).
EmbedFn = Callable[[list[str]], Any]


def _embedding_text(message: EmailMessage) -> str:
    """Mirror ``iris_personal.plugins.email_workflows.discovery.CorpusRow.to_embedding_text`` so a query and a
    stored message land in the same vector space (sender first — it carries the most
    category signal)."""
    domain = message.from_domain or ""
    return (
        f"From: {message.from_address} ({domain})\n"
        f"Subject: {message.subject}\n"
        f"Snippet: {message.snippet or ''}"
    )


@dataclass
class EmailSemanticIndex:
    """Persisted vector index over email snippets, backed by local ChromaDB.

    Degrades safely: if ChromaDB is unavailable, ``is_ready`` is False and every
    method no-ops / returns empty, so callers fall back to lexical search.
    """

    persist_dir: Path = field(default_factory=lambda: data_path("email_semantic"))
    embed_fn: EmbedFn | None = None

    def __post_init__(self) -> None:
        self._ok = False
        self._client: Any = None
        self._col: Any = None
        # Tests that don't exercise retrieval skip the model/Chroma entirely.
        if os.environ.get("IRIS_TEST_NULL_EMBEDDINGS"):
            return
        try:
            import chromadb

            from iris_harness.sdk.persistence import (
                collection_kwargs,
            )

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            # Shared embedding function (Phase 3): one ONNX model per process.
            self._col = self._client.get_or_create_collection(_COLLECTION, **collection_kwargs())
            self._ok = True
            logger.info("email semantic index ready (vectors=%d)", self._col.count())
        except Exception:  # degrade to lexical search, never crash
            logger.warning("email semantic index unavailable", exc_info=True)

    @property
    def is_ready(self) -> bool:
        return self._ok

    def count(self) -> int:
        return self._col.count() if self._ok else 0

    def _embed(self, texts: list[str]) -> list[list[float]]:
        fn = self.embed_fn
        if fn is None:
            from iris_harness.sdk.llm import embed_corpus  # lazy model load

            fn = embed_corpus
        return [[float(x) for x in row] for row in fn(texts)]

    def index_messages(self, messages: Iterable[EmailMessage], *, batch_size: int = 128) -> int:
        """Upsert messages (idempotent by id). Returns how many were indexed."""
        if not self._ok:
            return 0

        def _flush(group: list[EmailMessage]) -> int:
            if not group:
                return 0
            self._col.upsert(
                ids=[m.id for m in group],
                embeddings=self._embed([_embedding_text(m) for m in group]),
                metadatas=[
                    {
                        "account_id": m.account_id,
                        "ts": m.received_at.timestamp(),
                        "v": INDEX_VERSION,
                        **_domain_metadata(m),
                    }
                    for m in group
                ],
            )
            return len(group)

        total = 0
        batch: list[EmailMessage] = []
        for message in messages:
            batch.append(message)
            if len(batch) >= batch_size:
                total += _flush(batch)
                batch = []
        total += _flush(batch)
        return total

    def stale_ids(self, ids: list[str]) -> list[str]:
        """Of ``ids``, the ones not indexed yet or indexed by an older version."""
        if not self._ok or not ids:
            return []
        found = self._col.get(ids=ids, include=["metadatas"])
        current = {
            i
            for i, meta in zip(found.get("ids") or [], found.get("metadatas") or [], strict=False)
            if (meta or {}).get("v") == INDEX_VERSION
        }
        return [i for i in ids if i not in current]

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        account_id: str | None = None,
        since: datetime | None = None,
        domains: Iterable[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``(message_id, distance)`` nearest neighbours (smaller = closer).

        ``domains`` scopes the search to mail from those sender domains or their
        subdomains (an institution's ``sender_domains``)."""
        if not self._ok:
            return []
        count = self._col.count()
        if count == 0 or not query.strip():
            return []
        clauses: list[dict[str, Any]] = []
        if account_id:
            clauses.append({"account_id": account_id})
        if since is not None:
            clauses.append({"ts": {"$gte": since.timestamp()}})
        wanted = sorted({d.strip().lower() for d in domains or () if d and d.strip()})
        if wanted:
            clauses.append(
                {"$or": [{key: {"$in": wanted}} for key in ("domain", "parent2", "parent3")]}
            )
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses} if clauses else None

        res = self._col.query(
            query_embeddings=self._embed([query]),
            n_results=min(k, count),
            where=where,
        )
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return list(zip(ids, dists, strict=False))


def backfill_semantic_index(
    *,
    email_store: EmailStore,
    index: EmailSemanticIndex,
    per_account_limit: int = 10_000,
) -> int:
    """Index every stored message across all accounts (idempotent). Returns the count."""
    total = 0
    for account_id in email_store.list_accounts():
        messages = email_store.list_recent(account_id, limit=per_account_limit)
        total += index.index_messages(messages)
    return total


#: Emails embedded between checks for a conversation.
_CHUNK = 50


@dataclass
class RefreshSummary:
    indexed: int = 0
    left: int = 0  # still to index; the next run picks them up
    paused: bool = False  # stopped early for a conversation

    def __str__(self) -> str:
        text = f"indexed {self.indexed} email(s)"
        text += f", {self.left} left for the next run" if self.left else ""
        return text + (" (paused for a conversation)" if self.paused else "")


def refresh_semantic_index(
    *,
    email_store: EmailStore,
    index: EmailSemanticIndex,
    max_per_run: int = 500,
    per_account_limit: int = 10_000,
) -> RefreshSummary:
    """Index the stored mail the index lacks (or holds in an older version), newest
    first, at most ``max_per_run`` per call so a small host catches up over runs."""
    summary = RefreshSummary()
    if not index.is_ready:
        return summary
    stale: list[EmailMessage] = []
    for account_id in email_store.list_accounts():
        messages = email_store.list_recent(account_id, limit=per_account_limit)
        by_id = {m.id: m for m in messages}
        ids = list(by_id)
        for start in range(0, len(ids), 500):
            stale.extend(by_id[i] for i in index.stale_ids(ids[start : start + 500]))
    stale.sort(key=lambda m: m.received_at, reverse=True)
    todo = stale[:max_per_run]
    # Embedding shares the host's CPU with chat: index in small chunks and stop when a
    # conversation starts (foundation/activity.py); the rest waits for the next run.
    for start in range(0, len(todo), _CHUNK):
        if chat_in_progress():
            summary.paused = True
            break
        summary.indexed += index.index_messages(todo[start : start + _CHUNK])
    summary.left = max(0, len(stale) - summary.indexed)
    return summary


# ── Hybrid retrieval (ADR-0071 slice 2) ──────────────────────────────────────


def rrf_fuse(rankings: list[list[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal-rank fusion of several ranked id lists into one ranked list.

    ``score(id) = Σ 1/(k + rank)`` (rank 1-based) across the lists it appears in —
    the standard RRF (k=60). Rewards ids that rank well in *either* list, so lexical
    precision and semantic recall reinforce each other. Returns ``(id, score)``
    highest-first; ties break by id for determinism.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def hybrid_search(
    query: str,
    *,
    email_store: Any,
    semantic_index: EmailSemanticIndex | None,
    k: int = 15,
    account_id: str | None = None,
    since: datetime | None = None,
    category: str | None = None,
) -> list[str]:
    """Fuse lexical (FTS5) + semantic (vector) retrieval → ranked message ids.

    Each leg degrades independently: a malformed FTS query or an unavailable index
    just contributes nothing. Returns [] when both legs are empty (the caller then
    runs its dead-end gate). The semantic leg ignores ``category`` (the index keeps
    no category metadata in slice 1), so category-filtered asks lean lexical.
    """
    lexical_ids: list[str] = []
    try:
        accounts = [account_id] if account_id else email_store.list_accounts()
        for acct in accounts:
            lexical_ids.extend(
                hit.id
                for hit in email_store.search(
                    query, account_id=acct, category_prefix=category, since=since, limit=k
                )
            )
    except Exception as exc:  # noqa: BLE001 — bad FTS syntax etc.; lexical leg just empty
        _log_degraded("lexical search", exc)
        lexical_ids = []

    semantic_ids: list[str] = []
    if semantic_index is not None and semantic_index.is_ready:
        try:
            semantic_ids = [
                mid
                for mid, _ in semantic_index.search(query, k=k, account_id=account_id, since=since)
            ]
        except Exception as exc:  # noqa: BLE001
            _log_degraded("semantic search", exc)
            semantic_ids = []

    if not lexical_ids and not semantic_ids:
        return []
    return [doc_id for doc_id, _ in rrf_fuse([lexical_ids, semantic_ids])[:k]]


# ── Incremental hook: index new mail as the sweep delivers it ────────────────

_LAZY_INDEX: EmailSemanticIndex | None = None


def _lazy_index() -> EmailSemanticIndex:
    global _LAZY_INDEX
    if _LAZY_INDEX is None:
        _LAZY_INDEX = EmailSemanticIndex()
    return _LAZY_INDEX


def _handle_email_new_arrived_index(payload: Any) -> None:
    """Event handler — subscribed via :func:`subscribe_email_semantic_index`.

    Indexes the just-arrived messages so semantic search stays current without a
    backfill. Soft-fails: an indexing hiccup must not disturb the sweep or triage.
    """
    from iris_personal.email.events import EmailNewArrivedPayload  # avoid cycle

    if not isinstance(payload, EmailNewArrivedPayload):
        return
    try:
        index = _lazy_index()
        if not index.is_ready:
            return
        store = EmailStore()
        store.ensure_schema()
        messages = [m for mid in payload.new_message_ids if (m := store.get(mid)) is not None]
        if messages:
            index.index_messages(messages)
    except Exception:  # never break the event chain
        logger.warning("email semantic index: incremental index failed", exc_info=True)


def subscribe_email_semantic_index(bus: Any | None = None) -> None:
    """Wire incremental indexing to ``email.new_arrived`` (off unless the caller opts in)."""
    from iris_harness.sdk.events import get_default_bus
    from iris_personal.email.events import EMAIL_NEW_ARRIVED

    target_bus = bus if bus is not None else get_default_bus()
    target_bus.on(EMAIL_NEW_ARRIVED, _handle_email_new_arrived_index)
    logger.info("email semantic index subscribed to %s", EMAIL_NEW_ARRIVED)


__all__ = [
    "INDEX_VERSION",
    "RefreshSummary",
    "refresh_semantic_index",
    "semantic_search_enabled",
    "EmailSemanticIndex",
    "backfill_semantic_index",
    "subscribe_email_semantic_index",
    "rrf_fuse",
    "hybrid_search",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_LAZY_INDEX")
