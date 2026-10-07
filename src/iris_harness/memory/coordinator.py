"""Single write-seam for user-fact mutations across the three fact-homes.

A user fact lives in three representations:

* the SQLite store (``memory.db``) — the **source of truth** plus the reversible
  ``user_fact_history`` trail;
* the ChromaDB semantic index — a **derived** recall index;
* the ``## Auto-detected`` block of ``USER.md`` — a **derived**, prompt-loaded
  projection.

Historically the *write* path fanned out to all three (``_persist_fact``) but the
*forget/correct/restore* paths only touched the store, silently desyncing the two
derived homes — the drift that forced facts to be purged in three places by hand.
``FactCoordinator`` is the one seam every mutation goes through, so the derived
homes cannot drift from the truth on a mutation.

The store write is authoritative and runs first; derived-home updates are
best-effort (logged, never raised) so a Chroma/markdown hiccup can never lose a
fact from the truth. Any residual drift (un-migrated call sites, a failed derived
write, a hand-edited DB) is the job of ``iris memory doctor`` (``coherence.repair``).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from iris_harness.foundation.logsafe import log_safe
from iris_harness.memory.identity import append_user_fact_to_md, drop_user_fact_from_md
from iris_harness.memory.semantic_index import SemanticIndex
from iris_harness.memory.store import MemoryStore, UserFact

logger = logging.getLogger(__name__)


class FactCoordinator:
    """Owns every user-fact mutation, fanning out to all three fact-homes.

    ``index`` is optional (semantic recall degrades gracefully when ChromaDB is
    unavailable). ``project_md`` defaults to True; pass False to suppress the
    ``USER.md`` projection (e.g. headless/test contexts that don't want to touch
    the identity workspace).
    """

    def __init__(
        self,
        store: MemoryStore,
        index: SemanticIndex | None = None,
        *,
        project_md: bool = True,
    ) -> None:
        self._store = store
        self._index = index
        self._project_md = project_md

    # -- mutations ---------------------------------------------------------

    def record(
        self, key: str, value: str, confidence: float, source: str, *, confirmed: bool = False
    ) -> UserFact:
        """Upsert a fact (preserving first-seen / confirm count) and derive the rest.

        The derived homes are built from what the store ACTUALLY holds after the
        upsert, not from the incoming value. The store's confidence gate can refuse a
        write; this method used to index and project the refused value anyway, so
        ChromaDB and USER.md could disagree with the truth (ADR-0105 drift).
        """
        now = datetime.now(UTC)
        existing = self._store.fetch_user_fact(key)
        fact = UserFact(
            key=key,
            value=value,
            confidence=confidence,
            source=source,
            first_seen=existing.first_seen if existing else now,
            last_confirmed=now,
            times_confirmed=(existing.times_confirmed + 1) if existing else 1,
            confirmed=confirmed or bool(existing and existing.confirmed),
        )
        self._store.upsert_user_fact(fact)
        stored = self._store.fetch_user_fact(key) or fact
        if stored.value != value:
            logger.info(
                "fact write blocked by the confidence gate: key=%s kept=%r rejected=%r",
                key,
                stored.value,
                value,
            )
        # The index is the recall path and USER.md reads as the user's own profile, so
        # both hold confirmed facts only (a 0.3 mis-extraction once landed in the file);
        # both are rebuilt for the key from everything it holds (_rederive).
        self._rederive(key)
        logger.info(
            "recorded user fact: %s = %r (source=%s confirmed=%s)",
            key,
            stored.value,
            source,
            stored.confirmed,
        )
        return stored

    def approve_proposal(self, proposal_id: str) -> UserFact | None:
        """Approve a queued proposal: it becomes a confirmed fact, everywhere."""
        proposal = self._store.fetch_fact_proposal(proposal_id)
        if proposal is None or proposal.status != "pending":
            return None
        if proposal.subject is not None:
            # About someone else (one hop away): confirm that statement, and nothing
            # else — the owner's fact, USER.md and the recall index are about the owner.
            if not self._store.resolve_fact_proposal(proposal_id, "approved"):
                return None
            now = datetime.now(UTC)
            return UserFact(
                key=proposal.key,
                value=proposal.value,
                confidence=max(proposal.confidence, 1.0),
                source=f"{proposal.source}:approved",
                first_seen=proposal.created_at,
                last_confirmed=now,
                times_confirmed=proposal.seen_count,
                confirmed=True,
            )
        fact = self.record(
            proposal.key,
            proposal.value,
            # An approved fact is the owner's word, so it outranks whatever the
            # extractor guessed — otherwise the store's confidence gate could refuse
            # the very value the owner just approved.
            max(proposal.confidence, 1.0),
            f"{proposal.source}:approved",
            confirmed=True,
        )
        self._store.resolve_fact_proposal(proposal_id, "approved")
        return fact

    def reject_proposal(self, proposal_id: str) -> bool:
        """Reject a queued proposal. Nothing is written to the fact store."""
        return self._store.resolve_fact_proposal(proposal_id, "rejected")

    def confirm(self, key: str) -> UserFact | None:
        """Confirm a fact already in the store (the legacy rows queue up this way)."""
        if not self._store.set_fact_confirmed(key, True):
            return None
        fact = self._store.fetch_user_fact(key)
        self._rederive(key)
        return fact

    def forget(self, key: str, *, statement_id: str | None = None) -> bool:
        """Forget a fact — or, with ``statement_id``, one value of a key that holds
        several (one of two cards) — from the store and both derived homes."""
        removed = self._store.delete_user_fact(key, statement_id=statement_id)
        if removed:
            self._rederive(key)
        return removed

    def correct(
        self,
        key: str,
        value: str,
        *,
        confidence: float = 1.0,
        statement_id: str | None = None,
    ) -> bool:
        """Explicitly set a fact's value — with ``statement_id``, replace that one value
        of a key that holds several instead of adding beside it — then re-derive."""
        replaced = self._store.correct_user_fact(
            key, value, confidence=confidence, statement_id=statement_id
        )
        self._rederive(key)
        return replaced

    def restore(self, key: str) -> str | None:
        """Restore a fact's prior value, re-deriving both derived homes."""
        restored = self._store.restore_user_fact(key)
        if restored is not None:
            self._rederive(key)
        return restored

    def rederive(self, key: str) -> None:
        """Rebuild both derived homes for ``key`` after a change made outside this seam
        (a removal that withdrew or restored its statements, ADR-0119)."""
        self._rederive(key)

    # -- derived-home helpers (best-effort, never raise) -------------------

    def _rederive(self, key: str) -> None:
        """Rebuild the recall index and USER.md for ``key`` from what the store holds.

        Both homes keep a key once. A key with several confirmed values (two cards) is
        one line listing them all; with none, the key leaves both homes. Writing the
        value just changed instead — the old way — kept only the last card, and a
        forget of one card wiped the line for the other.
        """
        try:
            projection = next(
                (
                    p
                    for p in self._store.fetch_fact_projections(confirmed_only=True)
                    if p.key == key
                ),
                None,
            )
        except Exception:
            logger.exception(
                "failed to read the facts for %r; derived homes unchanged", log_safe(key)
            )
            return
        if projection is None:
            self._drop_index(key)
            self._drop_projection(key)
            return
        self._index_fact(projection)
        self._project(projection.key, projection.value, projection.confidence)

    def _index_fact(self, fact: UserFact) -> None:
        if self._index is not None:
            self._index.index_fact(fact)

    def _drop_index(self, key: str) -> None:
        if self._index is not None:
            self._index.drop_fact(key)

    def _project(self, key: str, value: str, confidence: float) -> None:
        if not self._project_md:
            return
        try:
            append_user_fact_to_md(key, value, confidence)
        except Exception:
            logger.exception("failed to project fact %r into USER.md", log_safe(key))

    def _drop_projection(self, key: str) -> None:
        if not self._project_md:
            return
        try:
            drop_user_fact_from_md(key)
        except Exception:
            logger.exception("failed to drop fact %r from USER.md", log_safe(key))
