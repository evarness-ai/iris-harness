"""Where memris keeps its records — a deliberately small persistence contract.

A :class:`GraphStore` saves and finds entities and statements. It holds no meaning:
supersede, end, retract, validation and the time rules all live in
:class:`memris.graph.MemoryGraph`, so a new backend (Neo4j, an RDF store, a remote
service) only has to implement these few methods to get the same behaviour.

``save`` is the one write, and it is atomic: a supersede closes the old statement and
adds the new one in a single call, so a crash can never leave both current.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from memris.model import Entity, EntityDecision, LearnedTerm, Statement
from memris.store.memory import InMemoryGraphStore
from memris.store.sqlite import SQLiteGraphStore


@runtime_checkable
class GraphStore(Protocol):
    def save(
        self, *, entities: Iterable[Entity] = (), statements: Iterable[Statement] = ()
    ) -> None:
        """Insert or replace every record given, all or nothing."""
        ...

    def get_entity(self, entity_id: str) -> Entity | None: ...

    def find_entities(self, *, label: str | None = None, class_: str | None = None) -> list[Entity]:
        """Entities whose label or an alias equals ``label`` (case-insensitive)."""
        ...

    def get_statement(self, statement_id: str) -> Statement | None: ...

    def statements(
        self,
        *,
        subject_id: str | None = None,
        predicates: Collection[str] | None = None,
        object_id: str | None = None,
    ) -> list[Statement]:
        """Every matching statement — current or not — oldest record first."""
        ...

    def save_decision(self, decision: EntityDecision) -> None:
        """Insert or replace one entity decision (same / distinct / candidate / undone)."""
        ...

    def decisions(self, entity_id: str | None = None) -> list[EntityDecision]:
        """Every decision, or those naming ``entity_id``, oldest first."""
        ...

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and the decisions naming it (MemoryGraph.delete_entity decides
        whether it may go; its statements are deleted first, with delete_statements)."""
        ...

    def delete_statements(self, ids: Collection[str]) -> int:
        """Delete statements outright (MemoryGraph.purge decides which may go)."""
        ...

    def used_terms(self) -> set[str]:
        """Every predicate and entity class stored data refers to (for check_usage)."""
        ...

    def save_term(self, term: LearnedTerm) -> None:
        """Insert or replace one learned term (decision 7), by name."""
        ...

    def get_term(self, name: str) -> LearnedTerm | None: ...

    def terms(self, status: str | None = None) -> list[LearnedTerm]:
        """Learned terms — all, or those with ``status`` — oldest first."""
        ...

    def delete_term(self, name: str) -> bool:
        """Forget a learned term outright (an expired candidate)."""
        ...


@runtime_checkable
class IndexedGraphStore(GraphStore, Protocol):
    """Optional extras a store can offer so reads need not scan every entity.

    Not part of :class:`GraphStore`: a backend without them still works, and
    :class:`memris.graph.MemoryGraph` falls back to scanning ``find_entities()``. Both
    stores shipped here implement them.
    """

    def merged_members(self, targets: Collection[str]) -> dict[str, list[str]]:
        """For each target id, the ids of entities merged directly into it, oldest first."""
        ...

    def live_entities(self, classes: Collection[str] | None = None) -> list[Entity]:
        """Entities neither merged nor removed (of ``classes``, when given), oldest first."""
        ...

    def session(self) -> AbstractContextManager[None]:
        """Share one connection across the calls made inside the block (reentrant)."""
        ...


__all__ = ["GraphStore", "InMemoryGraphStore", "IndexedGraphStore", "SQLiteGraphStore"]
