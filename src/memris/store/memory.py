"""A dict-backed GraphStore — for tests, notebooks, and anyone who wants no database."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator
from contextlib import contextmanager

from memris.model import Entity, EntityDecision, LearnedTerm, Statement, name_key


class InMemoryGraphStore:
    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._statements: dict[str, Statement] = {}
        self._decisions: dict[str, EntityDecision] = {}
        self._terms: dict[str, LearnedTerm] = {}

    def save(
        self, *, entities: Iterable[Entity] = (), statements: Iterable[Statement] = ()
    ) -> None:
        # Materialise first so a failing iterable cannot leave a partial write.
        new_entities = list(entities)
        new_statements = list(statements)
        self._entities.update((e.id, e) for e in new_entities)
        self._statements.update((s.id, s) for s in new_statements)

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def find_entities(self, *, label: str | None = None, class_: str | None = None) -> list[Entity]:
        wanted = name_key(label) if label is not None else None
        found = []
        for entity in self._entities.values():
            if class_ is not None and entity.class_ != class_:
                continue
            names = {name_key(n) for n in (entity.label, *entity.aliases)}
            if wanted is not None and wanted not in names:
                continue
            found.append(entity)
        return sorted(found, key=lambda e: (e.created_at, e.id))

    def merged_members(self, targets: Collection[str]) -> dict[str, list[str]]:
        """For each target id, the entities merged directly into it, oldest first."""
        wanted = set(targets)
        found: dict[str, list[str]] = {}
        for e in sorted(self._entities.values(), key=lambda e: (e.created_at, e.id)):
            if e.merged_into is not None and e.merged_into in wanted:
                found.setdefault(e.merged_into, []).append(e.id)
        return found

    def live_entities(self, classes: Collection[str] | None = None) -> list[Entity]:
        """Entities neither merged nor removed — of ``classes``, when given — oldest first."""
        wanted = set(classes) if classes is not None else None
        found = [
            e
            for e in self._entities.values()
            if e.merged_into is None and not e.removed and (wanted is None or e.class_ in wanted)
        ]
        return sorted(found, key=lambda e: (e.created_at, e.id))

    @contextmanager
    def session(self) -> Iterator[None]:
        """Nothing to share in memory: here for parity with SQLiteGraphStore.session."""
        yield

    def get_statement(self, statement_id: str) -> Statement | None:
        return self._statements.get(statement_id)

    def statements(
        self,
        *,
        subject_id: str | None = None,
        predicates: Collection[str] | None = None,
        object_id: str | None = None,
    ) -> list[Statement]:
        found = [
            s
            for s in self._statements.values()
            if (subject_id is None or s.subject_id == subject_id)
            and (predicates is None or s.predicate in predicates)
            and (object_id is None or s.object_id == object_id)
        ]
        return sorted(found, key=lambda s: (s.recorded_at, s.id))

    def save_decision(self, decision: EntityDecision) -> None:
        self._decisions[decision.id] = decision

    def decisions(self, entity_id: str | None = None) -> list[EntityDecision]:
        found = [
            d for d in self._decisions.values() if entity_id is None or entity_id in (d.a, d.b)
        ]
        return sorted(found, key=lambda d: (d.decided_at, d.id))

    def save_term(self, term: LearnedTerm) -> None:
        self._terms[term.name] = term

    def get_term(self, name: str) -> LearnedTerm | None:
        return self._terms.get(name)

    def terms(self, status: str | None = None) -> list[LearnedTerm]:
        found = [t for t in self._terms.values() if status is None or t.status == status]
        return sorted(found, key=lambda t: (t.first_seen, t.name))

    def delete_term(self, name: str) -> bool:
        return self._terms.pop(name, None) is not None

    def delete_entity(self, entity_id: str) -> bool:
        self._decisions = {k: d for k, d in self._decisions.items() if entity_id not in (d.a, d.b)}
        return self._entities.pop(entity_id, None) is not None

    def delete_statements(self, ids: Collection[str]) -> int:
        gone = [i for i in ids if self._statements.pop(i, None) is not None]
        return len(gone)

    def used_terms(self) -> set[str]:
        return {s.predicate for s in self._statements.values()} | {
            e.class_ for e in self._entities.values()
        }


__all__ = ["InMemoryGraphStore"]
