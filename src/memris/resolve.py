"""Which entity a name means — the resolution policy of ADR-0115 decision 4.

When unsure, keep them separate: a wrong merge corrupts every statement attached to
both entities; a missed merge is only a duplicate node. So:

- **exact / alias**: the name (folded by ``normalise``) is an existing entity's label or
  alias, in a compatible class → that entity. Automatic.
- **likely**: no exact match, but ``similarity`` scores an existing entity at or above
  ``ask_similarity`` → a NEW entity is made, and the pair is recorded as a *candidate*.
  The caller may ask the owner about it once (``MemoryGraph.mark_asked``, then
  ``accept_candidate`` / ``reject_candidate``); unanswered, evidence gathers — each
  distinct episode in which either name turns up — and at ``evidence_episodes`` the two
  are merged automatically, with the evidence recorded and one-step undo.
- **new**: nothing close → a new entity.

A pair someone marked distinct is never proposed again. memris calls no model: the
similarity is a callable the caller supplies (string, embedding, anything).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from memris.graph import MemoryGraph
from memris.model import Entity, EntityDecision, name_key
from memris.store import IndexedGraphStore

How = Literal["exact", "alias", "new"]


@dataclass(frozen=True)
class Resolution:
    entity_id: str
    how: How
    created: bool
    candidates: tuple[EntityDecision, ...] = ()
    merged: tuple[EntityDecision, ...] = field(default_factory=tuple)


def _identity(name: str) -> str:
    return name


class Resolver:
    def __init__(
        self,
        graph: MemoryGraph,
        *,
        similarity: Callable[[str, str], float] | None = None,
        normalise: Callable[[str], str] = _identity,
        ask_similarity: float = 0.8,
        evidence_episodes: int = 3,
    ) -> None:
        self.graph = graph
        self.similarity = similarity
        self.normalise = normalise
        self.ask_similarity = ask_similarity
        self.evidence_episodes = max(1, evidence_episodes)

    def _compatible(self, entity: Entity, class_name: str) -> bool:
        onto = self.graph.ontology
        wanted = onto.qualify(class_name)
        return onto.is_subclass(entity.class_, wanted) or onto.is_subclass(wanted, entity.class_)

    def _keys(self, name: str) -> set[str]:
        return {name_key(name), name_key(self.normalise(name))} - {""}

    def _live(self, class_name: str) -> list[Entity]:
        """Entities standing alone (not merged, not removed) of a compatible class, oldest
        first — asked of the store by class when it can, instead of loading every entity."""
        store = self.graph.store
        if isinstance(store, IndexedGraphStore):
            onto = self.graph.ontology
            wanted = onto.qualify(class_name)
            classes = [
                c
                for c in onto.classes
                if onto.is_subclass(c, wanted) or onto.is_subclass(wanted, c)
            ]
            return store.live_entities(classes)
        return [
            e
            for e in store.find_entities()
            if e.merged_into is None and not e.removed and self._compatible(e, class_name)
        ]

    def _named(self, keys: set[str], class_name: str) -> list[Entity]:
        """With no ``normalise``, the entities that may carry one of ``keys`` — through the
        store's name index — instead of every entity. Same order as :meth:`_live`."""
        found: dict[str, Entity] = {}
        for key in keys:
            for e in self.graph.store.find_entities(label=key):
                if e.merged_into is None and not e.removed and self._compatible(e, class_name):
                    found[e.id] = e
        return sorted(found.values(), key=lambda e: (e.created_at, e.id))

    def find(self, name: str, class_name: str) -> tuple[Entity, How] | None:
        """An existing entity this name exactly means (label or alias, folded), if any."""
        wanted = self._keys(name)
        if not wanted:
            return None
        # The store indexes names by name_key only, so it can answer alone when names are
        # not normalised; a caller's normalise can map any stored name onto ``wanted``.
        if self.normalise is _identity:
            return self._find_in(self._named(wanted, class_name), wanted)
        return self._find_in(self._live(class_name), wanted)

    def _find_in(self, live: list[Entity], wanted: set[str]) -> tuple[Entity, How] | None:
        for entity in live:
            if self._keys(entity.label) & wanted:
                return entity, "exact"
            if any(self._keys(alias) & wanted for alias in entity.aliases):
                return entity, "alias"
        return None

    def resolve(self, name: str, class_name: str, *, episode: str | None = None) -> Resolution:
        """The entity ``name`` means — found, or made (and its look-alikes recorded)."""
        live: list[Entity] | None = None
        if self.normalise is _identity:
            found = self.find(name, class_name)
        else:  # one pass over the live entities serves the lookup and the look-alikes
            live = self._live(class_name)
            wanted = self._keys(name)
            found = self._find_in(live, wanted) if wanted else None
        if found is not None:
            entity, how = found
            merged = self._gather(entity.id, episode)
            target = self.graph.canonical_id(entity.id)
            return Resolution(target, how, False, merged=merged)
        created = self.graph.add_entity(class_name, name)
        candidates: list[EntityDecision] = []
        if self.similarity is not None:
            for entity in live if live is not None else self._live(class_name):
                if entity.id == created.id:
                    continue
                score = max(
                    self.similarity(self.normalise(name), self.normalise(n))
                    for n in (entity.label, *entity.aliases)
                )
                if score >= self.ask_similarity:
                    noted = self.graph.note_candidate(
                        entity.id, created.id, score=score, episode=episode
                    )
                    if noted is not None:
                        candidates.append(noted)
        return Resolution(created.id, "new", True, tuple(candidates))

    def _gather(self, entity_id: str, episode: str | None) -> tuple[EntityDecision, ...]:
        """A name in use again: evidence for its open candidates; merge past the threshold."""
        merged: list[EntityDecision] = []
        for d in self.graph.decisions(entity_id):
            if d.decision != "candidate":
                continue
            noted = self.graph.note_candidate(d.a, d.b, score=d.score or 0.0, episode=episode)
            if noted is None or len(noted.evidence) < self.evidence_episodes:
                continue
            pair = [e for i in (d.a, d.b) if (e := self.graph.get_entity(i)) is not None]
            if len(pair) != 2 or any(e.merged_into is not None or e.removed for e in pair):
                continue
            keep, fold = sorted(pair, key=lambda e: (e.created_at, e.id))  # the older name stays
            merged.append(
                self.graph.merge(keep.id, fold.id, decided_by="evidence", evidence=noted.evidence)
            )
        return tuple(merged)


__all__ = ["How", "Resolution", "Resolver"]
