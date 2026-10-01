"""MemoryGraph — what a statement means, over any GraphStore.

Every write is checked against the compiled ontology (the predicate exists and is not
retired, the subject is in its domain, the object in its range) and every change of
belief follows ADR-0115 decision 5:

- **supersede**: a *confirmed* value for a property the shapes limit to one closes the
  current value (``valid_to`` = the new value's start) instead of deleting it. A
  *proposed* value never displaces a confirmed one; confirming it later does.
- **end**: the claim stopped being true, and nothing replaced it.
- **retract**: the claim was never true. It leaves recall but stays in the store; if
  it had superseded an older value, that value is current again — a claim that never
  held cannot have ended anything.

Reads follow decision 8: a query for a predicate also returns statements stored under
retired terms whose ``replaced_by`` names it, and under its subproperties. Stored
statements are never rewritten to a new term.

Known limit: record time is kept per statement, not per field. ``known_at`` answers
"did memris hold this claim then?", but a later ``end`` or supersede updates
``valid_to`` in place rather than as a new version.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime
from typing import Literal

from memris.model import OWNER_ID, Entity, EntityDecision, Statement, Status, new_id, utc
from memris.ontology import Constraint, Issue, Ontology, check_usage, with_learned
from memris.store import GraphStore, IndexedGraphStore

Direction = Literal["out", "in", "both"]


class StatementError(ValueError):
    """A write the ontology or the current state does not allow."""


def _now() -> datetime:
    return datetime.now(UTC)


class MemoryGraph:
    def __init__(
        self,
        ontology: Ontology,
        store: GraphStore,
        *,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        # What the YAML declares, and what is in use: that plus learned terms (decision 7).
        self.declared = ontology
        self.ontology = ontology
        self.store = store
        self._clock = clock
        self._children: dict[str, set[str]] = {}
        self._read_cache: dict[tuple[str, bool], frozenset[str]] = {}
        self._read_cache_for: object = None
        self._indexed_for: object = None
        self._indexed_store: IndexedGraphStore | None = None
        self.refresh_vocabulary()

    @property
    def _indexed(self) -> IndexedGraphStore | None:
        """The store, when it can answer merge and live-entity lookups without a scan."""
        store = self.store
        if self._indexed_for is not store:  # checked once per store, not per call
            self._indexed_for = store
            self._indexed_store = store if isinstance(store, IndexedGraphStore) else None
        return self._indexed_store

    def session(self) -> AbstractContextManager[None]:
        """One store connection for a batch of calls (a no-op for stores without one)."""
        indexed = self._indexed
        return indexed.session() if indexed is not None else nullcontext()

    def refresh_vocabulary(self) -> None:
        """Re-derive the vocabulary in use from the declared ontology and the store's
        learned terms — after a term is activated, rejected or imported."""
        self.ontology = with_learned(self.declared, self.store.terms())
        children: dict[str, set[str]] = {}
        for table in (self.ontology.relations, self.ontology.attributes):
            for name, term in table.items():
                if term.parent:
                    children.setdefault(term.parent, set()).add(name)
        self._children = children
        self._read_cache = {}  # the children changed too, even if the ontology object did not

    def _now(self) -> datetime:
        return utc(self._clock())

    # -- entities ----------------------------------------------------------------

    def _concrete_class(self, class_name: str) -> str:
        name = self.ontology.qualify(class_name)
        term = self.ontology.classes.get(name)
        if term is None:
            raise StatementError(f"'{class_name}' is not a declared class")
        if term.abstract:
            raise StatementError(f"'{name}' is abstract and cannot have instances")
        if term.deprecated:
            hint = f"; use {', '.join(term.replaced_by)}" if term.replaced_by else ""
            raise StatementError(f"'{name}' is deprecated{hint}")
        return name

    def ensure_owner(self, label: str, class_name: str | None = None) -> Entity:
        """The reserved owner entity, created on first use.

        Its class is configuration: ``class_name``, else the ontology's ``owner_class``.
        """
        existing = self.store.get_entity(OWNER_ID)
        if existing is not None:
            return existing
        chosen = class_name or self.ontology.owner_class
        if chosen is None:
            raise StatementError("no owner class: pass one or set ontology.owner_class")
        owner = Entity(OWNER_ID, self._concrete_class(chosen), label, created_at=self._now())
        self.store.save(entities=[owner])
        return owner

    def add_entity(self, class_name: str, label: str, aliases: Iterable[str] = ()) -> Entity:
        label = " ".join(label.split())
        if not label:
            raise StatementError("an entity needs a non-empty label")
        entity = Entity(
            new_id("ent"),
            self._concrete_class(class_name),
            label,
            tuple(dict.fromkeys(a for a in aliases if a and a != label)),
            self._now(),
        )
        self.store.save(entities=[entity])
        return entity

    def get_entity(self, entity_id: str) -> Entity | None:
        return self.store.get_entity(entity_id)

    def _entity(self, entity_id: str, role: str) -> Entity:
        entity = self.store.get_entity(entity_id)
        if entity is None:
            raise StatementError(f"{role} '{entity_id}' does not exist")
        if entity.merged_into is not None:
            raise StatementError(f"{role} '{entity_id}' was merged into '{entity.merged_into}'")
        if entity.removed:
            raise StatementError(f"{role} '{entity_id}' was removed; restore it first")
        return entity

    # -- removal (ADR-0119) --------------------------------------------------------

    def _about(self, entity_id: str) -> list[Statement]:
        """Every statement naming the entity (or one merged into it), as subject or object."""
        found: dict[str, Statement] = {}
        with self.session():
            for member in self.members(entity_id):
                for s in (
                    *self.store.statements(subject_id=member),
                    *self.store.statements(object_id=member),
                ):
                    found[s.id] = s
        return sorted(found.values(), key=lambda s: (s.recorded_at, s.id))

    def _holds(self, s: Statement, now: datetime) -> bool:
        return s.status == "proposed" or (s.status == "confirmed" and s.valid_at(now))

    def remove_entity(self, entity_id: str, *, reason: str = "removed") -> Entity:
        """Take an entity out of memory, with every claim that still holds about it.

        Each such statement is withdrawn (status ``retracted``, ``reason``), and the
        entity records which ones and how they stood, so :meth:`restore_entity` puts
        back exactly those. Unlike :meth:`retract`, nothing a withdrawn statement had
        superseded is reopened: removing "Northwind Bank" does not make the bank before it
        current again. The owner cannot be removed.
        """
        if entity_id == OWNER_ID:
            raise StatementError("the owner cannot be removed")
        entity = self._entity(entity_id, "entity")
        now = self._now()
        live = [s for s in self._about(entity.id) if self._holds(s, now)]
        withdrawn = [s.evolve(status="retracted", retracted_at=now, reason=reason) for s in live]
        removed = entity.evolve(
            removed_at=now,
            removed_reason=reason,
            removed_statements=tuple((s.id, s.status, s.reason) for s in live),
        )
        self.store.save(entities=[removed], statements=withdrawn)
        return removed

    def restore_entity(self, entity_id: str) -> Entity:
        """Undo :meth:`remove_entity`: the entity and exactly what went with it return.

        A statement purged since is skipped. A single-valued claim that was replaced
        while removed comes back closed at the replacement's start, never beside it.
        """
        entity = self.store.get_entity(entity_id)
        if entity is None:
            raise StatementError(f"entity '{entity_id}' does not exist")
        if not entity.removed:
            return entity
        restored: list[Statement] = []
        for statement_id, status, reason in entity.removed_statements:
            s = self.store.get_statement(statement_id)
            if s is None or s.status != "retracted":
                continue
            restored.append(s.evolve(status=status, retracted_at=None, reason=reason))
        restored = [r for back in restored for r in self._put_back(back)]
        standing = entity.evolve(removed_at=None, removed_reason=None, removed_statements=())
        self.store.save(entities=[standing], statements=restored)
        return standing

    def reinstate(
        self, statement_id: str, *, status: Status = "confirmed", reason: str | None = None
    ) -> Statement:
        """Undo a retraction: the statement holds again, as ``status``, with ``reason``.

        What its retraction reopened (the value it had superseded) is closed again, and a
        single-valued claim replaced since comes back closed at the replacement's start.
        """
        s = self._statement(statement_id)
        if s.status != "retracted":
            raise StatementError(f"statement '{statement_id}' is not retracted")
        if status not in ("proposed", "confirmed"):
            raise StatementError(f"a statement is reinstated proposed or confirmed, not {status!r}")
        for entity_id in (s.subject_id, s.object_id):
            if entity_id is not None:
                self._entity(entity_id, "entity")
        changed = self._put_back(s.evolve(status=status, retracted_at=None, reason=reason))
        self.store.save(statements=changed)
        return changed[-1]

    def _put_back(self, back: Statement) -> list[Statement]:
        """A statement returning to memory, and what its return closes: ``[..., back]``."""
        if back.status != "confirmed" or back.valid_to is not None:
            return [back]
        closed: list[Statement] = []
        if back.supersedes is not None:
            previous = self.store.get_statement(back.supersedes)
            if (
                previous is not None
                and previous.status == "confirmed"
                and previous.valid_to is None
                and previous.effective_from <= back.effective_from
            ):
                closed.append(previous.evolve(valid_to=back.effective_from))
        return [*closed, self._closed_by_newer(back)]

    def _closed_by_newer(self, back: Statement) -> Statement:
        subject = self.store.get_entity(back.subject_id)
        if subject is None:
            return back
        constraint = self._constraint(subject, back.predicate)
        if constraint is None or constraint.max_count != 1:
            return back
        newer = [
            s
            for s in self.current(back.subject_id, back.predicate, include_subproperties=False)
            if s.id != back.id
            and s.predicate == back.predicate
            and s.recorded_at > back.recorded_at
        ]
        if not newer:
            return back
        return back.evolve(valid_to=min(s.effective_from for s in newer))

    def delete_entity(self, entity_id: str) -> int:
        """Delete a removed entity for good, with every statement naming it.

        Refused unless the entity was removed first, while any statement about it still
        holds, and while other entities are merged into it. Returns how many statements
        went with it. What is deleted cannot be restored.
        """
        entity = self.store.get_entity(entity_id)
        if entity is None:
            raise StatementError(f"entity '{entity_id}' does not exist")
        if not entity.removed:
            raise StatementError(f"entity '{entity_id}' was not removed; remove it first")
        if len(self.members(entity.id)) > 1:
            raise StatementError(f"entities are merged into '{entity_id}'; unmerge them first")
        now = self._now()
        about = self._about(entity.id)
        if any(self._holds(s, now) for s in about):
            raise StatementError(f"a statement about '{entity_id}' still holds")
        deleted = self.store.delete_statements([s.id for s in about])
        self.store.delete_entity(entity.id)
        return deleted

    # -- writes ------------------------------------------------------------------

    def assert_(
        self,
        subject_id: str,
        predicate: str,
        *,
        object_id: str | None = None,
        literal: str | None = None,
        status: Status = "proposed",
        valid_from: datetime | None = None,
        confidence: float | None = None,
        source_episode: str | None = None,
        source_turn: str | None = None,
        extractor: str | None = None,
        evidence: str | None = None,
    ) -> Statement:
        """Record a claim. Returns the stored statement (an existing one if identical)."""
        if status not in ("proposed", "confirmed"):
            raise StatementError(f"a new statement is proposed or confirmed, not {status!r}")
        name = self.ontology.qualify(predicate)
        subject = self._entity(subject_id, "subject")
        datatype = self._check_shape_of_claim(name, subject, object_id, literal)
        if valid_from is not None:
            valid_from = utc(valid_from)

        same = [
            s
            for s in self.current(
                subject_id, name, include_proposed=True, include_subproperties=False
            )
            if s.predicate == name and s.object_id == object_id and s.literal == literal
        ]
        if same:
            existing = self.refresh(same[0].id, reinforced=same[0].reinforced + 1)
            if status == "confirmed" and existing.status == "proposed":
                return self.confirm(existing.id)
            return existing

        statement = Statement(
            id=new_id("st"),
            subject_id=subject_id,
            predicate=name,
            recorded_at=self._now(),
            object_id=object_id,
            literal=literal,
            datatype=datatype,
            valid_from=valid_from,
            status=status,
            confidence=confidence,
            source_episode=source_episode,
            source_turn=source_turn,
            extractor=extractor,
            ontology_version=self.ontology.version,
            evidence=evidence,
        )
        closed: list[Statement] = []
        if status == "confirmed":
            statement, closed = self._supersede(statement, subject)
        self.store.save(statements=[*closed, statement])
        return statement

    def _check_shape_of_claim(
        self, name: str, subject: Entity, object_id: str | None, literal: str | None
    ) -> str | None:
        onto = self.ontology
        kind = onto.kind_of(name)
        if kind not in ("relation", "attribute"):
            raise StatementError(f"'{name}' is not a declared property")
        term = onto.relations[name] if kind == "relation" else onto.attributes[name]
        if term.deprecated:
            hint = f"; use {', '.join(term.replaced_by)}" if term.replaced_by else ""
            raise StatementError(f"'{name}' is deprecated{hint}")
        if not onto.is_subclass(subject.class_, term.domain):
            raise StatementError(
                f"'{subject.class_}' is outside the domain of '{name}' ({term.domain})"
            )
        if kind == "attribute":
            if literal is None or object_id is not None:
                raise StatementError(f"'{name}' is an attribute: give a literal, not an object")
            return onto.attributes[name].datatype
        if object_id is None or literal is not None:
            raise StatementError(f"'{name}' is a relation: give an object, not a literal")
        target = self._entity(object_id, "object")
        allowed = onto.relations[name].range
        narrowed = self._constraint(subject, name)
        if narrowed is not None and narrowed.object_class is not None:
            allowed = narrowed.object_class
        if not onto.is_subclass(target.class_, allowed):
            raise StatementError(f"'{target.class_}' is outside the range of '{name}' ({allowed})")
        return None

    def _constraint(self, subject: Entity, predicate: str) -> Constraint | None:
        for cls in self.ontology.ancestors(subject.class_):
            constraint = self.ontology.shapes.get(cls, {}).get(predicate)
            if constraint is not None:
                return constraint
        return None

    def _supersede(self, new: Statement, subject: Entity) -> tuple[Statement, list[Statement]]:
        """Close what a single-valued property's new value replaces."""
        constraint = self._constraint(subject, new.predicate)
        if constraint is None or constraint.max_count != 1:
            return new, []
        start = new.effective_from
        closed = [
            s.evolve(valid_to=start)
            for s in self.current(new.subject_id, new.predicate, include_subproperties=False)
            if s.id != new.id and s.predicate == new.predicate and s.effective_from <= start
        ]
        if closed:
            new = new.evolve(supersedes=closed[-1].id)
        return new, closed

    def _statement(self, statement_id: str) -> Statement:
        statement = self.store.get_statement(statement_id)
        if statement is None:
            raise StatementError(f"statement '{statement_id}' does not exist")
        return statement

    def refresh(
        self,
        statement_id: str,
        *,
        reinforced: int | None = None,
        at: datetime | None = None,
        confidence: float | None = None,
        extractor: str | None = None,
        reason: str | None = None,
    ) -> Statement:
        """Update a statement's bookkeeping — never its claim, status or time bounds.

        ``reinforced`` sets the count (an importer knows it exactly); ``at`` is when it
        was last reinforced (default now).
        """
        statement = self._statement(statement_id)
        updated = statement.evolve(
            reinforced=statement.reinforced if reinforced is None else max(1, reinforced),
            last_reinforced_at=utc(at) if at is not None else self._now(),
            confidence=statement.confidence if confidence is None else confidence,
            extractor=statement.extractor if extractor is None else extractor,
            reason=statement.reason if reason is None else reason,
        )
        self.store.save(statements=[updated])
        return updated

    def unconfirm(self, statement_id: str) -> Statement:
        """Back to a proposal: out of what memory asserts, still awaiting a yes."""
        statement = self._statement(statement_id)
        if statement.status != "confirmed":
            return statement
        demoted = statement.evolve(status="proposed")
        self.store.save(statements=[demoted])
        return demoted

    def import_(self, statements: Iterable[Statement], entities: Iterable[Entity] = ()) -> None:
        """Write records made elsewhere — a migration, an import — exactly as given.

        Each is checked against the ontology (property declared, subject in its domain,
        object in its range, literal vs object) but nothing is superseded, reinforced or
        re-timed: the records carry their own history. All or nothing.
        """
        new_entities = list(entities)
        new_statements = list(statements)
        known = {e.id: e for e in new_entities}
        for entity in new_entities:
            if entity.id != OWNER_ID:
                self._concrete_class(entity.class_)
        for s in new_statements:
            subject = known.get(s.subject_id) or self._entity(s.subject_id, "subject")
            name = self.ontology.qualify(s.predicate)
            kind = self.ontology.kind_of(name)
            if kind not in ("relation", "attribute"):
                raise StatementError(f"'{s.predicate}' is not a declared property")
            term = (
                self.ontology.relations[name]
                if kind == "relation"
                else self.ontology.attributes[name]
            )
            if not self.ontology.is_subclass(subject.class_, term.domain):
                raise StatementError(f"'{subject.class_}' is outside the domain of '{name}'")
            if kind == "attribute" and (s.literal is None or s.object_id is not None):
                raise StatementError(f"'{name}' is an attribute: give a literal, not an object")
            if kind == "relation":
                if s.object_id is None or s.literal is not None:
                    raise StatementError(f"'{name}' is a relation: give an object, not a literal")
                target = known.get(s.object_id) or self._entity(s.object_id, "object")
                if not self.ontology.is_subclass(
                    target.class_, self.ontology.relations[name].range
                ):
                    raise StatementError(f"'{target.class_}' is outside the range of '{name}'")
        self.store.save(entities=new_entities, statements=new_statements)

    def confirm(self, statement_id: str) -> Statement:
        statement = self._statement(statement_id)
        if statement.status == "retracted":
            raise StatementError("a retracted statement cannot be confirmed; assert it again")
        if statement.status == "confirmed":
            return statement
        confirmed, closed = self._supersede(
            statement.evolve(status="confirmed"), self._entity(statement.subject_id, "subject")
        )
        self.store.save(statements=[*closed, confirmed])
        return confirmed

    def end(self, statement_id: str, at: datetime | None = None) -> Statement:
        statement = self._statement(statement_id)
        if statement.status == "retracted":
            raise StatementError("a retracted statement cannot end; it never held")
        if statement.valid_to is not None:
            raise StatementError(f"statement already ended at {statement.valid_to.isoformat()}")
        moment = utc(at) if at is not None else self._now()
        if moment <= statement.effective_from:
            raise StatementError("a statement cannot end before it started")
        ended = statement.evolve(valid_to=moment)
        self.store.save(statements=[ended])
        return ended

    def retract(self, statement_id: str, reason: str | None = None) -> Statement:
        statement = self._statement(statement_id)
        if statement.status == "retracted":
            return statement
        retracted = statement.evolve(
            status="retracted", retracted_at=self._now(), reason=reason or statement.reason
        )
        reopened: list[Statement] = []
        if statement.supersedes is not None:
            previous = self.store.get_statement(statement.supersedes)
            if (
                previous is not None
                and previous.status == "confirmed"
                and previous.valid_to == statement.effective_from
            ):
                reopened.append(previous.evolve(valid_to=None))
        self.store.save(statements=[*reopened, retracted])
        return retracted

    def refuse(
        self,
        subject_id: str,
        predicate: str,
        *,
        contradicts: str,
        object_id: str | None = None,
        literal: str | None = None,
        confidence: float | None = None,
        extractor: str | None = None,
        evidence: str | None = None,
    ) -> Statement:
        """Keep a claim that was received but not believed (e.g. lower confidence).

        It is recorded and retracted in the same instant — never known, never valid —
        with reason ``refused`` and the statement it lost against in ``contradicts``.
        The same refusal again, before anyone has reviewed it, reinforces the existing
        record instead of adding one: a model repeating a bad value must not flood review.
        """
        name = self.ontology.qualify(predicate)
        subject = self._entity(subject_id, "subject")
        datatype = self._check_shape_of_claim(name, subject, object_id, literal)
        self._statement(contradicts)
        for s in self.store.statements(subject_id=subject_id, predicates=[name]):
            if (
                s.reason == "refused"
                and s.reviewed_at is None
                and s.object_id == object_id
                and s.literal == literal
            ):
                return self.refresh(s.id, reinforced=s.reinforced + 1, confidence=confidence)
        now = self._now()
        refused = Statement(
            id=new_id("st"),
            subject_id=subject_id,
            predicate=name,
            recorded_at=now,
            object_id=object_id,
            literal=literal,
            datatype=datatype,
            retracted_at=now,
            status="retracted",
            confidence=confidence,
            extractor=extractor,
            ontology_version=self.ontology.version,
            evidence=evidence,
            reason="refused",
            contradicts=contradicts,
        )
        self.store.save(statements=[refused])
        return refused

    def review(self, statement_ids: Iterable[str], at: datetime | None = None) -> int:
        """Mark statements as seen by an owner (e.g. a contradiction acknowledged)."""
        moment = utc(at) if at is not None else self._now()
        updated = [
            s.evolve(reviewed_at=moment)
            for statement_id in dict.fromkeys(statement_ids)
            if (s := self.store.get_statement(statement_id)) is not None
        ]
        self.store.save(statements=updated)
        return len(updated)

    def purge(self, statement_ids: Iterable[str]) -> int:
        """Delete statements for good — only ones that no longer hold.

        A current or proposed statement is refused: forgetting something is a
        retraction; purging is for clearing old history the owner chose to drop.
        """
        now = self._now()
        doomed = []
        for statement_id in statement_ids:
            s = self._statement(statement_id)
            if s.status == "proposed" or (s.status == "confirmed" and s.valid_at(now)):
                raise StatementError(f"statement '{statement_id}' still holds; retract it first")
            doomed.append(statement_id)
        return self.store.delete_statements(doomed)

    # -- reads -------------------------------------------------------------------

    def _read_predicates(self, predicate: str, include_subproperties: bool) -> set[str]:
        """The term itself, retired terms that point to it, and (optionally) its subproperties."""
        key = (predicate, include_subproperties)
        if self._read_cache_for is not self.ontology:  # vocabulary changed: start over
            self._read_cache, self._read_cache_for = {}, self.ontology
        cached = self._read_cache.get(key)
        if cached is not None:
            return set(cached)
        name = self.ontology.qualify(predicate)
        wanted = {name}
        frontier = [name]
        while frontier:
            current = frontier.pop()
            found: set[str] = set()
            for table in (self.ontology.relations, self.ontology.attributes):
                for other, term in table.items():
                    if current in term.replaced_by:
                        found.add(other)
            if include_subproperties:
                found |= self._children.get(current, set())
            frontier.extend(found - wanted)
            wanted |= found
        self._read_cache[key] = frozenset(wanted)
        return wanted

    def current(
        self,
        subject_id: str | None = None,
        predicate: str | None = None,
        *,
        object_id: str | None = None,
        as_of: datetime | None = None,
        known_at: datetime | None = None,
        include_proposed: bool = False,
        include_subproperties: bool = True,
    ) -> list[Statement]:
        """Statements true at ``as_of`` (valid time) as memris knew them at ``known_at``.

        Both default to now. Retracted statements never appear; proposed ones only on
        request.
        """
        now = self._now()
        valid = utc(as_of) if as_of is not None else now
        known = utc(known_at) if known_at is not None else now
        predicates = (
            self._read_predicates(predicate, include_subproperties)
            if predicate is not None
            else None
        )
        # A statement retracted AFTER `known` was still held then (known_at() drops the
        # rest), so looking back in record time it counts; its earlier status is not kept.
        allowed = {"confirmed", "retracted", *(("proposed",) if include_proposed else ())}
        return [
            s
            for s in self._statements_for(subject_id, predicates, object_id)
            if s.known_at(known) and s.valid_at(valid) and s.status in allowed
        ]

    def neighbourhood(
        self,
        entity_id: str,
        *,
        predicate: str | None = None,
        direction: Direction = "both",
        hops: int = 1,
        as_of: datetime | None = None,
        include_proposed: bool = False,
    ) -> list[tuple[int, Statement]]:
        """Statements within ``hops`` of an entity, each with the hop it was found at.

        ``direction``: ``out`` follows statements the entity is the subject of, ``in``
        those it is the object of, ``both`` either. Hop 2 continues from the entities at
        the far end of hop 1 (literals end a path). ``predicate`` narrows every hop and
        reads through subproperties and retired terms like :meth:`current`. Merged
        entities are followed, so a name folded into another still answers.
        """
        if direction not in ("out", "in", "both"):
            raise StatementError(f"direction is out, in or both, not {direction!r}")
        with self.session():
            return self._neighbourhood(
                entity_id, predicate, direction, hops, as_of, include_proposed
            )

    def _neighbourhood(
        self,
        entity_id: str,
        predicate: str | None,
        direction: Direction,
        hops: int,
        as_of: datetime | None,
        include_proposed: bool,
    ) -> list[tuple[int, Statement]]:
        found: dict[str, tuple[int, Statement]] = {}
        frontier = [self.canonical_id(entity_id)]
        seen = set(frontier)
        for hop in range(1, max(1, hops) + 1):
            reached: list[str] = []
            for node in frontier:
                matches: list[tuple[Statement, str | None]] = []
                if direction in ("out", "both"):
                    matches += [
                        (s, s.object_id)
                        for s in self.current(
                            node, predicate, as_of=as_of, include_proposed=include_proposed
                        )
                    ]
                if direction in ("in", "both"):
                    matches += [
                        (s, s.subject_id)
                        for s in self.current(
                            None,
                            predicate,
                            object_id=node,
                            as_of=as_of,
                            include_proposed=include_proposed,
                        )
                    ]
                for statement, far in matches:
                    found.setdefault(statement.id, (hop, statement))
                    if far is not None:
                        far = self.canonical_id(far)
                        if far not in seen:
                            seen.add(far)
                            reached.append(far)
            frontier = reached
            if not frontier:
                break
        return sorted(found.values(), key=lambda pair: (pair[0], pair[1].recorded_at, pair[1].id))

    def history(self, subject_id: str, predicate: str | None = None) -> list[Statement]:
        """Everything ever recorded for the subject — superseded, ended and retracted too."""
        predicates = self._read_predicates(predicate, True) if predicate is not None else None
        return self._statements_for(subject_id, predicates, None)

    def _statements_for(
        self, subject_id: str | None, predicates: set[str] | None, object_id: str | None
    ) -> list[Statement]:
        """Statements about an entity AND every entity merged into it (read-through).

        A merge never rewrites statements (decision 8): what was said about "Northwind" stays
        about Northwind's id, and a read of "Northwind Bank" simply includes it.
        """
        found: dict[str, Statement] = {}
        with self.session():
            subjects = self.members(subject_id) if subject_id is not None else [None]
            objects = self.members(object_id) if object_id is not None else [None]
            for subj in subjects:
                for obj in objects:
                    for s in self.store.statements(
                        subject_id=subj, predicates=predicates, object_id=obj
                    ):
                        found[s.id] = s
        return sorted(found.values(), key=lambda s: (s.recorded_at, s.id))

    # -- identity (decision 4) -----------------------------------------------------

    def canonical_id(self, entity_id: str) -> str:
        """The entity ``entity_id`` stands for now: followed through any merges."""
        seen = {entity_id}
        with self.session():
            current = self.store.get_entity(entity_id)
            while current is not None and current.merged_into is not None:
                if current.merged_into in seen:
                    break
                seen.add(current.merged_into)
                current = self.store.get_entity(current.merged_into)
        return current.id if current is not None else entity_id

    def members(self, entity_id: str) -> list[str]:
        """``entity_id`` and every entity that was merged into it, directly or not."""
        indexed = self._indexed
        merged: dict[str, list[str]] = {}
        if indexed is None:  # a store without the lookup: scan every entity once
            for e in self.store.find_entities():
                if e.merged_into is not None:
                    merged.setdefault(e.merged_into, []).append(e.id)
        out, frontier = [entity_id], [entity_id]
        with self.session():
            while frontier:
                if indexed is not None:  # only the entities merged into this level
                    merged = indexed.merged_members(frontier)
                nxt = [m for f in frontier for m in merged.get(f, []) if m not in out]
                out.extend(nxt)
                frontier = nxt
        return out

    def decisions(self, entity_id: str | None = None) -> list[EntityDecision]:
        return self.store.decisions(entity_id)

    def _decision_between(self, a: str, b: str) -> EntityDecision | None:
        latest = None
        for d in self.store.decisions(a):
            if {d.a, d.b} == {a, b}:
                latest = d
        return latest

    def are_distinct(self, a: str, b: str) -> bool:
        d = self._decision_between(a, b)
        return d is not None and d.decision == "distinct"

    def merge(
        self,
        keep: str,
        into_it: str,
        *,
        decided_by: str,
        evidence: Iterable[str] = (),
        score: float | None = None,
    ) -> EntityDecision:
        """``into_it`` becomes part of ``keep``: its names become aliases; reversible.

        Refused when someone has said the two are distinct, when either is already
        merged, or when their classes are unrelated (a Person is never an Organization).
        """
        winner = self._entity(keep, "entity")
        loser = self._entity(into_it, "entity")
        if winner.id == loser.id:
            raise StatementError("an entity cannot be merged into itself")
        if self.are_distinct(winner.id, loser.id):
            raise StatementError(f"'{winner.label}' and '{loser.label}' were marked distinct")
        onto = self.ontology
        if not (
            onto.is_subclass(loser.class_, winner.class_)
            or onto.is_subclass(winner.class_, loser.class_)
        ):
            raise StatementError(f"'{loser.class_}' and '{winner.class_}' are unrelated classes")
        added = tuple(
            n
            for n in dict.fromkeys((loser.label, *loser.aliases))
            if n != winner.label and n not in winner.aliases
        )
        now = self._now()
        previous = self._decision_between(winner.id, loser.id)
        decision = EntityDecision(
            id=previous.id if previous is not None else new_id("dec"),
            a=winner.id,
            b=loser.id,
            decision="same",
            decided_at=now,
            decided_by=decided_by,
            score=score if score is not None else (previous.score if previous else None),
            evidence=tuple(dict.fromkeys((*(previous.evidence if previous else ()), *evidence))),
            added_aliases=added,
        )
        self.store.save(
            entities=[
                winner.__class__(**{**winner.__dict__, "aliases": (*winner.aliases, *added)}),
                loser.__class__(**{**loser.__dict__, "merged_into": winner.id}),
            ]
        )
        self.store.save_decision(decision)
        return decision

    def unmerge(self, decision_id: str, *, decided_by: str) -> EntityDecision:
        """Undo a merge: both entities stand alone again, as they were before."""
        decision = next((d for d in self.store.decisions() if d.id == decision_id), None)
        if decision is None or decision.decision != "same":
            raise StatementError(f"'{decision_id}' is not a merge that can be undone")
        winner = self.store.get_entity(decision.a)
        loser = self.store.get_entity(decision.b)
        if winner is None or loser is None:
            raise StatementError("a merged entity no longer exists")
        self.store.save(
            entities=[
                winner.__class__(
                    **{
                        **winner.__dict__,
                        "aliases": tuple(
                            a for a in winner.aliases if a not in decision.added_aliases
                        ),
                    }
                ),
                loser.__class__(**{**loser.__dict__, "merged_into": None}),
            ]
        )
        undone = decision.__class__(
            **{
                **decision.__dict__,
                "decision": "undone",
                "decided_at": self._now(),
                "decided_by": decided_by,
            }
        )
        self.store.save_decision(undone)
        return undone

    def mark_distinct(self, a: str, b: str, *, decided_by: str) -> EntityDecision:
        """Two different things with similar names: recorded, and never proposed again."""
        previous = self._decision_between(a, b)
        if previous is not None and previous.decision == "same":
            raise StatementError("these two are merged; undo the merge first")
        decision = EntityDecision(
            id=previous.id if previous is not None else new_id("dec"),
            a=a,
            b=b,
            decision="distinct",
            decided_at=self._now(),
            decided_by=decided_by,
            score=previous.score if previous is not None else None,
            evidence=previous.evidence if previous is not None else (),
        )
        self.store.save_decision(decision)
        return decision

    def note_candidate(
        self, a: str, b: str, *, score: float, episode: str | None
    ) -> EntityDecision | None:
        """Two entities that look alike: gather evidence (distinct episodes) toward a merge."""
        previous = self._decision_between(a, b)
        if previous is not None and previous.decision in ("distinct", "same"):
            return None
        evidence = previous.evidence if previous is not None else ()
        if episode is not None and episode not in evidence:
            evidence = (*evidence, episode)
        decision = EntityDecision(
            id=previous.id if previous is not None else new_id("dec"),
            a=previous.a if previous is not None else a,
            b=previous.b if previous is not None else b,
            decision="candidate",
            decided_at=self._now(),
            decided_by="similarity",
            score=max(score, previous.score or 0.0) if previous is not None else score,
            evidence=evidence,
            asked_at=previous.asked_at if previous is not None else None,
        )
        self.store.save_decision(decision)
        return decision

    def get_decision(self, decision_id: str) -> EntityDecision | None:
        return next((d for d in self.store.decisions() if d.id == decision_id), None)

    def open_candidates(self) -> list[EntityDecision]:
        """Look-alike pairs nobody has decided yet, both entities still standing alone."""
        found = []
        for d in self.store.decisions():
            if d.decision != "candidate":
                continue
            pair = [self.store.get_entity(i) for i in (d.a, d.b)]
            if all(e is not None and e.merged_into is None and not e.removed for e in pair):
                found.append(d)
        return found

    def mark_asked(self, decision_id: str) -> EntityDecision:
        """The owner was asked about this pair: it is never asked again (decision 4)."""
        decision = self._candidate(decision_id)
        asked = decision.__class__(**{**decision.__dict__, "asked_at": self._now()})
        self.store.save_decision(asked)
        return asked

    def accept_candidate(self, decision_id: str, *, decided_by: str) -> EntityDecision:
        """Someone said a look-alike pair is one thing: merge it, the older name kept."""
        decision = self._candidate(decision_id)
        pair = [self._entity(i, "entity") for i in (decision.a, decision.b)]
        keep, fold = sorted(pair, key=lambda e: (e.created_at, e.id))
        return self.merge(keep.id, fold.id, decided_by=decided_by, evidence=decision.evidence)

    def reject_candidate(self, decision_id: str, *, decided_by: str) -> EntityDecision:
        """Someone said a look-alike pair is two things: kept apart, never proposed again."""
        decision = self._candidate(decision_id)
        return self.mark_distinct(decision.a, decision.b, decided_by=decided_by)

    def _candidate(self, decision_id: str) -> EntityDecision:
        decision = self.get_decision(decision_id)
        if decision is None or decision.decision != "candidate":
            raise StatementError(f"'{decision_id}' is not an open look-alike pair")
        return decision

    def check_usage(self) -> list[Issue]:
        """Decision 8: every term stored data uses must still resolve in the ontology."""
        return check_usage(self.ontology, self.store.used_terms())


__all__ = ["Direction", "MemoryGraph", "StatementError"]
