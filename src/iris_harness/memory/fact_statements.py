"""User facts, kept as memris statements (memris plan PR 2b; ADR-0115 decision 3).

``MemoryStore``'s fact API still speaks ``key = value``; this adapter turns each call
into statements about the owner and back:

- a fact key is found in ``config/memory/mappings.yaml``; its mapping names the
  property, and whether the value is an entity (a relation) or a literal;
- a key with no mapping cannot be stored — :class:`FactKeyError` — because a closed
  vocabulary is the point (decision 6). Reading one simply finds nothing;
- ``first_seen`` / ``last_confirmed`` / ``times_confirmed`` / ``source`` live on the
  statement as ``recorded_at`` / ``last_reinforced_at`` / ``reinforced`` / ``extractor``;
- keys that share a property (``employer`` and ``company``) are one fact, reported under
  the mapping's first key.

A fact may also be about someone one hop from the owner (memris PR 3b; ADR-0115
decision 6): "my wife Petra works at Infosys" is the owner's ``spouse`` fact plus an
``employer`` fact whose subject is Petra. Those are proposals only — they reach the review
queue beside the owner's, and never the owner's own fact API (``get``/``all``/L0).

Behaviour the old table had, kept here: a new value only replaces a single-valued fact
when its confidence is at least the current one's (unless the owner corrects it);
confirmation only goes up on a re-capture; a new value inherits the current one's
confirmation. What is new: a property without a limit (two cards) keeps every value.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from iris_harness.foundation.clock import utc_now
from iris_harness.kernel.governance.identity_redaction import invalidates_owner_identity
from iris_harness.memory.ontology import fact_mappings, memory_ontology, normalise_fact_key
from memris.graph import MemoryGraph
from memris.model import OWNER_ID, Entity, Statement, Status, name_key
from memris.ontology import MappingRule, Ontology
from memris.resolve import Resolver
from memris.store import SQLiteGraphStore

if TYPE_CHECKING:
    from iris_harness.memory.store import (
        FactContradiction,
        FactHistoryEntry,
        FactProposal,
        UserFact,
    )

OWNER_LABEL = "Owner"


class FactKeyError(ValueError):
    """A fact key the ontology has no property for."""


class SubjectError(ValueError):
    """A fact about someone its property cannot describe (outside the property's domain)."""


@dataclass(frozen=True)
class Link:
    """A relation stated in the same message: ``subject`` (None = the owner) --key--> ``value``."""

    subject: str | None
    key: str
    value: str


@dataclass(frozen=True)
class PutResult:
    prior_value: str | None
    prior_confidence: float | None
    stored_value: str
    stored_confidence: float | None
    changed: bool  # the current value is different from before
    blocked: bool  # a lower-confidence value was refused


class FactStatements:
    def __init__(
        self,
        db_path: Path,
        ontology: Ontology | None = None,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.ontology = ontology or memory_ontology()
        self.graph = MemoryGraph(self.ontology, SQLiteGraphStore(db_path), clock=clock)
        self._mappings = fact_mappings(self.ontology)
        # A property's display key is the first key its mapping lists.
        self._declared_keys: dict[str, str] = {}
        for rule in self.ontology.mappings:
            if rule.source_type == "fact" and rule.keys:
                self._declared_keys.setdefault(rule.predicate, normalise_fact_key(rule.keys[0]))
        self._owner_ready = False
        self._resolver: Resolver | None = None

    # -- learned vocabulary (memris PR 7; ADR-0115 decision 7) ------------------------

    def _learned(self) -> dict[str, str]:
        """Learned properties in use (active, or since rejected but still read) → their key."""
        declared = {*self.ontology.relations, *self.ontology.attributes}
        in_use = {*self.graph.ontology.relations, *self.graph.ontology.attributes}
        return {name: name.partition(":")[2] or name for name in in_use - declared}

    @property
    def _key_for(self) -> dict[str, str]:
        """Property → the fact key it is reported under: the YAML's, then the learned ones."""
        return {**self._learned(), **self._declared_keys}

    def _learned_rule(self, key: str) -> MappingRule | None:
        """A fact key that is an ACTIVE learned term — a statement may be written with it.

        Read from the store each time, not from this process's vocabulary: a term can be
        activated or rejected elsewhere (the CLI, the API) while this graph is open.
        """
        from iris_harness.memory.vocabulary import learned_prefix

        if not key:
            return None
        name = f"{learned_prefix()}:{key}"
        term = self.graph.store.get_term(name)
        if term is None or term.status != "active":
            return None
        if name not in self.graph.ontology.attributes and name not in self.graph.ontology.relations:
            self.graph.refresh_vocabulary()
        return MappingRule(
            id=f"learned_{key}",
            source_type="fact",
            keys=(key,),
            subject="$owner",
            predicate=name,
            object_from="$value" if term.kind == "relation" else None,
            object_class=term.range if term.kind == "relation" else None,
            value=None if term.kind == "relation" else "$value",
        )

    # -- helpers -----------------------------------------------------------------

    def _owner(self) -> None:
        if not self._owner_ready:
            self.graph.ensure_owner(OWNER_LABEL)
            self._owner_ready = True

    def rule(self, key: str) -> MappingRule:
        rule = self._mappings.get(normalise_fact_key(key)) or self._learned_rule(
            normalise_fact_key(key)
        )
        if rule is None:
            raise FactKeyError(
                f"'{key}' is not a fact key memory keeps; known keys: "
                + ", ".join(sorted(self._key_for.values()))
            )
        return rule

    def knows(self, key: str) -> bool:
        wanted = normalise_fact_key(key)
        return wanted in self._mappings or self._learned_rule(wanted) is not None

    def _single_valued(self, predicate: str) -> bool:
        owner_class = self.ontology.owner_class
        for cls in self.ontology.ancestors(owner_class) if owner_class else []:
            constraint = self.ontology.shapes.get(cls, {}).get(predicate)
            if constraint is not None:
                return constraint.max_count == 1
        return False

    def key_of(self, statement: Statement) -> str | None:
        """The fact key a statement is reported under, or None for a property with none."""
        return self._key_for.get(statement.predicate)

    def value_of(self, statement: Statement) -> str:
        """A statement's value — for an entity, the name of what it stands for now (merges)."""
        if statement.object_id is None:
            return str(statement.literal)
        entity = self.graph.get_entity(self.graph.canonical_id(statement.object_id))
        return entity.label if entity is not None else statement.object_id

    def _fact(self, statement: Statement) -> UserFact:
        from iris_harness.memory.store import UserFact  # store imports us

        return UserFact(
            key=self._key_for.get(statement.predicate, statement.predicate),
            value=self.value_of(statement),
            confidence=statement.confidence if statement.confidence is not None else 1.0,
            source=statement.extractor or "memris",
            first_seen=statement.recorded_at,
            last_confirmed=statement.last_reinforced_at or statement.recorded_at,
            times_confirmed=statement.reinforced,
            confirmed=statement.status == "confirmed",
            statement_id=statement.id,
        )

    def _current(self, predicate: str, subject_id: str = OWNER_ID) -> list[Statement]:
        found = self.graph.current(
            subject_id, predicate, include_proposed=True, include_subproperties=False
        )
        return sorted(found, key=lambda s: (s.last_reinforced_at or s.recorded_at, s.recorded_at))

    def _same_value(self, rule: MappingRule, statement: Statement, value: str) -> bool:
        """Is ``value`` what ``statement`` already says? For an entity: the same entity once
        names are resolved ("Northwind Bank Ltd" is "Northwind Bank"), not the same spelling."""
        if self.value_of(statement) == value:
            return True
        if rule.object_class is None or statement.object_id is None:
            return False
        found = self._get_resolver().find(value, rule.object_class)
        return found is not None and self.graph.canonical_id(
            found[0].id
        ) == self.graph.canonical_id(statement.object_id)

    def _mentioned(self, rule: MappingRule, value: str) -> None:
        """A name used again is evidence: its open look-alike pairs count this conversation.

        Only resolution gathers evidence, and a repeated value never needs a new entity —
        so without this, "the same pair in three conversations" could never be seen.
        """
        if rule.object_class is not None:
            self._object(rule, value)

    def _get_resolver(self) -> Resolver:
        if self._resolver is None:
            from iris_harness.memory.resolution import resolver_for

            self._resolver = resolver_for(self.graph)
        return self._resolver

    def _object(self, rule: MappingRule, value: str) -> str:
        """The entity a relation's value names — resolved (ADR-0115 decision 4), or made."""
        assert rule.object_class is not None
        from iris_harness.memory.resolution import current_episode

        resolution = self._get_resolver().resolve(
            value, rule.object_class, episode=current_episode()
        )
        return resolution.entity_id

    # -- reads -------------------------------------------------------------------

    def get(self, key: str) -> UserFact | None:
        """The believed value; failing that, the newest pending one (callers check .confirmed)."""
        if not self.knows(key):
            return None
        current = self._current(self.rule(key).predicate)
        confirmed = [s for s in current if s.status == "confirmed"]
        chosen = confirmed or current
        return self._fact(chosen[-1]) if chosen else None

    def all(self, *, confirmed_only: bool = False) -> list[UserFact]:
        facts = [
            self._fact(s)
            for s in self.graph.current(OWNER_ID, include_proposed=not confirmed_only)
            if s.predicate in self._key_for
        ]
        return sorted(facts, key=lambda f: (f.key, f.first_seen))

    def count_unconfirmed(self) -> int:
        """What review owes the owner — facts about them and about anyone one hop away."""
        return sum(
            1
            for s in self.graph.current(include_proposed=True)
            if s.predicate in self._key_for and s.status == "proposed"
        )

    # -- writes ------------------------------------------------------------------

    # The owner's confirmed facts are an owner-identity source (ADR-0125): every write that
    # can change which of them hold invalidates the guards' corpus in this process.
    @invalidates_owner_identity
    def put(self, fact: UserFact, *, force: bool = False, reason: str | None = None) -> PutResult:
        """Store ``fact``. ``force`` is an owner correction: it overrides the confidence gate.

        ``reason`` records why the new value exists (``corrected``, ``restored``) — the
        history view reads it back.
        """
        rule = self.rule(fact.key)
        self._owner()
        current = self._current(rule.predicate)
        # The belief is the prior value and the gate; with no belief, the newest pending
        # value plays that part among unconfirmed writes, as the old table's rows did.
        believed = [s for s in current if s.status == "confirmed"]
        latest = believed[-1] if believed else (current[-1] if current else None)
        prior_value = self.value_of(latest) if latest else None
        prior_conf = latest.confidence if latest else None

        same = [s for s in current if self._same_value(rule, s, fact.value)]
        if same:
            self._mentioned(rule, fact.value)
            s = same[-1]
            reinforced = s.reinforced + 1 if force else max(s.reinforced, fact.times_confirmed)
            s = self.graph.refresh(
                s.id,
                reinforced=reinforced,
                at=fact.last_confirmed,
                confidence=fact.confidence,
                extractor=fact.source,
            )
            if fact.confirmed and s.status == "proposed":
                self.graph.confirm(s.id)
                if s.reason == "proposed":  # a queued question, answered by a plain statement
                    self.graph.refresh(s.id, reason="approved")
            return PutResult(prior_value, prior_conf, fact.value, fact.confidence, False, False)

        single = self._single_valued(rule.predicate)
        if (
            single
            and latest is not None
            and not force
            and fact.confidence < (latest.confidence if latest.confidence is not None else 0.0)
        ):
            # Refused, not forgotten: kept as a refused statement, so review can see it.
            if rule.object_class is not None:
                self.graph.refuse(
                    OWNER_ID,
                    rule.predicate,
                    contradicts=latest.id,
                    object_id=self._object(rule, fact.value),
                    confidence=fact.confidence,
                    extractor=fact.source,
                )
            else:
                self.graph.refuse(
                    OWNER_ID,
                    rule.predicate,
                    contradicts=latest.id,
                    literal=fact.value,
                    confidence=fact.confidence,
                    extractor=fact.source,
                )
            return PutResult(prior_value, prior_conf, prior_value or "", prior_conf, False, True)

        if not single:  # another value alongside the others (a second card), not a replacement
            prior_value, prior_conf = None, None
        inherits = single and bool(believed)
        status: Status = "confirmed" if fact.confirmed or inherits else "proposed"
        if rule.object_class is not None:
            new = self.graph.assert_(
                OWNER_ID,
                rule.predicate,
                object_id=self._object(rule, fact.value),
                status=status,
                confidence=fact.confidence,
                extractor=fact.source,
            )
        else:
            new = self.graph.assert_(
                OWNER_ID,
                rule.predicate,
                literal=fact.value,
                status=status,
                confidence=fact.confidence,
                extractor=fact.source,
            )
        if fact.times_confirmed > 1 or reason is not None:
            self.graph.refresh(
                new.id,
                reinforced=max(new.reinforced, fact.times_confirmed),
                at=fact.last_confirmed,
                reason=reason,
            )
        if single and status == "proposed":
            # An unconfirmed write replaces the previous unconfirmed value, as the old
            # table's row did. (A confirmed value superseded the old belief inside
            # assert_, and leaves pending proposals alone: they are still questions.)
            replaced = None
            for s in self._current(rule.predicate):
                if s.id != new.id and s.status == "proposed" and s.reason != "proposed":
                    replaced = self.graph.retract(s.id, reason="replaced")
            if replaced is not None and new.supersedes is None:
                # the unconfirmed value this one replaced, so history reads old → new
                linked = self.graph.store.get_statement(new.id)
                if linked is not None:
                    self.graph.store.save(statements=[linked.evolve(supersedes=replaced.id)])
        return PutResult(prior_value, prior_conf, fact.value, fact.confidence, True, False)

    # -- one hop from the owner (memris PR 3b) ----------------------------------------

    def _subject(self, rule: MappingRule, name: str, class_name: str | None) -> str:
        """The entity a one-hop fact is about — resolved like any name (decision 4)."""
        from iris_harness.memory.resolution import current_episode

        term = self.ontology.relations.get(rule.predicate) or self.ontology.attributes.get(
            rule.predicate
        )
        wanted = class_name or (term.domain if term is not None else None)
        if (
            term is None
            or wanted is None
            or not self.ontology.is_subclass(self.ontology.qualify(wanted), term.domain)
        ):
            raise SubjectError(f"'{rule.predicate}' cannot describe a {wanted or 'thing'}")
        resolution = self._get_resolver().resolve(name, wanted, episode=current_episode())
        entity = self.graph.get_entity(resolution.entity_id)
        if entity is None or not self.ontology.is_subclass(entity.class_, term.domain):
            raise SubjectError(f"'{rule.predicate}' cannot describe '{name}'")
        return resolution.entity_id

    def _names(self, name: str) -> set[str]:
        resolver = self._get_resolver()
        return {name_key(name), name_key(resolver.normalise(name))} - {""}

    def _named(self, entity: Entity, keys: set[str]) -> bool:
        return any(self._names(n) & keys for n in (entity.label, *entity.aliases))

    def reachable(self, name: str, links: list[Link], *, max_hops: int) -> str | None:
        """Is ``name`` within ``max_hops`` of the owner? Returns its (qualified) class, or None.

        A hop is a relation either stated in this message (``links``, already through
        capture's gates) or held as a confirmed statement. So "my wife Petra works at
        Infosys" reaches Petra through the ``spouse`` link it states, a later "Petra
        moved to Pune" through the confirmed one, and a public figure through neither.
        """
        wanted = self._names(name)
        if not wanted:
            return None
        # A node is (entity id or None, name or None, class): an entity already in the
        # graph, or a name only this message has introduced so far.
        frontier: list[tuple[str | None, str | None, str]] = [
            (OWNER_ID, None, self.ontology.owner_class or "")
        ]
        seen: set[str] = {OWNER_ID}
        for _hop in range(max_hops):
            found: list[tuple[str | None, str | None, str]] = []
            for entity_id, node_name, _cls in frontier:
                found.extend(self._message_hops(entity_id, node_name, links))
                if entity_id is not None:
                    found.extend(self._graph_hops(entity_id))
            nxt = []
            for entity_id, node_name, cls in found:
                marker = entity_id or f"name:{name_key(node_name or '')}"
                if marker in seen:
                    continue
                seen.add(marker)
                entity = self.graph.get_entity(entity_id) if entity_id else None
                if (entity is not None and self._named(entity, wanted)) or (
                    node_name is not None and self._names(node_name) & wanted
                ):
                    return self.ontology.qualify(cls)
                nxt.append((entity_id, node_name, cls))
            frontier = nxt
        return None

    def _message_hops(
        self, entity_id: str | None, node_name: str | None, links: list[Link]
    ) -> list[tuple[str | None, str | None, str]]:
        entity = self.graph.get_entity(entity_id) if entity_id else None
        out: list[tuple[str | None, str | None, str]] = []
        for link in links:
            if link.subject is None:
                if entity_id != OWNER_ID:
                    continue
            else:
                keys = self._names(link.subject)
                named = node_name is not None and bool(self._names(node_name) & keys)
                if not named and not (entity is not None and self._named(entity, keys)):
                    continue
            if not self.knows(link.key):
                continue
            rule = self.rule(link.key)
            if rule.object_class is None:  # an attribute is a value, not a hop
                continue
            found = self._get_resolver().find(link.value, rule.object_class)
            target = self.graph.canonical_id(found[0].id) if found else None
            out.append((target, link.value, rule.object_class))
        return out

    def _graph_hops(self, entity_id: str) -> list[tuple[str | None, str | None, str]]:
        out: list[tuple[str | None, str | None, str]] = []
        for s in self.graph.current(entity_id):  # confirmed only: a proposal is no hop
            if s.object_id is None:
                continue
            target = self.graph.get_entity(self.graph.canonical_id(s.object_id))
            if target is not None:
                out.append((target.id, None, target.class_))
        return out

    # -- the review queue (proposed statements) --------------------------------------

    # Not an identity write: a proposal is unconfirmed (and memris never lets a proposed
    # statement supersede a confirmed one), so it cannot change the owner-identity corpus.
    def propose(
        self,
        key: str,
        value: str,
        *,
        confidence: float,
        source: str,
        evidence: str = "",
        subject: str | None = None,
        subject_class: str | None = None,
    ) -> str | None:
        """Queue a proposed fact. Returns its id, or None when it is already believed.

        Restating a confirmed fact counts as reinforcement, not a new question; restating
        a pending one bumps its count (repetition can confirm it, ADR-0114).

        ``subject`` names who it is about when that is not the owner, resolved as a
        ``subject_class`` entity (found, or made) — the caller has checked it is in scope
        (:meth:`reachable`). :class:`SubjectError` when the property cannot describe it.
        """
        rule = self.rule(key)
        self._owner()
        subject_id = OWNER_ID
        if subject is not None:
            subject_id = self._subject(rule, subject, subject_class)
        for s in self._current(rule.predicate, subject_id):
            if not self._same_value(rule, s, value):
                continue
            self._mentioned(rule, value)
            refreshed = self.graph.refresh(s.id, reinforced=s.reinforced + 1)
            return None if refreshed.status == "confirmed" else refreshed.id
        common = {
            "status": "proposed",
            "confidence": confidence,
            "extractor": source,
            "evidence": evidence[:500],
        }
        # reason "proposed" marks a question in the review queue (not an unconfirmed
        # write): history leaves it out until it is approved.
        if rule.object_class is not None:
            new = self.graph.assert_(
                subject_id, rule.predicate, object_id=self._object(rule, value), **common  # type: ignore[arg-type]
            )
        else:
            new = self.graph.assert_(subject_id, rule.predicate, literal=value, **common)  # type: ignore[arg-type]
        self.graph.refresh(new.id, reinforced=new.reinforced, reason="proposed")
        return new.id

    def _proposal_status(self, s: Statement) -> str | None:
        if s.status == "proposed":
            return "pending"
        if s.status == "confirmed" and s.reason == "approved":
            return "approved"
        if s.status == "retracted" and s.reason in ("rejected", "expired"):
            return s.reason
        return None

    def _proposal(self, s: Statement) -> FactProposal | None:
        from iris_harness.memory.store import FactProposal  # store imports us

        status = self._proposal_status(s)
        if status is None or s.predicate not in self._key_for:
            return None
        value = self.value_of(s)
        believed = [
            b
            for b in self._current(s.predicate, s.subject_id)
            if b.status == "confirmed" and b.id != s.id
        ]
        subject = None
        if s.subject_id != OWNER_ID:
            entity = self.graph.get_entity(self.graph.canonical_id(s.subject_id))
            subject = entity.label if entity is not None else s.subject_id
        current_value = self.value_of(believed[-1]) if believed else None
        resolved = s.retracted_at if status in ("rejected", "expired") else None
        if status == "approved":
            resolved = s.last_reinforced_at
        return FactProposal(
            id=s.id,
            key=self._key_for[s.predicate],
            value=value,
            confidence=s.confidence if s.confidence is not None else 1.0,
            source=s.extractor or "memris",
            evidence=s.evidence or "",
            current_value=current_value if current_value != value else None,
            created_at=s.recorded_at,
            status=status,
            resolved_at=resolved,
            seen_count=s.reinforced,
            subject=subject,
        )

    def proposals(self, *, status: str = "pending", limit: int = 200) -> list[FactProposal]:
        # Every subject: a fact about someone one hop away is reviewed beside the owner's.
        pool = (
            self.graph.current(include_proposed=True)
            if status == "pending"
            else self.graph.store.statements()
        )
        found = [p for s in pool if (p := self._proposal(s)) is not None and p.status == status]
        return sorted(found, key=lambda p: (p.created_at, p.id), reverse=True)[:limit]

    def proposal(self, proposal_id: str) -> FactProposal | None:
        s = self.graph.store.get_statement(proposal_id)
        return self._proposal(s) if s is not None else None

    @invalidates_owner_identity
    def resolve(self, proposal_id: str, status: str) -> bool:
        """approved / rejected / expired. Approval expects the fact to be stored already."""
        s = self.graph.store.get_statement(proposal_id)
        if s is None or s.predicate not in self._key_for:
            return False
        if status == "approved":
            if s.status == "retracted" or s.reason == "approved":
                return False
            if s.status == "proposed":
                self.graph.confirm(s.id)
            self.graph.refresh(s.id, reason="approved")
            return True
        if status in ("rejected", "expired") and s.status == "proposed":
            self.graph.retract(s.id, reason=status)
            return True
        return False

    def expire(self, *, older_than: datetime) -> int:
        stale = [
            s
            for s in self.graph.current(include_proposed=True)
            if s.status == "proposed"
            and s.predicate in self._key_for
            and s.recorded_at < older_than
        ]
        for s in stale:
            self.graph.retract(s.id, reason="expired")
        return len(stale)

    # -- history and contradictions, read from the chain (PR 2c-ii) ------------------

    _NOT_HISTORY = frozenset({"proposed", "rejected", "expired", "refused"})
    _EVENT_REASON = {"corrected": "correct", "restored": "restore"}

    def _fact_statements(self, key: str | None) -> list[Statement]:
        if key is not None:
            if not self.knows(key):
                return []
            predicates = {self.rule(key).predicate}
        else:
            predicates = set(self._key_for)
        return [
            s for s in self.graph.store.statements(subject_id=OWNER_ID) if s.predicate in predicates
        ]

    def history(self, key: str | None = None) -> list[FactHistoryEntry]:
        """Every change to the fact(s), newest first — derived, not logged."""
        from iris_harness.memory.store import FactHistoryEntry

        statements = self._fact_statements(key)
        by_id = {s.id: s for s in statements}
        events: list[FactHistoryEntry] = []
        for s in statements:
            if s.reason in self._NOT_HISTORY:
                continue
            before = by_id.get(s.supersedes) if s.supersedes else None
            if s.reason in self._EVENT_REASON:
                kind = self._EVENT_REASON[s.reason]
            else:
                kind = "supersede" if before is not None else "capture"
            events.append(
                FactHistoryEntry(
                    id=s.id,
                    key=self._key_for[s.predicate],
                    old_value=self.value_of(before) if before else None,
                    old_confidence=before.confidence if before else None,
                    new_value=self.value_of(s),
                    new_confidence=s.confidence,
                    source=s.extractor or "memris",
                    reason=kind,
                    changed_at=s.recorded_at,
                )
            )
            if s.status == "retracted" and s.reason == "forgot" and s.retracted_at is not None:
                events.append(
                    FactHistoryEntry(
                        id=f"{s.id}#forget",
                        key=self._key_for[s.predicate],
                        old_value=self.value_of(s),
                        old_confidence=s.confidence,
                        new_value=None,
                        new_confidence=None,
                        source="user:forget",
                        reason="forget",
                        changed_at=s.retracted_at,
                    )
                )
        return sorted(events, key=lambda e: (e.changed_at, e.id), reverse=True)

    def _holds(self, s: Statement) -> bool:
        return s.status == "proposed" or (s.status == "confirmed" and s.valid_to is None)

    def retention_candidates(self, *, older_than: datetime, limit: int) -> list[FactHistoryEntry]:
        """History old enough to review for pruning — only statements that no longer hold."""
        holding = {s.id for s in self._fact_statements(None) if self._holds(s)}
        old = [
            e
            for e in self.history()
            if e.changed_at < older_than and e.id.split("#")[0] not in holding
        ]
        return sorted(old, key=lambda e: (e.changed_at, e.id))[:limit]

    def prune(self, entry_ids: list[str]) -> int:
        """Owner-chosen history goes for good (purge refuses anything still holding)."""
        candidates = {s.id: s for s in self._fact_statements(None)}
        chosen = [
            sid
            for sid in dict.fromkeys(i.split("#")[0] for i in entry_ids)
            if sid in candidates and not self._holds(candidates[sid])
        ]
        return self.graph.purge(chosen) if chosen else 0

    def contradictions(self, *, include_reviewed: bool, limit: int) -> list[FactContradiction]:
        """Conflicts review should see: a value that replaced another, or one refused."""
        from iris_harness.memory.store import FactContradiction

        statements = self._fact_statements(None)
        by_id = {s.id: s for s in statements}
        found: list[FactContradiction] = []
        for s in statements:
            if s.reason == "refused" and s.contradicts in by_id:
                other, resolution = by_id[s.contradicts], "blocked"
            elif s.supersedes in by_id and s.reason not in ("corrected", "restored", "refused"):
                other, resolution = by_id[s.supersedes or ""], "superseded"
            else:
                continue
            if s.reviewed_at is not None and not include_reviewed:
                continue
            found.append(
                FactContradiction(
                    id=s.id,
                    key=self._key_for[s.predicate],
                    stored_value=self.value_of(other),
                    stored_confidence=other.confidence,
                    incoming_value=self.value_of(s),
                    incoming_confidence=s.confidence,
                    resolution=resolution,
                    source=s.extractor or "memris",
                    detected_at=s.recorded_at,
                    acknowledged=s.reviewed_at is not None,
                    seen_count=s.reinforced if resolution == "blocked" else 1,
                )
            )
        return sorted(found, key=lambda c: (c.detected_at, c.id), reverse=True)[:limit]

    def acknowledge(self, ids: list[str]) -> int:
        known = {c.id for c in self.contradictions(include_reviewed=True, limit=100_000)}
        return self.graph.review([i for i in ids if i in known])

    @invalidates_owner_identity
    def forget(
        self, key: str, *, statement_id: str | None = None
    ) -> list[tuple[str, float | None]]:
        """Retract every current value — or only ``statement_id``, one value of a key
        that holds several; returns what was withdrawn."""
        if not self.knows(key):
            return []
        gone = []
        for s in self._current(self.rule(key).predicate):
            if statement_id is not None and s.id != statement_id:
                continue
            self.graph.retract(s.id, reason="forgot")
            gone.append((self.value_of(s), s.confidence))
        return gone

    @invalidates_owner_identity
    def replace(self, key: str, statement_id: str, fact: UserFact, *, reason: str) -> bool:
        """An owner edit of ONE value of a key that holds several (a card renamed).

        For a single-valued key a new value supersedes the old by itself; for a key
        with several values it would be added beside them, so the edited statement is
        retracted (``replaced``) and the new one linked to it, and history reads
        old → new. Returns False when ``statement_id`` is not a current value of ``key``.
        """
        rule = self.rule(key)
        if self._single_valued(rule.predicate):
            return False
        old = next((s for s in self._current(rule.predicate) if s.id == statement_id), None)
        if old is None:
            return False
        self.put(fact, force=True, reason=reason)
        new = next(
            (
                s
                for s in reversed(self._current(rule.predicate))
                if s.id != old.id and self._same_value(rule, s, fact.value)
            ),
            None,
        )
        if new is None:
            return False
        self.graph.retract(old.id, reason="replaced")
        linked = self.graph.store.get_statement(new.id)
        if linked is not None and linked.supersedes is None:
            self.graph.store.save(statements=[linked.evolve(supersedes=old.id)])
        return True

    @invalidates_owner_identity
    def set_confirmed(self, key: str, confirmed: bool) -> bool:
        if not self.knows(key):
            return False
        current = self._current(self.rule(key).predicate)
        for s in current:
            if confirmed:
                self.graph.confirm(s.id)
            else:
                self.graph.unconfirm(s.id)
        return bool(current)

    def reinforce_confirmed(self, key: str, value: str) -> bool:
        """Saying a confirmed fact again: count it, and report that nothing needs review."""
        if not self.knows(key):
            return False
        for s in self._current(self.rule(key).predicate):
            if s.status == "confirmed" and self._same_value(self.rule(key), s, value):
                self.graph.refresh(s.id, reinforced=s.reinforced + 1)
                return True
        return False


__all__ = ["OWNER_LABEL", "FactKeyError", "FactStatements", "Link", "PutResult", "SubjectError"]
