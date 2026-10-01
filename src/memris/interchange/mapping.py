"""Records from another system → memris statements, through a mappings file.

An importer is a READER plus a MAPPINGS FILE (ADR-0115, decision 11). The reader knows
the other system's format and yields :class:`SourceRecord` s; the mappings file — the
same DSL as ``config/memory/mappings.yaml``, compiled against the core ontology with
:func:`memris.ontology.load_mappings` — says which record becomes which statement.
This module is the part in between, and it knows neither side's vocabulary.

Lossless-or-declared: every record is either written or listed in the
:class:`MappingReport` with the reason (no mapping for its type, an endpoint without a
class, a class outside the property's domain or range …). What maps is written in one
``MemoryGraph.import_`` — validated, all or nothing — so a report is never a
half-written store.

Variables a mapping may use: ``$subject`` and ``$object`` (entity endpoints) and
``$value`` (a literal). A record may name its endpoints' classes itself; an object
without one takes the mapping's object class.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from memris.model import STATUSES, Entity, Statement, Status, name_key, new_id
from memris.ontology import MappingRule, Ontology

SUBJECT = "$subject"
OBJECT = "$object"
VALUE = "$value"


@dataclass(frozen=True)
class EntityRef:
    """An endpoint as the other system names it. ``class_`` is an ontology class or None."""

    name: str
    class_: str | None = None
    source_id: str | None = None


@dataclass(frozen=True)
class SourceRecord:
    """One record another system holds, in the shape a mapping can read."""

    source_type: str
    key: str | None
    source_id: str
    subject: EntityRef | None = None
    object: EntityRef | None = None
    value: str | None = None
    recorded_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    retracted_at: datetime | None = None
    episode: str | None = None
    evidence: str | None = None
    extractor: str | None = None
    statement_id: str | None = None  # a stable id makes a re-import replace, not duplicate
    notes: tuple[str, ...] = ()  # the reader's own approximations, carried into the report


@dataclass
class MappingReport:
    """What an import did, and every record it did not write (with why)."""

    records: int = 0
    statements: int = 0
    already_present: int = 0
    entities_created: int = 0
    entities_reused: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (source id, why)
    notes: list[tuple[str, str]] = field(default_factory=list)  # (source id, approximation)
    unmapped_types: dict[str, int] = field(default_factory=dict)  # "type:key" → count

    @property
    def lossless(self) -> bool:
        return not self.skipped

    def summary(self) -> str:
        return (
            f"{self.records} record(s): {self.statements} statement(s) written "
            f"({self.already_present} already present), {self.entities_created} entit(ies) "
            f"created, {self.entities_reused} reused, {len(self.skipped)} skipped, "
            f"{len(self.notes)} approximated"
        )


def _aware(moment: datetime | None) -> datetime | None:
    if moment is None:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


class _Applier:
    def __init__(self, graph: Any, rules: list[MappingRule], status: Status, now: datetime) -> None:
        self.graph = graph
        self.onto: Ontology = graph.ontology
        self.status = status
        self.now = now
        self.by_type: dict[tuple[str, str | None], MappingRule] = {}
        for rule in rules:
            for key in rule.keys or (None,):
                self.by_type.setdefault((rule.source_type, key), rule)
        self.entities: dict[tuple[str, str], Entity] = {}  # new ones, by (class, name)
        self.resolved: dict[tuple[str, str], str] = {}  # every (class, name) → entity id
        self.report = MappingReport()
        # Exact / alias resolution only: an import makes no guesses about look-alikes.
        from memris.resolve import Resolver  # resolve imports graph

        self.resolver = Resolver(graph)

    def rule_for(self, record: SourceRecord) -> MappingRule | None:
        return self.by_type.get((record.source_type, record.key)) or self.by_type.get(
            (record.source_type, None)
        )

    def entity(self, ref: EntityRef, class_name: str) -> str:
        """The id of the entity ``ref`` names: an existing one of that class, or a new one."""
        slot = (class_name, name_key(ref.name))
        if slot in self.resolved:
            return self.resolved[slot]
        found = self.resolver.find(ref.name, class_name)  # label or alias, merges followed
        if found is not None:
            self.report.entities_reused += 1
            self.resolved[slot] = self.graph.canonical_id(found[0].id)
            return self.resolved[slot]
        created = Entity(new_id("ent"), class_name, " ".join(ref.name.split()), (), self.now)
        self.entities[slot] = created
        self.resolved[slot] = created.id
        self.report.entities_created += 1
        return created.id

    def endpoint_class(self, ref: EntityRef, fallback: str | None) -> str | None:
        return self.onto.qualify(ref.class_) if ref.class_ else fallback

    def statement(self, record: SourceRecord) -> Statement | str:
        """The statement for ``record``, or the reason it cannot be written."""
        rule = self.rule_for(record)
        if rule is None:
            label = f"{record.source_type}:{record.key}" if record.key else record.source_type
            self.report.unmapped_types[label] = self.report.unmapped_types.get(label, 0) + 1
            return f"no mapping for {label!r}"
        if rule.subject != SUBJECT or record.subject is None:
            return f"mapping {rule.id!r} needs a {SUBJECT} the record does not carry"
        predicate = rule.predicate
        kind = self.onto.kind_of(predicate)
        term = self.onto.relations.get(predicate) or self.onto.attributes.get(predicate)
        if kind is None or term is None:
            return f"mapping {rule.id!r} names unknown property {predicate!r}"
        subject_class = self.endpoint_class(record.subject, None)
        if subject_class is None or self.onto.kind_of(subject_class) != "class":
            return f"subject {record.subject.name!r} has no known class ({record.subject.class_!r})"
        if self.onto.classes[subject_class].abstract:
            return f"subject class {subject_class} is abstract"
        if not self.onto.is_subclass(subject_class, term.domain):
            return f"{subject_class} is outside the domain of {predicate} ({term.domain})"
        object_id = literal = datatype = None
        if kind == "relation":
            if rule.object_from != OBJECT or record.object is None:
                return f"mapping {rule.id!r} needs an {OBJECT} the record does not carry"
            object_class = self.endpoint_class(record.object, rule.object_class)
            if object_class is None or self.onto.kind_of(object_class) != "class":
                return (
                    f"object {record.object.name!r} has no known class ({record.object.class_!r})"
                )
            if self.onto.classes[object_class].abstract:
                return f"object class {object_class} is abstract"
            if not self.onto.is_subclass(object_class, self.onto.relations[predicate].range):
                return f"{object_class} is outside the range of {predicate}"
            object_id = self.entity(record.object, object_class)
        else:
            if record.value is None:
                return f"mapping {rule.id!r} needs a {VALUE} the record does not carry"
            literal, datatype = record.value, self.onto.attributes[predicate].datatype
        subject_id = self.entity(record.subject, subject_class)
        recorded = _aware(record.recorded_at) or self.now
        retracted = _aware(record.retracted_at)
        return Statement(
            id=record.statement_id or new_id("st"),
            subject_id=subject_id,
            predicate=predicate,
            recorded_at=recorded,
            object_id=object_id,
            literal=literal,
            datatype=datatype,
            valid_from=_aware(record.valid_from),
            valid_to=_aware(record.valid_to),
            retracted_at=retracted,
            status="retracted" if retracted is not None else self.status,
            extractor=record.extractor,
            source_episode=record.episode,
            evidence=record.evidence,
            ontology_version=self.onto.version,
        )


def apply_mappings(
    records: Iterable[SourceRecord],
    rules: list[MappingRule],
    graph: Any,
    *,
    status: Status = "proposed",
    now: datetime | None = None,
) -> MappingReport:
    """Write every record ``rules`` can map into ``graph`` (a ``MemoryGraph``); report the rest.

    ``status`` is what an imported claim starts as. The default is ``proposed``: another
    system's belief is a question for the owner here, not a belief of this memory.
    """
    if status not in STATUSES or status == "retracted":
        raise ValueError(f"imported statements start proposed or confirmed, not {status!r}")
    applier = _Applier(graph, rules, status, now or datetime.now(UTC))
    statements: list[Statement] = []
    for record in records:
        applier.report.records += 1
        applier.report.notes.extend((record.source_id, n) for n in record.notes)
        made = applier.statement(record)
        if isinstance(made, str):
            applier.report.skipped.append((record.source_id, made))
            continue
        existing = graph.store.get_statement(made.id)
        if existing is not None:
            # The source owns its claim's time bounds and evidence; the owner owns the
            # decision about it. A re-import updates the first and never resets the
            # second — a proposal the owner confirmed, or a review, must survive.
            applier.report.already_present += 1
            made = existing.evolve(
                valid_from=made.valid_from,
                valid_to=made.valid_to,
                evidence=made.evidence,
                source_episode=made.source_episode,
                retracted_at=(
                    made.retracted_at if existing.status != "confirmed" else existing.retracted_at
                ),
                status=(
                    "retracted"
                    if made.status == "retracted" and existing.status == "proposed"
                    else existing.status
                ),
            )
        statements.append(made)
    graph.import_(statements, list(applier.entities.values()))
    applier.report.statements = len(statements)
    return applier.report


__all__ = [
    "OBJECT",
    "SUBJECT",
    "VALUE",
    "EntityRef",
    "MappingReport",
    "SourceRecord",
    "apply_mappings",
]
