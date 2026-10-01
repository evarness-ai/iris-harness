"""JSON-LD for memris, with plain ``json`` — no RDF library at runtime.

Shape of a document::

    {
      "@context": { prefixes…, one term per class/relation/attribute…, memris terms… },
      "memris:ontology": {"@id": <ontology id>, "memris:ontologyVersion": "0.1.0"},
      "@graph": [
        {"@id": "urn:memris:entity:ent_…", "@type": "<class>", "rdfs:label": "…", …},
        {"@id": "urn:memris:statement:st_…", "@type": "rdf:Statement",
         "rdf:subject": {"@id": …}, "rdf:predicate": {"@id": "<property>"},
         "rdf:object": {"@id": …} | {"@value": "…", "@type": "<xsd datatype>"},
         "prov:generatedAtTime": …, "memris:validFrom": …, …}
      ]
    }

A statement is its own node (``rdf:Statement``) because it carries time, status and
provenance that a bare triple cannot. Provenance uses PROV-O where a PROV term means
the same thing (``prov:generatedAtTime`` = when memris recorded it,
``prov:wasDerivedFrom`` = the episode it came from); the rest is memris's own
namespace. Nothing here names an ontology term: classes and properties come from the
compiled ontology the caller passes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from memris.model import STATUSES, Entity, LearnedTerm, Statement
from memris.ontology import Ontology
from memris.ontology.compiler import XSD

MEMRIS_NS = "urn:memris:ns#"
ENTITY_BASE = "urn:memris:entity:"
STATEMENT_BASE = "urn:memris:statement:"
EPISODE_BASE = "urn:memris:episode:"

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
PROV = "http://www.w3.org/ns/prov#"

# Engine terms: statement and entity bookkeeping. Standards vocabulary, not ontology.
_DATETIME = {"@type": XSD + "dateTime"}
_ENGINE_TERMS: dict[str, dict[str, str]] = {
    "memris:validFrom": _DATETIME,
    "memris:validTo": _DATETIME,
    "memris:retractedAt": _DATETIME,
    "memris:lastReinforcedAt": _DATETIME,
    "memris:reviewedAt": _DATETIME,
    "memris:createdAt": _DATETIME,
    "memris:removedAt": _DATETIME,
    "prov:generatedAtTime": _DATETIME,
    "memris:supersedes": {"@type": "@id"},
    "memris:contradicts": {"@type": "@id"},
    "memris:mergedInto": {"@type": "@id"},
    "prov:wasDerivedFrom": {"@type": "@id"},
}

# Keys a node may carry; anything else is reported, never ignored silently.
_ENTITY_KEYS = frozenset(
    {
        "@id",
        "@type",
        "rdfs:label",
        "memris:aliases",
        "memris:createdAt",
        "memris:mergedInto",
        "memris:removedAt",
        "memris:removedReason",
        "memris:removedStatements",
    }
)
_STATEMENT_KEYS = frozenset(
    {
        "@id",
        "@type",
        "rdf:subject",
        "rdf:predicate",
        "rdf:object",
        "prov:generatedAtTime",
        "prov:wasDerivedFrom",
        "memris:validFrom",
        "memris:validTo",
        "memris:retractedAt",
        "memris:status",
        "memris:confidence",
        "memris:sourceTurn",
        "memris:extractor",
        "memris:supersedes",
        "memris:ontologyVersion",
        "memris:reinforced",
        "memris:lastReinforcedAt",
        "memris:evidence",
        "memris:reason",
        "memris:contradicts",
        "memris:reviewedAt",
    }
)
_STATEMENT_TYPE = "rdf:Statement"


# --------------------------------------------------------------------------- context


def build_context(ontology: Ontology) -> dict[str, Any]:
    """The ``@context``: prefixes, then one term per ontology class and property.

    Generated, never written by hand — change the YAML and the context follows.
    Relations are typed ``@id`` (their objects are nodes); attributes carry their XSD
    datatype.
    """
    context: dict[str, Any] = {
        "@version": 1.1,
        "rdf": RDF,
        "rdfs": RDFS,
        "prov": PROV,
        "xsd": XSD,
        "memris": MEMRIS_NS,
    }
    for prefix, base in ontology.prefixes.items():
        context.setdefault(prefix, base)
    for name in ontology.classes:
        context[name] = {"@id": ontology.expand(name)}
    for name in ontology.relations:
        context[name] = {"@id": ontology.expand(name), "@type": "@id"}
    for name, term in ontology.attributes.items():
        context[name] = {"@id": ontology.expand(name), "@type": term.datatype}
    context.update(_ENGINE_TERMS)
    return context


# --------------------------------------------------------------------------- export


def _ts(moment: datetime | None) -> str | None:
    return None if moment is None else moment.isoformat()


def _ref(base: str, value: str | None) -> dict[str, str] | None:
    return None if value is None else {"@id": base + value}


def _entity_node(entity: Entity) -> dict[str, Any]:
    node: dict[str, Any] = {
        "@id": ENTITY_BASE + entity.id,
        "@type": entity.class_,
        "rdfs:label": entity.label,
        "memris:aliases": list(entity.aliases),
        "memris:createdAt": _ts(entity.created_at),
        "memris:mergedInto": _ref(ENTITY_BASE, entity.merged_into),
        "memris:removedAt": _ts(entity.removed_at),
        "memris:removedReason": entity.removed_reason,
        "memris:removedStatements": (
            [list(r) for r in entity.removed_statements] if entity.removed_statements else None
        ),
    }
    return {k: v for k, v in node.items() if v is not None}


def _statement_node(s: Statement) -> dict[str, Any]:
    obj: dict[str, Any]
    if s.object_id is not None:
        obj = {"@id": ENTITY_BASE + s.object_id}
    else:
        obj = {"@value": s.literal}
        if s.datatype is not None:
            obj["@type"] = s.datatype
    node: dict[str, Any] = {
        "@id": STATEMENT_BASE + s.id,
        "@type": _STATEMENT_TYPE,
        "rdf:subject": {"@id": ENTITY_BASE + s.subject_id},
        "rdf:predicate": {"@id": s.predicate},
        "rdf:object": obj,
        "prov:generatedAtTime": _ts(s.recorded_at),
        "prov:wasDerivedFrom": _ref(EPISODE_BASE, s.source_episode),
        "memris:validFrom": _ts(s.valid_from),
        "memris:validTo": _ts(s.valid_to),
        "memris:retractedAt": _ts(s.retracted_at),
        "memris:status": s.status,
        "memris:confidence": s.confidence,
        "memris:sourceTurn": s.source_turn,
        "memris:extractor": s.extractor,
        "memris:supersedes": _ref(STATEMENT_BASE, s.supersedes),
        "memris:ontologyVersion": s.ontology_version,
        "memris:reinforced": s.reinforced,
        "memris:lastReinforcedAt": _ts(s.last_reinforced_at),
        "memris:evidence": s.evidence,
        "memris:reason": s.reason,
        "memris:contradicts": _ref(STATEMENT_BASE, s.contradicts),
        "memris:reviewedAt": _ts(s.reviewed_at),
    }
    return {k: v for k, v in node.items() if v is not None}


def export_document(graph: Any) -> dict[str, Any]:
    """Every entity and statement in ``graph`` (a ``MemoryGraph``) as one JSON-LD document.

    Everything is exported, not just what is current: superseded, ended and retracted
    statements are the history, and an export that dropped them would not round-trip.
    """
    ontology: Ontology = graph.ontology
    entities = sorted(graph.store.find_entities(), key=lambda e: e.id)
    statements = sorted(graph.store.statements(), key=lambda s: s.id)
    return {
        "@context": build_context(ontology),
        "memris:ontology": {"@id": ontology.id, "memris:ontologyVersion": ontology.version},
        # The words memory learned (decision 7): every row, so an import can read the
        # statements that use them and keeps the candidates' counts.
        "memris:learnedTerms": [_term_node(t) for t in graph.store.terms()],
        "@graph": [_entity_node(e) for e in entities] + [_statement_node(s) for s in statements],
    }


_TERM_FIELDS = (
    "kind",
    "label",
    "domain",
    "range",
    "status",
    "alias_of",
    "observations",
    "episodes",
    "examples",
    "decided_by",
)
_TERM_TIMES = ("first_seen", "last_seen", "activated_at")


def _term_node(term: LearnedTerm) -> dict[str, Any]:
    node: dict[str, Any] = {"memris:name": term.name}
    for key in _TERM_FIELDS:
        value = getattr(term, key)
        node[f"memris:{key}"] = list(value) if isinstance(value, tuple) else value
    for key in _TERM_TIMES:
        node[f"memris:{key}"] = _ts(getattr(term, key))
    return node


def _read_term(node: dict[str, Any]) -> LearnedTerm:
    """A learned-term row back; raises (KeyError, ValueError) when it cannot be read."""
    first_seen = _dt(node["memris:first_seen"])
    last_seen = _dt(node["memris:last_seen"])
    if first_seen is None or last_seen is None:
        raise ValueError("a learned term needs first_seen and last_seen")
    status = node["memris:status"]
    kind = node["memris:kind"]
    if status not in ("candidate", "alias", "active", "rejected"):
        raise ValueError(f"unknown term status {status!r}")
    if kind not in ("relation", "attribute"):
        raise ValueError(f"unknown term kind {kind!r}")
    return LearnedTerm(
        name=str(node["memris:name"]),
        kind=kind,
        label=str(node["memris:label"]),
        domain=str(node["memris:domain"]),
        range=str(node["memris:range"]),
        status=status,
        first_seen=first_seen,
        last_seen=last_seen,
        alias_of=node.get("memris:alias_of"),
        observations=int(node.get("memris:observations", 1)),
        episodes=tuple(node.get("memris:episodes") or ()),
        examples=tuple(node.get("memris:examples") or ()),
        activated_at=_dt(node.get("memris:activated_at")),
        decided_by=node.get("memris:decided_by"),
    )


# --------------------------------------------------------------------------- import


@dataclass
class ImportReport:
    """What an import did, and everything it could not map (lossless-or-declared)."""

    entities: int = 0
    statements: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (node id, why)
    unmapped_fields: list[tuple[str, str]] = field(default_factory=list)  # (node id, key)

    @property
    def lossless(self) -> bool:
        return not self.skipped and not self.unmapped_fields


def _strip(value: Any, base: str) -> str | None:
    """``{"@id": base + x}`` (or a bare string) → ``x``; anything else → None."""
    if isinstance(value, dict):
        value = value.get("@id")
    if not isinstance(value, str):
        return None
    return value[len(base) :] if value.startswith(base) else value


def _dt(value: Any) -> datetime | None:
    """An ISO-8601 moment, always aware: a zone-less one from a foreign source is read as UTC.

    memris compares moments across statements; one naive value would make every later
    comparison raise. A malformed value raises ValueError for the caller to report.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("@value")
    moment = datetime.fromisoformat(str(value))
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _compact(ontology: Ontology, name: str) -> str:
    """A full IRI back to the ontology's ``prefix:local`` form; a compact name as is."""
    for prefix, base in sorted(ontology.prefixes.items(), key=lambda kv: -len(kv[1])):
        if name.startswith(base):
            return f"{prefix}:{name[len(base) :]}"
    return name


def import_document(document: dict[str, Any], graph: Any) -> ImportReport:
    """Read a JSON-LD document into ``graph`` (a ``MemoryGraph``), validated, all or nothing.

    Nodes whose class or property the ontology does not know are skipped and listed in
    the report, as are statements that depend on a skipped entity; keys memris does not
    read are listed too. Everything that maps is written in one ``MemoryGraph.import_``.
    """
    report = ImportReport()
    # Learned vocabulary first: the statements below may use it (decision 7).
    for node in document.get("memris:learnedTerms") or []:
        try:
            graph.store.save_term(_read_term(node))
        except (KeyError, ValueError, TypeError) as exc:
            report.skipped.append((str(node.get("memris:name")), f"unreadable learned term: {exc}"))
    graph.refresh_vocabulary()
    ontology: Ontology = graph.ontology
    nodes = document.get("@graph") or []
    entity_nodes = [n for n in nodes if n.get("@type") != _STATEMENT_TYPE]
    statement_nodes = [n for n in nodes if n.get("@type") == _STATEMENT_TYPE]

    def note_unmapped(node_id: str, node: dict[str, Any], allowed: frozenset[str]) -> None:
        report.unmapped_fields.extend((node_id, k) for k in sorted(node) if k not in allowed)

    entities: dict[str, Entity] = {}
    for node in entity_nodes:
        node_id = _strip(node.get("@id"), ENTITY_BASE)
        if node_id is None:
            report.skipped.append((str(node.get("@id")), "node without an @id"))
            continue
        class_name = _compact(ontology, str(node.get("@type")))
        if ontology.kind_of(class_name) != "class":
            report.skipped.append((node_id, f"unknown class {node.get('@type')!r}"))
            continue
        note_unmapped(node_id, node, _ENTITY_KEYS)
        try:
            created = _dt(node.get("memris:createdAt"))
            removed_at = _dt(node.get("memris:removedAt"))
        except ValueError as exc:
            report.skipped.append((node_id, f"unreadable timestamp: {exc}"))
            continue
        entities[node_id] = Entity(
            id=node_id,
            class_=class_name,
            label=str(node.get("rdfs:label", "")),
            aliases=tuple(node.get("memris:aliases") or ()),
            created_at=(
                created
                if created is not None
                else datetime.fromisoformat("1970-01-01T00:00:00+00:00")
            ),
            merged_into=_strip(node.get("memris:mergedInto"), ENTITY_BASE),
            removed_at=removed_at,
            removed_reason=node.get("memris:removedReason"),
            removed_statements=tuple(
                (str(r[0]), str(r[1]), None if r[2] is None else str(r[2]))
                for r in node.get("memris:removedStatements") or ()
            ),
        )

    known_ids = set(entities) | {e.id for e in graph.store.find_entities()}
    statements: list[Statement] = []
    for node in statement_nodes:
        node_id = _strip(node.get("@id"), STATEMENT_BASE)
        if node_id is None:
            report.skipped.append((str(node.get("@id")), "node without an @id"))
            continue
        predicate = _compact(ontology, str(_strip(node.get("rdf:predicate"), "")))
        if ontology.kind_of(predicate) not in ("relation", "attribute"):
            report.skipped.append((node_id, f"unknown property {predicate!r}"))
            continue
        subject = _strip(node.get("rdf:subject"), ENTITY_BASE)
        obj = node.get("rdf:object") or {}
        object_id = _strip(obj, ENTITY_BASE) if "@id" in obj else None
        missing = [x for x in (subject, object_id) if x is not None and x not in known_ids]
        if subject is None or missing:
            report.skipped.append(
                (node_id, f"refers to an entity not imported: {missing or subject}")
            )
            continue
        status = node.get("memris:status", "proposed")
        if status not in STATUSES:
            report.skipped.append((node_id, f"unknown status {status!r}"))
            continue
        note_unmapped(node_id, node, _STATEMENT_KEYS)
        try:
            times = {
                key: _dt(node.get(key))
                for key in (
                    "prov:generatedAtTime",
                    "memris:validFrom",
                    "memris:validTo",
                    "memris:retractedAt",
                    "memris:lastReinforcedAt",
                    "memris:reviewedAt",
                )
            }
        except ValueError as exc:
            report.skipped.append((node_id, f"unreadable timestamp: {exc}"))
            continue
        recorded = times["prov:generatedAtTime"]
        if recorded is None:
            report.skipped.append((node_id, "no prov:generatedAtTime"))
            continue
        confidence = node.get("memris:confidence")
        statements.append(
            Statement(
                id=node_id,
                subject_id=subject,
                predicate=predicate,
                recorded_at=recorded,
                object_id=object_id,
                literal=None if object_id is not None else obj.get("@value"),
                datatype=None if object_id is not None else obj.get("@type"),
                valid_from=times["memris:validFrom"],
                valid_to=times["memris:validTo"],
                retracted_at=times["memris:retractedAt"],
                status=status,
                confidence=None if confidence is None else float(confidence),
                source_episode=_strip(node.get("prov:wasDerivedFrom"), EPISODE_BASE),
                source_turn=node.get("memris:sourceTurn"),
                extractor=node.get("memris:extractor"),
                supersedes=_strip(node.get("memris:supersedes"), STATEMENT_BASE),
                ontology_version=node.get("memris:ontologyVersion"),
                reinforced=int(node.get("memris:reinforced", 1)),
                last_reinforced_at=times["memris:lastReinforcedAt"],
                evidence=node.get("memris:evidence"),
                reason=node.get("memris:reason"),
                contradicts=_strip(node.get("memris:contradicts"), STATEMENT_BASE),
                reviewed_at=times["memris:reviewedAt"],
            )
        )

    graph.import_(statements, list(entities.values()))
    report.entities, report.statements = len(entities), len(statements)
    return report


def iter_nodes(
    document: dict[str, Any], kind: Callable[[dict[str, Any]], bool]
) -> Iterable[dict[str, Any]]:
    """Nodes of a document matching ``kind`` — a convenience for callers and tests."""
    return (n for n in document.get("@graph") or [] if kind(n))


__all__ = [
    "ENTITY_BASE",
    "EPISODE_BASE",
    "MEMRIS_NS",
    "STATEMENT_BASE",
    "ImportReport",
    "build_context",
    "export_document",
    "import_document",
    "iter_nodes",
]
