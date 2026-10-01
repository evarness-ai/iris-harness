"""Moving a memory graph in and out of memris: JSON-LD (ADR-0115, decision 11).

``export_document(graph)`` writes every entity and every statement — current,
superseded, retracted — as JSON-LD whose ``@context`` is generated from the compiled
ontology, so the same file reads as plain JSON to one tool and as RDF to another.
``apply_mappings(records, rules, graph)`` is the other half of decision 11: an importer
for a foreign system is a reader yielding ``SourceRecord`` s plus a mappings file.
``import_document(document, graph)`` reads it back through ``MemoryGraph.import_``,
so each record is checked against the ontology, and returns a report of anything it
could not map instead of dropping it silently (lossless-or-declared).

memris → JSON-LD → memris reproduces entities and statements exactly; the round-trip
test is the CI gate for that promise.
"""

from __future__ import annotations

from memris.interchange.jsonld import (
    ENTITY_BASE,
    EPISODE_BASE,
    MEMRIS_NS,
    STATEMENT_BASE,
    ImportReport,
    build_context,
    export_document,
    import_document,
)
from memris.interchange.mapping import (
    EntityRef,
    MappingReport,
    SourceRecord,
    apply_mappings,
)

__all__ = [
    "ENTITY_BASE",
    "EntityRef",
    "MappingReport",
    "SourceRecord",
    "apply_mappings",
    "EPISODE_BASE",
    "MEMRIS_NS",
    "STATEMENT_BASE",
    "ImportReport",
    "build_context",
    "export_document",
    "import_document",
]
