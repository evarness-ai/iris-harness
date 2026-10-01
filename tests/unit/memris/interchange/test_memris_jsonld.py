"""JSON-LD interchange (ADR-0115 decisions 11 and 13.7).

The round trip is the CI gate: memris → JSON-LD → memris must reproduce every entity
and every statement exactly — ids, all time fields, status, provenance — on each store.
The ontology here is test data; memris knows none of it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from textwrap import dedent

import pytest

from memris.graph import MemoryGraph, StatementError
from memris.interchange import (
    ENTITY_BASE,
    STATEMENT_BASE,
    build_context,
    export_document,
    import_document,
)
from memris.interchange.__main__ import main
from memris.model import OWNER_ID
from memris.ontology import Ontology, load_or_raise
from memris.store import GraphStore, InMemoryGraphStore, SQLiteGraphStore

ONTOLOGY = """
ontology: { id: "urn:test:onto", version: "2.0.0", default_prefix: t, owner_class: Person }
prefixes: { t: "urn:test#", schema: "https://schema.org/" }
classes:
  Thing:   { abstract: true }
  Person:  { subclass_of: Thing, maps_to: schema:Person }
  Org:     { subclass_of: Thing }
  Country: { subclass_of: Thing }
relations:
  works_at:   { domain: Person, range: Org, inverse: employs }
  citizen_of: { domain: Person, range: Country }
attributes:
  email:    { domain: Person }
  birthday: { domain: Person, datatype: date }
"""
SHAPES = """
Person:
  properties:
    works_at: { max_count: 1 }
    email:    { max_count: 1 }
"""
T0 = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, days: int = 1) -> None:
        self.now += timedelta(days=days)


def _ontology(root: Path, text: str = ONTOLOGY) -> Ontology:
    root.mkdir(parents=True, exist_ok=True)
    (root / "ontology.yaml").write_text(dedent(text), encoding="utf-8")
    (root / "shapes.yaml").write_text(dedent(SHAPES), encoding="utf-8")
    return load_or_raise(root)


def _store(kind: str, path: Path) -> GraphStore:
    return InMemoryGraphStore() if kind == "memory" else SQLiteGraphStore(path)


@pytest.fixture(params=["memory", "sqlite"])
def kind(request: pytest.FixtureRequest) -> Iterator[str]:
    yield request.param


def _populated(onto: Ontology, store: GraphStore) -> MemoryGraph:
    """A graph exercising every field: history, retraction, proposal, provenance, merge."""
    clock = Clock()
    g = MemoryGraph(onto, store, clock=clock)
    g.ensure_owner("Me")
    barclays = g.add_entity("Org", "Barclays", aliases=["Barclays PLC"])
    litware = g.add_entity("Org", "Litware")
    old_litware = g.add_entity("Org", "Litware Bank")
    store.save(
        entities=[old_litware.__class__(**{**old_litware.__dict__, "merged_into": litware.id})]
    )
    india = g.add_entity("Country", "India")
    uk = g.add_entity("Country", "United Kingdom")

    first = g.assert_(
        OWNER_ID,
        "works_at",
        object_id=barclays.id,
        status="confirmed",
        confidence=0.9,
        source_episode="conv-1",
        source_turn="turn-3",
        extractor="conversation:llm",
        valid_from=T0 - timedelta(days=700),
        evidence="I work at Barclays",
    )
    clock.advance(30)
    g.assert_(OWNER_ID, "works_at", object_id=litware.id, status="confirmed", confidence=0.95)
    g.assert_(OWNER_ID, "citizen_of", object_id=india.id, status="confirmed")
    g.assert_(OWNER_ID, "citizen_of", object_id=uk.id)  # a proposal
    g.assert_(OWNER_ID, "birthday", literal="1990-04-02", status="confirmed")
    clock.advance()
    mail = g.assert_(OWNER_ID, "email", literal="old@example.org", status="confirmed")
    g.refresh(mail.id, reinforced=5, reason="approved")
    clock.advance()
    g.retract(mail.id, reason="forgot")
    wrong = g.assert_(OWNER_ID, "email", literal="junk@example.org", evidence="pasted text")
    store.save(
        statements=[
            wrong.evolve(contradicts=first.id, reviewed_at=clock.now, reason="refused"),
        ]
    )
    return g


def _snapshot(g: MemoryGraph) -> tuple[list, list]:  # type: ignore[type-arg]
    return (
        sorted(g.store.find_entities(), key=lambda e: e.id),
        sorted(g.store.statements(), key=lambda s: s.id),
    )


# --- the gate ---------------------------------------------------------------------


def test_round_trip_reproduces_every_record_exactly(tmp_path: Path, kind: str) -> None:
    onto = _ontology(tmp_path / "onto")
    source = _populated(onto, _store(kind, tmp_path / "a.db"))
    text = json.dumps(export_document(source))  # through real JSON, not a dict copy

    target = MemoryGraph(onto, _store(kind, tmp_path / "b.db"))
    report = import_document(json.loads(text), target)

    assert report.lossless, report
    assert _snapshot(target) == _snapshot(source)
    assert report.statements == len(_snapshot(source)[1]) >= 7
    # the history survived as history, not just the current values
    assert {s.status for s in target.store.statements()} == {"confirmed", "proposed", "retracted"}


def test_exporting_again_after_import_is_identical(tmp_path: Path) -> None:
    onto = _ontology(tmp_path / "onto")
    source = _populated(onto, InMemoryGraphStore())
    document = export_document(source)
    target = MemoryGraph(onto, SQLiteGraphStore(tmp_path / "b.db"))
    import_document(json.loads(json.dumps(document)), target)
    assert export_document(target) == document


# --- the document -----------------------------------------------------------------


def test_the_context_is_generated_from_the_ontology(tmp_path: Path) -> None:
    onto = _ontology(tmp_path / "onto")
    context = build_context(onto)
    for name in (*onto.classes, *onto.relations, *onto.attributes):
        assert context[name]["@id"] == onto.expand(name)
    assert context["t:works_at"]["@type"] == "@id"
    assert context["t:birthday"]["@type"].endswith("#date")
    assert context["t"] == "urn:test#" and context["schema"] == "https://schema.org/"

    grown = _ontology(
        tmp_path / "grown",
        ONTOLOGY.replace(
            "  citizen_of: { domain: Person, range: Country }\n",
            "  citizen_of: { domain: Person, range: Country }\n  mentors: { domain: Person, range: Person }\n",
        ),
    )
    assert "t:mentors" in build_context(grown) and "t:mentors" not in context


def test_statements_are_nodes_with_prov_time_and_provenance(tmp_path: Path) -> None:
    onto = _ontology(tmp_path / "onto")
    document = export_document(_populated(onto, InMemoryGraphStore()))
    first = next(n for n in document["@graph"] if n.get("memris:evidence") == "I work at Barclays")
    assert first["@type"] == "rdf:Statement"
    assert first["rdf:predicate"] == {"@id": "t:works_at"}
    assert first["rdf:subject"] == {"@id": ENTITY_BASE + OWNER_ID}
    assert first["prov:generatedAtTime"] == T0.isoformat()
    assert first["prov:wasDerivedFrom"] == {"@id": "urn:memris:episode:conv-1"}
    assert "memris:validTo" in first  # superseded: its end is in the document
    birthday = next(
        n for n in document["@graph"] if n.get("rdf:predicate") == {"@id": "t:birthday"}
    )
    assert birthday["rdf:object"]["@type"].endswith("#date")
    assert document["memris:ontology"] == {
        "@id": "urn:test:onto",
        "memris:ontologyVersion": "2.0.0",
    }


# --- lossless-or-declared ---------------------------------------------------------


def test_what_cannot_be_mapped_is_reported_not_dropped(tmp_path: Path) -> None:
    onto = _ontology(tmp_path / "onto")
    document = export_document(_populated(onto, InMemoryGraphStore()))
    document["@graph"] += [
        {"@id": ENTITY_BASE + "ent_alien", "@type": "t:Spaceship", "rdfs:label": "Enterprise"},
        {
            "@id": STATEMENT_BASE + "st_alien",
            "@type": "rdf:Statement",
            "rdf:subject": {"@id": ENTITY_BASE + OWNER_ID},
            "rdf:predicate": {"@id": "t:pilots"},
            "rdf:object": {"@id": ENTITY_BASE + "ent_alien"},
            "prov:generatedAtTime": T0.isoformat(),
        },
        {
            "@id": STATEMENT_BASE + "st_orphan",
            "@type": "rdf:Statement",
            "rdf:subject": {"@id": ENTITY_BASE + OWNER_ID},
            "rdf:predicate": {"@id": "t:works_at"},
            "rdf:object": {"@id": ENTITY_BASE + "ent_alien"},
            "prov:generatedAtTime": T0.isoformat(),
        },
    ]
    document["@graph"][0]["foo:colour"] = "blue"

    target = MemoryGraph(onto, InMemoryGraphStore())
    report = import_document(document, target)

    skipped = dict(report.skipped)
    assert "unknown class" in skipped["ent_alien"]
    assert "unknown property" in skipped["st_alien"]
    assert "not imported" in skipped["st_orphan"]
    assert (
        document["@graph"][0]["@id"].removeprefix(ENTITY_BASE),
        "foo:colour",
    ) in report.unmapped_fields
    assert not report.lossless
    assert report.statements >= 7  # everything that maps still came in


def test_full_iris_resolve_to_the_ontology_names(tmp_path: Path) -> None:
    """Another tool may write full IRIs; they fold back to the ontology's own names."""
    onto = _ontology(tmp_path / "onto")
    document = export_document(_populated(onto, InMemoryGraphStore()))
    for node in document["@graph"]:
        if node["@type"] != "rdf:Statement":
            node["@type"] = onto.expand(node["@type"])
        else:
            node["rdf:predicate"] = {"@id": onto.expand(node["rdf:predicate"]["@id"])}
    target = MemoryGraph(onto, InMemoryGraphStore())
    assert import_document(document, target).lossless
    assert {s.predicate for s in target.store.statements()} <= {
        "t:works_at",
        "t:citizen_of",
        "t:email",
        "t:birthday",
    }


def test_import_validates_against_the_ontology_all_or_nothing(tmp_path: Path) -> None:
    onto = _ontology(tmp_path / "onto")
    document = export_document(_populated(onto, InMemoryGraphStore()))
    stmt = next(n for n in document["@graph"] if n.get("rdf:predicate") == {"@id": "t:works_at"})
    country = next(n for n in document["@graph"] if n.get("@type") == "t:Country")
    stmt["rdf:object"] = {"@id": country["@id"]}  # a country is not an employer
    target = MemoryGraph(onto, InMemoryGraphStore())
    with pytest.raises(StatementError, match="outside the range"):
        import_document(document, target)
    assert target.store.statements() == [] and target.store.find_entities() == []


# --- the command ------------------------------------------------------------------


def test_the_command_round_trips_a_sqlite_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    onto_dir = tmp_path / "onto"
    onto = _ontology(onto_dir)
    _populated(onto, SQLiteGraphStore(tmp_path / "a.db"))
    out = tmp_path / "memory.jsonld"
    assert main(["export", str(onto_dir), str(tmp_path / "a.db"), "--out", str(out)]) == 0
    assert main(["import", str(onto_dir), str(tmp_path / "b.db"), str(out)]) == 0
    assert "imported" in capsys.readouterr().out
    a = MemoryGraph(onto, SQLiteGraphStore(tmp_path / "a.db"))
    b = MemoryGraph(onto, SQLiteGraphStore(tmp_path / "b.db"))
    assert _snapshot(a) == _snapshot(b)


def _foreign(node_id: str, **extra: object) -> dict[str, object]:
    return {
        "@id": STATEMENT_BASE + node_id,
        "@type": "rdf:Statement",
        "rdf:subject": {"@id": ENTITY_BASE + OWNER_ID},
        "rdf:predicate": {"@id": "t:email"},
        "rdf:object": {"@value": f"{node_id}@example.org"},
        "prov:generatedAtTime": T0.isoformat(),
        **extra,
    }


def test_foreign_documents_cannot_poison_the_store(tmp_path: Path) -> None:
    """A zone-less time is read as UTC; a bad status or timestamp is reported, not stored."""
    onto = _ontology(tmp_path / "onto")
    document = export_document(_populated(onto, InMemoryGraphStore()))
    document["@graph"] += [
        _foreign("st_naive", **{"prov:generatedAtTime": "2026-01-01T09:00:00"}),
        _foreign("st_status", **{"memris:status": "active"}),
        _foreign("st_garbled", **{"memris:validTo": "next tuesday"}),
    ]
    target = MemoryGraph(onto, InMemoryGraphStore())
    report = import_document(document, target)

    skipped = dict(report.skipped)
    assert "unknown status" in skipped["st_status"]
    assert "unreadable timestamp" in skipped["st_garbled"]
    naive = target.store.get_statement("st_naive")
    assert naive is not None and naive.recorded_at.tzinfo is not None
    # and the store still answers time questions after the import
    target.current(OWNER_ID, "email", include_proposed=True)
