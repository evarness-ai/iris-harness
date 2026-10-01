"""The mapping applier: foreign records → statements, lossless-or-declared (memris PR 9).

A toy ontology and a toy mappings file — the applier is tested on its own, with no
knowledge of any particular foreign system.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from textwrap import dedent

import pytest

from memris.graph import MemoryGraph
from memris.interchange import EntityRef, SourceRecord, apply_mappings
from memris.ontology import OntologyError, load_mappings, load_or_raise
from memris.store import InMemoryGraphStore

ONTOLOGY = """
ontology: { id: "urn:toy", version: "1.0.0", default_prefix: t, owner_class: Person }
prefixes: { t: "urn:toy#" }
classes:
  Thing:  { abstract: true }
  Person: { subclass_of: Thing }
  Org:    { subclass_of: Thing }
  Team:   { subclass_of: Org }
  Place:  { subclass_of: Thing }
relations:
  works_at: { domain: Person, range: Org }
  lives_in: { domain: Person, range: Place }
attributes:
  email: { domain: Person }
"""
MAPPINGS = """
mappings:
  - id: m_works
    when: { source_type: rel, key: [WORKS_AT, EMPLOYED_BY] }
    emit: { subject: $subject, predicate: works_at, object: { from: $object, class: Org } }
  - id: m_email
    when: { source_type: attr, key: EMAIL }
    emit: { subject: $subject, predicate: email, value: $value }
"""
T0 = datetime(2025, 3, 1, tzinfo=UTC)


@pytest.fixture
def setup(tmp_path: Path):  # type: ignore[no-untyped-def]
    (tmp_path / "ontology.yaml").write_text(dedent(ONTOLOGY), encoding="utf-8")
    (tmp_path / "connector.yaml").write_text(dedent(MAPPINGS), encoding="utf-8")
    onto = load_or_raise(tmp_path)
    graph = MemoryGraph(onto, InMemoryGraphStore())
    return graph, load_mappings(tmp_path / "connector.yaml", onto)


def _rel(sid: str, key: str, subj: EntityRef, obj: EntityRef, **kw: object) -> SourceRecord:
    return SourceRecord(source_type="rel", key=key, source_id=sid, subject=subj, object=obj, **kw)  # type: ignore[arg-type]


ANA = EntityRef("Ana", "Person")
ACME = EntityRef("Acme", "Org")


def test_a_record_becomes_a_statement_with_its_times_and_provenance(setup) -> None:  # type: ignore[no-untyped-def]
    graph, rules = setup
    report = apply_mappings(
        [
            _rel(
                "e1",
                "WORKS_AT",
                ANA,
                ACME,
                recorded_at=T0,
                valid_from=T0 - timedelta(days=30),
                valid_to=T0 + timedelta(days=5),
                episode="ep1",
                evidence="Ana works at Acme",
                extractor="toy",
                statement_id="st_toy_e1",
            )
        ],
        rules,
        graph,
    )
    assert report.lossless and report.statements == 1
    s = graph.store.get_statement("st_toy_e1")
    assert s is not None
    assert (s.predicate, s.status, s.recorded_at, s.valid_from, s.valid_to) == (
        "t:works_at",
        "proposed",
        T0,
        T0 - timedelta(days=30),
        T0 + timedelta(days=5),
    )
    assert (s.source_episode, s.evidence, s.extractor) == ("ep1", "Ana works at Acme", "toy")
    assert graph.get_entity(s.object_id).label == "Acme"  # type: ignore[union-attr, arg-type]


def test_every_unwritten_record_is_listed_with_why(setup) -> None:  # type: ignore[no-untyped-def]
    graph, rules = setup
    records = [
        _rel("ok", "WORKS_AT", ANA, ACME),
        _rel("u1", "FOUNDED", ANA, ACME),  # no mapping
        _rel("u2", "FOUNDED", ANA, ACME),
        _rel("c1", "WORKS_AT", EntityRef("Ana"), ACME),  # subject without a class
        _rel("d1", "WORKS_AT", EntityRef("Acme", "Org"), ACME),  # subject outside the domain
        _rel("r1", "WORKS_AT", ANA, EntityRef("Paris", "Place")),  # object outside the range
        _rel("a1", "WORKS_AT", EntityRef("?", "Thing"), ACME),  # abstract subject
        SourceRecord("attr", "EMAIL", "v1", subject=ANA),  # attribute without a value
    ]
    report = apply_mappings(records, rules, graph)
    why = dict(report.skipped)
    assert set(why) == {"u1", "u2", "c1", "d1", "r1", "a1", "v1"}
    assert "no mapping" in why["u1"] and "no known class" in why["c1"]
    assert "domain" in why["d1"] and "range" in why["r1"] and "abstract" in why["a1"]
    assert report.unmapped_types == {"rel:FOUNDED": 2}
    assert report.statements == 1 and not report.lossless  # the good one still landed


def test_entities_are_found_by_name_within_their_class(setup) -> None:  # type: ignore[no-untyped-def]
    graph, rules = setup
    known = graph.add_entity("Org", "Acme")
    report = apply_mappings(
        [
            _rel("e1", "WORKS_AT", ANA, ACME),
            _rel("e2", "EMPLOYED_BY", EntityRef("Bo", "Person"), ACME),
        ],
        rules,
        graph,
    )
    objects = {graph.store.get_statement(s.id).object_id for s in graph.store.statements()}  # type: ignore[union-attr]
    assert objects == {known.id}
    assert (report.entities_reused, report.entities_created) == (1, 2)  # Acme reused; Ana, Bo new


def test_an_object_may_narrow_the_mapping_class(setup) -> None:  # type: ignore[no-untyped-def]
    graph, rules = setup
    apply_mappings([_rel("e1", "WORKS_AT", ANA, EntityRef("Core", "Team"))], rules, graph)
    [s] = graph.store.statements()
    assert graph.get_entity(s.object_id).class_ == "t:Team"  # type: ignore[union-attr, arg-type]


def test_a_stable_id_makes_a_reimport_replace_not_duplicate(setup) -> None:  # type: ignore[no-untyped-def]
    graph, rules = setup
    record = _rel("e1", "WORKS_AT", ANA, ACME, statement_id="st_toy_e1")
    apply_mappings([record], rules, graph)
    again = apply_mappings([record], rules, graph)
    assert again.already_present == 1 and again.entities_created == 0
    assert len(graph.store.statements()) == 1 and len(graph.store.find_entities()) == 2


def test_a_reimport_updates_the_source_side_and_keeps_the_owners_decision(setup) -> None:  # type: ignore[no-untyped-def]
    """Confirmed or reviewed here stays so; a later end date from the source still lands."""
    graph, rules = setup
    first = _rel("e1", "WORKS_AT", ANA, ACME, statement_id="st_toy_e1", recorded_at=T0)
    apply_mappings([first], rules, graph)
    graph.confirm("st_toy_e1")
    graph.review(["st_toy_e1"])
    ended = _rel(
        "e1",
        "WORKS_AT",
        ANA,
        ACME,
        statement_id="st_toy_e1",
        recorded_at=T0,
        valid_to=T0 + timedelta(days=90),
        evidence="Ana left Acme",
    )
    apply_mappings([ended], rules, graph)  # default status is proposed
    kept = graph.store.get_statement("st_toy_e1")
    assert kept is not None
    assert kept.status == "confirmed" and kept.reviewed_at is not None  # the owner's decision
    assert kept.valid_to == T0 + timedelta(days=90) and kept.evidence == "Ana left Acme"


def test_imports_start_as_proposals_unless_confirmed(setup) -> None:  # type: ignore[no-untyped-def]
    graph, rules = setup
    apply_mappings([_rel("e1", "WORKS_AT", ANA, ACME)], rules, graph, status="confirmed")
    assert [s.status for s in graph.store.statements()] == ["confirmed"]
    with pytest.raises(ValueError, match="proposed or confirmed"):
        apply_mappings([], rules, graph, status="retracted")


def test_a_connector_mappings_file_is_checked_and_leaves_the_ontology_alone(tmp_path: Path) -> None:
    (tmp_path / "ontology.yaml").write_text(dedent(ONTOLOGY), encoding="utf-8")
    onto = load_or_raise(tmp_path)
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "mappings:\n  - id: x\n    when: { source_type: rel, key: K }\n"
        "    emit: { subject: $subject, predicate: flies, object: { from: $object, class: Org } }\n",
        encoding="utf-8",
    )
    with pytest.raises(OntologyError, match="flies"):
        load_mappings(bad, onto)
    (tmp_path / "good.yaml").write_text(dedent(MAPPINGS), encoding="utf-8")
    assert len(load_mappings(tmp_path / "good.yaml", onto)) == 2
    assert onto.mappings == []  # compiled beside the ontology, not into it
