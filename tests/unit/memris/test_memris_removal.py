"""Removing an entity, restoring it, deleting it for good (ADR-0119), on every store."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from test_memris_graph import Clock, _ontology

from memris.graph import MemoryGraph, StatementError
from memris.interchange import export_document, import_document
from memris.model import OWNER_ID
from memris.resolve import Resolver
from memris.store import GraphStore, InMemoryGraphStore, SQLiteGraphStore


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[GraphStore]:
    yield InMemoryGraphStore() if request.param == "memory" else SQLiteGraphStore(tmp_path / "g.db")


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def graph(tmp_path: Path, store: GraphStore, clock: Clock) -> MemoryGraph:
    g = MemoryGraph(_ontology(tmp_path), store, clock=clock)
    g.ensure_owner("Owner", "Person")
    return g


def _banks(graph: MemoryGraph) -> list[str]:
    return [s.object_id or "" for s in graph.current(OWNER_ID, "banks_with", include_proposed=True)]


def test_removing_an_entity_withdraws_what_holds_about_it_and_restore_puts_back_exactly_that(
    graph: MemoryGraph, clock: Clock
) -> None:
    northwind = graph.add_entity("Bank", "Northwind Bank")
    confirmed = graph.assert_(OWNER_ID, "banks_with", object_id=northwind.id, status="confirmed")
    stale = graph.assert_(OWNER_ID, "works_at", object_id=northwind.id)
    graph.retract(stale.id, reason="forgot")
    stale = graph.store.get_statement(stale.id)
    proposed = graph.assert_(OWNER_ID, "works_at", object_id=northwind.id)
    earlier = graph.assert_(OWNER_ID, "email", literal="a@x", status="confirmed")
    forgotten = graph.assert_(
        OWNER_ID, "citizen_of", object_id=graph.add_entity("Country", "India").id
    )
    graph.retract(forgotten.id, reason="forgot")
    clock.advance()

    removed = graph.remove_entity(northwind.id)

    assert removed.removed and removed.removed_reason == "removed"
    assert {r[0] for r in removed.removed_statements} == {confirmed.id, proposed.id}
    assert _banks(graph) == [] and graph.current(OWNER_ID, "works_at", include_proposed=True) == []
    assert graph.neighbourhood(northwind.id, include_proposed=True) == []
    assert graph.current(OWNER_ID, "email")[0].id == earlier.id  # untouched

    graph.restore_entity(northwind.id)

    back = {s.id: s for s in graph.store.statements()}
    assert back[confirmed.id].status == "confirmed" and back[confirmed.id].retracted_at is None
    assert back[proposed.id].status == "proposed"
    assert back[forgotten.id].status == "retracted" and back[forgotten.id].reason == "forgot"
    assert back[stale.id] == stale  # already withdrawn: not the removal's to touch
    assert not graph.get_entity(northwind.id).removed  # type: ignore[union-attr]
    assert _banks(graph) == [northwind.id]


def test_removal_does_not_reopen_what_the_withdrawn_claim_replaced(
    graph: MemoryGraph, clock: Clock
) -> None:
    old = graph.add_entity("Org", "Barclays")
    new = graph.add_entity("Org", "Litware")
    graph.assert_(OWNER_ID, "works_at", object_id=old.id, status="confirmed")
    clock.advance()
    graph.assert_(OWNER_ID, "works_at", object_id=new.id, status="confirmed")
    clock.advance()

    graph.remove_entity(new.id)

    assert graph.current(OWNER_ID, "works_at") == []


def test_a_single_valued_claim_replaced_while_removed_comes_back_closed(
    graph: MemoryGraph, clock: Clock
) -> None:
    first = graph.add_entity("Org", "Acme")
    graph.assert_(OWNER_ID, "works_at", object_id=first.id, status="confirmed")
    clock.advance()
    graph.remove_entity(first.id)
    clock.advance()
    second = graph.add_entity("Org", "Globex")
    graph.assert_(OWNER_ID, "works_at", object_id=second.id, status="confirmed")

    graph.restore_entity(first.id)

    assert [s.object_id for s in graph.current(OWNER_ID, "works_at")] == [second.id]


def test_a_removed_entity_is_out_of_writes_and_resolution(graph: MemoryGraph) -> None:
    northwind = graph.add_entity("Bank", "Northwind Bank")
    graph.remove_entity(northwind.id)

    with pytest.raises(StatementError, match="was removed; restore it first"):
        graph.assert_(OWNER_ID, "banks_with", object_id=northwind.id)
    resolver = Resolver(graph, similarity=lambda a, b: 1.0 if a[:4] == b[:4] else 0.0)
    assert resolver.find("Northwind Bank", "Bank") is None
    made = resolver.resolve("Northwind Bank", "Bank")
    assert made.created and made.entity_id != northwind.id and made.candidates == ()


def test_the_owner_cannot_be_removed(graph: MemoryGraph) -> None:
    with pytest.raises(StatementError, match="owner cannot be removed"):
        graph.remove_entity(OWNER_ID)


def test_delete_needs_a_removal_first_and_takes_every_statement_naming_it(
    graph: MemoryGraph, clock: Clock
) -> None:
    northwind = graph.add_entity("Bank", "Northwind Bank")
    other = graph.add_entity("Bank", "Northwind")
    graph.note_candidate(northwind.id, other.id, score=0.9, episode="e1")
    held = graph.assert_(OWNER_ID, "banks_with", object_id=northwind.id, status="confirmed")
    with pytest.raises(StatementError, match="was not removed"):
        graph.delete_entity(northwind.id)
    graph.remove_entity(northwind.id)

    assert graph.delete_entity(northwind.id) == 1

    assert graph.get_entity(northwind.id) is None
    assert graph.store.get_statement(held.id) is None
    assert graph.decisions(northwind.id) == []
    assert graph.get_entity(other.id) is not None


def test_delete_is_refused_while_a_statement_about_it_still_holds(graph: MemoryGraph) -> None:
    northwind = graph.add_entity("Bank", "Northwind Bank")
    graph.remove_entity(northwind.id)
    # Written behind the graph's back (an import, a second process): still holding.
    graph.store.save(
        statements=[
            graph.assert_(OWNER_ID, "email", literal="x").evolve(
                predicate="t:banks_with", object_id=northwind.id, literal=None
            )
        ]
    )
    with pytest.raises(StatementError, match="still holds"):
        graph.delete_entity(northwind.id)


def test_delete_is_refused_while_entities_are_merged_into_it(graph: MemoryGraph) -> None:
    northwind = graph.add_entity("Bank", "Northwind Bank")
    alias = graph.add_entity("Bank", "Northwind")
    graph.merge(northwind.id, alias.id, decided_by="owner")
    graph.remove_entity(northwind.id)
    with pytest.raises(StatementError, match="unmerge them first"):
        graph.delete_entity(northwind.id)


def test_a_removed_entity_survives_the_json_ld_round_trip(
    graph: MemoryGraph, tmp_path: Path
) -> None:
    northwind = graph.add_entity("Bank", "Northwind Bank")
    graph.assert_(OWNER_ID, "banks_with", object_id=northwind.id, status="confirmed")
    graph.remove_entity(northwind.id)
    text = json.dumps(export_document(graph))

    target = MemoryGraph(graph.declared, SQLiteGraphStore(tmp_path / "copy.db"))
    report = import_document(json.loads(text), target)

    assert report.lossless, report
    assert target.get_entity(northwind.id) == graph.get_entity(northwind.id)
    assert export_document(target) == json.loads(text)


def test_a_v5_database_gains_the_removal_columns(tmp_path: Path) -> None:
    path = tmp_path / "v5.db"
    SQLiteGraphStore(path)
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE memris_entities DROP COLUMN removed_statements")
        conn.execute("ALTER TABLE memris_entities DROP COLUMN removed_reason")
        conn.execute("ALTER TABLE memris_entities DROP COLUMN removed_at")
        conn.execute("UPDATE memris_meta SET value = '5' WHERE key = 'schema_version'")

    store = SQLiteGraphStore(path)
    graph = MemoryGraph(_ontology(tmp_path), store)
    graph.ensure_owner("Owner", "Person")

    assert store.get_meta("schema_version") == "6"
    assert graph.remove_entity(graph.add_entity("Org", "Acme").id).removed


def test_reinstate_undoes_a_retraction_and_closes_what_it_reopened(
    graph: MemoryGraph, clock: Clock
) -> None:
    old = graph.assert_(OWNER_ID, "email", literal="old@x", status="confirmed")
    clock.advance()
    new = graph.assert_(OWNER_ID, "email", literal="new@x", status="confirmed")
    graph.retract(new.id, reason="forgot")
    assert [s.id for s in graph.current(OWNER_ID, "email")] == [old.id]  # reopened

    back = graph.reinstate(new.id, reason="restored")

    assert back.status == "confirmed" and back.reason == "restored"
    assert [s.id for s in graph.current(OWNER_ID, "email")] == [new.id]


def test_reinstate_refuses_what_is_not_retracted_or_names_a_removed_entity(
    graph: MemoryGraph,
) -> None:
    northwind = graph.add_entity("Bank", "Northwind Bank")
    held = graph.assert_(OWNER_ID, "banks_with", object_id=northwind.id, status="confirmed")
    with pytest.raises(StatementError, match="not retracted"):
        graph.reinstate(held.id)
    graph.retract(held.id, reason="forgot")
    graph.remove_entity(northwind.id)
    with pytest.raises(StatementError, match="was removed"):
        graph.reinstate(held.id)
    with pytest.raises(StatementError, match="proposed or confirmed"):
        graph.reinstate(held.id, status="retracted")
