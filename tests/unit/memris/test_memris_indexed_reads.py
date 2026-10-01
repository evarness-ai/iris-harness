"""Indexed reads: the same answers as the full scans they replace, and fewer connections.

``MemoryGraph`` and ``Resolver`` ask an :class:`IndexedGraphStore` for merged members
and live entities instead of loading every entity; a store without those extras falls
back to the scan. These tests hold the two paths to the same answers on a graph with
chained merges, a removal and look-alike names. The ontology is test data.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from memris.graph import MemoryGraph
from memris.model import Statement
from memris.ontology import Ontology, load_or_raise
from memris.resolve import Resolver
from memris.store import GraphStore, IndexedGraphStore, InMemoryGraphStore, SQLiteGraphStore
from memris.store.sqlite import SCHEMA_VERSION

ONTOLOGY = """
ontology: { id: "urn:test", version: "1.0.0", default_prefix: t, owner_class: Person }
prefixes: { t: "urn:test#" }
classes:
  Thing:  { abstract: true }
  Person: { subclass_of: Thing }
  Org:    { subclass_of: Thing }
  Bank:   { subclass_of: Org }
relations:
  works_at:   { domain: Person, range: Org }
  banks_with: { domain: Person, range: Org }
  knows:      { domain: Person, range: Person }
"""
T0 = datetime(2026, 1, 1, tzinfo=UTC)
NEW_INDEXES = {
    "memris_statements_predicate",
    "memris_entities_merged",
    "memris_entities_class",
    "memris_entity_decisions_b",
}
_HIDDEN = {"merged_members", "live_entities", "session"}


class ScanOnlyStore:
    """A GraphStore without the indexed extras — the path any third-party store takes."""

    def __init__(self, inner: GraphStore) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        if name in _HIDDEN:
            raise AttributeError(name)
        return getattr(self._inner, name)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def _later() -> datetime:
    """A reader's clock: after everything _populate recorded."""
    return T0 + timedelta(days=1)


def _ontology(tmp_path: Path) -> Ontology:
    root = tmp_path / "onto"
    root.mkdir(exist_ok=True)
    (root / "ontology.yaml").write_text(dedent(ONTOLOGY), encoding="utf-8")
    return load_or_raise(root)


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _populate(graph: MemoryGraph) -> dict[str, str]:
    """Banks merged in a chain (c into b, b into a), a removed person, statements both ways."""
    graph.ensure_owner("Me")
    a = graph.add_entity("Bank", "Northwind Bank")
    b = graph.add_entity("Bank", "Northwind", aliases=["Northwind Ltd"])
    c = graph.add_entity("Org", "H.D.F.C.")
    wingtip = graph.add_entity("Bank", "Wingtip Bank")
    ann = graph.add_entity("Person", "Ann", aliases=["Annie"])
    bob = graph.add_entity("Person", "Bob")
    gone = graph.add_entity("Person", "Ann B")
    for subj, pred, obj in [
        ("owner", "t:banks_with", c.id),
        ("owner", "t:banks_with", wingtip.id),
        (ann.id, "t:works_at", b.id),
        (bob.id, "t:works_at", a.id),
        (ann.id, "t:knows", bob.id),
        ("owner", "t:knows", ann.id),
    ]:
        graph.assert_(subj, pred, object_id=obj, status="confirmed")
    graph.merge(b.id, c.id, decided_by="test")
    graph.merge(a.id, b.id, decided_by="test")
    graph.remove_entity(gone.id)
    return {"a": a.id, "b": b.id, "c": c.id, "wingtip": wingtip.id, "ann": ann.id, "bob": bob.id}


def _answers(graph: MemoryGraph, ids: dict[str, str]) -> dict[str, object]:
    plain = Resolver(graph)
    folded = Resolver(graph, normalise=lambda n: n.removesuffix(" Ltd"))

    def found(r: Resolver, name: str, cls: str) -> object:
        hit = r.find(name, cls)
        return None if hit is None else (hit[0].id, hit[1])

    def ids_of(statements: list[Statement]) -> list[str]:
        return [s.id for s in statements]

    return {
        "members": {k: graph.members(v) for k, v in ids.items()},
        "canonical": {k: graph.canonical_id(v) for k, v in ids.items()},
        "neighbourhood": [
            (h, s.id) for h, s in graph.neighbourhood("owner", hops=3, direction="both")
        ],
        "current_in": ids_of(graph.current(None, "t:works_at", object_id=ids["a"])),
        "history": ids_of(graph.history("owner")),
        "find": [
            found(r, name, cls)
            for r in (plain, folded)
            for name, cls in [
                ("northwind", "Bank"),
                ("Northwind Ltd", "Org"),
                ("H.D.F.C.", "Bank"),
                ("annie", "Person"),
                ("Ann B", "Person"),
                ("Wingtip", "Bank"),
                ("Wingtip Bank Ltd", "Org"),
                ("Bob", "Org"),
                ("   ", "Person"),
            ]
        ],
    }


@pytest.fixture(params=["memory", "sqlite"])
def inner(request: pytest.FixtureRequest, tmp_path: Path) -> GraphStore:
    return (
        InMemoryGraphStore() if request.param == "memory" else SQLiteGraphStore(tmp_path / "g.db")
    )


def test_both_shipped_stores_offer_the_indexed_reads(inner: GraphStore) -> None:
    assert isinstance(inner, IndexedGraphStore)
    assert not isinstance(ScanOnlyStore(inner), IndexedGraphStore)


def test_indexed_reads_answer_exactly_as_the_full_scan(tmp_path: Path, inner: GraphStore) -> None:
    graph = MemoryGraph(_ontology(tmp_path), inner, clock=Clock())
    ids = _populate(graph)
    indexed = _answers(MemoryGraph(graph.declared, inner, clock=_later), ids)
    scanning = _answers(MemoryGraph(graph.declared, ScanOnlyStore(inner), clock=_later), ids)
    assert indexed == scanning
    assert indexed["members"]["a"] == [ids["a"], ids["b"], ids["c"]]  # type: ignore[index]
    assert indexed["canonical"]["c"] == ids["a"]  # type: ignore[index]


def test_resolve_records_the_same_look_alikes_either_way(tmp_path: Path) -> None:
    def run(wrap: bool) -> tuple[object, ...]:
        store: GraphStore = SQLiteGraphStore(tmp_path / f"r{wrap}.db")
        graph = MemoryGraph(_ontology(tmp_path), store, clock=Clock())
        _populate(graph)
        if wrap:
            graph = MemoryGraph(graph.declared, ScanOnlyStore(store), clock=graph._clock)
        out: list[object] = []
        for normalise in (None, lambda n: n.removesuffix(" Ltd")):
            kw: dict[str, Any] = {"normalise": normalise} if normalise else {}
            r = Resolver(graph, similarity=_similar, ask_similarity=0.6, **kw)
            for name, cls in [
                ("Wingtip Bnk", "Bank"),
                ("Anne", "Person"),
                ("Northwind Ltd", "Org"),
            ]:
                got = r.resolve(name, cls, episode="ep1")
                named = graph.get_entity(got.entity_id)
                assert named is not None
                labels = []
                for d in got.candidates:
                    other = graph.get_entity(d.a)
                    assert other is not None
                    labels.append(other.label)
                out.append((named.label, got.how, got.created, labels))
        return tuple(out)

    assert run(wrap=False) == run(wrap=True)


def test_a_store_without_the_extras_still_works(tmp_path: Path) -> None:
    graph = MemoryGraph(_ontology(tmp_path), ScanOnlyStore(InMemoryGraphStore()), clock=Clock())
    ids = _populate(graph)
    with graph.session():
        assert graph.members(ids["a"]) == [ids["a"], ids["b"], ids["c"]]


# --- the SQLite indexes -----------------------------------------------------------------


def _indexes(path: Path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def test_a_new_database_has_the_lookup_indexes(tmp_path: Path) -> None:
    SQLiteGraphStore(tmp_path / "g.db")
    assert NEW_INDEXES <= _indexes(tmp_path / "g.db")


def test_an_existing_database_gains_the_indexes_in_place(tmp_path: Path) -> None:
    path = tmp_path / "g.db"
    graph = MemoryGraph(_ontology(tmp_path), SQLiteGraphStore(path), clock=Clock())
    ids = _populate(graph)
    before = _answers(MemoryGraph(graph.declared, graph.store, clock=_later), ids)
    conn = sqlite3.connect(path)  # a database made before these indexes existed
    with conn:
        for name in NEW_INDEXES:
            conn.execute(f"DROP INDEX {name}")
    conn.close()
    assert not NEW_INDEXES & _indexes(path)

    reopened = MemoryGraph(graph.declared, SQLiteGraphStore(path), clock=_later)
    assert NEW_INDEXES <= _indexes(path)
    assert _answers(reopened, ids) == before
    # Indexes change no row: the schema version stays, so an older memris still opens it.
    assert reopened.store.get_meta("schema_version") == str(SCHEMA_VERSION)  # type: ignore[attr-defined]


# --- one connection per unit of work -------------------------------------------------


def test_a_session_shares_one_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SQLiteGraphStore(tmp_path / "g.db")
    graph = MemoryGraph(_ontology(tmp_path), store, clock=Clock())
    ids = _populate(graph)
    opened: list[int] = []
    real_open = store._open

    def counting_open() -> sqlite3.Connection:
        opened.append(1)
        return real_open()

    monkeypatch.setattr(store, "_open", counting_open)
    graph.neighbourhood("owner", hops=3)
    assert len(opened) == 1
    opened.clear()
    with graph.session(), graph.session():  # nested: still the outer connection
        graph.members(ids["a"])
        graph.current("owner")
    assert len(opened) == 1


def test_a_failed_call_in_a_session_undoes_only_itself(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "g.db")
    graph = MemoryGraph(_ontology(tmp_path), store, clock=Clock())
    graph.ensure_owner("Me")
    with store.session():
        kept = graph.add_entity("Person", "Kept")
        with pytest.raises(sqlite3.IntegrityError), store._connect() as conn:
            conn.execute("DELETE FROM memris_entity_names WHERE entity_id = ?", (kept.id,))
            conn.execute("INSERT INTO memris_entity_names(entity_id, name_key) VALUES ('x', 'y')")
    assert store.get_entity(kept.id) is not None
    assert [e.id for e in store.find_entities(label="kept")] == [kept.id]
