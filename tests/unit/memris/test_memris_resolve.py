"""Entity resolution and merges (ADR-0115 decision 4), on every GraphStore.

The ontology and the similarity function here are test data; memris knows neither.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from textwrap import dedent

import pytest

from memris.graph import MemoryGraph, StatementError
from memris.model import OWNER_ID
from memris.ontology import load_or_raise
from memris.resolve import Resolver
from memris.store import GraphStore, InMemoryGraphStore, SQLiteGraphStore

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
"""
T0 = datetime(2026, 1, 1, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)  # every event a distinct moment
        return self.now


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


@pytest.fixture(params=["memory", "sqlite"])
def graph(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[MemoryGraph]:
    (tmp_path / "ontology.yaml").write_text(dedent(ONTOLOGY), encoding="utf-8")
    store: GraphStore = (
        InMemoryGraphStore() if request.param == "memory" else SQLiteGraphStore(tmp_path / "g.db")
    )
    g = MemoryGraph(load_or_raise(tmp_path), store, clock=Clock())
    g.ensure_owner("Me")
    yield g


def _resolver(graph: MemoryGraph, **kw: object) -> Resolver:
    return Resolver(graph, similarity=_similar, ask_similarity=0.7, evidence_episodes=3, **kw)  # type: ignore[arg-type]


# --- finding what a name means ----------------------------------------------------


def test_an_exact_name_ignoring_case_and_spacing_is_the_same_entity(graph: MemoryGraph) -> None:
    r = _resolver(graph)
    first = r.resolve("Northwind  Bank", "Bank")
    again = r.resolve("northwind bank", "Bank")
    assert first.created and not again.created
    assert again.entity_id == first.entity_id and again.how == "exact"


def test_an_alias_or_a_folded_name_is_the_same_entity(graph: MemoryGraph) -> None:
    bank = graph.add_entity("Bank", "Northwind Bank", aliases=["Northwind"])
    assert _resolver(graph).resolve("Northwind", "Bank").entity_id == bank.id
    folded = Resolver(graph, normalise=lambda n: n.removesuffix(" Ltd"))
    assert folded.resolve("Northwind Bank Ltd", "Bank").entity_id == bank.id


def test_a_name_in_an_unrelated_class_is_a_different_entity(graph: MemoryGraph) -> None:
    person = graph.add_entity("Person", "Jordan")
    assert _resolver(graph).resolve("Jordan", "Org").entity_id != person.id
    # a subclass is compatible: a Bank found when an Org is wanted
    bank = graph.add_entity("Bank", "Axis")
    assert _resolver(graph).resolve("Axis", "Org").entity_id == bank.id


# --- when unsure, keep them separate ----------------------------------------------


def test_a_look_alike_is_a_new_entity_and_a_recorded_candidate(graph: MemoryGraph) -> None:
    r = _resolver(graph)
    bank = r.resolve("Northwind Bank", "Bank", episode="e1")
    near = r.resolve("Northwind Banks", "Bank", episode="e1")
    assert near.created and near.entity_id != bank.entity_id  # separate, not merged
    [candidate] = near.candidates
    assert candidate.decision == "candidate" and candidate.evidence == ("e1",)
    assert r.resolve("Barclays", "Bank").candidates == ()  # nothing close, nothing noted


def test_evidence_across_episodes_merges_with_the_evidence_recorded(graph: MemoryGraph) -> None:
    r = _resolver(graph)
    old = r.resolve("Northwind Bank", "Bank", episode="e1").entity_id
    new = r.resolve("Northwind Banks", "Bank", episode="e1").entity_id
    r.resolve("Northwind Banks", "Bank", episode="e2")
    assert (
        graph.get_entity(new).merged_into is None
    )  # two episodes: not yet  # type: ignore[union-attr]
    third = r.resolve("Northwind Bank", "Bank", episode="e3")
    [merge] = third.merged
    assert (merge.decision, merge.decided_by, merge.a, merge.b) == ("same", "evidence", old, new)
    assert merge.evidence == ("e1", "e2", "e3")
    assert graph.get_entity(new).merged_into == old  # type: ignore[union-attr]
    assert "Northwind Banks" in graph.get_entity(old).aliases  # type: ignore[union-attr]
    assert (
        r.resolve("Northwind Banks", "Bank").entity_id == old
    )  # the folded name now resolves home


def test_a_pair_marked_distinct_is_never_proposed_or_merged(graph: MemoryGraph) -> None:
    r = _resolver(graph)
    a = r.resolve("Northwind Bank", "Bank", episode="e1").entity_id
    b = r.resolve("Northwind Banks", "Bank", episode="e1").entity_id
    graph.mark_distinct(a, b, decided_by="owner")
    for episode in ("e2", "e3", "e4"):
        r.resolve("Northwind Banks", "Bank", episode=episode)
    assert graph.get_entity(b).merged_into is None  # type: ignore[union-attr]
    with pytest.raises(StatementError, match="marked distinct"):
        graph.merge(a, b, decided_by="owner")


# --- merges: read-through, undo, limits --------------------------------------------


def test_reads_follow_a_merge_and_nothing_is_rewritten(graph: MemoryGraph) -> None:
    northwind = graph.add_entity("Bank", "Northwind")
    northwind_bank = graph.add_entity("Bank", "Northwind Bank")
    said = graph.assert_(OWNER_ID, "banks_with", object_id=northwind.id, status="confirmed")
    graph.merge(northwind_bank.id, northwind.id, decided_by="owner")
    assert [s.id for s in graph.current(object_id=northwind_bank.id)] == [said.id]
    assert graph.store.get_statement(said.id).object_id == northwind.id  # type: ignore[union-attr]
    assert graph.canonical_id(northwind.id) == northwind_bank.id
    with pytest.raises(StatementError, match="merged into"):
        graph.assert_(OWNER_ID, "banks_with", object_id=northwind.id)  # new writes use the survivor


def test_unmerge_restores_both_as_they_were(graph: MemoryGraph) -> None:
    northwind = graph.add_entity("Bank", "Northwind")
    northwind_bank = graph.add_entity("Bank", "Northwind Bank", aliases=["Northwind Bank Ltd"])
    decision = graph.merge(northwind_bank.id, northwind.id, decided_by="evidence")
    undone = graph.unmerge(decision.id, decided_by="owner")
    assert undone.decision == "undone"
    assert graph.get_entity(northwind.id).merged_into is None  # type: ignore[union-attr]
    assert graph.get_entity(northwind_bank.id).aliases == ("Northwind Bank Ltd",)  # type: ignore[union-attr]
    assert graph.current(object_id=northwind_bank.id) == []


def test_unrelated_classes_never_merge(graph: MemoryGraph) -> None:
    person = graph.add_entity("Person", "Jordan")
    org = graph.add_entity("Org", "Jordan")
    with pytest.raises(StatementError, match="unrelated classes"):
        graph.merge(org.id, person.id, decided_by="owner")


# --- the owner decides a look-alike (PR 4b) ----------------------------------------


def _pair(graph: MemoryGraph) -> tuple[str, str, str]:
    r = _resolver(graph)
    old = r.resolve("Northwind Bank", "Bank", episode="e1").entity_id
    near = r.resolve("Northwind Banks", "Bank", episode="e1")
    [candidate] = near.candidates
    return old, near.entity_id, candidate.id


def test_open_candidates_are_the_undecided_pairs(graph: MemoryGraph) -> None:
    old, new, cid = _pair(graph)
    assert [d.id for d in graph.open_candidates()] == [cid]
    graph.reject_candidate(cid, decided_by="owner")
    assert graph.open_candidates() == []


def test_yes_merges_the_pair_keeping_the_older_name(graph: MemoryGraph) -> None:
    old, new, cid = _pair(graph)
    done = graph.accept_candidate(cid, decided_by="owner")
    assert (done.id, done.decision, done.decided_by, done.a, done.b) == (
        cid,
        "same",
        "owner",
        old,
        new,
    )
    assert graph.canonical_id(new) == old
    graph.unmerge(done.id, decided_by="owner")  # one-step undo still works
    assert graph.get_entity(new).merged_into is None  # type: ignore[union-attr]


def test_no_keeps_the_pair_apart_for_good(graph: MemoryGraph) -> None:
    old, new, cid = _pair(graph)
    graph.reject_candidate(cid, decided_by="owner")
    assert graph.are_distinct(old, new)
    for episode in ("e2", "e3", "e4"):
        _resolver(graph).resolve("Northwind Banks", "Bank", episode=episode)
    assert graph.get_entity(new).merged_into is None  # type: ignore[union-attr]


def test_a_decided_pair_cannot_be_decided_again(graph: MemoryGraph) -> None:
    _, _, cid = _pair(graph)
    graph.reject_candidate(cid, decided_by="owner")
    with pytest.raises(StatementError, match="not an open look-alike"):
        graph.accept_candidate(cid, decided_by="owner")
    with pytest.raises(StatementError, match="not an open look-alike"):
        graph.mark_asked("dec_nope")


def test_asked_survives_later_evidence(graph: MemoryGraph) -> None:
    """Asked once means once: evidence arriving later must not reset the mark."""
    _, _, cid = _pair(graph)
    asked = graph.mark_asked(cid)
    assert asked.asked_at is not None
    _resolver(graph).resolve("Northwind Banks", "Bank", episode="e2")
    again = graph.get_decision(cid)
    assert again is not None and again.evidence == ("e1", "e2")
    assert again.asked_at == asked.asked_at


def test_a_pair_whose_entity_was_merged_elsewhere_is_not_open(graph: MemoryGraph) -> None:
    old, new, cid = _pair(graph)
    third = graph.add_entity("Bank", "Northwind Bank Limited")
    graph.merge(third.id, new, decided_by="owner")  # "Northwind Banks" folded elsewhere
    assert graph.open_candidates() == []
