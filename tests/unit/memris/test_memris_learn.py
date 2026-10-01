"""Learning vocabulary (ADR-0115 decision 7), run against every GraphStore.

The ontology here is test data; memris knows none of it.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import timedelta
from difflib import SequenceMatcher
from pathlib import Path
from textwrap import dedent

import pytest
from test_memris_graph import ONTOLOGY, SHAPES, Clock

from memris.graph import MemoryGraph, StatementError
from memris.interchange.jsonld import export_document, import_document
from memris.learn import Learner, term_local_name
from memris.model import OWNER_ID
from memris.ontology import load_or_raise
from memris.ontology.compiler import XSD
from memris.store import GraphStore, InMemoryGraphStore, SQLiteGraphStore
from memris.store.sqlite import SCHEMA_VERSION


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[GraphStore]:
    yield InMemoryGraphStore() if request.param == "memory" else SQLiteGraphStore(tmp_path / "g.db")


def _onto(root: Path, text: str = ONTOLOGY):  # type: ignore[no-untyped-def]
    root.mkdir(parents=True, exist_ok=True)
    (root / "ontology.yaml").write_text(dedent(text), encoding="utf-8")
    (root / "shapes.yaml").write_text(dedent(SHAPES), encoding="utf-8")
    return load_or_raise(root)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def graph(tmp_path: Path, store: GraphStore, clock: Clock) -> MemoryGraph:
    g = MemoryGraph(_onto(tmp_path / "onto"), store, clock=clock)
    g.ensure_owner("Owner", "Person")
    return g


def _learner(graph: MemoryGraph, **kw: object) -> Learner:
    settings: dict[str, object] = {
        "prefix": "learned",
        "min_observations": 3,
        "min_episodes": 2,
        "similarity": _ratio,
    }
    settings.update(kw)
    return Learner(graph, **settings)  # type: ignore[arg-type]


def _see(learner: Learner, raw: str, episode: str, example: str = "x"):  # type: ignore[no-untyped-def]
    return learner.observe(
        raw, domain="Person", range_=f"{XSD}string", example=example, episode=episode
    )


class TestCounting:
    def test_an_unknown_property_is_counted_not_stated(self, graph: MemoryGraph) -> None:
        seen = _see(_learner(graph), "mentors", "e1", "two juniors")

        assert seen is not None and seen.outcome == "counted" and seen.usable_as is None
        assert seen.term.name == "learned:mentors" and seen.term.examples == ("two juniors",)
        assert "learned:mentors" not in graph.ontology.attributes
        with pytest.raises(StatementError):
            graph.assert_(OWNER_ID, "learned:mentors", literal="two juniors")

    def test_it_activates_after_enough_observations_in_enough_episodes(
        self, graph: MemoryGraph
    ) -> None:
        learner = _learner(graph)
        outcomes = [_see(learner, "mentors", ep).outcome for ep in ("e1", "e1", "e1")]  # type: ignore[union-attr]
        assert outcomes == ["counted"] * 3  # three times, one episode: not yet

        seen = _see(learner, "mentors", "e2")

        assert seen is not None and seen.outcome == "activated"
        assert seen.usable_as == "learned:mentors"
        term = graph.ontology.attributes["learned:mentors"]
        assert (term.domain, term.datatype, term.label) == ("t:Person", f"{XSD}string", "mentors")
        graph.assert_(OWNER_ID, "learned:mentors", literal="two juniors", status="confirmed")
        assert [s.literal for s in graph.current(OWNER_ID, "learned:mentors")] == ["two juniors"]
        assert graph.check_usage() == []

    def test_enough_conversations_is_not_enough_without_enough_sightings(
        self, graph: MemoryGraph
    ) -> None:
        learner = _learner(graph)  # 3 sightings across 2 episodes
        outcomes = [_see(learner, "mentors", ep).outcome for ep in ("e1", "e2")]  # type: ignore[union-attr]
        assert outcomes == ["counted", "counted"]
        assert _see(learner, "mentors", "e2").outcome == "activated"  # type: ignore[union-attr]

    def test_examples_are_few_and_distinct(self, graph: MemoryGraph) -> None:
        learner = _learner(graph, max_examples=2, min_observations=99)
        for example in ("a", "a", "b", "c"):
            seen = _see(learner, "mentors", "e1", example)
        assert seen is not None and seen.term.examples == ("b", "c")
        assert seen.term.observations == 4

    def test_names_are_normalised(self) -> None:
        assert term_local_name(" Mentors-Juniors ") == "mentors_juniors"
        assert term_local_name("!!") == ""


class TestAlias:
    def test_a_near_synonym_becomes_an_alias_at_once(self, graph: MemoryGraph) -> None:
        seen = _see(_learner(graph), "work_at", "e1")

        assert seen is not None and seen.outcome == "alias"
        assert seen.usable_as == "t:works_at"
        assert "learned:work_at" not in graph.ontology.relations

    def test_not_across_domains(self, tmp_path: Path, store: GraphStore) -> None:
        graph = MemoryGraph(_onto(tmp_path / "o"), store)
        seen = Learner(graph, prefix="learned", similarity=_ratio).observe(
            "work_at", domain="Org", range_=f"{XSD}string", episode="e1"
        )
        assert seen is not None and seen.outcome == "counted"

    def test_without_a_similarity_nothing_is_an_alias(self, graph: MemoryGraph) -> None:
        seen = _see(_learner(graph, similarity=None), "works_at", "e1")
        assert seen is not None and seen.outcome == "counted"


class TestDecisions:
    def test_a_rejected_word_is_never_learned(self, graph: MemoryGraph) -> None:
        learner = _learner(graph)
        _see(learner, "topic", "e1")
        learner.reject("learned:topic", decided_by="owner")

        outcomes = {_see(learner, "topic", ep).outcome for ep in ("e2", "e3", "e4")}  # type: ignore[union-attr]

        assert outcomes == {"rejected"}
        assert "learned:topic" not in graph.ontology.attributes

    def test_rejecting_an_active_term_keeps_what_was_said_readable(
        self, graph: MemoryGraph
    ) -> None:
        learner = _learner(graph)
        learner.activate("learned:mentors", decided_by="owner")  # unknown yet: nothing
        _see(learner, "mentors", "e1")
        assert learner.activate("learned:mentors", decided_by="owner") is not None
        graph.assert_(OWNER_ID, "learned:mentors", literal="two", status="confirmed")

        learner.reject("learned:mentors", decided_by="owner")

        assert graph.check_usage() == []
        assert [s.literal for s in graph.current(OWNER_ID, "learned:mentors")] == ["two"]
        assert _see(learner, "mentors", "e9").outcome == "rejected"  # type: ignore[union-attr]

    def test_stale_candidates_expire_and_nothing_else_does(
        self, graph: MemoryGraph, clock: Clock
    ) -> None:
        learner = _learner(graph, expire_days=90, min_observations=1, min_episodes=1)
        _see(learner, "mentors", "e1")  # active at once with these thresholds
        stale = _learner(graph, expire_days=90)
        _see(stale, "sings_in", "e1")
        clock.now += timedelta(days=91)
        _see(stale, "plays", "e2")

        assert stale.expire() == ["learned:sings_in"]
        assert {t.name for t in graph.store.terms()} == {"learned:mentors", "learned:plays"}


class TestPromotion:
    def test_a_declared_term_that_matches_takes_over_and_old_statements_read_through(
        self, tmp_path: Path, store: GraphStore
    ) -> None:
        graph = MemoryGraph(_onto(tmp_path / "a"), store)
        graph.ensure_owner("Owner", "Person")
        learner = Learner(graph, prefix="learned", min_observations=1, min_episodes=1)
        _see(learner, "mentors", "e1")
        graph.assert_(OWNER_ID, "learned:mentors", literal="two", status="confirmed")

        promoted = ONTOLOGY.replace(
            "  email: { domain: Person }",
            "  email: { domain: Person }\n  mentors: { domain: Person }",
        )
        later = MemoryGraph(_onto(tmp_path / "b", promoted), store)

        assert later.ontology.attributes["learned:mentors"].replaced_by == ("t:mentors",)
        assert [s.literal for s in later.current(OWNER_ID, "t:mentors")] == ["two"]
        assert later.check_usage() == []


class TestPersistence:
    def test_a_v4_database_gains_the_terms_table(self, tmp_path: Path) -> None:
        path = tmp_path / "v4.db"
        SQLiteGraphStore(path)
        with sqlite3.connect(path) as conn:
            conn.execute("DROP TABLE memris_ontology_terms")
            conn.execute("UPDATE memris_meta SET value = '4' WHERE key = 'schema_version'")

        store = SQLiteGraphStore(path)

        assert store.terms() == []
        assert store.get_meta("schema_version") == str(SCHEMA_VERSION) == "6"

    def test_json_ld_carries_learned_terms_and_what_was_said_with_them(
        self, tmp_path: Path, graph: MemoryGraph
    ) -> None:
        learner = _learner(graph, min_observations=1, min_episodes=1)
        _see(learner, "mentors", "e1", "two juniors")
        _see(_learner(graph), "sings_in", "e1")  # a candidate, still counting
        graph.assert_(OWNER_ID, "learned:mentors", literal="two juniors", status="confirmed")
        text = json.dumps(export_document(graph))

        target = MemoryGraph(graph.declared, SQLiteGraphStore(tmp_path / "copy.db"))
        report = import_document(json.loads(text), target)

        assert report.lossless, report
        assert target.store.terms() == graph.store.terms()
        assert sorted(target.store.statements(), key=lambda s: s.id) == sorted(
            graph.store.statements(), key=lambda s: s.id
        )
        assert export_document(target) == json.loads(text)
