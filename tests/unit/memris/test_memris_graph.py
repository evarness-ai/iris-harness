"""MemoryGraph semantics (ADR-0115 decisions 3, 5, 8), run against every GraphStore.

The ontology here is test data written for these cases; memris knows none of it.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from textwrap import dedent

import pytest

from memris.graph import MemoryGraph, StatementError
from memris.model import OWNER_ID, Statement, new_id
from memris.ontology import Ontology, load_or_raise
from memris.store import GraphStore, InMemoryGraphStore, SQLiteGraphStore

ONTOLOGY = """
ontology: { id: "urn:test", version: "1.0.0", default_prefix: t }
prefixes: { t: "urn:test#" }
classes:
  Thing:   { abstract: true }
  Person:  { subclass_of: Thing }
  Org:     { subclass_of: Thing }
  Bank:    { subclass_of: Org }
  Place:   { subclass_of: Thing }
  City:    { subclass_of: Place }
  Country: { subclass_of: Place }
  Old:     { subclass_of: Thing, deprecated: true, replaced_by: Org }
relations:
  works_at:     { domain: Person, range: Org }
  banks_with:   { domain: Person, range: Org }
  lives_in:     { domain: Person, range: Place }
  in_city:      { domain: Person, range: City, subproperty_of: lives_in }
  in_country:   { domain: Person, range: Country, subproperty_of: lives_in }
  citizen_of:   { domain: Person, range: Country }
  employed_by:  { domain: Person, range: Org, deprecated: true, replaced_by: works_at }
attributes:
  email: { domain: Person }
"""

SHAPES = """
Person:
  properties:
    works_at:   { max_count: 1 }
    in_city:    { max_count: 1 }
    in_country: { max_count: 1 }
    email:      { max_count: 1 }
    banks_with: { class: Bank }
"""

T0 = datetime(2026, 1, 1, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, days: int = 1) -> datetime:
        self.now += timedelta(days=days)
        return self.now


def _ontology(tmp_path: Path, text: str = ONTOLOGY, shapes: str = SHAPES) -> Ontology:
    root = tmp_path / "onto"
    root.mkdir(parents=True, exist_ok=True)
    (root / "ontology.yaml").write_text(dedent(text), encoding="utf-8")
    (root / "shapes.yaml").write_text(dedent(shapes), encoding="utf-8")
    return load_or_raise(root)


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


def _values(graph: MemoryGraph, statements: list[Statement]) -> list[str]:
    out = []
    for s in statements:
        if s.object_id is None:
            out.append(str(s.literal))
        else:
            entity = graph.get_entity(s.object_id)
            assert entity is not None
            out.append(entity.label)
    return out


# --- entities ---------------------------------------------------------------------


def test_the_owner_class_comes_from_the_ontology_when_not_given(tmp_path: Path) -> None:
    onto = _ontology(
        tmp_path,
        ONTOLOGY.replace("default_prefix: t }", "default_prefix: t, owner_class: Person }"),
    )
    assert MemoryGraph(onto, InMemoryGraphStore()).ensure_owner("Me").class_ == "t:Person"
    with pytest.raises(StatementError, match="no owner class"):
        MemoryGraph(_ontology(tmp_path / "bare"), InMemoryGraphStore()).ensure_owner("Me")


def test_the_owner_is_created_once_with_the_class_it_is_given(graph: MemoryGraph) -> None:
    again = graph.ensure_owner("Someone else")
    assert again.id == OWNER_ID and again.label == "Owner" and again.class_ == "t:Person"


@pytest.mark.parametrize(
    ("cls", "why"),
    [("Thing", "abstract"), ("Nope", "not a declared"), ("Old", "deprecated; use t:Org")],
)
def test_entities_need_a_concrete_live_class(graph: MemoryGraph, cls: str, why: str) -> None:
    with pytest.raises(StatementError, match=why):
        graph.add_entity(cls, "X")


def test_entities_are_found_by_label_or_alias_ignoring_case_and_spacing(graph: MemoryGraph) -> None:
    bank = graph.add_entity("Bank", "Northwind  Bank", aliases=["Northwind"])
    assert [e.id for e in graph.store.find_entities(label="northwind bank")] == [bank.id]
    assert [e.id for e in graph.store.find_entities(label=" Northwind ")] == [bank.id]
    assert graph.store.find_entities(label="Northwind", class_="t:Person") == []


# --- what a write must satisfy ----------------------------------------------------


def test_a_claim_is_checked_against_the_ontology(graph: MemoryGraph) -> None:
    acme = graph.add_entity("Org", "Acme").id
    london = graph.add_entity("City", "London").id
    cases = [
        ({"predicate": "nope", "object_id": acme}, "not a declared property"),
        ({"predicate": "employed_by", "object_id": acme}, "deprecated; use t:works_at"),
        ({"predicate": "works_at", "object_id": london}, "outside the range"),
        ({"predicate": "works_at", "literal": "Acme"}, "a relation: give an object"),
        ({"predicate": "email", "object_id": acme}, "an attribute: give a literal"),
        (
            {"predicate": "banks_with", "object_id": acme},
            "outside the range of 't:banks_with' \\(t:Bank\\)",
        ),
    ]
    for kwargs, why in cases:
        with pytest.raises(StatementError, match=why):
            graph.assert_(OWNER_ID, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(StatementError, match="outside the domain"):
        graph.assert_(acme, "works_at", object_id=acme)


def test_a_merged_entity_cannot_take_new_claims(graph: MemoryGraph) -> None:
    a = graph.add_entity("Org", "A")
    b = graph.add_entity("Org", "B")
    graph.store.save(entities=[a.__class__(**{**a.__dict__, "merged_into": b.id})])
    with pytest.raises(StatementError, match="merged into"):
        graph.assert_(OWNER_ID, "works_at", object_id=a.id)


def test_naive_datetimes_are_refused(graph: MemoryGraph) -> None:
    acme = graph.add_entity("Org", "Acme").id
    with pytest.raises(ValueError, match="naive"):
        graph.assert_(OWNER_ID, "works_at", object_id=acme, valid_from=datetime(2020, 1, 1))


def test_saying_the_same_thing_twice_keeps_one_statement(graph: MemoryGraph) -> None:
    acme = graph.add_entity("Org", "Acme").id
    first = graph.assert_(OWNER_ID, "works_at", object_id=acme)
    second = graph.assert_(OWNER_ID, "works_at", object_id=acme)
    assert second.id == first.id and second.status == "proposed"
    confirmed = graph.assert_(OWNER_ID, "works_at", object_id=acme, status="confirmed")
    assert confirmed.id == first.id and confirmed.status == "confirmed"
    assert confirmed.reinforced == 3  # said three times
    assert len(graph.history(OWNER_ID)) == 1


# --- supersede --------------------------------------------------------------------


def test_a_new_confirmed_value_supersedes_and_history_answers_as_of(
    graph: MemoryGraph, clock: Clock
) -> None:
    barclays = graph.add_entity("Org", "Barclays").id
    litware = graph.add_entity("Org", "Litware").id
    old = graph.assert_(OWNER_ID, "works_at", object_id=barclays, status="confirmed")
    moved = clock.advance(30)
    new = graph.assert_(OWNER_ID, "works_at", object_id=litware, status="confirmed")

    assert _values(graph, graph.current(OWNER_ID, "works_at")) == ["Litware"]
    assert new.supersedes == old.id
    closed = graph.store.get_statement(old.id)
    assert closed is not None and closed.valid_to == moved
    assert _values(graph, graph.current(OWNER_ID, "works_at", as_of=moved - timedelta(days=1))) == [
        "Barclays"
    ]
    assert graph.current(OWNER_ID, "works_at", as_of=T0 - timedelta(days=1)) == []
    assert len(graph.history(OWNER_ID, "works_at")) == 2


def test_a_proposal_never_displaces_a_confirmed_value_until_it_is_confirmed(
    graph: MemoryGraph, clock: Clock
) -> None:
    barclays = graph.add_entity("Org", "Barclays").id
    litware = graph.add_entity("Org", "Litware").id
    graph.assert_(OWNER_ID, "works_at", object_id=barclays, status="confirmed")
    clock.advance()
    proposal = graph.assert_(OWNER_ID, "works_at", object_id=litware)
    assert _values(graph, graph.current(OWNER_ID, "works_at")) == ["Barclays"]
    assert _values(graph, graph.current(OWNER_ID, "works_at", include_proposed=True)) == [
        "Barclays",
        "Litware",
    ]
    graph.confirm(proposal.id)
    assert _values(graph, graph.current(OWNER_ID, "works_at")) == ["Litware"]


def test_values_coexist_where_the_shapes_set_no_limit(graph: MemoryGraph, clock: Clock) -> None:
    india = graph.add_entity("Country", "India").id
    uk = graph.add_entity("Country", "United Kingdom").id
    graph.assert_(OWNER_ID, "citizen_of", object_id=india, status="confirmed")
    clock.advance()
    graph.assert_(OWNER_ID, "citizen_of", object_id=uk, status="confirmed")
    assert sorted(_values(graph, graph.current(OWNER_ID, "citizen_of"))) == [
        "India",
        "United Kingdom",
    ]


def test_an_attribute_supersedes_too(graph: MemoryGraph, clock: Clock) -> None:
    graph.assert_(OWNER_ID, "email", literal="a@example.org", status="confirmed")
    clock.advance()
    graph.assert_(OWNER_ID, "email", literal="b@example.org", status="confirmed")
    assert _values(graph, graph.current(OWNER_ID, "email")) == ["b@example.org"]


def test_residence_parts_supersede_separately(graph: MemoryGraph, clock: Clock) -> None:
    """Moving city does not erase the country; a lives_in query returns both parts."""
    graph.assert_(
        OWNER_ID, "in_country", object_id=graph.add_entity("Country", "UK").id, status="confirmed"
    )
    graph.assert_(
        OWNER_ID, "in_city", object_id=graph.add_entity("City", "London").id, status="confirmed"
    )
    clock.advance()
    graph.assert_(
        OWNER_ID, "in_city", object_id=graph.add_entity("City", "Leeds").id, status="confirmed"
    )
    assert sorted(_values(graph, graph.current(OWNER_ID, "lives_in"))) == ["Leeds", "UK"]
    assert graph.current(OWNER_ID, "lives_in", include_subproperties=False) == []


# --- end and retract --------------------------------------------------------------


def test_end_closes_without_a_replacement(graph: MemoryGraph, clock: Clock) -> None:
    acme = graph.add_entity("Org", "Acme").id
    job = graph.assert_(OWNER_ID, "works_at", object_id=acme, status="confirmed")
    left = clock.advance(10)
    graph.end(job.id)
    assert graph.current(OWNER_ID, "works_at") == []
    assert _values(graph, graph.current(OWNER_ID, "works_at", as_of=left - timedelta(days=1))) == [
        "Acme"
    ]
    with pytest.raises(StatementError, match="already ended"):
        graph.end(job.id)


def test_a_statement_cannot_end_before_it_started(graph: MemoryGraph) -> None:
    job = graph.assert_(
        OWNER_ID, "works_at", object_id=graph.add_entity("Org", "Acme").id, status="confirmed"
    )
    with pytest.raises(StatementError, match="before it started"):
        graph.end(job.id, at=T0 - timedelta(days=1))


def test_retract_removes_from_recall_but_keeps_the_record(graph: MemoryGraph, clock: Clock) -> None:
    job = graph.assert_(
        OWNER_ID, "works_at", object_id=graph.add_entity("Org", "DOJ").id, status="confirmed"
    )
    before = clock.advance()
    clock.advance()
    graph.retract(job.id)
    assert graph.current(OWNER_ID, "works_at") == []
    assert [s.status for s in graph.history(OWNER_ID)] == ["retracted"]
    # looking back in record time, it was held until the retraction
    assert len(graph.current(OWNER_ID, "works_at", known_at=before)) == 1
    with pytest.raises(StatementError, match="cannot be confirmed"):
        graph.confirm(job.id)


def test_retracting_a_supersede_reopens_what_it_closed(graph: MemoryGraph, clock: Clock) -> None:
    """ "I moved to Litware" was wrong, so Barclays never stopped being current."""
    graph.assert_(
        OWNER_ID, "works_at", object_id=graph.add_entity("Org", "Barclays").id, status="confirmed"
    )
    clock.advance()
    wrong = graph.assert_(
        OWNER_ID, "works_at", object_id=graph.add_entity("Org", "Litware").id, status="confirmed"
    )
    clock.advance()
    graph.retract(wrong.id)
    assert _values(graph, graph.current(OWNER_ID, "works_at")) == ["Barclays"]


# --- ontology change (decision 8) -------------------------------------------------


def test_a_query_reads_through_retired_terms(graph: MemoryGraph, clock: Clock) -> None:
    """Old data stored under employed_by answers a works_at query; nothing is rewritten."""
    acme = graph.add_entity("Org", "Acme").id
    legacy = Statement(
        id=new_id("st"),
        subject_id=OWNER_ID,
        predicate="t:employed_by",
        recorded_at=clock(),
        object_id=acme,
        status="confirmed",
    )
    graph.store.save(statements=[legacy])
    assert [s.id for s in graph.current(OWNER_ID, "works_at")] == [legacy.id]
    stored = graph.store.get_statement(legacy.id)
    assert stored is not None and stored.predicate == "t:employed_by"
    assert graph.check_usage() == []


def test_usage_check_fails_when_a_used_term_vanishes(graph: MemoryGraph, tmp_path: Path) -> None:
    graph.assert_(OWNER_ID, "email", literal="a@example.org", status="confirmed")
    trimmed = _ontology(
        tmp_path / "v2",
        ONTOLOGY.replace("  email: { domain: Person }\n", "  other: { domain: Person }\n"),
        SHAPES.replace("    email:      { max_count: 1 }\n", ""),
    )
    later = MemoryGraph(trimmed, graph.store)
    assert [i.code for i in later.check_usage()] == ["term-vanished"]


# --- the SQLite store in particular -----------------------------------------------


def test_sqlite_persists_across_instances(tmp_path: Path) -> None:
    onto = _ontology(tmp_path)
    path = tmp_path / "persist.db"
    first = MemoryGraph(onto, SQLiteGraphStore(path))
    first.ensure_owner("Owner", "Person")
    stored = first.assert_(
        OWNER_ID, "email", literal="a@example.org", status="confirmed", valid_from=T0
    )
    second = MemoryGraph(onto, SQLiteGraphStore(path))
    [again] = second.current(OWNER_ID, "email", as_of=T0 + timedelta(days=1))
    assert again == stored


def test_sqlite_refuses_a_newer_schema(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "future.db"
    SQLiteGraphStore(path)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE memris_meta SET value = '999' WHERE key = 'schema_version'")
    with pytest.raises(RuntimeError, match="schema 999"):
        SQLiteGraphStore(path)


def test_sqlite_refuses_a_statement_with_both_object_and_literal(tmp_path: Path) -> None:
    import sqlite3

    store = SQLiteGraphStore(tmp_path / "check.db")
    bad = Statement(
        id="st_x",
        subject_id=OWNER_ID,
        predicate="t:email",
        recorded_at=T0,
        object_id="e",
        literal="x",
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.save(statements=[bad])


# --- bookkeeping, demotion and import (PR 2b) ------------------------------------


def test_refresh_changes_bookkeeping_only(graph: MemoryGraph, clock: Clock) -> None:
    job = graph.assert_(
        OWNER_ID, "works_at", object_id=graph.add_entity("Org", "Acme").id, status="confirmed"
    )
    later = clock.advance(3)
    done = graph.refresh(job.id, reinforced=7, confidence=0.4, extractor="user:correction")
    assert (done.reinforced, done.last_reinforced_at, done.confidence, done.extractor) == (
        7,
        later,
        0.4,
        "user:correction",
    )
    assert (done.status, done.valid_to, done.object_id) == (job.status, job.valid_to, job.object_id)


def test_unconfirm_turns_a_belief_back_into_a_proposal(graph: MemoryGraph) -> None:
    job = graph.assert_(
        OWNER_ID, "works_at", object_id=graph.add_entity("Org", "Acme").id, status="confirmed"
    )
    graph.unconfirm(job.id)
    assert graph.current(OWNER_ID, "works_at") == []
    assert len(graph.current(OWNER_ID, "works_at", include_proposed=True)) == 1


def test_import_keeps_records_exactly_and_still_checks_the_ontology(graph: MemoryGraph) -> None:
    acme = graph.add_entity("Org", "Acme")
    old = Statement(
        id="st_old",
        subject_id=OWNER_ID,
        predicate="t:works_at",
        recorded_at=T0 - timedelta(days=400),
        object_id=acme.id,
        status="confirmed",
        valid_to=T0 - timedelta(days=10),
        reinforced=4,
    )
    graph.import_([old])
    assert graph.store.get_statement("st_old") == old  # nothing superseded, re-timed or reinforced
    with pytest.raises(StatementError, match="outside the range"):
        graph.import_(
            [
                Statement(
                    id="st_bad",
                    subject_id=OWNER_ID,
                    predicate="t:works_at",
                    recorded_at=T0,
                    object_id=graph.add_entity("City", "Leeds").id,
                )
            ]
        )
    assert graph.store.get_statement("st_bad") is None  # all or nothing


def test_sqlite_upgrades_a_version_1_file_in_place(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "v1.db"
    SQLiteGraphStore(path)
    with sqlite3.connect(path) as conn:  # rebuild the v1 shape: no reinforcement columns
        conn.execute("DROP TABLE memris_statements")
        conn.execute(
            "CREATE TABLE memris_statements (id TEXT PRIMARY KEY, subject_id TEXT NOT NULL, predicate TEXT NOT NULL, "
            "object_id TEXT, literal TEXT, datatype TEXT, valid_from TEXT, valid_to TEXT, recorded_at TEXT NOT NULL, "
            "retracted_at TEXT, status TEXT NOT NULL, confidence REAL, source_episode TEXT, source_turn TEXT, "
            "extractor TEXT, supersedes TEXT, ontology_version TEXT)"
        )
        conn.execute(
            "INSERT INTO memris_statements(id, subject_id, predicate, literal, recorded_at, status) "
            "VALUES ('st_v1', 'owner', 't:email', 'a@example.org', ?, 'confirmed')",
            (T0.isoformat(),),
        )
        conn.execute("UPDATE memris_meta SET value = '1' WHERE key = 'schema_version'")
    store = SQLiteGraphStore(path)
    kept = store.get_statement("st_v1")
    assert kept is not None and kept.reinforced == 1 and kept.literal == "a@example.org"
    from memris.store.sqlite import SCHEMA_VERSION

    assert store.get_meta("schema_version") == str(SCHEMA_VERSION)  # v1 → every later version
    assert kept.evidence is None and kept.reason is None and kept.reviewed_at is None


def test_a_v3_database_gains_asked_at_on_its_entity_decisions(tmp_path: Path) -> None:
    """v3 made memris_entity_decisions without asked_at; opening it adds the column."""
    import sqlite3

    path = tmp_path / "v3.db"
    SQLiteGraphStore(path)
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE memris_entity_decisions")
        conn.execute(
            "CREATE TABLE memris_entity_decisions (id TEXT PRIMARY KEY, a TEXT NOT NULL, "
            "b TEXT NOT NULL, decision TEXT NOT NULL, decided_at TEXT NOT NULL, decided_by TEXT, "
            "score REAL, evidence TEXT NOT NULL DEFAULT '[]', "
            "added_aliases TEXT NOT NULL DEFAULT '[]')"
        )
        conn.execute(
            "INSERT INTO memris_entity_decisions(id, a, b, decision, decided_at) "
            "VALUES ('dec_v3', 'e1', 'e2', 'candidate', ?)",
            (T0.isoformat(),),
        )
        conn.execute("UPDATE memris_meta SET value = '3' WHERE key = 'schema_version'")
    store = SQLiteGraphStore(path)
    [kept] = store.decisions()
    assert (kept.id, kept.decision, kept.asked_at) == ("dec_v3", "candidate", None)
    from memris.store.sqlite import SCHEMA_VERSION

    assert store.get_meta("schema_version") == str(SCHEMA_VERSION)


def test_a_meta_claim_can_be_won_only_once(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "claim.db")
    assert store.claim_meta("job", "a") is True
    assert store.claim_meta("job", "b") is False
    assert store.get_meta("job") == "a"


# --- reasons, evidence and purge (PR 2c) ----------------------------------------


def test_evidence_and_reasons_are_kept(graph: MemoryGraph) -> None:
    acme = graph.add_entity("Org", "Acme").id
    job = graph.assert_(OWNER_ID, "works_at", object_id=acme, evidence="I work at Acme")
    assert graph.store.get_statement(job.id).evidence == "I work at Acme"  # type: ignore[union-attr]
    graph.refresh(job.id, reason="approved")
    gone = graph.retract(job.id, reason="rejected")
    assert (gone.status, gone.reason) == ("retracted", "rejected")


def test_purge_deletes_only_what_no_longer_holds(graph: MemoryGraph, clock: Clock) -> None:
    barclays = graph.add_entity("Org", "Barclays").id
    old = graph.assert_(OWNER_ID, "works_at", object_id=barclays, status="confirmed")
    clock.advance()
    new = graph.assert_(
        OWNER_ID, "works_at", object_id=graph.add_entity("Org", "Litware").id, status="confirmed"
    )
    pending = graph.assert_(OWNER_ID, "email", literal="a@example.org")
    for still_holds in (new.id, pending.id):
        with pytest.raises(StatementError, match="still holds"):
            graph.purge([still_holds])
    assert graph.purge([old.id]) == 1  # superseded: history, and the owner may drop it
    assert graph.store.get_statement(old.id) is None
    assert graph.store.get_statement(new.id) is not None


def test_a_refused_claim_is_kept_never_believed_and_deduplicated(graph: MemoryGraph) -> None:
    kept = graph.assert_(OWNER_ID, "email", literal="a@example.org", status="confirmed")
    first = graph.refuse(OWNER_ID, "email", literal="wrong@example.org", contradicts=kept.id)
    assert (first.status, first.reason, first.contradicts) == ("retracted", "refused", kept.id)
    assert first.recorded_at == first.retracted_at  # received, never held
    assert [s.literal for s in graph.current(OWNER_ID, "email")] == ["a@example.org"]
    again = graph.refuse(OWNER_ID, "email", literal="wrong@example.org", contradicts=kept.id)
    assert (again.id, again.reinforced) == (first.id, 2)
    graph.review([first.id])
    fresh = graph.refuse(OWNER_ID, "email", literal="wrong@example.org", contradicts=kept.id)
    assert fresh.id != first.id  # after review, a recurrence is news again
    with pytest.raises(StatementError, match="an attribute"):
        graph.refuse(OWNER_ID, "email", object_id=kept.id, contradicts=kept.id)


# --- neighbourhood (decision 9: the graph tool's traversal) ----------------------------


def _hood(graph: MemoryGraph, entity_id: str, **kw: object) -> list[tuple[int, str]]:
    return [
        (hop, f"{s.subject_id}>{s.predicate}>{s.object_id or s.literal}")
        for hop, s in graph.neighbourhood(entity_id, **kw)  # type: ignore[arg-type]
    ]


@pytest.fixture
def family(graph: MemoryGraph) -> dict[str, str]:
    """The owner and Petra both work at Acme; Petra has an email."""
    petra = graph.add_entity("Person", "Petra").id
    acme = graph.add_entity("Org", "Acme").id
    graph.assert_(OWNER_ID, "works_at", object_id=acme, status="confirmed")
    graph.assert_(petra, "works_at", object_id=acme, status="confirmed")
    graph.assert_(petra, "email", literal="p@x.example", status="confirmed")
    return {"petra": petra, "acme": acme}


def test_one_hop_both_ways(graph: MemoryGraph, family: dict[str, str]) -> None:
    acme = family["acme"]
    assert sorted(_hood(graph, acme)) == sorted(
        [(1, f"{OWNER_ID}>t:works_at>{acme}"), (1, f"{family['petra']}>t:works_at>{acme}")]
    )


def test_direction_narrows(graph: MemoryGraph, family: dict[str, str]) -> None:
    petra = family["petra"]
    # same clock tick: order among equals is by id, so compare as a set
    assert {s for _h, s in _hood(graph, petra, direction="out")} == {
        f"{petra}>t:works_at>{family['acme']}",
        f"{petra}>t:email>p@x.example",
    }
    assert _hood(graph, petra, direction="in") == []
    with pytest.raises(StatementError):
        graph.neighbourhood(petra, direction="sideways")  # type: ignore[arg-type]


def test_two_hops_reach_through_a_shared_entity(graph: MemoryGraph, family: dict[str, str]) -> None:
    one = _hood(graph, OWNER_ID, hops=1)
    two = _hood(graph, OWNER_ID, hops=2)

    assert [h for h, _s in one] == [1]
    assert (2, f"{family['petra']}>t:works_at>{family['acme']}") in two
    assert (2, f"{family['petra']}>t:email>p@x.example") not in two  # three hops away


def test_predicate_and_as_of(graph: MemoryGraph, family: dict[str, str], clock: Clock) -> None:
    before = clock.now
    clock.advance()
    other = graph.add_entity("Org", "Globex").id
    graph.assert_(family["petra"], "works_at", object_id=other, status="confirmed")

    now = _hood(graph, family["petra"], predicate="works_at", direction="out")
    then = _hood(graph, family["petra"], predicate="works_at", direction="out", as_of=before)

    assert now == [(1, f"{family['petra']}>t:works_at>{other}")]
    assert then == [(1, f"{family['petra']}>t:works_at>{family['acme']}")]


def test_proposals_only_on_request_and_merges_are_followed(
    graph: MemoryGraph, family: dict[str, str]
) -> None:
    twin = graph.add_entity("Person", "Petra S").id
    graph.assert_(twin, "email", literal="ps@x.example")  # proposed

    assert _hood(graph, twin) == []
    assert len(_hood(graph, twin, include_proposed=True)) == 1
    graph.merge(family["petra"], twin, decided_by="owner")
    assert (1, f"{twin}>t:email>ps@x.example") in _hood(
        graph, family["petra"], include_proposed=True
    )
