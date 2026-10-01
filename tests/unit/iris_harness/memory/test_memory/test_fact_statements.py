"""User facts kept as memris statements (memris plan PR 2b).

The store's fact API is unchanged for callers; these tests pin what moved underneath:
facts are statements, the old behaviours hold, the vocabulary is closed, and a legacy
database moves over by itself, once, with a backup.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_harness.memory.fact_statements import FactKeyError
from iris_harness.memory.ontology import memory_ontology
from iris_harness.memory.statement_migration import CLAIM_KEY, MIGRATED_KEY, apply_migration
from iris_harness.memory.store import MemoryStore, UserFact
from memris.model import OWNER_ID

# Fact keys such as `bank` and `credit_card` come from the test vocabulary fragment
# (tests/fixtures/test_vocabulary, installed by the `test_vocabulary` fixture), not
# from whichever domain plugin the tree happens to carry.
pytestmark = pytest.mark.usefixtures("test_vocabulary")

NOW = datetime(2026, 9, 1, tzinfo=UTC)


def _fact(key: str, value: str, conf: float = 0.9, *, confirmed: bool = True) -> UserFact:
    return UserFact(key, value, conf, "conversation:llm", NOW, NOW, 1, confirmed)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


def _history(store: MemoryStore, key: str) -> list[tuple[str | None, str | None, str]]:
    return [(e.old_value, e.new_value, e.reason) for e in reversed(store.fetch_fact_history(key))]


def _contradictions(store: MemoryStore) -> list[tuple[str, str, str]]:
    return [(c.stored_value, c.incoming_value, c.resolution) for c in store.fetch_contradictions()]


# --- facts are statements -----------------------------------------------------------


def test_a_fact_is_a_statement_about_the_owner(store: MemoryStore) -> None:
    store.upsert_user_fact(_fact("employer", "Barclays"))
    graph = store._facts().graph
    [statement] = graph.current(OWNER_ID, "works_at")
    entity = graph.get_entity(statement.object_id or "")
    assert entity is not None and (entity.label, entity.class_) == ("Barclays", "mem:Organization")
    fact = store.fetch_user_fact("employer")
    assert fact is not None and (fact.key, fact.value, fact.confirmed) == (
        "employer",
        "Barclays",
        True,
    )


def test_the_legacy_table_is_no_longer_written(store: MemoryStore) -> None:
    store.upsert_user_fact(_fact("employer", "Barclays"))
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM user_facts").fetchone()[0] == 0


def test_keys_that_share_a_property_are_one_fact(store: MemoryStore) -> None:
    store.upsert_user_fact(_fact("company", "Acme"))
    fact = store.fetch_user_fact("employer")
    assert fact is not None and (fact.key, fact.value) == ("employer", "Acme")


# --- behaviour the old table had ------------------------------------------------------


def test_a_lower_confidence_value_is_blocked_and_says_so(store: MemoryStore) -> None:
    store.upsert_user_fact(_fact("blog", "web3notes.example", 0.9))
    store.upsert_user_fact(_fact("blog", "site", 0.3))
    fact = store.fetch_user_fact("blog")
    assert fact is not None and fact.value == "web3notes.example"
    assert _contradictions(store) == [("web3notes.example", "site", "blocked")]
    assert _history(store, "blog") == [(None, "web3notes.example", "capture")]


def test_a_higher_confidence_value_supersedes_and_inherits_confirmation(store: MemoryStore) -> None:
    store.upsert_user_fact(_fact("employer", "Barclays", 0.8))
    store.upsert_user_fact(_fact("employer", "Litware", 0.9, confirmed=False))
    fact = store.fetch_user_fact("employer")
    assert fact is not None and (fact.value, fact.confirmed) == ("Litware", True)
    assert _history(store, "employer")[-1] == ("Barclays", "Litware", "supersede")
    assert _contradictions(store) == [("Barclays", "Litware", "superseded")]
    # the old value is closed, not deleted
    assert [
        store._facts().value_of(s) for s in store._facts().graph.history(OWNER_ID, "works_at")
    ] == [
        "Barclays",
        "Litware",
    ]


def test_saying_it_again_counts(store: MemoryStore) -> None:
    store.upsert_user_fact(_fact("city", "Leeds"))
    assert store.add_fact_proposal(key="city", value="Leeds", confidence=0.9, source="x") is None
    fact = store.fetch_user_fact("city")
    assert fact is not None and fact.times_confirmed == 2


def test_forget_retracts_and_restore_brings_it_back(store: MemoryStore) -> None:
    store.upsert_user_fact(_fact("email", "me@example.org"))
    assert store.delete_user_fact("email") is True
    assert store.fetch_user_fact("email") is None
    assert store.delete_user_fact("email") is False
    assert store.restore_user_fact("email") == "me@example.org"
    fact = store.fetch_user_fact("email")
    assert fact is not None and fact.value == "me@example.org"


def test_a_correction_overrides_the_gate(store: MemoryStore) -> None:
    store.upsert_user_fact(_fact("blog", "web3notes.example", 0.99))
    assert store.correct_user_fact("blog", "tech4talk.com", confidence=0.5) is True
    fact = store.fetch_user_fact("blog")
    assert fact is not None and fact.value == "tech4talk.com"


def test_unconfirmed_facts_are_what_review_owes(store: MemoryStore) -> None:
    store.upsert_user_fact(_fact("hobby", "chess", confirmed=False))
    assert store.count_pending_review() == 1
    assert store.fetch_all_user_facts(confirmed_only=True) == []
    store.set_fact_confirmed("hobby", True)
    assert store.count_pending_review() == 0
    assert [f.key for f in store.fetch_all_user_facts(confirmed_only=True)] == ["hobby"]


# --- what is new ------------------------------------------------------------------


def test_a_property_without_a_limit_keeps_both_values(store: MemoryStore) -> None:
    store.upsert_user_fact(_fact("credit_card", "Discover"))
    store.upsert_user_fact(_fact("card", "Wingtip Bank Credit Card"))
    values = sorted(f.value for f in store.fetch_all_user_facts() if f.key == "credit_card")
    assert values == ["Discover", "Wingtip Bank Credit Card"]
    assert _contradictions(store) == []  # a second card is an addition, not a conflict


def test_the_vocabulary_is_closed(store: MemoryStore) -> None:
    with pytest.raises(FactKeyError, match="not a fact key memory keeps"):
        store.upsert_user_fact(_fact("favorite_topic", "graphs"))
    assert store.fetch_user_fact("favorite_topic") is None


# --- the automatic migration --------------------------------------------------------


def _legacy(path: Path, rows: list[tuple[str, str]]) -> None:
    MemoryStore(db_path=path)._ensure_tables()
    with sqlite3.connect(path) as conn:
        for key, value in rows:
            conn.execute(
                "INSERT INTO user_facts(key, value, confidence, source, first_seen, "
                "last_confirmed, times_confirmed, confirmed) VALUES (?, ?, 0.9, 'x', ?, ?, 2, 1)",
                (key, value, NOW.isoformat(), NOW.isoformat()),
            )
            conn.execute(
                "INSERT INTO user_fact_history(key, old_value, old_confidence, new_value, "
                "new_confidence, source, reason, changed_at) VALUES (?, NULL, NULL, ?, 0.9, 'x', "
                "'capture', ?)",
                (key, value, NOW.isoformat()),
            )


def test_a_legacy_database_moves_over_on_first_open_with_a_backup(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _legacy(db, [("employer", "Barclays"), ("to-do", "buy milk")])

    store = MemoryStore(db_path=db)
    facts = {f.key: f for f in store.fetch_all_user_facts()}

    assert set(facts) == {"employer"}  # the off-list fact is dropped by default
    assert facts["employer"].times_confirmed == 2  # bookkeeping survived the move
    assert list((tmp_path / "backups").glob("memory-before-memris-*.db"))
    [report] = (tmp_path / "memris-migration").glob("report-applied-*.md")
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Fact → statement migration: applied")
    assert "nothing was written" not in text and "memory-before-memris-" in text
    with sqlite3.connect(db) as conn:  # the archive stays
        assert conn.execute("SELECT COUNT(*) FROM user_facts").fetchone()[0] == 2


def test_the_migration_runs_once(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _legacy(db, [("employer", "Barclays")])
    MemoryStore(db_path=db).fetch_all_user_facts()
    again = apply_migration(
        db, memory_ontology(), out_dir=tmp_path / "o", backup_dir=tmp_path / "b"
    )
    assert (again.applied, again.reason) == (False, "already migrated")
    assert len(MemoryStore(db_path=db).fetch_all_user_facts()) == 1


def test_a_fresh_database_is_marked_without_a_backup(tmp_path: Path) -> None:
    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.ensure_schema()
    assert store._facts().graph.store.get_meta(MIGRATED_KEY)  # type: ignore[attr-defined]
    assert not (tmp_path / "backups").exists()


def test_a_claimed_migration_is_not_run_twice(tmp_path: Path) -> None:
    from memris.store import SQLiteGraphStore

    db = tmp_path / "memory.db"
    _legacy(db, [("employer", "Barclays")])
    SQLiteGraphStore(db).claim_meta(CLAIM_KEY, "another-process")
    with pytest.raises(RuntimeError, match="another process claimed"):
        apply_migration(
            db,
            memory_ontology(),
            out_dir=tmp_path / "o",
            backup_dir=tmp_path / "b",
            wait_seconds=0.2,
        )
    assert not (tmp_path / "b").exists()  # nothing touched while someone else holds it


def test_the_migrated_facts_keep_their_history_as_closed_statements(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _legacy(db, [("employer", "Barclays")])
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE user_facts SET value = 'Litware' WHERE key = 'employer'")
        conn.execute(
            "INSERT INTO user_fact_history(key, old_value, old_confidence, new_value, "
            "new_confidence, source, reason, changed_at) VALUES ('employer', 'Barclays', 0.9, "
            "'Litware', 0.9, 'x', 'supersede', ?)",
            ((NOW + timedelta(days=1)).isoformat(),),
        )
    store = MemoryStore(db_path=db)
    fact = store.fetch_user_fact("employer")
    assert fact is not None and fact.value == "Litware"
    past = store._facts().graph.history(OWNER_ID, "works_at")
    assert [store._facts().value_of(s) for s in past] == ["Barclays", "Litware"]
    assert past[0].valid_to is not None


# --- the review queue is proposed statements (PR 2c-i) ------------------------------


def _coordinator(store: MemoryStore):  # type: ignore[no-untyped-def]
    from iris_harness.memory.coordinator import FactCoordinator

    return FactCoordinator(store, None)


def test_a_proposal_is_a_proposed_statement_with_its_evidence(store: MemoryStore) -> None:
    pid = store.add_fact_proposal(
        key="city",
        value="Springfield",
        confidence=0.8,
        source="llm",
        evidence="I moved to Springfield",
    )
    assert isinstance(pid, str) and pid.startswith("st_")
    statement = store._facts().graph.store.get_statement(pid)
    assert statement is not None and statement.status == "proposed"
    [proposal] = store.fetch_fact_proposals()
    assert (proposal.id, proposal.evidence, proposal.seen_count) == (
        pid,
        "I moved to Springfield",
        1,
    )


def test_saying_a_pending_fact_again_bumps_the_same_proposal(store: MemoryStore) -> None:
    first = store.add_fact_proposal(key="city", value="Springfield", confidence=0.8, source="llm")
    again = store.add_fact_proposal(key="city", value="Springfield", confidence=0.8, source="llm")
    assert again == first
    assert store.fetch_fact_proposal(first).seen_count == 2  # type: ignore[union-attr]


def test_approving_confirms_that_statement_and_keeps_other_questions_open(
    store: MemoryStore,
) -> None:
    leeds = store.add_fact_proposal(key="city", value="Leeds", confidence=0.8, source="llm")
    london = store.add_fact_proposal(key="city", value="London", confidence=0.8, source="llm")
    fact = _coordinator(store).approve_proposal(london)
    assert fact is not None and (fact.value, fact.confirmed) == ("London", True)
    approved = store.fetch_fact_proposal(london)
    assert approved is not None and approved.status == "approved"
    # the other proposal is still a question for the owner, not silently withdrawn
    assert [p.id for p in store.fetch_fact_proposals()] == [leeds]
    # one statement per value: approval confirmed the proposal itself
    values = [
        store._facts().value_of(s)
        for s in store._facts().graph.history(OWNER_ID, "resides_in_city")
    ]
    assert sorted(values) == ["Leeds", "London"]


def test_reject_and_expire_retract_with_a_reason(store: MemoryStore) -> None:
    no = store.add_fact_proposal(key="city", value="Leeds", confidence=0.8, source="llm")
    old = store.add_fact_proposal(key="hobby", value="chess", confidence=0.8, source="llm")
    assert _coordinator(store).reject_proposal(no) is True
    assert _coordinator(store).reject_proposal(no) is False  # only pending ones
    with sqlite3.connect(store.db_path) as conn:  # age the second one past the window
        conn.execute(
            "UPDATE memris_statements SET recorded_at = ? WHERE id = ?",
            ((NOW - timedelta(days=400)).isoformat(), old),
        )
    assert store.expire_fact_proposals(older_than_days=30) == 1
    assert [p.id for p in store.fetch_fact_proposals(status="rejected")] == [no]
    assert [p.id for p in store.fetch_fact_proposals(status="expired")] == [old]
    assert store.fetch_fact_proposals() == []


def test_legacy_pending_proposals_move_into_the_queue(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _legacy(db, [("employer", "Barclays")])
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO fact_proposals(key, value, confidence, source, evidence, current_value, "
            "created_at, status, seen_count) VALUES ('city', 'Leeds', 0.7, 'llm', 'I live in Leeds', "
            "NULL, ?, 'pending', 2)",
            (NOW.isoformat(),),
        )
    store = MemoryStore(db_path=db)
    [proposal] = store.fetch_fact_proposals()
    assert (proposal.key, proposal.value, proposal.seen_count, proposal.evidence) == (
        "city",
        "Leeds",
        2,
        "I live in Leeds",
    )


def test_a_new_confirmed_value_leaves_pending_questions_open(store: MemoryStore) -> None:
    """ "My city is Paris" while "Leeds?" waits: Paris is believed, Leeds is still asked."""
    leeds = store.add_fact_proposal(key="city", value="Leeds", confidence=0.8, source="llm")
    _coordinator(store).record("city", "Paris", 0.95, "user", confirmed=True)
    fact = store.fetch_user_fact("city")
    assert fact is not None and (fact.value, fact.confirmed) == ("Paris", True)
    assert [p.id for p in store.fetch_fact_proposals()] == [leeds]


# --- history and contradictions come from the chain (PR 2c-ii) ----------------------


def test_the_old_log_tables_are_no_longer_written(store: MemoryStore) -> None:
    store.upsert_user_fact(_fact("city", "Leeds"))
    store.upsert_user_fact(_fact("city", "London"))
    store.upsert_user_fact(_fact("city", "Paris", 0.1))  # refused
    store.delete_user_fact("city")
    with sqlite3.connect(store.db_path) as conn:
        for table in ("user_fact_history", "user_fact_contradictions"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0  # noqa: S608
    assert [e.reason for e in store.fetch_fact_history("city")] == [
        "forget",
        "supersede",
        "capture",
    ]


def test_a_refused_value_is_a_statement_never_a_belief(store: MemoryStore) -> None:
    store.upsert_user_fact(_fact("blog", "web3notes.example", 0.9))
    store.upsert_user_fact(_fact("blog", "wrong.com", 0.2))
    [refused] = [s for s in store._facts().graph.history(OWNER_ID, "blog") if s.reason == "refused"]
    assert refused.recorded_at == refused.retracted_at and refused.contradicts is not None
    assert [f.value for f in store.fetch_all_user_facts()] == ["web3notes.example"]
    assert "wrong.com" not in {e.new_value for e in store.fetch_fact_history("blog")}


def test_a_correction_is_history_but_not_a_contradiction(store: MemoryStore) -> None:
    store.upsert_user_fact(_fact("city", "Leeds"))
    store.correct_user_fact("city", "London")
    [event, _] = store.fetch_fact_history("city")
    assert (event.reason, event.old_value, event.new_value) == ("correct", "Leeds", "London")
    assert store.fetch_contradictions() == []


def test_pending_proposals_are_not_history_until_approved(store: MemoryStore) -> None:
    pid = store.add_fact_proposal(key="city", value="Leeds", confidence=0.8, source="llm")
    assert store.fetch_fact_history("city") == []
    _coordinator(store).approve_proposal(pid)
    assert [e.new_value for e in store.fetch_fact_history("city")] == ["Leeds"]


def test_legacy_logs_move_onto_the_chain(tmp_path: Path) -> None:
    """A blocked conflict becomes a refused statement; acknowledgements carry over."""
    db = tmp_path / "memory.db"
    _legacy(db, [("blog", "web3notes.example")])
    later = (NOW + timedelta(days=1)).isoformat()
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO user_fact_contradictions(key, stored_value, stored_confidence, "
            "incoming_value, incoming_confidence, resolution, source, detected_at, acknowledged, "
            "seen_count) VALUES ('blog', 'web3notes.example', 0.9, 'wrong.com', 0.2, 'blocked', "
            "'llm', ?, 1, 3)",
            (later,),
        )
    store = MemoryStore(db_path=db)
    assert store.fetch_contradictions() == []  # it was acknowledged
    [c] = store.fetch_contradictions(include_acknowledged=True)
    assert (c.stored_value, c.incoming_value, c.resolution, c.seen_count) == (
        "web3notes.example",
        "wrong.com",
        "blocked",
        3,
    )


def test_migrated_history_keeps_its_chain_and_reasons(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _legacy(db, [("city", "Leeds")])
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE user_facts SET value = 'London' WHERE key = 'city'")
        conn.execute(
            "INSERT INTO user_fact_history(key, old_value, old_confidence, new_value, "
            "new_confidence, source, reason, changed_at) VALUES ('city', 'Leeds', 0.9, "
            "'London', 1.0, 'user:correction', 'correct', ?)",
            ((NOW + timedelta(days=2)).isoformat(),),
        )
    history = MemoryStore(db_path=db).fetch_fact_history("city")
    assert [(e.reason, e.old_value, e.new_value) for e in history] == [
        ("correct", "Leeds", "London"),
        ("capture", None, "Leeds"),
    ]


# --- entity resolution (PR 4a) ------------------------------------------------------


def _in_session(session_id: str):  # type: ignore[no-untyped-def]
    from iris_harness.foundation.observability.session_log import session_scope

    return session_scope(session_id)


def test_a_folded_name_is_the_same_fact_not_a_new_one(store: MemoryStore) -> None:
    """ "Northwind Bank Ltd" after "Northwind Bank": the same bank (entity_aliases.yaml suffixes)."""
    store.upsert_user_fact(_fact("bank", "Northwind Bank"))
    store.upsert_user_fact(_fact("bank", "Northwind Bank Ltd"))
    [fact] = store.fetch_all_user_facts()  # one fact, one statement
    assert fact.value == "Northwind Bank"
    assert len(store.memory_graph().history(OWNER_ID, "tv:banks_with")) == 1
    assert store.fetch_contradictions() == []  # not a supersede of itself
    assert (
        store.add_fact_proposal(key="bank", value="northwind bank ltd", confidence=0.9, source="x")
        is None
    )


def test_a_look_alike_waits_for_evidence_then_merges_and_can_be_undone(store: MemoryStore) -> None:
    graph = store.memory_graph()
    with _in_session("s1"):
        store.upsert_user_fact(_fact("credit_card", "Wingtip Bank Credit Card"))
        store.upsert_user_fact(_fact("credit_card", "Wingtip Bank Credit Cards"))
    [candidate] = [d for d in graph.decisions() if d.decision == "candidate"]
    assert len({f.value for f in store.fetch_all_user_facts()}) == 2  # separate, for now
    for session in ("s2", "s3"):
        with _in_session(session):
            store.upsert_user_fact(_fact("credit_card", "Wingtip Bank Credit Cards"))
    [merge] = [d for d in graph.decisions() if d.decision == "same"]
    assert merge.decided_by == "evidence" and len(merge.evidence) == 3
    assert {f.value for f in store.fetch_all_user_facts()} == {"Wingtip Bank Credit Card"}
    graph.unmerge(merge.id, decided_by="owner")
    assert len({f.value for f in store.fetch_all_user_facts()}) == 2


def test_the_cli_lists_merges_and_undoes_them(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from iris_harness.cli.memory import memory_app

    db = tmp_path / "memory.db"
    s = MemoryStore(db_path=db)
    graph = s.memory_graph()
    a = graph.add_entity("mem:Organization", "Acme")
    b = graph.add_entity("mem:Organization", "Acme Inc")
    merge = graph.merge(a.id, b.id, decided_by="evidence", evidence=["s1", "s2", "s3"])
    listed = CliRunner().invoke(memory_app, ["entities", "--db-path", str(db)])
    assert listed.exit_code == 0 and merge.id in listed.output and "Acme" in listed.output
    undone = CliRunner().invoke(memory_app, ["unmerge", merge.id, "--db-path", str(db)])
    assert undone.exit_code == 0, undone.output
    assert graph.get_entity(b.id).merged_into is None  # type: ignore[union-attr]
    distinct = CliRunner().invoke(memory_app, ["distinct", merge.id, "--db-path", str(db)])
    assert distinct.exit_code == 0 and graph.are_distinct(a.id, b.id)


def test_the_cli_settles_a_look_alike_as_the_same(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from iris_harness.cli.memory import memory_app

    db = tmp_path / "memory.db"
    graph = MemoryStore(db_path=db).memory_graph()
    old = graph.add_entity("mem:Organization", "Acme")
    new = graph.add_entity("mem:Organization", "Acmee")
    candidate = graph.note_candidate(old.id, new.id, score=0.9, episode="s1")
    assert candidate is not None
    same = CliRunner().invoke(memory_app, ["same", candidate.id, "--db-path", str(db)])
    assert same.exit_code == 0, same.output
    assert "unmerge" in same.output  # the undo is printed with it
    assert graph.canonical_id(new.id) == old.id
    again = CliRunner().invoke(memory_app, ["same", candidate.id, "--db-path", str(db)])
    assert again.exit_code == 1  # already decided
