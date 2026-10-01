"""The memory graph's acceptance suite (ADR-0115 decision 13; memris plan "How we will know
it works"), deterministic, on the real ontology and the real capture path.

Scenario 8 (no vocabulary in code) is ``tests/unit/memris/ontology/test_memris_no_vocabulary.py``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from iris_harness.memory.coordinator import FactCoordinator
from iris_harness.memory.graph_context import GraphContext, graph_context
from iris_harness.memory.ontology import memory_config_dir
from iris_harness.memory.store import MemoryStore, UserFact
from iris_harness.runtime.turn_capture import TurnCapture
from memris.graph import MemoryGraph
from memris.interchange.jsonld import export_document, import_document
from memris.model import OWNER_ID
from memris.ontology import load_or_raise
from memris.store import SQLiteGraphStore

# Fact keys such as `bank` and `credit_card` come from the test vocabulary fragment
# (tests/fixtures/test_vocabulary, installed by the `test_vocabulary` fixture), not
# from whichever domain plugin the tree happens to carry.
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("test_vocabulary")]


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


def _say(store: MemoryStore, key: str, value: str) -> None:
    now = datetime.now(UTC)
    store.upsert_user_fact(UserFact(key, value, 0.9, "test", now, now, 1, True))


@contextmanager
def _conversation(session_id: str) -> Iterator[None]:
    from iris_harness.foundation.observability.session_log import session_scope

    with session_scope(session_id):
        yield


def _capture(store: MemoryStore, monkeypatch: pytest.MonkeyPatch, facts: list[tuple]) -> Any:
    host = SimpleNamespace(
        config_dir=Path("config"),
        memory_store=store,
        semantic_index=None,
        signal_collector=SimpleNamespace(record_metric=lambda **kw: None),
        tier_router=SimpleNamespace(),
    )
    capture = TurnCapture(host)  # type: ignore[arg-type]
    monkeypatch.setattr(capture, "_extract_facts_via_llm", lambda _msg: facts)
    return capture


# 1 --------------------------------------------------------------------------------------


def test_1_supersede_keeps_the_past_answerable(store: MemoryStore) -> None:
    graph = store.memory_graph()
    clock = {"now": datetime(2026, 3, 1, tzinfo=UTC)}
    graph._clock = lambda: clock["now"]  # the store's graph, on a clock the test turns
    _say(store, "bank", "Barclays")
    clock["now"] = datetime(2026, 9, 1, tzinfo=UTC)
    _say(store, "bank", "Litware")
    ctx = graph_context(store)

    now = ctx.query("user", relation="bank")
    before_the_move = ctx.query("user", relation="bank", as_of="2026-06-01")
    before_anything = ctx.query("user", relation="bank", as_of="2026-01-01")

    assert "Litware" in now and "Barclays" not in now
    assert "Barclays" in before_the_move and "until 2026-09-01" in before_the_move
    assert "Litware" not in before_the_move
    assert "Nothing confirmed" in before_anything
    [old] = [s for s in graph.history(OWNER_ID, "tv:banks_with") if s.valid_to is not None]
    assert ctx.label(old.object_id or "") == "Barclays"  # closed, not deleted


# 2 --------------------------------------------------------------------------------------


def test_2_one_hop_is_captured_and_a_public_figure_is_not(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    wife = [("spouse", "Petra", 0.9, "test"), ("employer", "Infosys", 0.9, "test", "Petra")]
    _capture(store, monkeypatch, wife).extract_and_store_facts(
        "my wife Petra works at Infosys", "s1"
    )
    coordinator = FactCoordinator(store, None)
    for proposal in store.fetch_fact_proposals():
        coordinator.approve_proposal(proposal.id)

    linked = graph_context(store).linked("is Petra home?", max_tokens=400).text
    assert "the user: spouse of Petra" in linked
    assert "Petra: works at Infosys" in linked

    celebrity = [("employer", "Tesla", 0.9, "test", "Elon")]
    _capture(store, monkeypatch, celebrity).extract_and_store_facts(
        "I read that Elon works at Tesla", "s2"
    )
    assert store.fetch_fact_proposals() == []
    assert graph_context(store).named_in("Elon") == []


# 3 --------------------------------------------------------------------------------------


def test_3_an_unknown_kind_of_fact_is_counted_then_learned(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    mentors = [("mentors", "two juniors", 0.8, "test")]
    capture = _capture(store, monkeypatch, mentors)

    capture.extract_and_store_facts("I mentor two juniors", "s1")
    assert store.fetch_fact_proposals() == []  # no statement...
    [term] = store.vocabulary().terms()
    assert (term.name, term.status, term.observations) == ("learned:mentors", "candidate", 1)

    for session in ("s1", "s2", "s2", "s3"):  # past learning.yaml's 5 across 3
        capture.extract_and_store_facts("I mentor two juniors", session)

    [term] = store.vocabulary().terms()
    assert term.status == "active"
    assert "learned:mentors" in store.memory_graph().ontology.attributes
    [proposal] = store.fetch_fact_proposals()
    assert (proposal.key, proposal.value) == ("mentors", "two juniors")


# 4 --------------------------------------------------------------------------------------


def test_4_a_look_alike_waits_merges_on_evidence_and_undoes(store: MemoryStore) -> None:
    graph = store.memory_graph()
    with _conversation("s1"):
        _say(store, "credit_card", "Wingtip Bank Credit Card")
        _say(store, "credit_card", "Wingtip Bank Credit Cards")
    [candidate] = graph.open_candidates()
    ctx = graph_context(store)
    assert len(ctx.named_in("Wingtip Bank Credit Card and Wingtip Bank Credit Cards")) == 2

    for session in ("s2", "s3"):
        with _conversation(session):
            _say(store, "credit_card", "Wingtip Bank Credit Cards")
    [merge] = [d for d in graph.decisions() if d.decision == "same"]
    assert merge.decided_by == "evidence" and len(merge.evidence) == 3
    assert len(ctx.named_in("Wingtip Bank Credit Cards")) == 1  # one entity now

    graph.unmerge(merge.id, decided_by="owner")
    assert len(ctx.named_in("Wingtip Bank Credit Card and Wingtip Bank Credit Cards")) == 2
    assert candidate.id  # the pair was a candidate before it was a merge


# 5 --------------------------------------------------------------------------------------


def test_5_a_retraction_leaves_recall_and_the_prompt_but_not_the_audit_trail(
    store: MemoryStore,
) -> None:
    _say(store, "spouse", "Petra")
    FactCoordinator(store, None).forget("spouse")
    ctx = graph_context(store)

    assert ctx.linked("Petra", max_tokens=400).text == ""
    assert "Nothing confirmed" in ctx.query("Petra")
    [kept] = ctx.graph.history(OWNER_ID, "spouse")
    assert kept.status == "retracted" and kept.reason == "forgot"


# 6 --------------------------------------------------------------------------------------


def _edited_ontology(tmp_path: Path, edit: Any) -> Path:
    copy = tmp_path / "onto"
    shutil.copytree(memory_config_dir(), copy)
    for name in ("ontology.yaml", "shapes.yaml", "mappings.yaml"):
        path = copy / name
        path.write_text(edit(name, path.read_text(encoding="utf-8")), encoding="utf-8")
    return copy


def test_6_a_retired_term_reads_through_and_a_vanished_one_fails_the_check(
    store: MemoryStore, tmp_path: Path
) -> None:
    _say(store, "hobby", "chess")

    def retire(name: str, text: str) -> str:
        if name != "ontology.yaml":
            return text
        return text.replace(
            "  hobby:                    { domain: Person, datatype: string }",
            "  hobby:                    { domain: Person, datatype: string, deprecated: true,"
            " replaced_by: interest }",
        )

    renamed = load_or_raise(_edited_ontology(tmp_path / "a", retire))
    graph = MemoryGraph(renamed, SQLiteGraphStore(store.db_path))
    assert [s.literal for s in graph.current(OWNER_ID, "interest")] == ["chess"]
    assert "chess" in GraphContext(graph).query("user", relation="interest")
    assert graph.check_usage() == []

    def delete(name: str, text: str) -> str:
        return "\n".join(
            line for line in text.splitlines() if "hobby" not in line and "fact_hobby" not in line
        )

    vanished = load_or_raise(_edited_ontology(tmp_path / "b", delete))
    issues = MemoryGraph(vanished, SQLiteGraphStore(store.db_path)).check_usage()
    assert any("hobby" in str(issue) for issue in issues)


# 7 --------------------------------------------------------------------------------------


def test_7_json_ld_round_trips_the_owner_s_graph(store: MemoryStore, tmp_path: Path) -> None:
    _say(store, "spouse", "Petra")
    _say(store, "bank", "Barclays")
    _say(store, "bank", "Litware")
    store.add_fact_proposal(
        key="employer",
        value="Infosys",
        confidence=0.9,
        source="test",
        subject="Petra",
        subject_class="Person",
    )
    source = store.memory_graph()
    text = json.dumps(export_document(source))

    target = MemoryGraph(source.ontology, SQLiteGraphStore(tmp_path / "copy.db"))
    report = import_document(json.loads(text), target)

    assert report.lossless, report
    assert export_document(target) == json.loads(text)
