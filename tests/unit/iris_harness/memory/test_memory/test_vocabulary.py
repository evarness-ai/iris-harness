"""Memory learns words it had none for (memris PR 7; ADR-0115 decision 7).

"I mentor two juniors" has no key. It is not stored: it is counted. Said often enough,
in enough conversations, ``mentors`` becomes a learned term, and what is said with it
from then on is a fact like any other — reviewed, confirmed, recalled.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from iris_harness.cli.ontology import ontology_app
from iris_harness.foundation.auth import auth_headers
from iris_harness.memory.coordinator import FactCoordinator
from iris_harness.memory.fact_extractor import keys_block
from iris_harness.memory.graph_context import graph_context
from iris_harness.memory.store import MemoryStore
from iris_harness.runtime.turn_capture import TurnCapture
from iris_harness.server.iris_api.main import create_app

MENTOR = "I mentor two juniors at work"
SAYS_MENTORS = [("mentors", "two juniors", 0.8, "test")]


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


@pytest.fixture
def capture(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> Any:
    host = SimpleNamespace(
        config_dir=Path("config"),
        memory_store=store,
        semantic_index=None,
        signal_collector=SimpleNamespace(record_metric=lambda **kw: None),
        tier_router=SimpleNamespace(),
    )
    cap = TurnCapture(host)  # type: ignore[arg-type]
    monkeypatch.setattr(cap, "_extract_facts_via_llm", lambda msg: [])
    return cap


def _says(capture: Any, monkeypatch: pytest.MonkeyPatch, facts: list[tuple]) -> None:
    monkeypatch.setattr(capture, "_extract_facts_via_llm", lambda msg: facts)


def _mentor_in(capture: Any, monkeypatch: pytest.MonkeyPatch, sessions: list[str]) -> None:
    _says(capture, monkeypatch, SAYS_MENTORS)
    for session in sessions:
        capture.extract_and_store_facts(MENTOR, session)


def _term(store: MemoryStore, name: str = "learned:mentors") -> Any:
    return next((t for t in store.vocabulary().terms() if t.name == name), None)


class TestCounting:
    def test_an_unknown_key_is_counted_not_stored(
        self, capture: Any, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mentor_in(capture, monkeypatch, ["s1"])

        assert store.fetch_fact_proposals() == []
        assert store.fetch_all_user_facts() == []
        term = _term(store)
        assert (term.status, term.observations, term.examples) == (
            "candidate",
            1,
            ("two juniors",),
        )

    def test_it_is_learned_once_it_keeps_coming_up_and_then_reviewed_like_any_fact(
        self, capture: Any, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mentor_in(capture, monkeypatch, ["s1", "s1", "s2", "s2"])  # 4x, 2 conversations
        assert _term(store).status == "candidate"
        assert store.fetch_fact_proposals() == []

        _mentor_in(capture, monkeypatch, ["s3"])  # 5x, 3 conversations (learning.yaml)

        assert _term(store).status == "active"
        [proposal] = store.fetch_fact_proposals()
        assert (proposal.key, proposal.value) == ("mentors", "two juniors")
        FactCoordinator(store, None).approve_proposal(proposal.id)
        facts = {f.key: f.value for f in store.fetch_all_user_facts(confirmed_only=True)}
        assert facts == {"mentors": "two juniors"}
        assert store.learned_fact_keys() == ["mentors"]

    def test_a_learned_key_is_offered_to_the_extractor(self) -> None:
        assert keys_block(("mentors",)).splitlines()[-1] == "- mentors"
        assert keys_block(("hobby",)).count("- hobby") == 1  # declared keys once

    def test_nothing_is_learned_from_what_the_gates_refuse(
        self, capture: Any, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _says(capture, monkeypatch, [("mentors", "three seniors", 0.8, "test")])  # not said
        capture.extract_and_store_facts(MENTOR, "s1")
        _says(capture, monkeypatch, [("topic", "juniors", 0.8, "test")])  # never_learn
        capture.extract_and_store_facts(MENTOR, "s1")

        assert store.vocabulary().terms() == []


class TestAlias:
    def test_a_near_synonym_is_written_with_the_existing_key(
        self, capture: Any, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _says(capture, monkeypatch, [("emails", "sk@x.example", 0.9, "test")])

        capture.extract_and_store_facts("my emails: sk@x.example", "s1")

        term = _term(store, "learned:emails")
        assert (term.status, term.alias_of) == ("alias", "mem:email")
        [proposal] = store.fetch_fact_proposals()
        assert (proposal.key, proposal.value) == ("email", "sk@x.example")

    def test_an_alias_is_written_under_its_term_s_fact_key_not_its_name(
        self, capture: Any, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _says(capture, monkeypatch, [("work_at", "Infosys", 0.9, "test")])

        capture.extract_and_store_facts("I work at Infosys", "s1")

        assert _term(store, "learned:work_at").alias_of == "mem:works_at"
        fact = store.fetch_user_fact("employer")  # works_at's key, not "works_at"
        assert fact is not None and fact.value == "Infosys"


class TestDecisions:
    def test_never_retracts_what_was_said_and_stops_learning(
        self, capture: Any, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.vocabulary().activate("mentors")  # nothing to activate yet
        _mentor_in(capture, monkeypatch, ["s1"])
        store.vocabulary().activate("mentors")
        _mentor_in(capture, monkeypatch, ["s2"])
        assert len(store.fetch_fact_proposals()) == 1

        store.vocabulary().reject("mentors")
        _mentor_in(capture, monkeypatch, ["s3", "s4", "s5"])

        assert store.fetch_fact_proposals() == []
        assert store.learned_fact_keys() == []
        assert _term(store).status == "rejected"
        assert store.memory_graph().check_usage() == []

    def test_the_agent_can_ask_about_a_learned_word(
        self, capture: Any, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mentor_in(capture, monkeypatch, ["s1"])
        store.vocabulary().activate("mentors")
        _mentor_in(capture, monkeypatch, ["s2"])
        FactCoordinator(store, None).approve_proposal(store.fetch_fact_proposals()[0].id)
        ctx = graph_context(store)

        assert "learned:mentors" in ctx.tool_description()
        assert "the user: mentors = two juniors" in ctx.query("user", relation="mentors")


class TestSurfaces:
    @pytest.fixture
    def learned(
        self, capture: Any, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> MemoryStore:
        _mentor_in(capture, monkeypatch, ["s1", "s2", "s3", "s4", "s5"])
        _says(capture, monkeypatch, [("sings_in", "a choir", 0.8, "test")])
        capture.extract_and_store_facts("I sing in a choir", "s1")
        return store

    def test_the_cli_lists_promotes_exports_and_rejects(self, learned: MemoryStore) -> None:
        db = ["--db-path", str(learned.db_path)]
        run = CliRunner().invoke

        listed = run(ontology_app, ["terms", *db])
        promoted = run(ontology_app, ["promote", "mentors", *db])
        exported = run(ontology_app, ["export", *db])
        rejected = run(ontology_app, ["reject", "sings_in", *db])
        missing = run(ontology_app, ["promote", "nothing", *db])

        assert listed.exit_code == 0 and "learned:mentors" in listed.output
        assert "learned:sings_in" in listed.output
        assert promoted.exit_code == 0
        assert '  mentors: { domain: Person, datatype: string, label: "mentors" }' in (
            promoted.output
        )
        assert "key: mentors" in promoted.output
        assert exported.exit_code == 0 and "attributes:" in exported.output
        assert rejected.exit_code == 0 and _term(learned, "learned:sings_in").status == "rejected"
        assert missing.exit_code == 1

    @pytest.fixture
    def client(self, learned: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
        monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
        runtime = SimpleNamespace(memory_store=learned, semantic_index=None)
        with TestClient(
            create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
        ) as c:
            yield c

    def test_the_api_lists_decides_and_promotes(
        self, client: TestClient, learned: MemoryStore
    ) -> None:
        listed = client.get("/memory/terms").json()
        active = client.get("/memory/terms", params={"status": "active"}).json()
        promotion = client.get("/memory/terms/mentors/promotion").json()
        activated = client.post("/memory/terms/sings_in/activate").json()
        rejected = client.post("/memory/terms/learned:mentors/reject").json()

        assert listed["count"] == 2
        assert [t["name"] for t in active["terms"]] == ["learned:mentors"]
        assert active["terms"][0]["conversations"] == 5
        assert "mentors: { domain: Person" in promotion["yaml"]
        assert activated["status"] == "active"
        assert rejected["status"] == "rejected"
        assert client.post("/memory/terms/nothing/reject").status_code == 404
        assert client.get("/memory/terms/nothing/promotion").status_code == 404

    def test_deciding_is_a_gated_write(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "0")

        assert client.post("/memory/terms/mentors/reject").status_code == 403
        assert client.get("/memory/terms").status_code == 200


def test_a_term_activated_elsewhere_is_seen_by_an_open_store(tmp_path: Path) -> None:
    """The runtime's store stays open while the CLI decides: its next write must see it."""
    running = MemoryStore(db_path=tmp_path / "memory.db")
    running.ensure_schema()
    running.memory_graph()  # the vocabulary is loaded now, before the term exists
    elsewhere = MemoryStore(db_path=tmp_path / "memory.db")
    elsewhere.vocabulary().observe(
        "mentors", domain="mem:Person", datatype="x", example="two", episode="s1"
    )
    elsewhere.vocabulary().activate("mentors")

    pid = running.add_fact_proposal(key="mentors", value="two", confidence=0.9, source="t")

    assert pid is not None
    assert running.learned_fact_keys() == ["mentors"]
    elsewhere.vocabulary().reject("mentors")
    with pytest.raises(ValueError):
        running.add_fact_proposal(key="mentors", value="three", confidence=0.9, source="t")


def test_a_learned_key_follows_the_confirmation_tiers(
    capture: Any, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mentor_in(capture, monkeypatch, ["s1"])
    store.vocabulary().activate("mentors")
    _says(capture, monkeypatch, [("mentors", "two juniors", 0.9, "test")])

    capture.extract_and_store_facts("remember that I mentor two juniors", "s2")

    assert capture.take_notices() == ["mentors: two juniors"]  # tier A, like any key
    assert [f.key for f in store.fetch_all_user_facts(confirmed_only=True)] == ["mentors"]


def test_the_extractor_is_offered_the_learned_keys(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.vocabulary().observe(
        "mentors", domain="mem:Person", datatype="x", example="a", episode="s"
    )
    store.vocabulary().activate("mentors")
    offered: list[tuple[str, ...]] = []

    def _extract(message: str, *, client: Any, extra_keys: tuple[str, ...] = ()) -> list[Any]:
        offered.append(extra_keys)
        return []

    monkeypatch.setattr("iris_harness.memory.fact_extractor.extract_facts_with_llm", _extract)
    monkeypatch.setattr("iris_harness.llm.client.CodingLLMClient", lambda cfg: object())
    host = SimpleNamespace(
        memory_store=store, tier_router=SimpleNamespace(get_llm_config=lambda _i: None)
    )

    TurnCapture(host)._extract_facts_via_llm("I mentor two juniors")  # type: ignore[arg-type]

    assert offered == [("mentors",)]
