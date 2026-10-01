"""Removing things from memory reaches every reader, and comes back exactly (ADR-0119)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from iris_harness.memory import graph as graph_module
from iris_harness.memory.graph import build_memory_graph
from iris_harness.memory.graph_context import GraphContext
from iris_harness.memory.removal import MemoryRemoval, NotFoundError, RemovalError, Target
from iris_harness.memory.store import MemoryStore, UserFact

# Fact keys such as `bank` and `credit_card` come from the test vocabulary fragment
# (tests/fixtures/test_vocabulary, installed by the `test_vocabulary` fixture), not
# from whichever domain plugin the tree happens to carry.
pytestmark = pytest.mark.usefixtures("test_vocabulary")


@pytest.fixture(autouse=True)
def _fresh_config() -> None:
    graph_module.reset_config_cache()


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


@pytest.fixture
def rederived() -> list[str]:
    return []


@pytest.fixture
def removal(store: MemoryStore, rederived: list[str]) -> MemoryRemoval:
    return MemoryRemoval(store, rederive=rederived.append)


def _fact(key: str, value: str, confirmed: bool = True) -> UserFact:
    now = datetime.now(UTC)
    return UserFact(
        key=key,
        value=value,
        confidence=1.0,
        source="test",
        first_seen=now,
        last_confirmed=now,
        confirmed=confirmed,
    )


def _summary(store: MemoryStore, session: str, referenced: str) -> None:
    store.save_conversation_turns(session, [("user", f"about {referenced}"), ("assistant", "ok")])
    store.save_conversation_summary(session, f"Goal: test\nReferenced: {referenced}")


def _labels(store: MemoryStore) -> set[str]:
    return {n["label"] for n in build_memory_graph(store)["nodes"]}


def _node(store: MemoryStore, label: str) -> dict[str, Any]:
    return next(n for n in build_memory_graph(store)["nodes"] if n["label"] == label)


def _entity_id(store: MemoryStore, label: str) -> str:
    return str(_node(store, label)["ref"]["id"])


def _values(store: MemoryStore) -> set[str]:
    return {f.value for f in store.fetch_all_user_facts()}


class TestRemovingAnEntity:
    def test_it_leaves_the_map_facts_the_graph_tool_and_linking_then_restore_brings_it_back(
        self, store: MemoryStore, removal: MemoryRemoval, rederived: list[str]
    ) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind Bank"))
        _summary(store, "s1", "Northwind Bank")
        northwind = _entity_id(store, "Northwind Bank")
        context = GraphContext(store.memory_graph())
        assert context.linked("what about Northwind Bank?", max_tokens=500).shown

        preview = removal.preview([Target("entity", northwind)])
        assert preview[0]["lines"][0] == "forgets 1 fact: bank: Northwind Bank"
        [item] = removal.remove([Target("entity", northwind)])

        assert item["kind"] == "entity" and item["cascade"][0]["text"] == "bank: Northwind Bank"
        assert "Northwind Bank" not in _labels(store)  # nor through the summary mention
        assert "Northwind Bank" not in _values(store)
        assert rederived == ["bank"]
        assert store.memory_graph().neighbourhood(northwind) == []
        assert (
            not GraphContext(store.memory_graph()).linked("Northwind Bank?", max_tokens=500).shown
        )

        removal.restore(item["id"])

        assert "Northwind Bank" in _labels(store) and "Northwind Bank" in _values(store)
        assert rederived == ["bank", "bank"]
        assert removal.items() == []

    def test_removing_twice_is_one_removal(
        self, store: MemoryStore, removal: MemoryRemoval
    ) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind Bank"))
        northwind = _entity_id(store, "Northwind Bank")
        first = removal.remove([Target("entity", northwind)])
        again = removal.remove([Target("entity", northwind)])
        assert first == again and len(removal.items()) == 1

    def test_an_unknown_target_removes_nothing(
        self, store: MemoryStore, removal: MemoryRemoval
    ) -> None:
        _summary(store, "s1", "Walgreens")
        with pytest.raises(NotFoundError):
            removal.remove([Target("name", "Walgreens"), Target("session", "nope")])
        assert removal.items() == []


class TestSuppressedNames:
    def test_a_removed_name_stays_hidden_through_a_new_summary_until_restored(
        self, store: MemoryStore, removal: MemoryRemoval
    ) -> None:
        # Two sessions each: a one-off mention is not drawn at all (min_sessions, PR 1).
        _summary(store, "s0", "Walgreens, Anurag Sutton")
        _summary(store, "s1", "Walgreens, Anurag Sutton")
        [item] = removal.remove([Target("name", "Walgreens")])
        _summary(store, "s2", "Walgreens Inc")  # folded to the same name

        labels = _labels(store)
        assert "Walgreens" not in labels and "Anurag Sutton" in labels

        removal.restore(item["id"])
        assert "Walgreens" in _labels(store)

    def test_a_confirmed_fact_wins_over_a_removed_name_and_says_so(
        self, store: MemoryStore, removal: MemoryRemoval
    ) -> None:
        _summary(store, "s1", "Barclays")
        removal.remove([Target("name", "Barclays")])
        store.upsert_user_fact(_fact("bank", "Barclays", confirmed=False))
        assert "Barclays" not in _labels(store)  # a proposal does not bring it back

        store.upsert_user_fact(_fact("bank", "Barclays"))

        node = _node(store, "Barclays")
        assert node["previously_removed"] is True and node["ref"]["kind"] == "entity"
        # ... and the summaries that mention it link to it again.
        edges = {(e["source"], e["target"]) for e in build_memory_graph(store)["edges"]}
        session = _node(store, "s1")["id"]
        assert (session, node["id"]) in edges

    def test_a_removed_entity_is_not_brought_back_by_a_summary_but_a_new_fact_is_new(
        self, store: MemoryStore, removal: MemoryRemoval
    ) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind Bank"))
        old = _entity_id(store, "Northwind Bank")
        removal.remove([Target("entity", old)])
        _summary(store, "s1", "Northwind Bank")
        assert "Northwind Bank" not in _labels(store)

        store.upsert_user_fact(_fact("bank", "Northwind Bank"))

        node = _node(store, "Northwind Bank")
        assert node["ref"]["id"] != old and node["previously_removed"] is True

    def test_an_alias_of_a_removed_entity_is_hidden_too(
        self, store: MemoryStore, removal: MemoryRemoval
    ) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind Bank"))
        graph = store.memory_graph()
        northwind = graph.get_entity(_entity_id(store, "Northwind Bank"))
        assert northwind is not None
        graph.store.save(entities=[northwind.evolve(aliases=("Northwind Savings Bank",))])
        removal.remove([Target("entity", northwind.id)])
        _summary(store, "s1", "Northwind Savings Bank")

        assert "Northwind Savings Bank" not in _labels(store)

    def test_a_removed_entity_is_not_drawn_even_if_a_claim_names_it(
        self, store: MemoryStore, removal: MemoryRemoval
    ) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind Bank"))
        northwind = _entity_id(store, "Northwind Bank")
        [item] = removal.remove([Target("entity", northwind)])
        graph = store.memory_graph()
        [withdrawn] = item["cascade"]
        s = graph.store.get_statement(withdrawn["statement_id"])
        assert s is not None
        # Written behind the graph's back (an import, a second process).
        graph.store.save(statements=[s.evolve(status="confirmed", retracted_at=None)])
        store.delete_removal(item["id"])  # and its name no longer suppressed

        assert "Northwind Bank" not in _labels(store)


class TestRemovingASession:
    def test_it_leaves_the_map_and_recall_until_restored(
        self, store: MemoryStore, removal: MemoryRemoval
    ) -> None:
        _summary(store, "cal6-verify", "Budget Review Meeting")
        _summary(store, "real", "Anurag Sutton")
        assert "cal6-verify" in _labels(store)

        [item] = removal.remove([Target("session", "cal6-verify")])

        assert {"cal6-verify", "Budget Review Meeting"}.isdisjoint(_labels(store))
        assert store.search_summaries("Budget") == []
        assert store.search_turns("Budget") == []
        assert store.search_summaries("Budget", include_removed=True)
        assert store.removed_session_ids() == {"cal6-verify"}

        removal.restore(item["id"])
        assert "cal6-verify" in _labels(store) and store.search_summaries("Budget")

    def test_nodes_carry_what_remove_passes_back(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("nickname", "Robin"))
        _summary(store, "s1", "Walgreens")
        _summary(store, "s2", "Walgreens")  # a one-off mention is not drawn (PR 1)
        refs = {n["label"]: n["ref"] for n in build_memory_graph(store)["nodes"]}
        assert refs["s1"] == {"kind": "session", "id": "s1"}
        assert refs["Walgreens"] == {"kind": "name", "id": "Walgreens"}
        assert refs["You"] is None
        fact_ref = next(r for label, r in refs.items() if label.endswith("Robin"))
        assert fact_ref is not None and fact_ref["kind"] == "fact"


class TestRemovingAFact:
    def test_it_is_forgotten_listed_and_restored(
        self, store: MemoryStore, removal: MemoryRemoval, rederived: list[str]
    ) -> None:
        store.upsert_user_fact(_fact("nickname", "Robin"))
        statement = next(
            n["ref"]["id"] for n in build_memory_graph(store)["nodes"] if n["kind"] == "fact"
        )

        [item] = removal.remove([Target("fact", statement)])

        assert item["id"] == statement and "Robin" not in _values(store)
        assert [i["id"] for i in removal.items()] == [statement]
        removal.restore(statement)
        assert "Robin" in _values(store) and removal.items() == []
        assert rederived == ["nickname", "nickname"]

    def test_a_forgotten_fact_said_again_leaves_the_list(
        self, store: MemoryStore, removal: MemoryRemoval
    ) -> None:
        store.upsert_user_fact(_fact("nickname", "Robin"))
        store.delete_user_fact("nickname")
        store.upsert_user_fact(_fact("nickname", "Robin"))
        assert removal.items() == []

    def test_a_fact_forgotten_elsewhere_is_listed_too(
        self, store: MemoryStore, removal: MemoryRemoval
    ) -> None:
        store.upsert_user_fact(_fact("nickname", "Robin"))
        store.delete_user_fact("nickname")
        assert [i["kind"] for i in removal.items()] == ["fact"]


class TestDeletingForGood:
    def test_it_needs_the_word(self, removal: MemoryRemoval) -> None:
        with pytest.raises(RemovalError, match="type 'delete'"):
            removal.delete([], confirm="yes")

    def test_a_deleted_entity_stays_suppressed_until_its_row_is_restored(
        self, store: MemoryStore, removal: MemoryRemoval
    ) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind Bank"))
        northwind = _entity_id(store, "Northwind Bank")
        [item] = removal.remove([Target("entity", northwind)])
        _summary(store, "s1", "Northwind Bank")
        _summary(store, "s2", "Northwind Bank")  # a one-off mention is not drawn (PR 1)

        result = removal.delete([item["id"], "rm_unknown"], confirm="delete")

        assert result["deleted"] == [item["id"]]
        assert result["refused"] == [{"id": "rm_unknown", "reason": "not in the Removed list"}]
        assert store.memory_graph().get_entity(northwind) is None
        assert removal.items()[0]["permanent"] is True
        assert "Northwind Bank" not in _labels(store)
        assert removal.delete([item["id"]], confirm="delete")["refused"][0]["reason"] == (
            "already deleted"
        )

        removal.restore(item["id"])  # lifts the suppression; the entity stays gone
        assert "Northwind Bank" in _labels(store)  # the summary mention, as a plain name

    def test_a_name_has_no_permanent_delete(
        self, store: MemoryStore, removal: MemoryRemoval
    ) -> None:
        _summary(store, "s1", "Walgreens")
        [item] = removal.remove([Target("name", "Walgreens")])
        refused = removal.delete([item["id"]], confirm="delete")["refused"]
        assert "restore it" in refused[0]["reason"]

    def test_a_deleted_session_is_gone(self, store: MemoryStore, removal: MemoryRemoval) -> None:
        _summary(store, "cascade", "Budget Review Meeting")
        [item] = removal.remove([Target("session", "cascade")])
        removal.delete([item["id"]], confirm="delete")
        assert store.load_conversation_summary("cascade") == ""
        assert store.fetch_turn_ids("cascade") == []

    def test_a_forgotten_fact_is_purged(self, store: MemoryStore, removal: MemoryRemoval) -> None:
        store.upsert_user_fact(_fact("nickname", "Robin"))
        store.delete_user_fact("nickname")
        [item] = removal.items()
        assert removal.delete([item["id"]], confirm="delete")["deleted"] == [item["id"]]
        assert removal.items() == []


class TestLessonsAndPatternsSayWhereToEditThem:
    def test_meta_file(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from iris_harness.memory import identity
        from iris_harness.memory.identity import loader
        from iris_harness.memory.identity.loader import Behavior, EpisodicPattern

        lesson = tmp_path / "behaviors" / "reminders.md"
        monkeypatch.setattr(
            identity,
            "list_behaviors",
            lambda: [Behavior("reminders", "d", ("general",), (), "b", lesson)],
        )
        monkeypatch.setattr(loader, "EPISODIC_MD_PATH", tmp_path / "episodic.md")
        monkeypatch.setattr(
            identity, "list_episodic_patterns", lambda: [EpisodicPattern("p1", "asks early", 3)]
        )

        nodes = build_memory_graph(store)["nodes"]
        files = {
            n["label"]: n["meta"].get("file") for n in nodes if n["kind"] in ("lesson", "pattern")
        }

        assert files == {"reminders": str(lesson), "asks early": str(tmp_path / "episodic.md")}
        assert all(n["ref"] is None for n in nodes if n["kind"] in ("lesson", "pattern"))


class TestCrossSessionRecall:
    def test_a_removed_session_is_never_recalled_into_another(
        self, store: MemoryStore, removal: MemoryRemoval
    ) -> None:
        from types import SimpleNamespace

        from iris_harness.memory.retriever import MemoryRetriever
        from iris_harness.memory.semantic_index import RetrievedTurn

        turns = [
            RetrievedTurn("1", "cascade", "user", "budget review at 11"),
            RetrievedTurn("2", "mine", "user", "budget for June"),
        ]
        index = SimpleNamespace(
            is_ready=True,
            query_facts=lambda q, n: [],
            query_turns_detailed=lambda q, **kw: list(turns),
            query_episodic=lambda q, n: [],
        )
        _summary(store, "cascade", "Budget Review Meeting")
        removal.remove([Target("session", "cascade")])

        context = MemoryRetriever(store=store, index=index).build_context(  # type: ignore[arg-type]
            query="budget", session_id="now"
        )

        assert context.related_turns == ("user: budget for June",)


class TestTheCommandLine:
    def test_remove_list_restore_and_delete(self, store: MemoryStore) -> None:
        from typer.testing import CliRunner

        from iris_harness.cli.memory import memory_app

        store.upsert_user_fact(_fact("bank", "Northwind Bank"))
        db = ["--db-path", str(store.db_path)]
        run = CliRunner().invoke

        removed = run(memory_app, ["remove", "entity", "Northwind Bank", "--yes", *db])
        assert removed.exit_code == 0, removed.output
        assert "forgets 1 fact: bank: Northwind Bank" in removed.output
        assert "Northwind Bank" not in _values(store)

        [item] = MemoryRemoval(store).items()
        listed = run(memory_app, ["removed", *db])
        assert item["id"] in listed.output

        restored = run(memory_app, ["restore", item["id"], *db])
        assert restored.exit_code == 0 and "Northwind Bank" in _values(store)

        run(memory_app, ["remove", "name", "Walgreens", "--yes", *db])
        [name] = MemoryRemoval(store).items()
        no_word = run(memory_app, ["delete", name["id"], *db])
        kept = run(memory_app, ["delete", name["id"], "--confirm", "delete", *db])
        assert no_word.exit_code == 1 and "type 'delete'" in no_word.output
        assert kept.exit_code == 1 and "restore it" in kept.output

    def test_an_unknown_target_is_an_error(self, store: MemoryStore) -> None:
        from typer.testing import CliRunner

        from iris_harness.cli.memory import memory_app

        result = CliRunner().invoke(
            memory_app, ["remove", "session", "nope", "--yes", "--db-path", str(store.db_path)]
        )
        assert result.exit_code == 1 and "no session" in result.output
