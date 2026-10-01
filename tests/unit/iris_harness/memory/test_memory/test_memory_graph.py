"""The memory graph is a view, not a store — and it knows no vocabulary.

The wiki was the last attempt: 2,505 regex-built pages, 2,175 typed "person", read
back zero times, drawn as one unpaginated layout. So: nothing extracted, nothing
stored, opens small, and what the cap hides says so.

Since memris PR 5 the graph is drawn from memris statements, and every word on it comes
from config: edge labels from ontology.yaml, what connects to what from mappings.yaml,
colour groups from entity_aliases.yaml. The last class of tests changes the config and
watches the graph follow.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from iris_harness.memory import graph as graph_module
from iris_harness.memory.graph import build_memory_graph, canonical_entity
from iris_harness.memory.ontology import memory_config_dir
from iris_harness.memory.store import MemoryStore, UserFact
from memris.ontology import Ontology, load_or_raise

# Fact keys such as `bank` and `credit_card` come from the test vocabulary fragment
# (tests/fixtures/test_vocabulary, installed by the `test_vocabulary` fixture), not
# from whichever domain plugin the tree happens to carry.
pytestmark = pytest.mark.usefixtures("test_vocabulary")


@pytest.fixture(autouse=True)
def _fresh_config() -> None:
    graph_module.reset_config_cache()


@pytest.fixture(autouse=True)
def _empty_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Map also draws the workspace's lessons and patterns (``_read_behaviors`` /
    ``_read_patterns``), which live under the session-wide IRIS_HOME, not in ``store``.
    Another test in the same worker that writes one there added a node here: the node
    counts below ran one high only in some orders. Each test starts with none."""
    from iris_harness.memory.identity import loader

    monkeypatch.setattr(loader, "EPISODIC_MD_PATH", tmp_path / "workspace" / "episodic.md")
    monkeypatch.setattr(loader, "BEHAVIORS_DIR", tmp_path / "workspace" / "behaviors")


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


def _fact(key: str, value: str, confirmed: bool = True) -> UserFact:
    now = datetime.now(UTC)
    return UserFact(
        key=key,
        value=value,
        confidence=0.9,
        source="test",
        first_seen=now,
        last_confirmed=now,
        confirmed=confirmed,
    )


def _ids(graph: dict[str, Any]) -> set[str]:
    return {n["id"] for n in graph["nodes"]}


def _labels(graph: dict[str, Any]) -> set[str]:
    return {n["label"] for n in graph["nodes"]}


def _node(graph: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [n for n in graph["nodes"] if n["label"] == label]
    assert len(matches) == 1, f"{label!r}: {len(matches)} nodes"
    return matches[0]


def _edges(graph: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {(e["source"], e["target"], e["label"]) for e in graph["edges"]}


def _summary(store: MemoryStore, session: str, text: str) -> None:
    store.save_conversation_turns(session, [("user", "q"), ("assistant", "a")])
    store.save_conversation_summary(session, text)


class TestBuiltFromWhatIsStored:
    def test_it_centres_on_you(self, store: MemoryStore) -> None:
        graph = build_memory_graph(store)

        assert "you" in _ids(graph)

    def test_a_relation_fact_points_at_its_entity(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind Bank"))

        graph = build_memory_graph(store)
        bank = _node(graph, "Northwind Bank")

        assert bank["kind"] == "entity"
        assert bank["meta"]["class"] == "tv:Institution"
        assert ("you", bank["id"], "banks with") in _edges(graph)

    def test_an_attribute_fact_is_its_own_node(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("blog", "tech4talk.com"))

        graph = build_memory_graph(store)
        blog = _node(graph, "blog: tech4talk.com")

        assert blog["kind"] == "fact"
        assert ("you", blog["id"], "blog") in _edges(graph)

    def test_a_summary_contributes_its_referenced_names(self, store: MemoryStore) -> None:
        _summary(
            store,
            "s1",
            "Goal: the flat\nDecisions: rent on the 5th\nOpen items: none\n"
            "Referenced: Petra Sutton, Northwind Bank, Elmwood",
        )
        _summary(store, "s2", "Referenced: Petra Sutton")

        graph = build_memory_graph(store)
        petra = _node(graph, "Petra Sutton")

        assert ("you", "session:s1", "talked about") in _edges(graph)
        assert ("session:s1", petra["id"], "mentioned") in _edges(graph)

    def test_a_listed_non_name_is_not_a_node(self, store: MemoryStore) -> None:
        _summary(store, "s1", "Referenced: none")

        assert "none" not in {label.lower() for label in _labels(build_memory_graph(store))}

    def test_a_session_without_a_summary_is_not_drawn(self, store: MemoryStore) -> None:
        store.save_conversation_turns("s1", [("user", "q"), ("assistant", "a")])

        assert "session:s1" not in _ids(build_memory_graph(store))

    def test_forgetting_a_fact_removes_its_node(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind Bank"))
        assert "Northwind Bank" in _labels(build_memory_graph(store))

        store.delete_user_fact("bank")

        assert "Northwind Bank" not in _labels(build_memory_graph(store))

    def test_an_unconfirmed_fact_is_drawn_but_marked(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind Bank", confirmed=False))

        assert _node(build_memory_graph(store), "Northwind Bank")["confirmed"] is False

    def test_confirmed_only_leaves_it_out(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind Bank", confirmed=False))

        graph = build_memory_graph(store, confirmed_only=True)

        assert "Northwind Bank" not in _labels(graph)


class TestTime:
    def test_as_of_draws_what_held_then(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("employer", "Barclays"))
        between = datetime.now(UTC)
        store.upsert_user_fact(_fact("employer", "Litware"))

        assert "Litware" in _labels(build_memory_graph(store))
        assert "Barclays" not in _labels(build_memory_graph(store))
        then = build_memory_graph(store, as_of=between)
        assert "Barclays" in _labels(then) and "Litware" not in _labels(then)
        assert then["as_of"] == between.isoformat()

    def test_an_edge_carries_its_time_bounds(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("employer", "Barclays"))

        graph = build_memory_graph(store)
        edge = next(e for e in graph["edges"] if e["label"] == "works at")

        assert edge["meta"]["recorded_at"] is not None
        assert edge["meta"]["valid_to"] is None
        assert edge["meta"]["predicate"] == "mem:works_at"


class TestNames:
    def test_company_suffixes_fold_together(self) -> None:
        assert canonical_entity("Acme Bank Ltd") == canonical_entity("Acme Bank")

    def test_an_alias_from_config_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            graph_module,
            "_CONFIG_CACHE",
            {"aliases": {"northwind": "Northwind Bank"}, "suffixes": [], "groups": {}},
        )

        assert canonical_entity("Northwind") == "Northwind Bank"

    def test_the_same_entity_from_two_sources_is_one_node(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind Bank"))
        _summary(store, "s1", "Referenced: Northwind Bank Ltd")

        graph = build_memory_graph(store)
        bank = _node(graph, "Northwind Bank")  # exactly one

        assert ("session:s1", bank["id"], "mentioned") in _edges(graph)


class TestOnlyWhatEarnsAPlaceIsDrawn:
    """Map cleanup plan decisions 7 and 8: a one-off mention, an ID-shaped item and a
    test run are summary noise, not memory (ADR-0119)."""

    def test_a_name_listed_once_is_not_drawn(self, store: MemoryStore) -> None:
        _summary(store, "s1", "Referenced: Pythagorean theorem")

        assert "Pythagorean theorem" not in _labels(build_memory_graph(store))

    def test_a_name_listed_in_two_sessions_is_drawn_once(self, store: MemoryStore) -> None:
        _summary(store, "s1", "Referenced: Anurag Sutton")
        _summary(store, "s2", "Referenced: Anurag Sutton")

        graph = build_memory_graph(store)
        anurag = _node(graph, "Anurag Sutton")

        assert anurag["meta"]["sessions"] == 2
        assert ("session:s1", anurag["id"], "mentioned") in _edges(graph)
        assert ("session:s2", anurag["id"], "mentioned") in _edges(graph)

    def test_listing_a_name_twice_in_one_session_is_still_once(self, store: MemoryStore) -> None:
        _summary(store, "s1", "Referenced: Anurag Sutton\nReferenced: Anurag Sutton")

        assert "Anurag Sutton" not in _labels(build_memory_graph(store))

    def test_spellings_that_fold_together_count_as_one_name(self, store: MemoryStore) -> None:
        _summary(store, "s1", "Referenced: Acme Corp Ltd")
        _summary(store, "s2", "Referenced: Acme Corp")

        assert "Acme Corp" in _labels(build_memory_graph(store))

    def test_a_memory_entity_needs_no_second_session(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind Bank"))
        _summary(store, "s1", "Referenced: Northwind Bank")

        graph = build_memory_graph(store)

        assert ("session:s1", _node(graph, "Northwind Bank")["id"], "mentioned") in _edges(graph)

    @pytest.mark.parametrize(
        "item",
        [
            "approval ID: 61407eff-2a3b",
            "evt-20260624023314-dentist-appointment-a11e4b",
            "owner@example.com",
            "design.md",
            "/Users/owner/calendar_ics/evt.ics",
            "2026-05-08 16:10",
            "morning_brief",
            "insurance dues",
            "Here Are 4 Major Lawsuits That Have Shaped The Artificial Intelligence Debate",
        ],
    )
    def test_an_item_shaped_like_an_id_address_or_file_is_never_drawn(
        self, store: MemoryStore, item: str
    ) -> None:
        for session in ("s1", "s2", "s3"):
            _summary(store, session, f"Referenced: {item}, Anurag Sutton")

        labels = _labels(build_memory_graph(store))

        assert item not in labels
        assert "Anurag Sutton" in labels  # the same line's real name still is

    @pytest.mark.parametrize(
        "name",
        [
            "St. Louis County",
            "Charles Schwab & Co.",
            "AT&T",
            "U.S. Bank",
            "Facade Studio",
            "Anthem Blue Cross and Blue Shield Communications",
        ],
    )
    def test_ordinary_names_are_not_mistaken_for_shapes(
        self, store: MemoryStore, name: str
    ) -> None:
        for session in ("s1", "s2"):
            _summary(store, session, f"Referenced: {name}")

        assert name in _labels(build_memory_graph(store))

    def test_a_lower_case_memory_entity_is_still_drawn(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("project", "portfolio"))
        _summary(store, "s1", "Referenced: portfolio")

        assert "portfolio" in _labels(build_memory_graph(store))  # from its fact

    def test_a_test_run_is_not_drawn(self, store: MemoryStore) -> None:
        _summary(store, "playground-memory-1", "Referenced: Anurag Sutton")
        _summary(store, "probe3-0", "Referenced: Anurag Sutton")

        graph = build_memory_graph(store)

        assert not {"session:playground-memory-1", "session:probe3-0"} & _ids(graph)
        assert "Anurag Sutton" not in _labels(graph)  # its mentions do not count either

    def test_a_bad_pattern_in_config_is_skipped_not_fatal(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for session in ("s1", "s2"):
            _summary(store, session, "Referenced: Anurag Sutton, design.md")
        config = {**graph_module.graph_config()}
        mentions = config["drawn_mentions"]
        config["drawn_mentions"] = {
            **mentions,
            "never_draw": ["(unclosed", *mentions["never_draw"]],
        }
        monkeypatch.setattr(graph_module, "_CONFIG_CACHE", config)

        labels = _labels(build_memory_graph(store))

        assert "Anurag Sutton" in labels
        assert "design.md" not in labels


class TestItOpensSmall:
    def _many_facts(self, store: MemoryStore, n: int) -> None:
        # Many distinct facts need a property with no one-value limit (the vocabulary is
        # closed, so made-up keys are refused): n cards, each its own entity node.
        for i in range(n):
            store.upsert_user_fact(_fact("credit_card", f"card {i}"))

    def test_the_cap_holds_and_says_what_it_hid(self, store: MemoryStore) -> None:
        self._many_facts(store, 60)

        graph = build_memory_graph(store, node_cap=20)

        assert len(graph["nodes"]) == 21  # 20 + the "+N more" marker
        more = next(n for n in graph["nodes"] if n["kind"] == "more")
        assert more["label"] == "+41 more"
        assert graph["stats"]["total_nodes"] == 61

    def test_you_is_never_the_node_that_gets_cut(self, store: MemoryStore) -> None:
        self._many_facts(store, 60)

        graph = build_memory_graph(store, node_cap=10)

        assert "you" in _ids(graph)

    def test_focus_keeps_one_node_and_its_neighbours(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind Bank"))
        _summary(store, "s1", "Referenced: Northwind Bank")
        store.upsert_user_fact(_fact("blog", "tech4talk.com"))
        bank_id = _node(build_memory_graph(store), "Northwind Bank")["id"]

        graph = build_memory_graph(store, focus=bank_id, depth=1)

        assert bank_id in _ids(graph)
        assert "session:s1" in _ids(graph)  # one hop away
        assert "blog: tech4talk.com" not in _labels(graph)  # two hops, via You

    def test_filtering_by_kind(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind Bank"))
        store.upsert_user_fact(_fact("blog", "tech4talk.com"))

        graph = build_memory_graph(store, kinds={"entity"})

        assert "blog: tech4talk.com" not in _labels(graph)
        assert "Northwind Bank" in _labels(graph)


def test_an_empty_store_still_returns_you(store: MemoryStore) -> None:
    graph = build_memory_graph(store)

    assert _ids(graph) >= {"you"}
    assert graph["stats"]["by_kind"]["you"] == 1


class TestTheConfigDrawsTheGraph:
    """Change the YAML, not the code: the graph follows."""

    @staticmethod
    def _ontology(tmp_path: Path, edit: tuple[str, str, str]) -> Ontology:
        root = tmp_path / "onto"
        shutil.copytree(memory_config_dir(), root)
        name, old, new = edit
        text = (root / name).read_text(encoding="utf-8")
        assert old in text
        (root / name).write_text(text.replace(old, new), encoding="utf-8")
        from iris_harness.memory.ontology import vocabulary_fragments

        return load_or_raise(root, vocabulary_fragments())

    def test_an_edge_label_is_the_ontology_label(self, store: MemoryStore, tmp_path: Path) -> None:
        store.upsert_user_fact(_fact("employer", "Infosys"))
        onto = self._ontology(
            tmp_path, ("ontology.yaml", 'label: "works at"', 'label: "is employed by"')
        )

        graph = build_memory_graph(store, ontology=onto)

        assert "is employed by" in {e["label"] for e in graph["edges"]}

    def test_a_record_kind_is_drawn_only_if_a_mapping_says_so(
        self, store: MemoryStore, tmp_path: Path
    ) -> None:
        _summary(store, "s1", "Referenced: Petra Sutton")
        _summary(store, "s2", "Referenced: Petra Sutton")
        assert "Petra Sutton" in _labels(build_memory_graph(store))
        onto = self._ontology(
            tmp_path,
            ("mappings.yaml", "source_type: summary_section", "source_type: nothing_reads_this"),
        )

        graph = build_memory_graph(store, ontology=onto)

        assert "Petra Sutton" not in _labels(graph)  # the mapping is what drew it
        assert "session:s1" in _ids(graph)  # the session mapping still does

    def test_a_node_kind_is_its_class_group(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind Bank"))
        config = {**graph_module.graph_config()}
        config["groups"] = {**config["groups"], "tv:Institution": "money"}
        monkeypatch.setattr(graph_module, "_CONFIG_CACHE", config)

        assert _node(build_memory_graph(store), "Northwind Bank")["kind"] == "money"

    def test_a_summary_section_is_found_by_its_configured_label(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from iris_harness.memory import compactor

        sections = [{"key": "referenced", "label": "People", "hint": "a, b"}]
        monkeypatch.setattr(compactor, "summary_config", lambda: {"sections": sections})
        _summary(store, "s1", "People: Petra Sutton")
        _summary(store, "s2", "People: Petra Sutton")

        assert "Petra Sutton" in _labels(build_memory_graph(store))

    def test_the_session_threshold_is_config(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _summary(store, "s1", "Referenced: Petra Sutton")
        assert "Petra Sutton" not in _labels(build_memory_graph(store))
        config = {**graph_module.graph_config()}
        config["drawn_mentions"] = {**config["drawn_mentions"], "min_sessions": 1}
        monkeypatch.setattr(graph_module, "_CONFIG_CACHE", config)

        assert "Petra Sutton" in _labels(build_memory_graph(store))

    def test_a_never_draw_shape_is_config(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for session in ("s1", "s2"):
            _summary(store, session, "Referenced: Petra Sutton, Quarterly Plan")
        config = {**graph_module.graph_config()}
        config["drawn_mentions"] = {**config["drawn_mentions"], "never_draw": ["^Quarterly "]}
        monkeypatch.setattr(graph_module, "_CONFIG_CACHE", config)

        labels = _labels(build_memory_graph(store))
        assert "Quarterly Plan" not in labels
        assert "Petra Sutton" in labels


def test_the_first_open_of_a_legacy_database_draws_its_facts(tmp_path: Path) -> None:
    """The map must not read statements before the one-time migration has run."""
    import sqlite3

    from iris_harness.memory.store import MemoryStore as _Store

    db = tmp_path / "legacy.db"
    _Store(db_path=db)._ensure_tables()  # the old shape only; nothing migrated yet
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO user_facts(key, value, confidence, source, first_seen, "
            "last_confirmed, times_confirmed, confirmed) VALUES ('bank', 'Northwind Bank', 0.9, "
            "'x', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 1, 1)"
        )
    graph = build_memory_graph(_Store(db_path=db))
    assert "Northwind Bank" in {n["label"] for n in graph["nodes"]}
