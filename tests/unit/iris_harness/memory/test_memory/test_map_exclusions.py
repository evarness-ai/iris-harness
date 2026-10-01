"""A plugin can tell the memory Map which names never to draw (ADR-0119, decision 9).

The core knows no email; the plugin that classified the mail knows which senders are
shops. The registry is keyed like ``runtime.api_routes``; the Map asks it once per draw
and drops only summary mentions — never an entity a confirmed fact points at.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from iris_harness.memory import graph as graph_module
from iris_harness.memory.graph import build_memory_graph, canonical_entity
from iris_harness.memory.map_exclusions import (
    clear_map_exclusions,
    excluded_names,
    register_map_exclusions,
)
from iris_harness.memory.store import MemoryStore, UserFact

# Fact keys such as `bank` and `credit_card` come from the test vocabulary fragment
# (tests/fixtures/test_vocabulary, installed by the `test_vocabulary` fixture), not
# from whichever domain plugin the tree happens to carry.
pytestmark = pytest.mark.usefixtures("test_vocabulary")


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    graph_module.reset_config_cache()
    clear_map_exclusions()
    yield
    clear_map_exclusions()


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


def _summary(store: MemoryStore, session: str, text: str) -> None:
    store.save_conversation_turns(session, [("user", "q"), ("assistant", "a")])
    store.save_conversation_summary(session, text)


def _labels(graph: dict[str, Any]) -> set[str]:
    return {n["label"] for n in graph["nodes"]}


def _identity(name: str) -> str:
    return name


class TestTheRegistry:
    def test_names_are_folded_and_lowercased(self) -> None:
        register_map_exclusions("p", lambda: ["  Shop  ", "SHOP", ""])

        assert excluded_names(str.strip) == frozenset({"shop"})

    def test_a_key_is_replaced_not_added(self) -> None:
        register_map_exclusions("p", lambda: ["one"])
        register_map_exclusions("p", lambda: ["two"])

        assert excluded_names(_identity) == frozenset({"two"})

    def test_every_key_contributes(self) -> None:
        register_map_exclusions("a", lambda: ["one"])
        register_map_exclusions("b", lambda: ["two"])

        assert excluded_names(_identity) == frozenset({"one", "two"})

    def test_a_raising_provider_contributes_nothing(self) -> None:
        def broken() -> list[str]:
            raise RuntimeError("plugin bug")

        register_map_exclusions("broken", broken)
        register_map_exclusions("ok", lambda: ["one"])

        assert excluded_names(_identity) == frozenset({"one"})

    def test_clear_drops_every_provider(self) -> None:
        register_map_exclusions("p", lambda: ["one"])
        clear_map_exclusions()

        assert excluded_names(_identity) == frozenset()


class TestTheMap:
    def test_an_excluded_mention_is_not_drawn(self, store: MemoryStore) -> None:
        _summary(store, "s1", "Referenced: Walgreens, Petra Sutton")
        _summary(
            store, "s2", "Referenced: Walgreens, Petra Sutton"
        )  # one-off mentions are not drawn
        register_map_exclusions("mail", lambda: ["Walgreens"])

        labels = _labels(build_memory_graph(store))

        assert "Walgreens" not in labels
        assert "Petra Sutton" in labels

    def test_the_match_uses_the_maps_own_fold(self, store: MemoryStore) -> None:
        # "Ltd" is a configured suffix: the mention and the excluded name fold alike,
        # whichever side carries it.
        assert canonical_entity("Acme Ltd") == "Acme"
        _summary(store, "s1", "Referenced: ACME")
        _summary(store, "s2", "Referenced: ACME")  # one-off mentions are not drawn
        register_map_exclusions("mail", lambda: ["Acme Ltd"])

        assert "ACME" not in _labels(build_memory_graph(store))

    def test_an_entity_a_confirmed_fact_names_is_still_drawn(self, store: MemoryStore) -> None:
        now = datetime.now(UTC)
        store.upsert_user_fact(
            UserFact(
                key="bank",
                value="Northwind Bank",
                confidence=0.9,
                source="test",
                first_seen=now,
                last_confirmed=now,
                confirmed=True,
            )
        )
        _summary(store, "s1", "Referenced: Northwind Bank")
        register_map_exclusions("mail", lambda: ["Northwind Bank"])

        graph = build_memory_graph(store)

        assert "Northwind Bank" in _labels(graph)
        # ...but the summary's mention edge is gone with the exclusion.
        bank = next(n["id"] for n in graph["nodes"] if n["label"] == "Northwind Bank")
        assert not [e for e in graph["edges"] if e["target"] == bank and e["source"] != "you"]

    def test_the_providers_are_asked_once_per_draw(self, store: MemoryStore) -> None:
        _summary(store, "s1", "Referenced: A1, B2, C3")
        _summary(store, "s2", "Referenced: D4, E5")
        calls: list[int] = []

        def provider() -> list[str]:
            calls.append(1)
            return []

        register_map_exclusions("mail", provider)
        build_memory_graph(store)

        assert len(calls) == 1

    def test_no_provider_draws_every_mention(self, store: MemoryStore) -> None:
        _summary(store, "s1", "Referenced: Walgreens, ACME")
        _summary(store, "s2", "Referenced: Walgreens, ACME")  # one-off mentions are not drawn

        labels = _labels(build_memory_graph(store))

        assert {"Walgreens", "ACME"} <= labels
