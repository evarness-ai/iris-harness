"""The memory graph as the agent sees it (memris PR 6; ADR-0115 decision 9).

Linking: names in the message bring their confirmed one-hop statements into the prompt,
within a token budget, with provenance. The tool: an entity, a relation, a direction,
hops and an as-of date — no query language.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_harness.memory.fact_statements import FactStatements
from iris_harness.memory.graph_context import (
    LINKED_HEADER,
    GraphContext,
    graph_tool_max_hops,
    linking_max_tokens,
)
from iris_harness.memory.store import UserFact
from memris.model import OWNER_ID

# Fact keys such as `bank` and `credit_card` come from the test vocabulary fragment
# (tests/fixtures/test_vocabulary, installed by the `test_vocabulary` fixture), not
# from whichever domain plugin the tree happens to carry.
pytestmark = pytest.mark.usefixtures("test_vocabulary")

NOW = datetime(2026, 9, 16, 9, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def facts(tmp_path: Path, clock: Clock) -> FactStatements:
    return FactStatements(tmp_path / "memory.db", clock=clock)


def _confirm(facts: FactStatements, key: str, value: str) -> None:
    facts.put(UserFact(key, value, 0.9, "test", NOW, NOW, 1, True))


def _about(facts: FactStatements, who: str, key: str, value: str) -> None:
    pid = facts.propose(
        key, value, confidence=0.9, source="test", subject=who, subject_class="Person"
    )
    assert pid is not None
    facts.resolve(pid, "approved")


@pytest.fixture
def petra(facts: FactStatements, clock: Clock) -> GraphContext:
    """The owner is married to Petra, who works at Infosys; the owner banks with Northwind.

    A minute apart, as separate turns would be: statements recorded in the same
    millisecond have no defined order.
    """
    _confirm(facts, "spouse", "Petra")
    clock.now += timedelta(minutes=1)
    _about(facts, "Petra", "employer", "Infosys")
    clock.now += timedelta(minutes=1)
    _confirm(facts, "bank", "Northwind Bank")
    return GraphContext(facts.graph)


class TestLinking:
    def test_a_name_brings_its_one_hop_statements_with_provenance(
        self, petra: GraphContext
    ) -> None:
        linked = petra.linked("is Petra free on Friday?", max_tokens=400)

        assert linked.text == (
            f"{LINKED_HEADER}\n"
            "- the user: spouse of Petra (told 2026-09-16, confirmed)\n"
            "- Petra: works at Infosys (told 2026-09-16, confirmed)"
        )
        assert linked.entities == ("Petra",)
        assert linked.pointer is None

    def test_names_are_matched_whole_and_in_any_case(self, petra: GraphContext) -> None:
        assert petra.named_in("PETRA said hi") != []
        assert petra.named_in("Priyanka said hi") == []

    def test_a_merged_name_names_it_too(self, petra: GraphContext) -> None:
        graph = petra.graph
        [bank] = [e for e in graph.store.find_entities() if e.label == "Northwind Bank"]
        old = graph.add_entity(bank.class_, "Northwind")
        graph.merge(bank.id, old.id, decided_by="owner")

        assert petra.linked("move money out of northwind", max_tokens=400).entities == (
            "Northwind Bank",
        )

    def test_a_configured_alias_names_it_too(
        self, petra: GraphContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = {"aliases": {"Hubby's office": "Infosys"}, "suffixes": []}
        monkeypatch.setattr("iris_harness.memory.graph.graph_config", lambda: config)

        assert petra.linked("near hubby's office", max_tokens=400).entities == ("Infosys",)

    def test_nothing_named_nothing_added(self, petra: GraphContext) -> None:
        assert petra.linked("what's the weather?", max_tokens=400).text == ""

    def test_the_owner_is_not_linked_by_their_placeholder_name(self, petra: GraphContext) -> None:
        assert petra.named_in("the owner of this laptop") == []

    def test_bookkeeping_entities_are_not_linked(self, petra: GraphContext) -> None:
        onto = petra.ontology
        system = next(n for n, c in onto.classes.items() if c.system and not c.abstract)
        petra.graph.add_entity(system, "Standup")

        assert petra.named_in("the Standup went long") == []

    def test_a_proposal_is_not_something_memory_knows(
        self, facts: FactStatements, petra: GraphContext
    ) -> None:
        facts.propose(
            "city",
            "Pune",
            confidence=0.9,
            source="test",
            subject="Petra",
            subject_class="Person",
        )

        assert "Pune" not in petra.linked("Petra", max_tokens=400).text

    def test_the_budget_keeps_what_fits_and_points_at_the_rest(self, petra: GraphContext) -> None:
        tight = petra.linked("Petra and Infosys", max_tokens=30)

        assert tight.shown == 1
        assert tight.left_out == 1
        assert tight.pointer is not None and "memory_graph" in tight.pointer
        assert "Petra" in tight.pointer

    def test_zero_budget_is_off(self, petra: GraphContext) -> None:
        assert petra.linked("Petra", max_tokens=0).text == ""

    def test_the_budget_is_configured(self) -> None:
        assert linking_max_tokens() == 400  # config/memory/learning.yaml
        assert graph_tool_max_hops() == 2


class TestTool:
    def test_an_entity_and_its_relations(self, petra: GraphContext) -> None:
        out = petra.query("Petra")

        assert "Petra: works at Infosys (told 2026-09-16, confirmed)" in out
        assert "the user: spouse of Petra" in out

    def test_the_user_is_an_entity_too(self, petra: GraphContext) -> None:
        assert "the user: banks with Northwind Bank" in petra.query("user", relation="bank")

    def test_relation_by_term_or_label(self, petra: GraphContext) -> None:
        by_term = petra.query("Petra", relation="works_at", direction="out")
        by_label = petra.query("Petra", relation="works at", direction="out")

        assert by_term == by_label == "- Petra: works at Infosys (told 2026-09-16, confirmed)"

    def test_two_hops_are_marked(self, petra: GraphContext) -> None:
        out = petra.query("Infosys", hops=2)

        assert "[hop 2]" in out and "spouse of Petra" in out

    def test_hops_are_capped_by_config(self, petra: GraphContext) -> None:
        assert "[hop 3]" not in petra.query("Infosys", hops=9)

    def test_as_of_answers_for_an_earlier_date(self, facts: FactStatements, clock: Clock) -> None:
        _confirm(facts, "city", "Chennai")
        clock.now = NOW + timedelta(days=30)
        _confirm(facts, "city", "London")
        ctx = GraphContext(facts.graph)

        assert "London" in ctx.query("user", relation="city")
        earlier = ctx.query("user", relation="city", as_of="2026-09-20")
        assert "Chennai" in earlier and "until 2026-10-16" in earlier

    def test_unknowns_say_so(self, petra: GraphContext) -> None:
        assert petra.query("Nobody") == "Nothing remembered about 'Nobody'."
        assert petra.query("Petra", relation="mentors").startswith("Error:")
        assert petra.query("Petra", direction="up").startswith("Error:")
        assert petra.query("Petra", as_of="last week").startswith("Error:")
        assert petra.query("").startswith("Error:")

    def test_the_description_is_generated_from_the_ontology(self, petra: GraphContext) -> None:
        described = petra.tool_description()
        ontology = petra.ontology

        for name in ("works_at", "spouse", "tv:banks_with"):
            assert ontology.qualify(name) in {*ontology.relations, *ontology.attributes}
            assert name in described
        assert "1-2" in described  # graph_tool.max_hops
        assert "employs" not in described  # synthesized inverse terms are not offered
        assert "participated_in" not in described  # bookkeeping (system classes), not facts


def test_the_owner_is_rendered_as_the_user(petra: GraphContext) -> None:
    assert petra.label(OWNER_ID) == "the user"
