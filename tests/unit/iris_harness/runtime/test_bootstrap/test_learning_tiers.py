"""The capture side of the tiers: what is remembered, what is asked, what is said.

A fact IRIS keeps quietly is one the user cannot correct, so anything confirmed during a
turn is announced in that turn's reply with a three-word undo.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from iris_harness.memory.store import MemoryStore
from iris_harness.runtime.turn_capture import TurnCapture

# Fact keys such as `bank` and `credit_card` come from the test vocabulary fragment
# (tests/fixtures/test_vocabulary, installed by the `test_vocabulary` fixture), not
# from whichever domain plugin the tree happens to carry.
pytestmark = pytest.mark.usefixtures("test_vocabulary")

SID = "tiers"


@pytest.fixture
def capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.ensure_schema()
    host = SimpleNamespace(
        config_dir=Path("config"),
        memory_store=store,
        semantic_index=None,
        signal_collector=SimpleNamespace(record_metric=lambda **kw: None),
        tier_router=SimpleNamespace(),
    )
    cap = TurnCapture(host)  # type: ignore[arg-type]
    # The LLM extractor is stubbed per test; the deterministic paths still run.
    monkeypatch.setattr(cap, "_extract_facts_via_llm", lambda msg: [])
    return cap


def _extracts(capture: Any, monkeypatch: pytest.MonkeyPatch, facts: list[tuple]) -> None:
    monkeypatch.setattr(capture, "_extract_facts_via_llm", lambda msg: facts)


class TestRememberedOutright:
    def test_a_plain_statement_is_confirmed_and_announced(
        self, capture: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _extracts(capture, monkeypatch, [("city", "Springfield", 0.9, "test")])

        capture.extract_and_store_facts("I live in Springfield", SID)

        stored = capture._host.memory_store.fetch_all_user_facts(confirmed_only=True)
        assert [(f.key, f.value) for f in stored] == [("city", "Springfield")]
        assert capture.take_notices() == ["city: Springfield"]

    def test_notices_are_taken_once(self, capture: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        _extracts(capture, monkeypatch, [("city", "Springfield", 0.9, "test")])
        capture.extract_and_store_facts("I live in Springfield", SID)

        assert capture.take_notices()
        assert capture.take_notices() == []


class TestAskedOnce:
    def test_an_inferred_fact_becomes_the_session_s_one_question(
        self, capture: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _extracts(capture, monkeypatch, [("bank", "Northwind", 0.8, "test")])

        capture.extract_and_store_facts("i pay by UPI from Northwind", SID)

        assert capture.take_notices() == []  # nothing believed yet
        assert capture.take_question(SID) == "Should I remember that your bank is Northwind?"
        assert capture._host.memory_store.fetch_all_user_facts(confirmed_only=True) == []

    def test_only_one_question_per_session(
        self, capture: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _extracts(capture, monkeypatch, [("bank", "Northwind", 0.8, "test")])
        capture.extract_and_store_facts("i pay by UPI from Northwind", SID)
        first = capture.pending_question(SID)

        _extracts(capture, monkeypatch, [("broker", "Tailspin", 0.8, "test")])
        capture.extract_and_store_facts("i invest through Tailspin", SID)

        assert capture.pending_question(SID) == first  # the second waits in the queue

    def test_yes_confirms_it(self, capture: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        _extracts(capture, monkeypatch, [("bank", "Northwind", 0.8, "test")])
        capture.extract_and_store_facts("i pay by UPI from Northwind", SID)

        reply = capture.resolve_question(SID, "yes")

        assert reply is not None and "remembered" in reply
        stored = capture._host.memory_store.fetch_all_user_facts(confirmed_only=True)
        assert [(f.key, f.value) for f in stored] == [("bank", "Northwind")]

    def test_no_rejects_it(self, capture: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        _extracts(capture, monkeypatch, [("bank", "Northwind", 0.8, "test")])
        capture.extract_and_store_facts("i pay by UPI from Northwind", SID)

        reply = capture.resolve_question(SID, "no")

        assert reply is not None and "won't remember" in reply
        assert capture._host.memory_store.fetch_all_user_facts(confirmed_only=True) == []

    def test_an_unrelated_message_leaves_the_question_open(
        self, capture: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _extracts(capture, monkeypatch, [("bank", "Northwind", 0.8, "test")])
        capture.extract_and_store_facts("i pay by UPI from Northwind", SID)

        assert capture.resolve_question(SID, "what's the weather tomorrow?") is None
        assert capture.pending_question(SID) is not None

    def test_nothing_to_answer_is_not_an_answer(self, capture: Any) -> None:
        assert capture.resolve_question(SID, "yes") is None


class TestRepetitionConfirms:
    def test_the_same_thing_three_times_stops_waiting(
        self, capture: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """People repeat what is true about them."""
        _extracts(capture, monkeypatch, [("broker", "Tailspin", 0.5, "test")])

        for _ in range(3):
            capture.extract_and_store_facts("i sold some shares on Tailspin today", "s-new")

        stored = capture._host.memory_store.fetch_all_user_facts(confirmed_only=True)
        assert [(f.key, f.value) for f in stored] == [("broker", "Tailspin")]


class TestLookAlikeAskedOnce:
    """ADR-0115 decision 4 (memris PR 4b): "is X the same as Y?" asked in the
    conversation that turned up the pair, sharing the one-question budget with facts."""

    def _say(self, capture: Any, monkeypatch: pytest.MonkeyPatch, sid: str, bank: str) -> None:
        from iris_harness.foundation.observability.session_log import session_scope

        _extracts(capture, monkeypatch, [("bank", bank, 0.9, "test")])
        with session_scope(sid):  # the conversation is the resolution's evidence
            capture.extract_and_store_facts(f"my bank is {bank}", sid)

    def _lookalike_turn(self, capture: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        self._say(capture, monkeypatch, "s1", "Fabrikam Meridian")
        capture.take_notices()
        self._say(capture, monkeypatch, "s2", "Fabrikam Meridien")

    def test_a_look_alike_becomes_the_conversation_s_question(
        self, capture: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._lookalike_turn(capture, monkeypatch)
        assert capture.take_question("s2") == "Is Fabrikam Meridien the same as Fabrikam Meridian?"
        assert capture.take_question("s1") is None  # not the conversation that saw it

    def test_yes_merges_them(self, capture: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        self._lookalike_turn(capture, monkeypatch)
        reply = capture.resolve_question("s2", "yes")
        assert reply is not None and "the same" in reply
        graph = capture._host.memory_store.memory_graph()
        [done] = graph.decisions()
        assert (done.decision, done.decided_by) == ("same", "owner")
        assert graph.get_entity(done.a).label == "Fabrikam Meridian"  # the older name kept
        assert capture.take_question("s2") is None  # answered; nothing left to ask

    def test_no_keeps_them_apart(self, capture: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        self._lookalike_turn(capture, monkeypatch)
        reply = capture.resolve_question("s2", "no")
        assert reply is not None and "apart" in reply
        [done] = capture._host.memory_store.memory_graph().decisions()
        assert (done.decision, done.decided_by) == ("distinct", "owner")

    def test_asked_once_even_when_unanswered(
        self, capture: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._lookalike_turn(capture, monkeypatch)
        assert capture.resolve_question("s2", "what's the weather?") is None  # still open
        self._say(capture, monkeypatch, "s3", "Fabrikam Meridien")  # the pair again, elsewhere
        assert capture.take_question("s3") is None

    def test_a_fact_question_spends_the_budget_first(
        self, capture: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from iris_harness.foundation.observability.session_log import session_scope

        self._say(capture, monkeypatch, "s1", "Fabrikam Meridian")
        # inferred (tier B) and a look-alike in the same turn: the fact is asked
        _extracts(capture, monkeypatch, [("bank", "Fabrikam Meridien", 0.8, "test")])
        with session_scope("s2"):
            capture.extract_and_store_facts("i pay from Fabrikam Meridien", "s2")
        assert capture.take_question("s2") == (
            "Should I remember that your bank is Fabrikam Meridien?"
        )
        capture.resolve_question("s2", "yes")
        assert capture.take_question("s2") is None  # one question a conversation
        # never asked, so a later conversation that sees the pair may ask
        self._say(capture, monkeypatch, "s3", "Fabrikam Meridien")
        assert capture.take_question("s3") == "Is Fabrikam Meridien the same as Fabrikam Meridian?"

    def test_decided_elsewhere_meanwhile_drops_the_question(
        self, capture: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._lookalike_turn(capture, monkeypatch)
        graph = capture._host.memory_store.memory_graph()
        [candidate] = graph.open_candidates()
        graph.reject_candidate(candidate.id, decided_by="owner")  # the Memory page, say
        assert capture.take_question("s2") is None
        assert capture.resolve_question("s2", "yes") is None

    def test_a_pair_from_another_conversation_is_not_asked_here(
        self, capture: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from iris_harness.foundation.observability.session_log import session_scope

        self._say(capture, monkeypatch, "s1", "Fabrikam Meridian")
        _extracts(capture, monkeypatch, [("bank", "Fabrikam Meridien", 0.8, "test")])
        with session_scope("s2"):  # the pair appears, but a fact question spends s2
            capture.extract_and_store_facts("i pay from Fabrikam Meridien", "s2")
        self._say(capture, monkeypatch, "s5", "Springfield")  # s5 never saw the pair
        assert capture.take_question("s5") is None


class TestOneHop:
    """ "My wife Petra works at Infosys" — a fact about someone one hop away (PR 3b)."""

    WIFE = "my wife Petra works at Infosys"

    def _about_priya(self, capture: Any) -> list[Any]:
        return [p for p in capture._host.memory_store.fetch_fact_proposals() if p.subject]

    def test_the_relation_and_her_fact_are_both_captured(
        self, capture: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _extracts(
            capture,
            monkeypatch,
            [
                ("employer", "Infosys", 0.9, "test", "Petra"),  # listed first on purpose
                ("spouse", "Petra", 0.9, "test"),
            ],
        )

        capture.extract_and_store_facts(self.WIFE, SID)

        [hers] = self._about_priya(capture)
        assert (hers.subject, hers.key, hers.value) == ("Petra", "employer", "Infosys")
        store = capture._host.memory_store
        assert store.fetch_user_fact("employer") is None  # never the owner's employer
        assert capture.take_notices() == []  # a fact about her is never confirmed on the spot

    def test_someone_the_owner_is_not_linked_to_is_dropped(
        self, capture: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _extracts(capture, monkeypatch, [("employer", "Infosys", 0.9, "test", "Sundar")])

        capture.extract_and_store_facts("my friend said Sundar works at Infosys", SID)

        assert capture._host.memory_store.fetch_fact_proposals() == []

    def test_a_subject_not_in_the_message_is_dropped(
        self, capture: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _extracts(capture, monkeypatch, [("spouse", "Petra", 0.9, "test")])
        capture.extract_and_store_facts("my wife is Petra", SID)  # in scope from here on
        _extracts(capture, monkeypatch, [("employer", "Infosys", 0.9, "test", "Petra")])

        capture.extract_and_store_facts("my wife works at Infosys", "later")  # no "Petra"

        assert self._about_priya(capture) == []

    def test_a_plain_statement_about_her_is_still_only_a_proposal(
        self, capture: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _extracts(
            capture,
            monkeypatch,
            [("spouse", "Petra", 0.9, "test"), ("employer", "Infosys", 0.9, "test", "Petra")],
        )

        # "remember that" confirms the owner's own facts on the spot — not hers
        capture.extract_and_store_facts("remember that my wife Petra works at Infosys", SID)

        [hers] = self._about_priya(capture)
        assert hers.status == "pending"
        assert capture.take_notices() == ["spouse: Petra"]

    def test_a_confirmed_relation_carries_a_later_message(
        self, capture: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _extracts(capture, monkeypatch, [("spouse", "Petra", 0.9, "test")])
        capture.extract_and_store_facts("my wife is Petra", SID)  # tier A: confirmed

        _extracts(capture, monkeypatch, [("city", "Pune", 0.9, "test", "Petra")])
        capture.extract_and_store_facts("my wife Petra moved to Pune", "later")

        [hers] = self._about_priya(capture)
        assert (hers.subject, hers.key, hers.value) == ("Petra", "city", "Pune")

    def test_the_question_names_her_and_yes_leaves_the_owner_alone(
        self, capture: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _extracts(capture, monkeypatch, [("spouse", "Petra", 0.9, "test")])
        capture.extract_and_store_facts("my wife is Petra", SID)
        _extracts(capture, monkeypatch, [("employer", "Infosys", 0.9, "test", "Petra")])

        capture.extract_and_store_facts(self.WIFE, "next")

        assert (
            capture.take_question("next") == "Should I remember that Petra's employer is Infosys?"
        )
        reply = capture.resolve_question("next", "yes")
        assert reply == "Got it — remembered that Petra's employer is Infosys."
        assert capture._host.memory_store.fetch_user_fact("employer") is None
        assert self._about_priya(capture) == []  # approved, off the queue

    def test_no_to_her_fact_says_so(self, capture: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        _extracts(capture, monkeypatch, [("spouse", "Petra", 0.9, "test")])
        capture.extract_and_store_facts("my wife is Petra", SID)
        _extracts(capture, monkeypatch, [("employer", "Infosys", 0.9, "test", "Petra")])
        capture.extract_and_store_facts(self.WIFE, "next")

        assert capture.resolve_question("next", "no") == "Okay — I won't remember Petra's employer."

    def test_hop_limit_zero_keeps_capture_to_the_owner(
        self, capture: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("iris_harness.runtime.turn_capture.subject_max_hops", lambda: 0)
        _extracts(
            capture,
            monkeypatch,
            [("spouse", "Petra", 0.9, "test"), ("employer", "Infosys", 0.9, "test", "Petra")],
        )

        capture.extract_and_store_facts(self.WIFE, SID)

        assert self._about_priya(capture) == []
