"""memory_search scopes, recall_conversation, and ranked behavior matching.

The shape this replaces: `wiki_search` was called 0 times in four months because a
tool the shortlist drops is a tool the model never sees. So lessons and past sessions
are reachable through `memory_search`, which is one of only two tools kept on every
turn, rather than through new names competing for a slot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.memory import identity
from iris_harness.memory.store import MemoryStore
from iris_harness.runtime.react_tools import builtin_react_tools


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


@pytest.fixture
def behaviors_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "iris-home"
    (home / "behaviors").mkdir(parents=True)
    monkeypatch.setattr(identity.loader, "IRIS_HOME", home)
    monkeypatch.setattr(identity.loader, "BEHAVIORS_DIR", home / "behaviors")
    return home


def _tools(store: MemoryStore) -> dict[str, object]:
    specs = builtin_react_tools(semantic_index=None, wiki=None, repo_root=None, memory_store=store)
    return {s.name: s for s in specs}


def _call(store: MemoryStore, name: str, args: dict[str, object]) -> str:
    return str(_tools(store)[name].call(args))  # type: ignore[attr-defined]


class TestMemorySearchScopes:
    def test_the_tool_documents_its_scopes(self, store: MemoryStore) -> None:
        description = _tools(store)["memory_search"].description  # type: ignore[attr-defined]

        for scope in ("facts", "patterns", "behaviors", "sessions"):
            assert scope in description

    def test_an_unknown_scope_is_rejected(self, store: MemoryStore) -> None:
        assert "scope must be one of" in _call(
            store, "memory_search", {"query": "x", "scope": "everything"}
        )

    def test_sessions_scope_finds_a_past_summary(self, store: MemoryStore) -> None:
        store.save_conversation_summary("s1", "Decisions: paid the Northwind card on the 5th")

        out = _call(store, "memory_search", {"query": "Northwind", "scope": "sessions"})

        assert "past session s1" in out and "Northwind" in out

    def test_behaviors_scope_lists_lessons_by_headline(
        self, store: MemoryStore, behaviors_home: Path
    ) -> None:
        identity.write_behavior(
            "card-dues",
            "When: dues. Do: call finance_lookup first.",
            match_keywords=("dues", "card"),
            description="check finance before the inbox",
        )

        out = _call(store, "memory_search", {"query": "card dues", "scope": "behaviors"})

        assert "lesson — card-dues" in out
        assert "check finance before the inbox" in out

    def test_all_is_the_default_and_labels_each_hit(self, store: MemoryStore) -> None:
        store.save_conversation_summary("s1", "Decisions: paid the Northwind card")

        out = _call(store, "memory_search", {"query": "Northwind"})

        assert "past session" in out

    def test_nothing_found_says_so(self, store: MemoryStore, behaviors_home: Path) -> None:
        # behaviors_home keeps the real ~/.iris recipes out of the result.
        assert "No stored memory matched" in _call(
            store, "memory_search", {"query": "nothing like this exists"}
        )


class TestRecallConversation:
    def test_it_returns_earlier_turns_word_for_word(self, store: MemoryStore) -> None:
        store.save_conversation_turns(
            "s1", [("user", "the deposit was 180000"), ("assistant", "noted")]
        )

        out = _call(store, "recall_conversation", {"query": "deposit"})

        assert "the deposit was 180000" in out

    def test_a_session_id_reads_that_session_back(self, store: MemoryStore) -> None:
        store.save_conversation_turns("s1", [("user", "first"), ("assistant", "second")])

        out = _call(store, "recall_conversation", {"session_id": "s1"})

        assert "first" in out and "second" in out

    def test_it_asks_for_something_to_go_on(self, store: MemoryStore) -> None:
        from iris_harness.runtime.turn_context import set_current_session_id

        set_current_session_id("")  # no live turn to fall back on
        assert "needs a 'query'" in _call(store, "recall_conversation", {})

    def test_nothing_matching_says_so(self, store: MemoryStore) -> None:
        assert "Nothing stored matches" in _call(
            store, "recall_conversation", {"query": "never said this"}
        )


class TestRankedBehaviorMatching:
    def _write(self, name: str, intents: tuple[str, ...], keywords: tuple[str, ...]) -> None:
        identity.write_behavior(
            name, f"recipe for {name}", match_intents=intents, match_keywords=keywords
        )

    def test_a_catch_all_intent_alone_no_longer_matches(self, behaviors_home: Path) -> None:
        """reminders.md claims `general`, which nearly every turn carries."""
        self._write("reminders", ("calendar", "general"), ("remind", "follow up"))

        assert identity.match_behavior("general", "what is the weather today") is None

    def test_a_specific_intent_still_matches_on_its_own(self, behaviors_home: Path) -> None:
        self._write("reminders", ("calendar", "general"), ("remind", "follow up"))

        matched = identity.match_behavior("calendar", "what is on tomorrow")

        assert matched is not None and matched.name == "reminders"

    def test_the_best_match_wins_not_the_first_file(self, behaviors_home: Path) -> None:
        self._write("aaa-generic", ("general",), ("card",))
        self._write("zzz-specific", ("finance",), ("card", "dues"))

        matched = identity.match_behavior("finance", "what are my card dues?")

        assert matched is not None and matched.name == "zzz-specific"

    def test_the_others_are_offered_as_pointers(self, behaviors_home: Path) -> None:
        self._write("aaa-generic", ("general",), ("card",))
        self._write("zzz-specific", ("finance",), ("card", "dues"))

        others = identity.other_matching_behaviors("finance", "what are my card dues?")

        assert [b.name for b in others] == ["aaa-generic"]


class TestRecallReachesCooledConversations:
    """Cooling a session REPLACES its turns with its summary (MemoryStore.cool_session).

    Until 2026-09-20 recall_conversation searched only raw turns, so once a session was
    cooled it answered "Nothing stored matches that." about content the system had
    deliberately kept — the acceptance test for the memory chain had been red on main
    for exactly this reason.
    """

    def test_a_cooled_session_is_still_recalled_from_its_summary(self, store: MemoryStore) -> None:
        store.save_conversation_summary(
            "flat", "Decisions & outcomes: rent is due on the 5th; deposit was 180000."
        )

        out = _call(store, "recall_conversation", {"query": "deposit"})

        assert "180000" in out
        assert "summary" in out.lower(), "and it does not pass the summary off as verbatim"

    def test_raw_turns_are_preferred_while_they_exist(self, store: MemoryStore) -> None:
        store.save_conversation_turns(
            "flat",
            [
                ("user", "the deposit was 180000 and Remy handles maintenance"),
                ("assistant", "Noted."),
            ],
        )
        store.save_conversation_summary("flat", "deposit was 180000")

        out = _call(store, "recall_conversation", {"query": "deposit"})

        assert "Remy handles maintenance" in out, "the verbatim turn, not the summary"
        assert "cooled" not in out

    def test_a_genuine_miss_still_says_nothing_matches(self, store: MemoryStore) -> None:
        store.save_conversation_summary("flat", "deposit was 180000")

        assert _call(store, "recall_conversation", {"query": "zeppelin"}) == (
            "Nothing stored matches that."
        )


class TestATestRunIsNotTheOwnersPast:
    """Map cleanup plan decision 7: a playground or test session (retention.yaml's
    ``ephemeral_session_prefixes``) is not recalled as one of the owner's conversations —
    unless it is the conversation asking."""

    def test_sessions_scope_skips_a_test_run(self, store: MemoryStore) -> None:
        store.save_conversation_summary("probe3-0", "Decisions: paid the Northwind card")
        store.save_conversation_summary("s1", "Decisions: Northwind statement arrived")

        out = _call(store, "memory_search", {"query": "Northwind", "scope": "sessions"})

        assert "past session s1" in out
        assert "probe3-0" not in out

    def test_recall_skips_a_test_runs_turns_and_summary(self, store: MemoryStore) -> None:
        store.save_conversation_turns("playground-x", [("user", "the deposit was 99")])
        store.save_conversation_summary("smoke-1", "deposit was 42")

        assert _call(store, "recall_conversation", {"query": "deposit"}) == (
            "Nothing stored matches that."
        )

    def test_a_test_run_still_recalls_itself(self, store: MemoryStore) -> None:
        from iris_harness.runtime.turn_context import set_current_session_id

        store.save_conversation_turns("playground-x", [("user", "the deposit was 99")])
        set_current_session_id("playground-x")
        try:
            out = _call(store, "recall_conversation", {"query": "deposit"})
        finally:
            set_current_session_id("")

        assert "deposit was 99" in out

    def test_filtering_does_not_starve_the_limit(self, store: MemoryStore) -> None:
        for i in range(3):
            store.save_conversation_summary(f"probe{i}", "Northwind noise")
        store.save_conversation_summary("s1", "Decisions: Northwind statement arrived")

        out = _call(store, "memory_search", {"query": "Northwind", "scope": "sessions", "n": 1})

        assert "past session s1" in out
