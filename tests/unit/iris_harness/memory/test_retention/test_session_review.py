"""The one-time "Looks like a test session" review (ADR-0119, Map cleanup plan decision 7).

Old test runs were named by hand (cal6-verify, cascade, clr-probe …) and match none of
retention.yaml's ephemeral prefixes, so they sat in the Map, recall and the chat list as
the owner's conversations. Rather than guess their names, retention.yaml says what a
REAL conversation's id looks like; everything else is offered for removal — never
removed without the owner, and always through the restorable Removed list.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.memory import retention
from iris_harness.memory.removal import MemoryRemoval, Target
from iris_harness.memory.retention import flag_test_sessions, real_session_shapes
from iris_harness.memory.store import MemoryStore
from iris_harness.server.iris_api.main import create_app


@pytest.fixture(autouse=True)
def _fresh_config() -> Iterator[None]:
    retention.reset_config_cache()
    yield
    retention.reset_config_cache()


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


def _session(store: MemoryStore, session_id: str, turns: int = 6, summary: str = "") -> None:
    store.save_conversation_turns(
        session_id, [("user" if i % 2 == 0 else "assistant", f"t{i}") for i in range(turns)]
    )
    if summary:
        store.save_conversation_summary(session_id, summary)


def _flagged(store: MemoryStore) -> set[str]:
    return {f.session_id for f in flag_test_sessions(store)}


def _review(monkeypatch: pytest.MonkeyPatch, review: dict[str, Any]) -> None:
    config = {**retention.retention_config(), "test_session_review": review}
    monkeypatch.setattr(retention, "_CONFIG_CACHE", config)


class TestTheShippedRules:
    """config/memory/retention.yaml as shipped, on ids from the owner's real store."""

    @pytest.mark.parametrize(
        "session_id", ["dc43912781ed", "web-0a7b2754", "telegram:123456789", "default"]
    )
    def test_a_real_conversation_is_never_flagged(
        self, store: MemoryStore, session_id: str
    ) -> None:
        _session(store, session_id)

        assert _flagged(store) == set()

    @pytest.mark.parametrize(
        "session_id",
        ["cal6-verify", "cascade", "clr-probe", "clean1", "web-g4", "web:validate", "q"],
    )
    def test_a_hand_named_run_is_flagged(self, store: MemoryStore, session_id: str) -> None:
        _session(store, session_id)

        assert _flagged(store) == {session_id}

    def test_a_near_miss_of_a_real_shape_is_flagged(self, store: MemoryStore) -> None:
        for session_id in ("dc43912781e", "dc43912781edf", "web-0a7b275", "telegram:abc"):
            _session(store, session_id)

        assert _flagged(store) == {"dc43912781e", "dc43912781edf", "web-0a7b275", "telegram:abc"}

    def test_the_shapes_are_described_in_words(self) -> None:
        assert "a Telegram chat" in real_session_shapes()


class TestWhatIsNotOfferedAgain:
    def test_a_session_already_removed(self, store: MemoryStore) -> None:
        _session(store, "cascade")
        MemoryRemoval(store).remove([Target("session", "cascade")])

        assert _flagged(store) == set()

    def test_a_run_the_ephemeral_prefixes_already_keep_out(self, store: MemoryStore) -> None:
        _session(store, "playground-memory-1")
        _session(store, "probe3-0")

        assert _flagged(store) == set()

    def test_restoring_offers_it_again(self, store: MemoryStore) -> None:
        _session(store, "cascade")
        removal = MemoryRemoval(store)
        [item] = removal.remove([Target("session", "cascade")])
        removal.restore(item["id"])

        assert _flagged(store) == {"cascade"}


class TestEachFlagSaysWhy:
    def test_the_id_reason_and_a_short_session(self, store: MemoryStore) -> None:
        _session(store, "cascade", turns=2, summary="Goal: budget review\nDecisions: none")

        [flag] = flag_test_sessions(store)

        assert flag.reasons == ("its id is not shaped like a real conversation's", "only 2 turns")
        assert flag.summary_goal == "Goal: budget review"
        assert flag.turns == 2

    def test_a_longer_session_has_only_the_id_reason(self, store: MemoryStore) -> None:
        _session(store, "cascade", turns=5)

        [flag] = flag_test_sessions(store)

        assert flag.reasons == ("its id is not shaped like a real conversation's",)
        assert flag.summary_goal == ""

    def test_one_turn_is_singular(self, store: MemoryStore) -> None:
        _session(store, "cascade", turns=1)

        assert flag_test_sessions(store)[0].reasons[-1] == "only 1 turn"

    def test_newest_first(self, store: MemoryStore) -> None:
        _session(store, "older")
        _session(store, "newer")

        assert [f.session_id for f in flag_test_sessions(store)][:1] == ["newer"]


class TestTheRulesAreConfig:
    def test_a_pattern_added_in_yaml_stops_the_flag(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _session(store, "cascade")
        _review(monkeypatch, {"real_session_patterns": [{"pattern": "^casc", "what": "x"}]})

        assert _flagged(store) == set()

    def test_a_bare_string_pattern_works(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _session(store, "cascade")
        _session(store, "keep-me")
        _review(monkeypatch, {"real_session_patterns": ["^keep-"]})

        assert _flagged(store) == {"cascade"}
        assert real_session_shapes() == []

    def test_no_patterns_flags_nothing(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _session(store, "cascade")
        _review(monkeypatch, {"real_session_patterns": []})

        assert _flagged(store) == set()

    def test_a_bad_pattern_is_skipped_not_fatal(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _session(store, "cascade")
        _session(store, "keep-me")
        _review(monkeypatch, {"real_session_patterns": ["(unclosed", "^keep-"]})

        assert _flagged(store) == {"cascade"}

    def test_the_short_session_threshold(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _session(store, "cascade", turns=6)
        _review(monkeypatch, {"real_session_patterns": ["^keep-"], "few_turns": 6})
        assert flag_test_sessions(store)[0].reasons[-1] == "only 6 turns"

        _review(monkeypatch, {"real_session_patterns": ["^keep-"], "few_turns": 5})
        assert len(flag_test_sessions(store)[0].reasons) == 1

    def test_a_non_number_threshold_is_ignored(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _session(store, "cascade", turns=2)
        _review(monkeypatch, {"real_session_patterns": ["^keep-"], "few_turns": "lots"})

        assert len(flag_test_sessions(store)[0].reasons) == 1


def test_the_endpoint_lists_them_and_removal_is_the_removed_api(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
    _session(store, "cascade", turns=2, summary="Goal: budget review")
    _session(store, "dc43912781ed")
    runtime = SimpleNamespace(memory_store=store, semantic_index=None)
    with TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    ) as client:
        body = client.get("/memory/review/test-sessions").json()
        [row] = body["sessions"]
        assert row["session_id"] == "cascade"
        assert row["summary_goal"] == "Goal: budget review"
        assert row["turns"] == 2 and row["reasons"][-1] == "only 2 turns"
        assert set(row) == {"session_id", "summary_goal", "turns", "last_activity", "reasons"}
        assert "a Telegram chat" in body["real_shapes"]

        target = {"kind": "session", "id": "cascade"}
        client.post("/memory/removed", json={"targets": [target]}).raise_for_status()
        assert client.get("/memory/review/test-sessions").json()["sessions"] == []
        [item] = client.get("/memory/removed").json()["items"]
        assert item["label"] == "cascade"


def test_the_endpoint_is_a_read_even_with_writes_off(store: MemoryStore) -> None:
    _session(store, "cascade")
    runtime = SimpleNamespace(memory_store=store, semantic_index=None)
    with TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    ) as client:
        response = client.get("/memory/review/test-sessions")

    assert response.status_code == 200
    assert [r["session_id"] for r in response.json()["sessions"]] == ["cascade"]


class TestTheCli:
    def _run(self, store: MemoryStore, *args: str, stdin: str = "") -> Any:
        from typer.testing import CliRunner

        from iris_harness.cli.memory import memory_app

        return CliRunner(env={"COLUMNS": "200"}).invoke(
            memory_app, ["test-sessions", *args, "--db-path", str(store.db_path)], input=stdin
        )

    def test_it_lists_without_removing(self, store: MemoryStore) -> None:
        _session(store, "cascade", turns=2)
        _session(store, "dc43912781ed")

        result = self._run(store)

        assert result.exit_code == 0, result.output
        assert "cascade" in result.output and "only 2 turns" in result.output
        assert "dc43912781ed" not in result.output
        assert "--remove" in result.output
        assert _flagged(store) == {"cascade"}

    def test_remove_asks_first_and_no_keeps_them(self, store: MemoryStore) -> None:
        _session(store, "cascade")

        result = self._run(store, "--remove", stdin="n\n")

        assert "cancelled" in result.output
        assert _flagged(store) == {"cascade"}

    def test_remove_yes_sends_them_to_the_removed_list(self, store: MemoryStore) -> None:
        _session(store, "cascade")
        _session(store, "clr-probe")

        result = self._run(store, "--remove", "--yes")

        assert result.exit_code == 0, result.output
        assert "removed 2 session(s)" in result.output
        assert _flagged(store) == set()
        assert {i["label"] for i in MemoryRemoval(store).items()} == {"cascade", "clr-probe"}

    def test_nothing_to_review_says_so(self, store: MemoryStore) -> None:
        _session(store, "dc43912781ed")

        assert "no sessions look like test runs" in self._run(store).output
