"""The summary roll runs after the reply, and the window survives whatever happens.

Compaction used to run inline in ``record_turn``, which is inside the chat finalizer.
That was free only because the "summary" was a string slice; with a real summarizer
wired it would add seconds to the turn the user is waiting on.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from iris_harness.memory.compactor import ConversationCompactor
from iris_harness.memory.store import MemoryStore
from iris_harness.runtime.session_memory import SessionMemory

SID = "async-compaction"


def _sessions(tmp_path: Path, llm_call, *, threshold: int = 2) -> SessionMemory:  # type: ignore[no-untyped-def]
    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.ensure_schema()
    host = SimpleNamespace(
        memory_store=store,
        semantic_index=None,
        compactor=ConversationCompactor(
            compaction_threshold=threshold, keep_recent=2, llm_call=llm_call
        ),
        learning=SimpleNamespace(behavior_miner=None),
    )
    return SessionMemory(host)  # type: ignore[arg-type]


def _wait_for(predicate, timeout: float = 5.0) -> bool:  # type: ignore[no-untyped-def]
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_record_turn_does_not_wait_for_the_summarizer(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def _slow_llm(prompt: str) -> str:
        started.set()
        release.wait(timeout=5)
        return "Goal: rolled."

    sessions = _sessions(tmp_path, _slow_llm)
    for i in range(3):
        sessions.record_turn(SID, f"question {i}", f"answer {i}")

    # The turn returned while the summarizer is still running.
    assert started.wait(timeout=5)
    assert sessions.compaction_in_flight(SID) is True
    assert sessions._session_summaries.get(SID) in (None, "")
    assert len(sessions.conversations[SID]) == 6  # window untouched until it lands

    release.set()
    assert _wait_for(lambda: not sessions.compaction_in_flight(SID))
    assert sessions._session_summaries[SID] == "Goal: rolled."
    assert len(sessions.conversations[SID]) < 6  # now the window is swapped


def test_a_turn_taken_during_the_roll_is_not_lost(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def _slow_llm(prompt: str) -> str:
        started.set()
        release.wait(timeout=5)
        return "Goal: rolled."

    sessions = _sessions(tmp_path, _slow_llm)
    for i in range(3):
        sessions.record_turn(SID, f"question {i}", f"answer {i}")
    assert started.wait(timeout=5)

    sessions.record_turn(SID, "question during the roll", "answer during the roll")
    release.set()
    assert _wait_for(lambda: not sessions.compaction_in_flight(SID))

    kept = " ".join(t.content for t in sessions.conversations[SID])
    assert "question during the roll" in kept


def test_a_failing_summarizer_leaves_the_session_alone(tmp_path: Path) -> None:
    def _boom(prompt: str) -> str:
        raise RuntimeError("model down")

    sessions = _sessions(tmp_path, _boom)
    sessions._session_summaries[SID] = "Goal: the summary we already had."
    for i in range(3):
        sessions.record_turn(SID, f"question {i}", f"answer {i}")

    assert _wait_for(lambda: not sessions.compaction_in_flight(SID))
    assert sessions._session_summaries[SID] == "Goal: the summary we already had."
    assert len(sessions.conversations[SID]) >= 2  # history intact, nothing dropped


def test_one_roll_at_a_time_per_session(tmp_path: Path) -> None:
    calls: list[str] = []
    release = threading.Event()

    def _slow_llm(prompt: str) -> str:
        calls.append(prompt)
        release.wait(timeout=5)
        return "Goal: rolled."

    sessions = _sessions(tmp_path, _slow_llm)
    for i in range(6):
        sessions.record_turn(SID, f"question {i}", f"answer {i}")

    assert _wait_for(lambda: len(calls) >= 1)
    assert len(calls) == 1  # later turns do not pile up more rolls
    release.set()
    assert _wait_for(lambda: not sessions.compaction_in_flight(SID))


def test_compact_now_stays_synchronous(tmp_path: Path) -> None:
    """The explicit control returns a result, so it summarizes in-line.

    The automatic trigger is kept out of the way (threshold 100). With it at 2, the
    test raced the background rolls: under a loaded -n auto run each roll finished
    between turns and left only the recent window, so compact_now had nothing left to
    summarize — a pass or a fail decided by the scheduler, not by the code under test.
    """
    sessions = _sessions(tmp_path, lambda p: "Goal: rolled by hand.", threshold=100)
    for i in range(6):
        sessions.record_turn(SID, f"question {i}", f"answer {i}")
    assert not sessions.compaction_in_flight(SID)  # nothing ran in the background

    result = sessions.compact_now(SID)

    assert result["compacted"] is True
    assert sessions._session_summaries[SID] == "Goal: rolled by hand."


def test_reload_fills_the_window_budget_not_ten_rows(tmp_path: Path) -> None:
    sessions = _sessions(tmp_path, lambda p: "Goal: rolled.")
    sessions._host.compactor.token_budget = 4000  # type: ignore[attr-defined]
    store = sessions._host.memory_store  # type: ignore[attr-defined]
    store.save_conversation_turns(
        SID, [(role, f"{role} line {i}") for i in range(20) for role in ("user", "assistant")]
    )

    sessions.build_memory_context("anything", session_id=SID)

    assert len(sessions.conversations[SID]) > 10


def test_a_playground_session_leaves_no_rows(tmp_path: Path) -> None:
    """Runs are not memory: 1,420 of 3,730 stored turns came from sessions like these,
    and cross-session recall served them back as the user's own past conversations."""
    sessions = _sessions(tmp_path, lambda p: "Goal: rolled.")
    store = sessions._host.memory_store  # type: ignore[attr-defined]

    sessions.record_turn("playground-brief-inj", "scenario question", "scenario answer")
    sessions.record_turn("default", "real question", "real answer")

    assert store.fetch_turn_ids("playground-brief-inj") == []
    assert len(store.fetch_turn_ids("default")) == 2
    # The run still keeps its own live window — it needs its history within the run.
    assert len(sessions.conversations["playground-brief-inj"]) == 2


# ── the closing summary of an idle session (retention, 2026-09-19) ───────────


def test_close_session_summarizes_a_short_conversation(tmp_path: Path) -> None:
    """compact_now said "nothing to compact" for this (all of it fits the kept window),
    so idle sessions never got the summary that cooling needs."""
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return "Goal: asked about the Discover due date."

    sessions = _sessions(tmp_path, llm, threshold=100)
    store = sessions._host.memory_store  # type: ignore[attr-defined]
    store.save_conversation_turns(
        "web-idle", [("user", "when is discover due?"), ("assistant", "the 12th")]
    )
    assert sessions.compact_now("web-idle")["compacted"] is False  # the old path: nothing

    result = sessions.close_session("web-idle")

    assert result["closed"] is True and result["turns"] == 2
    assert store.load_conversation_summary("web-idle") == "Goal: asked about the Discover due date."
    assert "when is discover due?" in prompts[-1]  # every turn went to the summarizer


def test_close_session_rolls_an_existing_summary_forward(tmp_path: Path) -> None:
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return "Goal: newer."

    sessions = _sessions(tmp_path, llm, threshold=100)
    store = sessions._host.memory_store  # type: ignore[attr-defined]
    store.save_conversation_turns("web-idle", [("user", "q"), ("assistant", "a")])
    store.save_conversation_summary("web-idle", "Goal: the earlier part.")

    assert sessions.close_session("web-idle")["closed"] is True
    assert "Goal: the earlier part." in prompts[-1]
    assert store.load_conversation_summary("web-idle") == "Goal: newer."


def test_close_session_without_a_summarizer_closes_nothing(tmp_path: Path) -> None:
    sessions = _sessions(tmp_path, None, threshold=100)
    store = sessions._host.memory_store  # type: ignore[attr-defined]
    store.save_conversation_turns("web-idle", [("user", "q"), ("assistant", "a")])
    result = sessions.close_session("web-idle")
    assert result["closed"] is False and store.load_conversation_summary("web-idle") == ""
    assert sessions.close_session("web-empty") == {
        "closed": False,
        "reason": "no turns",
        "session_id": "web-empty",
    }


def test_close_session_does_not_load_the_session_into_the_live_cache(tmp_path: Path) -> None:
    sessions = _sessions(tmp_path, lambda p: "Goal: closed.", threshold=100)
    sessions._host.memory_store.save_conversation_turns(  # type: ignore[attr-defined]
        "web-idle", [("user", "q"), ("assistant", "a")]
    )
    sessions.close_session("web-idle")
    assert "web-idle" not in sessions.conversations  # 20 an hour would pile up otherwise


def test_a_failed_roll_over_an_existing_summary_is_not_a_close(tmp_path: Path) -> None:
    """The summarizer hands back the previous summary when it fails; that is no roll."""

    def broken(prompt: str) -> str:
        raise RuntimeError("ollama down")

    sessions = _sessions(tmp_path, broken, threshold=100)
    store = sessions._host.memory_store  # type: ignore[attr-defined]
    store.save_conversation_turns("web-idle", [("user", "q"), ("assistant", "a")])
    store.save_conversation_summary("web-idle", "Goal: the earlier part.")

    assert sessions.close_session("web-idle")["closed"] is False
    assert store.load_conversation_summary("web-idle") == "Goal: the earlier part."
