"""Tests for the per-turn correlation scope (learning-observability.md §4.1)."""

from __future__ import annotations

import json
from pathlib import Path

from iris_harness.foundation.observability import session_log
from iris_harness.foundation.observability.session_log import (
    current_turn_id,
    log_user_message,
    session_scope,
    turn_scope,
)


def test_turn_scope_mints_and_restores() -> None:
    assert current_turn_id() is None
    with turn_scope() as outer:
        assert current_turn_id() == outer
        assert outer  # non-empty minted id
        with turn_scope("explicit-id") as inner:
            assert inner == "explicit-id"
            assert current_turn_id() == "explicit-id"
        # inner restores to outer, not to None
        assert current_turn_id() == outer
    assert current_turn_id() is None


def test_turn_id_is_stamped_on_session_events(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    with session_scope("sess-1"), turn_scope("turn-42"):
        log_user_message("sess-1", text="hello")

    line = (tmp_path / "session-sess-1.jsonl").read_text(encoding="utf-8").splitlines()[0]
    event = json.loads(line)
    assert event["kind"] == "user_message"
    assert event["turn_id"] == "turn-42"
