"""Chat comes first: whether a chat turn is running (2026-09-25 Telegram timeout)."""

from __future__ import annotations

import pytest

from iris_harness.foundation import activity


@pytest.fixture(autouse=True)
def _clean() -> None:
    activity._reset_for_tests()
    yield
    activity._reset_for_tests()


def test_a_turn_is_in_progress_while_it_runs_and_briefly_after() -> None:
    assert not activity.chat_in_progress()
    with activity.chat_turn():
        assert activity.chat_in_progress()
    assert activity.chat_in_progress()  # the quiet window after an answer
    assert not activity.chat_in_progress(quiet_after=0)


def test_overlapping_turns_count_until_the_last_ends() -> None:
    with activity.chat_turn():
        with activity.chat_turn():
            pass
        assert activity.chat_in_progress(quiet_after=0)
    assert not activity.chat_in_progress(quiet_after=0)


def test_an_exception_still_ends_the_turn() -> None:
    with pytest.raises(RuntimeError), activity.chat_turn():
        raise RuntimeError("boom")
    assert not activity.chat_in_progress(quiet_after=0)


def test_run_turn_marks_the_turn() -> None:
    from iris_harness.runtime.turn.pipeline import run_turn

    seen: list[bool] = []

    def stage(runtime, state):  # type: ignore[no-untyped-def]
        seen.append(activity.chat_in_progress(quiet_after=0))
        from iris_harness.runtime.types import ChatResult

        state.result = ChatResult(response="ok", session_id="s", intent="general", agent_type="x")
        return iter(())

    from iris_harness.runtime.turn.state import TurnRequest

    list(run_turn(object(), TurnRequest(message="hi", session_id="s"), stages=(("s", stage),)))  # type: ignore[arg-type]
    assert seen == [True]
    assert not activity.chat_in_progress(quiet_after=0)
