"""ADR-0106 Tier B (M5.C5b) — the loop can stop and ask.

`ask_user` is a control-flow action, not a tool: the loop never calls it, it stops
on it, writes a resumable checkpoint, and puts the question to the user. Both loops,
because both answer real turns.

What this slice does *not* do is resume from that checkpoint: that is C5c, and it
lives in ``test_tier_b_resume.py``. Everything here is about the pause itself — the
question reaching the user, the run stopping cleanly on it, and the checkpoint being
resumable — which must keep holding whether or not anything resumes from it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.agent.agentic_core import (
    ASK_USER_ACTION,
    ASK_USER_TOOL,
    AgenticCore,
    AgenticCoreConfig,
    ReactStep,
    ToolSpec,
    _ask_user_question,
)
from iris_harness.memory.state import CheckpointStore


def _echo_tool() -> ToolSpec:
    return ToolSpec(name="echo", description="echo", call=lambda a: f"observed: {a.get('q', '?')}")


class _ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def __call__(self, _prompt: str) -> str:
        if not self._responses:
            return "Thought: done\nFinal Answer: finished"
        return self._responses.pop(0)


def _core(
    responses: list[str], *, allow: bool = True, store: CheckpointStore | None = None
) -> AgenticCore:
    return AgenticCore(
        config=AgenticCoreConfig(max_iterations=4, allow_ask_user=allow),
        llm_call=_ScriptedLLM(responses),
        tools=[_echo_tool()],
        checkpoint_store=store,
        session_id="s1",
        agent_type="chat",
    )


_ASK = (
    "Thought: I need their call on this\n"
    'Action: ask_user\nAction Input: {"question": "Neo4j or Stardog?"}'
)


# ── the question ──────────────────────────────────────────────────────────────


def test_the_question_is_read_from_any_of_the_usual_keys() -> None:
    for key in ("question", "prompt", "text", "q"):
        step = ReactStep(thought="t", action=ASK_USER_ACTION, action_input={key: "Which one?"})
        assert _ask_user_question(step) == "Which one?"


def test_a_question_less_ask_falls_back_to_the_thought_then_a_default() -> None:
    thought_only = ReactStep(thought="Should I use Neo4j?", action=ASK_USER_ACTION)
    assert _ask_user_question(thought_only) == "Should I use Neo4j?"

    bare = ReactStep(thought="", action=ASK_USER_ACTION, action_input={})
    assert "how you'd like to proceed" in _ask_user_question(bare)


def test_the_tool_is_never_executable() -> None:
    """It is declared so the prompt advertises it; routing it to _execute_tool is a bug."""
    with pytest.raises(AssertionError, match="intercepted by the ReAct loop"):
        ASK_USER_TOOL.call({})


# ── the sync loop ─────────────────────────────────────────────────────────────


def test_sync_loop_stops_and_asks(tmp_path: Path) -> None:
    store = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    core = _core([_ASK], store=store)

    trace = core.run("pick a graph database")

    assert trace.final_answer == "Neo4j or Stardog?"
    assert trace.halted_by == "ask_user"
    assert trace.halt_reason == "awaiting_user_input"
    # Resumable, which is what makes this Tier B rather than a re-prompt.
    cp = store.get_latest(trace.run_id)
    assert cp.signal == "awaiting_user_input"
    assert cp.session_id == "s1"
    assert cp.payload["iteration"] == 1


def test_the_loop_does_not_run_on_past_the_question() -> None:
    """Asking ends the turn — the next scripted reply must never be consumed."""
    core = _core([_ASK, 'Thought: carry on\nAction: echo\nAction Input: {"q": "x"}'])

    trace = core.run("pick a graph database")

    assert len(trace.steps) == 1
    assert trace.steps[0].action == ASK_USER_ACTION


def test_ask_user_is_inert_when_not_offered() -> None:
    """A model that invents the action where it was never offered falls through to
    normal unknown-tool handling rather than gaining a way to stop the turn."""
    core = _core([_ASK], allow=False)

    trace = core.run("pick a graph database")

    assert trace.halted_by != "ask_user"
    assert trace.final_answer != "Neo4j or Stardog?"


def test_asking_still_works_without_a_checkpoint_store() -> None:
    """The checkpoint buys the mid-loop resume, not the question. Losing it must not
    cost the user their answer."""
    core = _core([_ASK], store=None)

    trace = core.run("pick a graph database")

    assert trace.final_answer == "Neo4j or Stardog?"
    assert trace.checkpoint_id is None


# ── the streaming loop ────────────────────────────────────────────────────────


def test_streaming_loop_stops_and_asks(tmp_path: Path) -> None:
    store = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    core = _core([_ASK], store=store)

    chunks = list(core.run_stream("pick a graph database"))
    text = [c for c in chunks if isinstance(c, str)]
    meta = [c for c in chunks if isinstance(c, dict)][-1]

    assert text[-1] == "Neo4j or Stardog?"
    assert meta["reason"] == "awaiting_user_input"
    assert meta["success"] is True  # asking is a complete turn, not a failure
    assert meta["paused_at_step"] == 0
    assert meta["run_id"]
    assert store.get_latest(str(meta["run_id"])).signal == "awaiting_user_input"


def test_streaming_reports_no_pause_point_on_an_ordinary_turn() -> None:
    core = _core(["Thought: easy\nFinal Answer: Neo4j."])

    meta = [c for c in core.run_stream("pick one") if isinstance(c, dict)][-1]

    assert meta["paused_at_step"] is None
    assert meta["reason"] == "final_answer"
