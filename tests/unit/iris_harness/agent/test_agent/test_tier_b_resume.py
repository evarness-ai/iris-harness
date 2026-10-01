"""ADR-0106 Tier B (M5.C5c) — the mid-loop resume.

C5b taught the loop to stop and ask (`test_ask_user_pause.py`). This is the other
half: the human answers, and the *same run* continues from the step that asked,
rather than the reply being answered as a turn of its own.

Two things are pinned here beyond "it resumes":

- **One decoder, two entry points.** `resume_seed_from_checkpoint` is the only thing
  that knows how a checkpoint is shaped; `run_from_seed` and `run_stream(resume=)`
  both start from its output. The two loop bodies are still separate (ADR-0107 option
  B was deliberately not started), so a resume that worked on only one of them is the
  exact drift ADR-0107 was written about.
- **The resumed run re-enters the original task**, not the reply that unblocked it.
  The reply is loop *input* — the paused step's observation — and a resume that
  treated it as the query would restart the work with the answer as the question.
"""

from __future__ import annotations

from pathlib import Path

from iris_harness.agent.agentic_core import (
    ASK_USER_ACTION,
    AgenticCore,
    AgenticCoreConfig,
    ToolSpec,
    resume_seed_from_checkpoint,
)
from iris_harness.memory.state import CheckpointStore

_ASK = (
    "Thought: I need their call on this\n"
    'Action: ask_user\nAction Input: {"question": "Neo4j or Stardog?"}'
)
_FINAL = "Thought: they picked one\nFinal Answer: Going with Stardog."
_QUERY = "pick a graph database"


def _echo_tool() -> ToolSpec:
    return ToolSpec(name="echo", description="echo", call=lambda a: f"observed: {a.get('q', '?')}")


class _RecordingLLM:
    """Scripted, but keeps every prompt so the test can assert what the loop saw."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            return "Thought: done\nFinal Answer: finished"
        return self._responses.pop(0)


def _core(
    responses: list[str], *, store: CheckpointStore | None = None, max_iterations: int = 4
) -> AgenticCore:
    return AgenticCore(
        config=AgenticCoreConfig(max_iterations=max_iterations, allow_ask_user=True),
        llm_call=_RecordingLLM(responses),
        tools=[_echo_tool()],
        checkpoint_store=store,
        session_id="s1",
        agent_type="chat",
    )


def _paused(tmp_path: Path, *, streaming: bool = False) -> tuple[CheckpointStore, str]:
    """Drive a run to an `ask_user` pause; return its store and run id."""
    store = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    core = _core([_ASK], store=store)
    if streaming:
        meta = [c for c in core.run_stream(_QUERY) if isinstance(c, dict)][-1]
        return store, str(meta["run_id"])
    return store, core.run(_QUERY).run_id


# ── the seed ──────────────────────────────────────────────────────────────────


def test_the_seed_re_enters_at_the_step_after_the_pause(tmp_path: Path) -> None:
    store, run_id = _paused(tmp_path)
    seed = resume_seed_from_checkpoint(store.get(run_id=run_id, step_id=0))

    assert seed.run_id == run_id
    assert seed.query == _QUERY
    assert seed.start_iteration == 1  # paused at 0, resumes at N+1
    assert len(seed.steps) == 1
    assert seed.steps[0].action == ASK_USER_ACTION


def test_the_reply_lands_as_the_observation_of_the_step_that_asked(tmp_path: Path) -> None:
    """LangGraph's `Command(resume=v)`, in the one place the loop will read it."""
    store, run_id = _paused(tmp_path)
    seed = resume_seed_from_checkpoint(
        store.get(run_id=run_id, step_id=0), user_reply="  Stardog, please  "
    )

    observation = seed.steps[-1].observation or ""
    assert "Neo4j or Stardog?" in observation  # what was asked survives
    assert "User answered: Stardog, please" in observation  # trimmed
    # The claim rule is "the next message resumes", so a change of subject arrives
    # here too and the loop must be able to drop the old task.
    assert "abandon the previous task" in observation
    assert "User answered:" in "\n".join(seed.history)


def test_no_reply_leaves_the_observation_untouched(tmp_path: Path) -> None:
    """An operator-driven resume (`iris run resume`-shaped, no human answer) must not
    invent one."""
    store, run_id = _paused(tmp_path)
    seed = resume_seed_from_checkpoint(store.get(run_id=run_id, step_id=0))

    assert "User answered:" not in (seed.steps[-1].observation or "")


# ── the sync loop ─────────────────────────────────────────────────────────────


def test_sync_resume_continues_the_same_run(tmp_path: Path) -> None:
    store, run_id = _paused(tmp_path)
    core = _core([_FINAL], store=store)
    seed = resume_seed_from_checkpoint(store.get(run_id=run_id, step_id=0), user_reply="Stardog")

    trace = core.run_from_seed(seed)

    assert trace.final_answer == "Going with Stardog."
    assert trace.run_id == run_id  # the same run, not a new one
    # The paused step is still on the trace — resumed, not restarted.
    assert trace.steps[0].action == ASK_USER_ACTION
    assert trace.query == _QUERY


def test_the_resumed_loop_is_shown_the_answer(tmp_path: Path) -> None:
    store, run_id = _paused(tmp_path)
    core = _core([_FINAL], store=store)
    seed = resume_seed_from_checkpoint(store.get(run_id=run_id, step_id=0), user_reply="Stardog")

    core.run_from_seed(seed)

    prompt = core._llm.prompts[-1]  # type: ignore[attr-defined]
    assert "User answered: Stardog" in prompt
    assert _QUERY in prompt  # still working the original task


def test_resume_from_checkpoint_still_works_and_takes_a_reply(tmp_path: Path) -> None:
    """The pre-C5c entry point (and the CLI's) keeps its shape."""
    store, run_id = _paused(tmp_path)
    core = _core([_FINAL], store=store)

    trace = core.resume_from_checkpoint(store.get(run_id=run_id, step_id=0), user_reply="Stardog")

    assert trace.final_answer == "Going with Stardog."
    assert "User answered: Stardog" in core._llm.prompts[-1]  # type: ignore[attr-defined]


# ── the streaming loop ────────────────────────────────────────────────────────


def test_streaming_resume_continues_the_same_run(tmp_path: Path) -> None:
    store, run_id = _paused(tmp_path, streaming=True)
    core = _core([_FINAL], store=store)
    seed = resume_seed_from_checkpoint(store.get(run_id=run_id, step_id=0), user_reply="Stardog")

    chunks = list(core.run_stream("Stardog", resume=seed))
    text = [c for c in chunks if isinstance(c, str)]
    meta = [c for c in chunks if isinstance(c, dict)][-1]

    assert text[-1] == "Going with Stardog."
    assert meta["success"] is True
    assert meta["reason"] == "final_answer"
    assert meta["run_id"] == run_id
    assert meta["paused_at_step"] is None  # answered, no longer waiting
    # Seeded from the checkpoint, so the paused step is counted, not lost.
    assert meta["iterations"] == 2


def test_streaming_resume_works_the_original_task_not_the_reply(tmp_path: Path) -> None:
    """`run_stream`'s `query` argument is the reply on a resume; the seed's query wins."""
    store, run_id = _paused(tmp_path, streaming=True)
    core = _core([_FINAL], store=store)
    seed = resume_seed_from_checkpoint(store.get(run_id=run_id, step_id=0), user_reply="Stardog")

    list(core.run_stream("Stardog", resume=seed))

    prompt = core._llm.prompts[-1]  # type: ignore[attr-defined]
    assert _QUERY in prompt
    assert "User answered: Stardog" in prompt


def test_an_exhausted_resume_ends_cleanly(tmp_path: Path) -> None:
    """A seed whose start is already past the iteration cap leaves the loop body
    unentered. The metadata below it reads the iteration counter, so this must not
    raise — it must report an ordinary exhausted run."""
    store, run_id = _paused(tmp_path)
    core = _core([_FINAL], store=store, max_iterations=1)
    seed = resume_seed_from_checkpoint(store.get(run_id=run_id, step_id=0), user_reply="Stardog")
    assert seed.start_iteration == 1

    chunks = list(core.run_stream("Stardog", resume=seed))
    meta = [c for c in chunks if isinstance(c, dict)][-1]

    assert meta["reason"] == "max_iterations"
    assert meta["success"] is False
    assert isinstance(meta["iterations"], int)
