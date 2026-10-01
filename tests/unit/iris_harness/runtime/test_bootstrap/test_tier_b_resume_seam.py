"""ADR-0106 Tier B (M5.C5c) — `_resume_seed_for`, the one decision both handlers share.

The react handler and its streaming twin must agree on whether a turn resumes, and
every way of *not* resuming has to land on the same safe answer: run the reply as its
own turn, which is exactly what the harness did before Tier B existed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from iris_harness.agent.agent_executor import AgentTask
from iris_harness.memory.state import CheckpointStore
from iris_harness.memory.state.chat import ChatCheckpointPayload, CheckpointStep
from iris_harness.runtime.handlers.react import _resume_seed_for


class _Store:
    """A real CheckpointStore, wrapped so the test can see whether it was opened."""

    def __init__(self, tmp_path: Path, *, write: bool = True) -> None:
        self.store = CheckpointStore(db_path=tmp_path / "checkpoints.db")
        self.opened = 0
        if write:
            self.store.write(
                run_id="run-1",
                step_id=0,
                agent_type="chat",
                payload=ChatCheckpointPayload(
                    query="pick a graph database",
                    steps=(
                        CheckpointStep(
                            thought="ask them",
                            action="ask_user",
                            action_input={"question": "Neo4j or Stardog?"},
                            observation="Asked the user: Neo4j or Stardog?",
                        ),
                    ),
                    iteration=1,
                ).to_payload(),
                signal="awaiting_user_input",
                session_id="s1",
            )

    def __call__(self) -> CheckpointStore:
        self.opened += 1
        return self.store


def _task(**kwargs: Any) -> AgentTask:
    return AgentTask(query="use Stardog", agent_type="system", **kwargs)


def test_a_task_with_no_resume_point_never_opens_the_store(tmp_path: Path) -> None:
    """The ordinary turn, which is every turn. It must not open checkpoints.db just to
    find out it is not resuming — hence a factory rather than a store."""
    store = _Store(tmp_path)

    assert _resume_seed_for(store, _task()) is None
    assert store.opened == 0


def test_the_reply_is_handed_over_as_the_answer(tmp_path: Path) -> None:
    store = _Store(tmp_path)

    seed = _resume_seed_for(
        store,
        _task(resume_run_id="run-1", resume_step_id=0, resume_reply="use Stardog"),
    )

    assert seed is not None
    assert seed.start_iteration == 1
    assert "User answered: use Stardog" in (seed.steps[-1].observation or "")
    # The original task, which is what the core must be built for — not the reply.
    assert seed.query == "pick a graph database"


def test_a_resume_with_no_reply_injects_nothing(tmp_path: Path) -> None:
    """An approved *governance* halt: the human answered an approval, not a question,
    and the paused step's observation is a real tool result. Overwriting it with
    "User answered: ..." would destroy the work the resume exists to continue — so
    `resume_reply`, not `query`, is what gets injected, and here it is absent."""
    store = _Store(tmp_path)

    seed = _resume_seed_for(store, _task(resume_run_id="run-1", resume_step_id=0))

    assert seed is not None
    observation = seed.steps[-1].observation or ""
    assert "User answered:" not in observation
    assert observation == "Asked the user: Neo4j or Stardog?"  # untouched


def test_a_missing_checkpoint_degrades_to_a_fresh_run(tmp_path: Path) -> None:
    """Expired (7-day TTL), swept, or undecodable — all the same answer."""
    store = _Store(tmp_path, write=False)

    assert _resume_seed_for(store, _task(resume_run_id="run-1", resume_step_id=0)) is None


def test_a_half_specified_resume_point_is_ignored(tmp_path: Path) -> None:
    """Both halves or neither: a run id with no step is not a resume point, and
    guessing a step would resume at the wrong place rather than fail."""
    store = _Store(tmp_path)

    assert _resume_seed_for(store, _task(resume_run_id="run-1")) is None
    assert _resume_seed_for(store, _task(resume_step_id=0)) is None
    assert store.opened == 0


def test_step_zero_is_a_real_resume_point(tmp_path: Path) -> None:
    """`step_id=0` is falsy and is the commonest pause of all — the first step."""
    store = _Store(tmp_path)

    assert _resume_seed_for(store, _task(resume_run_id="run-1", resume_step_id=0)) is not None
