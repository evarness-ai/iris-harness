"""The stage set an approved governance halt is resumed through.

Approving from the web could have called the react loop directly — it is the shortest
path and it would have worked. It is also the one path whose entire premise is that
governance stopped this run once already, so skipping ``curate`` there would drop the
ResponseCurator and every safety judge at exactly the wrong moment. That is the
2026-07-06 red-team rule the intercept chain documents, and it applies to a resume as
much as to an intercept.

So a resume runs the real pipeline with one stage removed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from iris_harness.agent.intent_router import IntentResult
from iris_harness.memory.state.continuations import ContinuationStore
from iris_harness.runtime.continuations import ContinuationRegistry
from iris_harness.runtime.turn import RESUME_STAGES, STAGES
from iris_harness.runtime.turn.stages import classify
from iris_harness.runtime.turn.state import TurnRequest, TurnState

# ── the stage set ─────────────────────────────────────────────────────────────


def test_a_resume_drops_only_the_screen_and_intercept_stages() -> None:
    """No new user message: nothing to screen (it was screened when it arrived) and
    nothing a deterministic handler may take over from the run being resumed."""
    names = [name for name, _ in RESUME_STAGES]
    assert names == [name for name, _ in STAGES if name not in {"screen", "intercept"}]


def test_a_resume_still_runs_the_curator_and_the_record() -> None:
    """`curate` is the judges; `record` is what puts the answer in the session log, and
    so is the only reason a resumed answer reaches the web transcript at all."""
    names = [name for name, _ in RESUME_STAGES]
    assert "curate" in names
    assert "record" in names


def test_a_resume_does_not_log_a_user_message() -> None:
    """`intercept` is where the user's message is logged. Nobody sent one here — the
    human answered an approval — and forging one would put words in their mouth."""
    assert "intercept" not in [name for name, _ in RESUME_STAGES]


# ── the request carries the point ─────────────────────────────────────────────


def test_a_request_with_both_halves_has_a_resume_point() -> None:
    request = TurnRequest(message="m", resume_run_id="run-1", resume_step_id=2)
    assert request.resume_point == ("run-1", 2)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"resume_run_id": "run-1"}, {"resume_step_id": 2}],
)
def test_a_half_specified_request_has_none(kwargs: dict[str, object]) -> None:
    assert TurnRequest(message="m", **kwargs).resume_point is None  # type: ignore[arg-type]


def test_step_zero_is_a_real_resume_point() -> None:
    """`0` is falsy and is the commonest halt of all — the first step."""
    assert TurnRequest(message="m", resume_run_id="run-1", resume_step_id=0).resume_point == (
        "run-1",
        0,
    )


# ── and it keeps classify's hands off someone else's question ──────────────────


class _SteerRuntime:
    def __init__(self, registry: ContinuationRegistry) -> None:
        self.continuations = registry


def _state(**request_kwargs: object) -> TurnState:
    state = TurnState(
        request=TurnRequest(message="pick a graph database", session_id="s1", **request_kwargs)  # type: ignore[arg-type]
    )
    state.intent_result = IntentResult(
        intent="general", agent_type="general", confidence=0.4, source="fallback"
    )
    return state


@pytest.fixture()
def registry(tmp_path: Path) -> ContinuationRegistry:
    return ContinuationRegistry(store=ContinuationStore(db_path=tmp_path / "checkpoints.db"))


def test_a_resume_does_not_claim_a_pending_continuation(
    registry: ContinuationRegistry,
) -> None:
    """A resumed turn has no user message behind it, so there is nothing here that could
    be an answer. Claiming the session's open question would close it on the strength of
    a reply that was never sent — and Tier B claims unconditionally, so without this
    guard it would."""
    registry.ask(
        "s1",
        "system",
        question="Neo4j or Stardog?",
        kind="question",
        intent="system",
        run_id="other-run",
        step_id=0,
    )
    state = _state(resume_run_id="run-1", resume_step_id=1)
    before = replace(state.intent_result)  # type: ignore[arg-type]

    classify._honour_continuation(_SteerRuntime(registry), state)

    assert state.continuation is None
    assert state.intent_result == before
    assert registry.pending("s1") is not None  # still someone else's, still open


def test_an_ordinary_turn_still_claims_it(registry: ContinuationRegistry) -> None:
    registry.ask(
        "s1",
        "system",
        question="Neo4j or Stardog?",
        kind="question",
        intent="system",
        run_id="run-9",
        step_id=0,
    )
    state = _state()

    classify._honour_continuation(_SteerRuntime(registry), state)

    assert state.continuation is not None
