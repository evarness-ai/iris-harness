"""Tests for the goal_drift evaluator signal (story 12.gov-3.6 follow-up)."""

from __future__ import annotations

from typing import Any

import pytest

from iris_harness.kernel.governance.evaluator import StepRecord
from iris_harness.kernel.governance.evaluator.signals import GoalDriftSignal


class _StubEmbedder:
    """Deterministic stub: maps each text to a fixed vector."""

    def __init__(self, *, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        if text not in self._vectors:
            raise KeyError(f"stub embedder has no vector for {text!r}")
        return list(self._vectors[text])


class _StubWriter:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **payload: Any) -> None:
        self.rows.append(payload)


def _step(thought: str | None, *, original: str | None = "do the thing") -> StepRecord:
    return StepRecord(
        run_id="r",
        step_id=0,
        agent_type="chat",
        thought=thought,
        original_task=original,
    )


def _judging_state(**extra: Any) -> dict[str, Any]:
    """Per-run state with the first-step exemption already spent.

    The opening thought of a run is exempt (it has no heading to have turned away
    from), so a test that wants the distance check itself has to start from a run
    that has already judged a step.
    """
    return {"judged_a_step": True, **extra}


def test_no_drift_when_thought_matches_task() -> None:
    embedder = _StubEmbedder(vectors={"do the thing": [1.0, 0.0]})
    signal = GoalDriftSignal(embedder=embedder, max_distance=0.65)
    state: dict[str, Any] = _judging_state()
    # Same text → cosine=1, distance=0.
    result = signal(_step("do the thing"), state=state)
    assert result.verdict == "ok"
    assert result.audit_metadata["distance"] == pytest.approx(0.0)


def test_orthogonal_thought_requires_approval() -> None:
    """Cosine distance 1.0 > 0.65 → require_approval."""
    embedder = _StubEmbedder(vectors={"do the thing": [1.0, 0.0], "off topic": [0.0, 1.0]})
    writer = _StubWriter()
    signal = GoalDriftSignal(embedder=embedder, max_distance=0.65, flagged_writer=writer)
    result = signal(_step("off topic"), state=_judging_state())
    assert result.verdict == "require_approval"
    assert result.severity == "warn"
    assert "drifted" in result.reason
    assert result.audit_metadata["distance"] == pytest.approx(1.0)
    assert len(writer.rows) == 1
    assert writer.rows[0]["signal"] == "goal_drift"
    assert writer.rows[0]["thought"] == "off topic"


def test_non_flagged_thought_does_not_persist() -> None:
    embedder = _StubEmbedder(vectors={"task": [1.0, 0.0], "near": [1.0, 0.0]})
    writer = _StubWriter()
    signal = GoalDriftSignal(embedder=embedder, flagged_writer=writer)

    result = signal(_step("near", original="task"), state=_judging_state())
    assert result.verdict == "ok"
    assert writer.rows == []


def test_just_under_max_distance_is_ok() -> None:
    """Distance exactly at the threshold is still ok (>, not >=)."""
    # Construct vectors with cosine = 0.35 → distance = 0.65.
    # The check is distance > max_distance, so 0.65 should pass.
    import math

    angle = math.acos(0.35)
    embedder = _StubEmbedder(
        vectors={
            "task": [1.0, 0.0],
            "near": [math.cos(angle), math.sin(angle)],
        }
    )
    signal = GoalDriftSignal(embedder=embedder, max_distance=0.65)
    result = signal(_step("near", original="task"), state=_judging_state())
    assert result.verdict == "ok"
    assert result.audit_metadata["distance"] == pytest.approx(0.65, rel=1e-6)


def test_original_task_embedded_only_once_per_run() -> None:
    """The original task vector is cached on per-run state."""
    embedder = _StubEmbedder(
        vectors={
            "task": [1.0, 0.0],
            "thought1": [1.0, 0.0],
            "thought2": [1.0, 0.0],
            "thought3": [1.0, 0.0],
        }
    )
    signal = GoalDriftSignal(embedder=embedder, max_distance=0.65)
    state: dict[str, Any] = _judging_state()
    signal(_step("thought1", original="task"), state=state)
    signal(_step("thought2", original="task"), state=state)
    signal(_step("thought3", original="task"), state=state)
    # "task" embedded exactly once; each thought once.
    assert embedder.calls.count("task") == 1
    assert embedder.calls.count("thought1") == 1
    assert embedder.calls.count("thought2") == 1
    assert embedder.calls.count("thought3") == 1


def test_changed_original_task_rebuilds_cache() -> None:
    """If the caller changes original_task mid-run, the cache invalidates."""
    embedder = _StubEmbedder(
        vectors={
            "task1": [1.0, 0.0],
            "task2": [0.0, 1.0],
            "thought": [1.0, 0.0],
        }
    )
    signal = GoalDriftSignal(embedder=embedder)
    state: dict[str, Any] = _judging_state()
    signal(_step("thought", original="task1"), state=state)
    signal(_step("thought", original="task2"), state=state)
    assert embedder.calls.count("task1") == 1
    assert embedder.calls.count("task2") == 1


def test_missing_original_task_is_noop() -> None:
    embedder = _StubEmbedder(vectors={})
    signal = GoalDriftSignal(embedder=embedder)
    result = signal(_step("anything", original=None), state={})
    assert result.verdict == "ok"
    assert "no original_task" in result.reason
    # Empty original_task too.
    blank = signal(_step("anything", original="   "), state={})
    assert blank.verdict == "ok"


def test_empty_thought_is_noop() -> None:
    embedder = _StubEmbedder(vectors={"task": [1.0, 0.0]})
    signal = GoalDriftSignal(embedder=embedder)
    result = signal(_step(None, original="task"), state={})
    assert result.verdict == "ok"


def test_embedder_failure_on_thought_does_not_halt() -> None:
    """A buggy embedder on the thought must degrade to warn."""
    embedder = _StubEmbedder(vectors={"task": [1.0, 0.0]})  # no "thought" entry
    signal = GoalDriftSignal(embedder=embedder)
    result = signal(_step("thought", original="task"), state=_judging_state())
    assert result.verdict == "warn"
    assert "embedder error" in result.reason


def test_embedder_failure_on_original_is_warn() -> None:
    embedder = _StubEmbedder(vectors={"thought": [1.0, 0.0]})  # no "task"
    signal = GoalDriftSignal(embedder=embedder)
    result = signal(_step("thought", original="task"), state=_judging_state())
    assert result.verdict == "warn"
    assert "original_task" in result.reason


def test_invalid_max_distance_rejected() -> None:
    embedder = _StubEmbedder(vectors={})
    with pytest.raises(ValueError):
        GoalDriftSignal(embedder=embedder, max_distance=-0.1)
    with pytest.raises(ValueError):
        GoalDriftSignal(embedder=embedder, max_distance=3.0)


def test_signal_protocol_compliance() -> None:
    from iris_harness.kernel.governance.evaluator.types import Signal

    embedder = _StubEmbedder(vectors={})
    sig = GoalDriftSignal(embedder=embedder)
    assert isinstance(sig, Signal)
    # Priority later than loop_detect (40) so the extra embed call runs last.
    assert sig.priority > 40


def test_signal_via_registry_end_to_end() -> None:
    from iris_harness.kernel.governance.evaluator import EvaluatorRegistry

    embedder = _StubEmbedder(vectors={"task": [1.0, 0.0], "off": [0.0, 1.0]})
    reg = EvaluatorRegistry()
    reg.register(GoalDriftSignal(embedder=embedder, max_distance=0.5))
    reg.init_lock()

    # The registry owns per-run state; spend the first-step exemption before the
    # step under test so this exercises the distance check, not the exemption.
    reg.evaluate(_step("task", original="task"))
    results = reg.evaluate(_step("off", original="task"))
    worst = reg.worst(results)
    assert worst is not None
    assert worst.verdict == "require_approval"
    assert worst.name == "goal_drift"


# ── a repeated tool call is not drift ─────────────────────────────────────────
#
# The regression these pin: session `web-6d670ccd`, run `b7c928e2`. The loop
# re-issued an identical `research` call and paraphrased its own thought more
# tersely; the paraphrase embedded 0.769 away from a long multi-clause question and
# halted a turn that had not drifted. `action_repeat` owns identical calls.


def _tool_step(thought: str, *, tool: str = "research", args_hash: str = "abc123") -> StepRecord:
    return StepRecord(
        run_id="r",
        step_id=0,
        agent_type="chat",
        thought=thought,
        tool_name=tool,
        tool_args_hash=args_hash,
        original_task="do the thing",
    )


def test_repeat_of_a_judged_action_is_not_drift() -> None:
    embedder = _StubEmbedder(
        vectors={"do the thing": [1.0, 0.0], "on task": [1.0, 0.0], "off topic": [0.0, 1.0]}
    )
    signal = GoalDriftSignal(embedder=embedder, max_distance=0.65)
    state: dict[str, Any] = {}

    first = signal(_tool_step("on task"), state=state)
    assert first.verdict == "ok"

    # Same (tool, args) again, with a thought that WOULD trip the threshold alone.
    repeat = signal(_tool_step("off topic"), state=state)
    assert repeat.verdict == "ok"
    assert repeat.audit_metadata["skipped"] == "repeat_action"
    # Skipped before the embed, so the off-topic thought was never embedded.
    assert "off topic" not in embedder.calls


def test_a_different_action_is_still_judged() -> None:
    """The skip is keyed on (tool, args) — it must not blanket-exempt later steps."""
    embedder = _StubEmbedder(
        vectors={"do the thing": [1.0, 0.0], "on task": [1.0, 0.0], "off topic": [0.0, 1.0]}
    )
    signal = GoalDriftSignal(embedder=embedder, max_distance=0.65)
    state: dict[str, Any] = {}

    assert signal(_tool_step("on task"), state=state).verdict == "ok"
    drifted = signal(_tool_step("off topic", args_hash="different"), state=state)
    assert drifted.verdict == "require_approval"


def test_a_first_call_that_drifts_still_requires_approval() -> None:
    """The skip needs a *prior* judgement — the first sight of an action is judged."""
    embedder = _StubEmbedder(vectors={"do the thing": [1.0, 0.0], "off topic": [0.0, 1.0]})
    signal = GoalDriftSignal(embedder=embedder, max_distance=0.65)
    state = _judging_state()
    assert signal(_tool_step("off topic"), state=state).verdict == "require_approval"


def test_a_thoughtless_step_records_nothing() -> None:
    """No thought → ok, and the action must NOT be banked as judged: otherwise a step
    the signal never actually examined would exempt its repeat."""
    embedder = _StubEmbedder(vectors={"do the thing": [1.0, 0.0], "off topic": [0.0, 1.0]})
    signal = GoalDriftSignal(embedder=embedder, max_distance=0.65)
    state: dict[str, Any] = _judging_state()

    assert signal(_tool_step(""), state=state).verdict == "ok"
    assert signal(_tool_step("off topic"), state=state).verdict == "require_approval"


# ── the opening step of a run is not drift ────────────────────────────────────
#
# The regression these pin: run `9e444de1`. Asked "wondering how my day looks like ?",
# the loop reasoned its way to `daily_plan` and was halted at distance 0.782 on step 0,
# before a second step existed. Measured against the production embedder every correct
# opening thought for that request scores 0.61-0.86: a six-word question embeds
# diffusely, so a fixed cutoff cannot separate them. A first step has no heading to
# have turned away from, so it is not this signal's to judge.


def test_the_first_step_of_a_run_is_exempt() -> None:
    embedder = _StubEmbedder(vectors={"do the thing": [1.0, 0.0], "off topic": [0.0, 1.0]})
    writer = _StubWriter()
    signal = GoalDriftSignal(embedder=embedder, max_distance=0.65, flagged_writer=writer)

    result = signal(_step("off topic"), state={})

    assert result.verdict == "ok"
    assert result.audit_metadata["skipped"] == "first_step"
    # Exempt before the embed: neither text is embedded, and nothing is persisted.
    assert embedder.calls == []
    assert writer.rows == []


def test_the_second_step_of_a_run_is_judged() -> None:
    """The exemption is one step wide, not a blanket pass for the run."""
    embedder = _StubEmbedder(vectors={"do the thing": [1.0, 0.0], "off topic": [0.0, 1.0]})
    signal = GoalDriftSignal(embedder=embedder, max_distance=0.65)
    state: dict[str, Any] = {}

    assert signal(_step("on topic"), state=state).verdict == "ok"
    assert signal(_step("off topic"), state=state).verdict == "require_approval"


def test_a_thoughtless_first_step_does_not_spend_the_exemption() -> None:
    """A step the signal never examined must not consume the one exemption."""
    embedder = _StubEmbedder(vectors={"do the thing": [1.0, 0.0], "off topic": [0.0, 1.0]})
    signal = GoalDriftSignal(embedder=embedder, max_distance=0.65)
    state: dict[str, Any] = {}

    assert signal(_step(None), state=state).verdict == "ok"
    # The exemption is still unspent, so this — the first thought — takes it.
    assert signal(_step("off topic"), state=state).audit_metadata["skipped"] == "first_step"


def test_the_exempt_step_still_banks_its_action() -> None:
    """Otherwise a repeat of the opening tool call would be judged as if it were new,
    which is exactly the case `action_repeat` owns."""
    embedder = _StubEmbedder(vectors={"do the thing": [1.0, 0.0], "off topic": [0.0, 1.0]})
    signal = GoalDriftSignal(embedder=embedder, max_distance=0.65)
    state: dict[str, Any] = {}

    assert signal(_tool_step("on task"), state=state).audit_metadata["skipped"] == "first_step"
    repeat = signal(_tool_step("off topic"), state=state)
    assert repeat.verdict == "ok"
    assert repeat.audit_metadata["skipped"] == "repeat_action"


def test_the_exemption_is_per_run() -> None:
    """Two runs each get their own opening step; state is keyed per run by the
    registry, so a fresh state dict must behave like a fresh run."""
    embedder = _StubEmbedder(vectors={"do the thing": [1.0, 0.0], "off topic": [0.0, 1.0]})
    signal = GoalDriftSignal(embedder=embedder, max_distance=0.65)

    for _ in range(2):
        assert signal(_step("off topic"), state={}).audit_metadata["skipped"] == "first_step"


def test_the_default_threshold_is_the_measured_one() -> None:
    """0.65 came from the design note and halts correct turns: measured against the
    production embedder, on-task thoughts reach 0.86 on a short original task while real
    drift starts at 0.88. Pinned so it cannot drift back silently."""
    embedder = _StubEmbedder(vectors={})
    assert GoalDriftSignal(embedder=embedder)._max_distance == pytest.approx(0.90)
