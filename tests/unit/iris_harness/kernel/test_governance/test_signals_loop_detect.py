"""Tests for the loop_detect evaluator signal (story 12.gov-3.6)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from iris_harness.kernel.governance.evaluator import StepRecord
from iris_harness.kernel.governance.evaluator.embeddings import cosine_similarity
from iris_harness.kernel.governance.evaluator.signals import LoopDetectSignal


class _StubEmbedder:
    """Deterministic stub: maps each unique text to a fixed unit vector.

    Tests that want "highly similar" thoughts share the same text;
    tests that want orthogonal thoughts use the orthogonal-vector helper.
    """

    def __init__(self, *, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def __call__(self, text: str) -> list[float]:
        if text not in self._vectors:
            raise KeyError(f"stub embedder has no vector for {text!r}")
        return list(self._vectors[text])


class _StubWriter:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **payload: Any) -> None:
        self.rows.append(payload)


def _step(step_id: int, thought: str | None) -> StepRecord:
    return StepRecord(run_id="r", step_id=step_id, agent_type="chat", thought=thought)


def test_window_collection_under_threshold_count() -> None:
    """Fewer than ``window`` thoughts → ok with a 'collected N/W' reason."""
    embedder = _StubEmbedder(vectors={"hi": [1.0, 0.0]})
    signal = LoopDetectSignal(embedder=embedder, window=3, threshold=0.92)
    state: dict[str, Any] = {}

    first = signal(_step(0, "hi"), state=state)
    assert first.verdict == "ok"
    assert "1/3" in first.reason

    second = signal(_step(1, "hi"), state=state)
    assert second.verdict == "ok"
    assert "2/3" in second.reason


def test_three_identical_thoughts_halt() -> None:
    """Three pairwise-similar thoughts at the threshold → halt."""
    embedder = _StubEmbedder(vectors={"loop": [1.0, 0.0, 0.0]})
    writer = _StubWriter()
    signal = LoopDetectSignal(
        embedder=embedder,
        window=3,
        threshold=0.92,
        flagged_writer=writer,
    )
    state: dict[str, Any] = {}

    assert signal(_step(0, "loop"), state=state).verdict == "ok"
    assert signal(_step(1, "loop"), state=state).verdict == "ok"
    third = signal(_step(2, "loop"), state=state)
    assert third.verdict == "halt"
    assert third.severity == "warn"
    assert third.audit_metadata["step_ids"] == [0, 1, 2]
    assert third.audit_metadata["min_similarity"] == pytest.approx(1.0)
    assert third.audit_metadata["threshold"] == 0.92
    assert [row["step_id"] for row in writer.rows] == [0, 1, 2]
    assert all(row["signal"] == "loop_detect" for row in writer.rows)


def test_orthogonal_thoughts_do_not_halt() -> None:
    """Three orthogonal thoughts → cosine 0.0 < 0.92 → ok."""
    embedder = _StubEmbedder(
        vectors={
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0],
            "c": [0.0, 0.0, 1.0],
        }
    )
    signal = LoopDetectSignal(embedder=embedder, window=3, threshold=0.92)
    state: dict[str, Any] = {}

    assert signal(_step(0, "a"), state=state).verdict == "ok"
    assert signal(_step(1, "b"), state=state).verdict == "ok"
    third = signal(_step(2, "c"), state=state)
    assert third.verdict == "ok"
    assert third.audit_metadata["min_similarity"] == pytest.approx(0.0)


def test_non_flagged_window_does_not_persist() -> None:
    embedder = _StubEmbedder(
        vectors={
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0],
            "c": [0.0, 0.0, 1.0],
        }
    )
    writer = _StubWriter()
    signal = LoopDetectSignal(embedder=embedder, window=3, flagged_writer=writer)
    state: dict[str, Any] = {}

    signal(_step(0, "a"), state=state)
    signal(_step(1, "b"), state=state)
    result = signal(_step(2, "c"), state=state)
    assert result.verdict == "ok"
    assert writer.rows == []


def test_one_outlier_in_window_does_not_halt() -> None:
    """t1,t2 nearly identical but t3 orthogonal → min pairwise sim is 0 → ok."""
    # Two nearly-identical thoughts and one orthogonal one. min cosine = 0,
    # so the pairwise-min gate keeps the signal from tripping.
    embedder = _StubEmbedder(
        vectors={
            "same1": [1.0, 0.0],
            "same2": [1.0, 0.0],
            "diff": [0.0, 1.0],
        }
    )
    signal = LoopDetectSignal(embedder=embedder, window=3, threshold=0.92)
    state: dict[str, Any] = {}

    assert signal(_step(0, "same1"), state=state).verdict == "ok"
    assert signal(_step(1, "same2"), state=state).verdict == "ok"
    third = signal(_step(2, "diff"), state=state)
    assert third.verdict == "ok"
    assert third.audit_metadata["min_similarity"] == pytest.approx(0.0)


def test_window_slides_after_break() -> None:
    """Once the buffer fills, only the most recent ``window`` thoughts count."""
    embedder = _StubEmbedder(
        vectors={
            "different": [0.0, 1.0],
            "loop": [1.0, 0.0],
        }
    )
    signal = LoopDetectSignal(embedder=embedder, window=3, threshold=0.92)
    state: dict[str, Any] = {}

    # Start with an orthogonal thought, then three identical ones.
    assert signal(_step(0, "different"), state=state).verdict == "ok"
    assert signal(_step(1, "loop"), state=state).verdict == "ok"
    assert signal(_step(2, "loop"), state=state).verdict == "ok"
    # At step 3 the window is [loop, loop, loop] — the leading "different"
    # has slid out — so this should halt.
    fourth = signal(_step(3, "loop"), state=state)
    assert fourth.verdict == "halt"
    assert fourth.audit_metadata["step_ids"] == [1, 2, 3]


def test_empty_thought_is_noop() -> None:
    embedder = _StubEmbedder(vectors={})
    signal = LoopDetectSignal(embedder=embedder, window=3)
    state: dict[str, Any] = {}

    result = signal(_step(0, None), state=state)
    assert result.verdict == "ok"
    assert "no thought" in result.reason

    result_blank = signal(_step(1, "   "), state=state)
    assert result_blank.verdict == "ok"


def test_embedder_failure_does_not_halt() -> None:
    """A buggy embedder must degrade to warn, not halt the run."""

    def _broken(_text: str) -> list[float]:
        raise RuntimeError("model offline")

    signal = LoopDetectSignal(embedder=_broken, window=3)
    state: dict[str, Any] = {}

    result = signal(_step(0, "anything"), state=state)
    assert result.verdict == "warn"
    assert "embedder error" in result.reason
    assert result.audit_metadata["error"] == "model offline"


def test_invalid_window_rejected() -> None:
    embedder = _StubEmbedder(vectors={})
    with pytest.raises(ValueError):
        LoopDetectSignal(embedder=embedder, window=1)


def test_invalid_threshold_rejected() -> None:
    embedder = _StubEmbedder(vectors={})
    with pytest.raises(ValueError):
        LoopDetectSignal(embedder=embedder, threshold=1.5)


def test_cosine_similarity_helper_basics() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    # zero-norm guard
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    # mismatched shapes
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0
    # arbitrary, just check it's a real number in range
    sim = cosine_similarity([3.0, 4.0], [4.0, 3.0])
    assert -1.0 <= sim <= 1.0
    assert sim == pytest.approx(24 / 25, rel=1e-6)


def test_signal_via_registry_resets_per_run_state() -> None:
    """End-to-end registry check: a halted run drops its embeddings buffer."""
    from iris_harness.kernel.governance.evaluator import EvaluatorRegistry

    embedder = _StubEmbedder(vectors={"loop": [1.0, 0.0]})
    reg = EvaluatorRegistry()
    reg.register(LoopDetectSignal(embedder=embedder, window=3, threshold=0.92))
    reg.init_lock()

    # Two steps with the same thought — not yet halted.
    for sid in (0, 1):
        results = reg.evaluate(_step(sid, "loop"))
        assert all(r.verdict == "ok" for r in results)

    # Third identical thought — should halt.
    results = reg.evaluate(_step(2, "loop"))
    assert reg.worst(results).verdict == "halt"  # type: ignore[union-attr]

    # After reset (the kernel hook does this on halt) the buffer is gone.
    reg.reset_run_state("r")
    fresh = reg.evaluate(_step(3, "loop"))
    assert all(r.verdict == "ok" for r in fresh)
    # And the result reason should reflect the fresh count.
    assert any("1/3" in r.reason for r in fresh)


def test_signal_protocol_compliance() -> None:
    """LoopDetectSignal should satisfy the runtime ``Signal`` protocol."""
    from iris_harness.kernel.governance.evaluator.types import Signal

    embedder = _StubEmbedder(vectors={})
    sig = LoopDetectSignal(embedder=embedder)
    assert isinstance(sig, Signal)
    # priority is later than the cheap signals so its embedding cost runs last
    assert sig.priority > 30
    # threshold default matches the design doc
    sentinel = math.isclose(sig._threshold, 0.92)  # type: ignore[attr-defined]
    assert sentinel


# --- the step is embedded, not the thought alone (multi-step loop plan, decision 5) ---


class _RecordingEmbedder:
    """Unit vector per distinct text; remembers every text it was asked to embed."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self._index: dict[str, int] = {}

    def __call__(self, text: str) -> list[float]:
        self.texts.append(text)
        idx = self._index.setdefault(text, len(self._index))
        vec = [0.0] * 8
        vec[idx] = 1.0
        return vec


def _action_step(step_id: int, thought: str, tool: str | None, args: str | None) -> StepRecord:
    return StepRecord(
        run_id="r",
        step_id=step_id,
        agent_type="chat",
        thought=thought,
        tool_name=tool,
        tool_args_text=args,
    )


def test_embedded_text_is_thought_plus_action_plus_args() -> None:
    embedder = _RecordingEmbedder()
    signal = LoopDetectSignal(embedder=embedder, window=3, threshold=0.92)
    signal(
        _action_step(0, "remind for each due", "create_reminder", '{"date": "2026-09-27"}'),
        state={},
    )
    assert embedder.texts == ['remind for each due\nAction: create_reminder {"date": "2026-09-27"}']


def test_step_without_tool_embeds_thought_alone() -> None:
    embedder = _RecordingEmbedder()
    signal = LoopDetectSignal(embedder=embedder, window=3, threshold=0.92)
    signal(_action_step(0, "I have enough to answer", None, None), state={})
    assert embedder.texts == ["I have enough to answer"]


def test_same_thought_over_different_items_does_not_halt() -> None:
    """A fan-out: the goal-level thought repeats, the action moves to the next item."""
    embedder = _RecordingEmbedder()
    signal = LoopDetectSignal(embedder=embedder, window=3, threshold=0.92)
    state: dict[str, Any] = {}
    verdicts = [
        signal(
            _action_step(
                i,
                "create a reminder for each due",
                "create_reminder",
                f'{{"date": "2026-09-{20 + i}"}}',
            ),
            state=state,
        ).verdict
        for i in range(4)
    ]
    assert verdicts == ["ok", "ok", "ok", "ok"]


def test_same_thought_and_same_call_still_halts() -> None:
    """The loop the signal exists for: same thought, same tool, same args, three times."""
    embedder = _RecordingEmbedder()
    signal = LoopDetectSignal(embedder=embedder, window=3, threshold=0.92)
    state: dict[str, Any] = {}
    results = [
        signal(
            _action_step(i, "find the dues", "search_inbox", '{"query": "amount due"}'), state=state
        )
        for i in range(3)
    ]
    assert [r.verdict for r in results] == ["ok", "ok", "halt"]
    assert "steps pairwise cosine" in results[-1].reason
