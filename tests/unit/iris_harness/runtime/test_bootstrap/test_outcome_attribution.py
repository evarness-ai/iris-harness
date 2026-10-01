"""Cross-turn user_correction attribution (§4.2).

Drives ``TurnCapture.evaluate_prior_turn_outcome`` directly with a lightweight
stand-in for ``self`` so the cross-turn logic is testable without building a full
runtime — the model-driven detector + confidence floor + correlation-to-prior-turn.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from iris_harness.agent.outcomes import CorrectionVerdict, OutcomeConfig
from iris_harness.memory.retriever import MemoryContext
from iris_harness.memory.semantic_index import RetrievedTurn
from iris_harness.runtime.bootstrap import IrisRuntime
from iris_harness.runtime.turn_capture import TurnCapture


class _RecordingCollector:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record_metric(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _runtime(detector, collector, *, floor: float = 0.6) -> Any:
    """A capture on a stand-in host, with its detector and config stubbed on the
    instance (the resolution the methods had as mixin members of a stand-in runtime)."""
    rt = TurnCapture(SimpleNamespace(signal_collector=collector))  # type: ignore[arg-type]
    rt._last_turn_outcome = {
        "s1": {
            "turn_id": "t0",
            "trace_id": "tr0",
            "span_id": "sp0",
            "resolved_tier": "tier1",
            "resolved_agent": "general",
            "intent": "general",
            "query": "what is the capital of australia?",
            "response": "Sydney.",
            "has_errors": False,
        }
    }
    rt._correction_detector = lambda: detector  # type: ignore[method-assign]
    rt._outcome_config = lambda: OutcomeConfig(  # type: ignore[method-assign]
        user_correction_enabled=True, correction_confidence_floor=floor
    )
    return rt


def test_correction_recorded_against_prior_turn() -> None:
    detector = lambda **_kw: CorrectionVerdict(  # noqa: E731
        is_correction=True, confidence=0.9, reason="re-ask"
    )
    collector = _RecordingCollector()
    rt = _runtime(detector, collector)

    rt.evaluate_prior_turn_outcome("s1", "no, it's Canberra")

    assert len(collector.calls) == 1
    call = collector.calls[0]
    assert call["metric_name"] == "user_correction"
    assert call["value"] == 1.0
    assert call["success"] is False
    # Correlated to the PRIOR turn that produced the outcome.
    assert call["turn_id"] == "t0"
    assert call["resolved_tier"] == "tier1"
    # Descriptor consumed (evaluated at most once).
    assert rt._last_turn_outcome.get("s1") is None


def test_below_confidence_floor_records_zero() -> None:
    detector = lambda **_kw: CorrectionVerdict(  # noqa: E731
        is_correction=True, confidence=0.3, reason="maybe"
    )
    collector = _RecordingCollector()
    rt = _runtime(detector, collector, floor=0.6)

    rt.evaluate_prior_turn_outcome("s1", "no, it's Canberra")
    assert collector.calls[0]["value"] == 0.0  # below floor → not a correction


def test_prefilter_skips_obvious_non_correction() -> None:
    called = {"n": 0}

    def detector(**_kw: Any) -> CorrectionVerdict:
        called["n"] += 1
        return CorrectionVerdict(is_correction=True, confidence=1.0)

    collector = _RecordingCollector()
    rt = _runtime(detector, collector)

    # No correction cue → pre-filter skips the LLM call entirely, no signal.
    rt.evaluate_prior_turn_outcome("s1", "thanks, what about New Zealand?")
    assert called["n"] == 0
    assert collector.calls == []


def test_no_detector_emits_nothing() -> None:
    collector = _RecordingCollector()
    rt = _runtime(detector=None, collector=collector)
    rt._correction_detector = lambda: None
    rt.evaluate_prior_turn_outcome("s1", "no, that's wrong")
    assert collector.calls == []


def test_no_prior_turn_is_a_noop() -> None:
    collector = _RecordingCollector()
    rt = _runtime(detector=lambda **_kw: None, collector=collector)
    rt._last_turn_outcome = {}
    rt.evaluate_prior_turn_outcome("s1", "no, that's wrong")
    assert collector.calls == []


# --- downstream_reuse (§4.2) ---


def test_downstream_reuse_recorded_for_recalled_turns() -> None:
    collector = _RecordingCollector()
    rt = SimpleNamespace(signal_collector=collector)
    ctx = MemoryContext(
        reused_turn_refs=(
            RetrievedTurn(
                row_id="42",
                session_id="old_sess",
                role="assistant",
                content="Canberra.",
                turn_id="t-old",
            ),
        )
    )
    IrisRuntime._record_downstream_reuse(rt, ctx)

    assert len(collector.calls) == 1
    call = collector.calls[0]
    assert call["metric_name"] == "downstream_reuse"
    assert call["value"] == 1.0
    # Correlated to the REUSED prior turn.
    assert call["turn_id"] == "t-old"
    assert call["session_id"] == "old_sess"
    assert call["metadata"]["reused_row_id"] == "42"


def test_downstream_reuse_noop_without_refs() -> None:
    collector = _RecordingCollector()
    rt = SimpleNamespace(signal_collector=collector)
    IrisRuntime._record_downstream_reuse(rt, MemoryContext())
    assert collector.calls == []


# --- pointer recall mode telemetry (IRIS_MEMORY_RECALL_MODE=pointer) ---


def test_a_recall_pointer_is_recorded_with_its_sessions_only() -> None:
    collector = _RecordingCollector()
    rt = SimpleNamespace(signal_collector=collector)
    IrisRuntime._record_downstream_reuse(
        rt, MemoryContext(recall_pointer_sessions=("car-sess", "dentist-sess"))
    )
    [call] = collector.calls
    assert call["metric_name"] == "recall_pointer_offered"
    assert call["value"] == 2.0
    assert call["metadata"]["sessions"] == ["car-sess", "dentist-sess"]


def test_the_backstop_is_recorded_beside_its_reused_turns() -> None:
    collector = _RecordingCollector()
    rt = SimpleNamespace(signal_collector=collector)
    ctx = MemoryContext(
        recall_backstop=True,
        reused_turn_refs=(
            RetrievedTurn(row_id="7", session_id="old", role="user", content="x", turn_id="t"),
        ),
    )
    IrisRuntime._record_downstream_reuse(rt, ctx)
    assert [c["metric_name"] for c in collector.calls] == ["recall_backstop", "downstream_reuse"]
