"""Grounding judge (Phase 5 P2) wired into ResponseCurator.

No retrieved context → skipped (non-RAG answers never penalized). With context:
LLM judge if configured, else strict-mode heuristic. Unsupported → retry; a
guard failure → warn; never halt.
"""

from __future__ import annotations

from iris_harness.agent.agent_executor import AgentResult
from iris_harness.agent.response_curator import ResponseCurator

CONTEXT = "[research]\nThe Eiffel Tower is 330 metres tall and located in Paris."


class _Judge:
    """Mock GroundingJudgeClient: returns a fixed JSON payload, or raises."""

    def __init__(self, payload: str | None = None, *, exc: Exception | None = None) -> None:
        self._payload = payload
        self._exc = exc
        self.calls: list[tuple[str, str, str]] = []

    async def judge(self, *, query: str, response: str, retrieved_context: str) -> str:
        if self._exc is not None:
            raise self._exc
        self.calls.append((query, response, retrieved_context))
        assert self._payload is not None
        return self._payload


def _curator(judge: _Judge | None = None) -> ResponseCurator:
    return ResponseCurator(grounding_judge=judge)


def _judge(curator: ResponseCurator, text: str, *, context: str = CONTEXT, strict: bool = False):
    return curator._judge_grounding(
        text,
        query="how tall is the eiffel tower",
        metadata={"retrieved_context": context} if context else {},
        strict=strict,
    )


def test_no_context_skips() -> None:
    sig = _judge(_curator(), "anything", context="")
    assert sig.verdict == "skipped"
    assert "no retrieved context" in sig.reason


def test_no_judge_non_strict_skips() -> None:
    sig = _judge(_curator(None), "The tower is 330m.", strict=False)
    assert sig.verdict == "skipped"


def test_heuristic_strict_pass_on_overlap() -> None:
    sig = _judge(_curator(None), "The Eiffel Tower is 330 metres tall in Paris.", strict=True)
    assert sig.verdict == "pass"
    assert sig.metadata.get("judge_mode") == "heuristic"


def test_heuristic_strict_retry_on_low_overlap() -> None:
    sig = _judge(_curator(None), "Bananas are an excellent source of potassium.", strict=True)
    assert sig.verdict == "retry"
    assert sig.retryable is True


def test_llm_judge_grounded_passes() -> None:
    judge = _Judge('{"grounded": true, "confidence": 0.95, "unsupported": ""}')
    sig = _judge(_curator(judge), "The Eiffel Tower is 330m tall.")
    assert sig.verdict == "pass"
    assert sig.metadata.get("judge_mode") == "llm"
    assert judge.calls and judge.calls[0][2] == CONTEXT  # context passed through


def test_llm_judge_ungrounded_retries() -> None:
    judge = _Judge('{"grounded": false, "confidence": 0.9, "unsupported": "height of 500m"}')
    sig = _judge(_curator(judge), "The Eiffel Tower is 500m tall.")
    assert sig.verdict == "retry"
    assert "500m" in sig.reason


def test_llm_judge_failure_warns_not_halts() -> None:
    sig = _judge(_curator(_Judge(exc=RuntimeError("judge down"))), "x")
    assert sig.verdict == "warn"  # never halt


def test_llm_judge_unparseable_warns() -> None:
    sig = _judge(_curator(_Judge("not json")), "x")
    assert sig.verdict == "warn"


def test_curate_grounding_skipped_without_context() -> None:
    curated = _curator(_Judge('{"grounded": true, "confidence": 1.0, "unsupported": ""}')).curate(
        [AgentResult(agent_type="chat", output="hello there", success=True)],
        query="hi",
    )
    bundle = curated.metadata["judge_bundle"]
    grounding = [s for s in bundle["signals"] if s["name"] == "grounding"]
    assert grounding and grounding[0]["verdict"] == "skipped"


def test_curate_grounding_runs_with_context() -> None:
    judge = _Judge('{"grounded": true, "confidence": 0.9, "unsupported": ""}')
    curated = _curator(judge).curate(
        [
            AgentResult(
                agent_type="chat",
                output="The Eiffel Tower is 330m tall.",
                success=True,
                metadata={"retrieved_context": CONTEXT},
            )
        ],
        query="how tall is the eiffel tower",
    )
    bundle = curated.metadata["judge_bundle"]
    grounding = [s for s in bundle["signals"] if s["name"] == "grounding"]
    assert grounding and grounding[0]["verdict"] == "pass"
