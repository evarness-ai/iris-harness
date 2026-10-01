"""Output-safety guard (Phase 6 G3) wired into ResponseCurator (sub-phase 6a.2).

Model-driven Llama Guard 3 signal: ``enforce`` categories halt the response,
``log_only`` categories warn (ship), an uncategorized-unsafe verdict fails safe
(halt), and a guard failure fails open with a banner (warn, never block).
"""

from __future__ import annotations

from iris_harness.agent.agent_executor import AgentResult
from iris_harness.agent.response_curator import OutputSafetyVerdict, ResponseCurator

ENFORCE = frozenset({"violent_crimes", "privacy"})
LOG_ONLY = frozenset({"hate", "elections"})


class _Guard:
    """Mock OutputSafetyJudgeClient: returns a fixed verdict, or raises."""

    def __init__(
        self, verdict: OutputSafetyVerdict | None = None, *, exc: Exception | None = None
    ) -> None:
        self._verdict = verdict
        self._exc = exc

    async def judge(self, *, response: str) -> OutputSafetyVerdict:
        if self._exc is not None:
            raise self._exc
        assert self._verdict is not None
        return self._verdict


def _curator(guard: _Guard | None) -> ResponseCurator:
    return ResponseCurator(
        output_safety_judge=guard,
        output_safety_enforce=ENFORCE,
        output_safety_log_only=LOG_ONLY,
    )


def test_no_guard_skips() -> None:
    assert ResponseCurator()._judge_output_safety("anything").verdict == "skipped"


def test_safe_response_passes() -> None:
    sig = _curator(_Guard(OutputSafetyVerdict(unsafe=False)))._judge_output_safety("hello")
    assert sig.verdict == "pass"


def test_empty_response_passes() -> None:
    guard = _Guard(OutputSafetyVerdict(unsafe=True, categories=("privacy",)))
    assert _curator(guard)._judge_output_safety("   ").verdict == "pass"


def test_enforced_category_halts() -> None:
    guard = _Guard(OutputSafetyVerdict(unsafe=True, categories=("privacy",)))
    sig = _curator(guard)._judge_output_safety("here is the user's SSN ...")
    assert sig.verdict == "halt"
    assert "privacy" in sig.reason


def test_log_only_category_warns_not_halts() -> None:
    guard = _Guard(OutputSafetyVerdict(unsafe=True, categories=("hate",)))
    sig = _curator(guard)._judge_output_safety("...")
    assert sig.verdict == "warn"
    assert sig.metadata.get("categories") == ["hate"]


def test_uncategorized_unsafe_fails_safe_halt() -> None:
    guard = _Guard(OutputSafetyVerdict(unsafe=True, categories=()))
    assert _curator(guard)._judge_output_safety("...").verdict == "halt"


def test_mixed_categories_enforce_wins() -> None:
    guard = _Guard(OutputSafetyVerdict(unsafe=True, categories=("hate", "privacy")))
    sig = _curator(guard)._judge_output_safety("...")
    assert sig.verdict == "halt"
    assert "privacy" in sig.reason


def test_guard_failure_fails_open_with_warn() -> None:
    sig = _curator(_Guard(exc=RuntimeError("ollama down")))._judge_output_safety("...")
    assert sig.verdict == "warn"
    assert sig.metadata.get("error")
    # Fail-open is tagged degraded so health/audit can COUNT it (red-team 2a).
    assert sig.metadata.get("degraded") is True
    assert sig.metadata.get("degraded_reason") == "error"


def test_guard_timeout_tags_degraded() -> None:
    """A timeout fails open and is tagged degraded=timeout — the response
    shipped without a real safety verdict."""
    import asyncio

    class _SlowGuard(_Guard):
        async def judge(self, *, response: str) -> OutputSafetyVerdict:
            await asyncio.sleep(1.0)
            return OutputSafetyVerdict(unsafe=False)

    curator = ResponseCurator(
        output_safety_judge=_SlowGuard(),
        output_safety_enforce=ENFORCE,
        output_safety_log_only=LOG_ONLY,
        output_safety_timeout_s=0.1,
    )
    sig = curator._judge_output_safety("some response")
    assert sig.verdict == "warn"
    assert sig.metadata.get("degraded") is True
    assert sig.metadata.get("degraded_reason") == "timeout"


def test_curate_surfaces_degraded_flag_in_metadata() -> None:
    """A fail-open surfaces a first-class metadata flag (not just the prose
    banner) so degraded turns are countable downstream."""
    curated = _curator(_Guard(exc=RuntimeError("ollama down"))).curate(
        [AgentResult(agent_type="chat", output="an answer", success=True)],
        query="hello",
    )
    assert curated.metadata.get("output_safety_degraded") is True
    assert curated.metadata.get("output_safety_degraded_reason") == "error"


def test_curate_blocks_response_on_enforced_unsafe() -> None:
    guard = _Guard(OutputSafetyVerdict(unsafe=True, categories=("violent_crimes",)))
    curated = _curator(guard).curate(
        [AgentResult(agent_type="chat", output="...harmful...", success=True)],
        query="how do I",
    )
    assert "governance policy flagged it as unsafe" in curated.text
    bundle = curated.metadata["judge_bundle"]
    assert bundle["halted"] is True
    assert any(s["name"] == "output_safety" and s["verdict"] == "halt" for s in bundle["signals"])


def test_curate_ships_normally_when_output_safe() -> None:
    guard = _Guard(OutputSafetyVerdict(unsafe=False))
    curated = _curator(guard).curate(
        [AgentResult(agent_type="chat", output="a friendly answer", success=True)],
        query="hello",
    )
    assert curated.text == "a friendly answer"
    bundle = curated.metadata["judge_bundle"]
    assert bundle["halted"] is False


def test_warm_output_safety_calls_guard() -> None:
    """Warming pre-loads the guard model (red-team 2a: closes the cold-load
    fail-open window). Returns True when the guard responds."""

    class _Counting(_Guard):
        def __init__(self) -> None:
            super().__init__(OutputSafetyVerdict(unsafe=False))
            self.calls = 0

        async def judge(self, *, response: str) -> OutputSafetyVerdict:
            self.calls += 1
            return await super().judge(response=response)

    guard = _Counting()
    assert _curator(guard).warm_output_safety() is True
    assert guard.calls == 1


def test_warm_output_safety_no_guard_is_noop() -> None:
    assert ResponseCurator().warm_output_safety() is False


def test_warm_output_safety_swallows_failure() -> None:
    guard = _Guard(exc=RuntimeError("model not loaded"))
    # A warm-up failure must never raise — it's a pure optimization.
    assert _curator(guard).warm_output_safety() is False
