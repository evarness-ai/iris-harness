from __future__ import annotations

from typing import Any

from iris_harness.kernel.governance.evaluator import EvaluatorHook, EvaluatorRegistry, StepRecord
from iris_harness.kernel.governance.evaluator.types import SignalResult
from iris_harness.kernel.governance.hooks.types import HookContext, HookPoint


def _ctx(
    *,
    payload: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    step_id: int | None = 3,
) -> HookContext:
    return HookContext(
        hook_point=HookPoint.POST_STEP,
        run_id="run-eval",
        agent_type="chat",
        step_id=step_id,
        payload=payload or {},
        metadata=metadata or {},
    )


class _StubSignal:
    def __init__(self, verdict: str, *, name: str = "stub", severity: str = "info") -> None:
        self.name = name
        self.priority = 10
        self._verdict = verdict
        self._severity = severity
        self.seen_step: StepRecord | None = None

    def __call__(self, step: StepRecord, *, state: dict[str, Any]) -> SignalResult:
        self.seen_step = step
        return SignalResult(
            name=self.name,
            verdict=self._verdict,  # type: ignore[arg-type]
            reason=f"verdict={self._verdict}",
            severity=self._severity,  # type: ignore[arg-type]
        )


async def test_empty_registry_allows() -> None:
    registry = EvaluatorRegistry()
    registry.init_lock()
    decision = await EvaluatorHook(registry=registry)(_ctx())
    assert decision.outcome == "allow"
    assert "no signals registered" in decision.reason


async def test_halt_signal_translates_to_deny() -> None:
    registry = EvaluatorRegistry()
    registry.register(_StubSignal("halt", severity="critical"))
    registry.init_lock()

    decision = await EvaluatorHook(registry=registry)(_ctx())
    assert decision.outcome == "deny"
    assert decision.severity == "critical"
    assert "stub -> halt" in decision.reason


async def test_require_approval_passes_through() -> None:
    registry = EvaluatorRegistry()
    registry.register(_StubSignal("require_approval"))
    registry.init_lock()

    decision = await EvaluatorHook(registry=registry)(_ctx())
    assert decision.outcome == "require_approval"


async def test_warn_is_allow_with_warn_severity_via_audit() -> None:
    registry = EvaluatorRegistry()
    registry.register(_StubSignal("warn", severity="warn"))
    registry.init_lock()

    decision = await EvaluatorHook(registry=registry)(_ctx())
    assert decision.outcome == "allow"
    assert decision.severity == "warn"
    signals_audit = decision.audit_metadata["signals"]
    assert signals_audit[0]["verdict"] == "warn"


async def test_step_record_carries_payload_and_metadata() -> None:
    signal = _StubSignal("ok")
    registry = EvaluatorRegistry()
    registry.register(signal)
    registry.init_lock()

    await EvaluatorHook(registry=registry)(
        _ctx(
            payload={
                "thought": "let me try X",
                "tool_name": "research",
                "tool_args_hash": "abc123",
                "tool_args_text": '{"query": "news"}',
            },
            metadata={"original_task": "summarize the news", "user_id": "u-1"},
            step_id=7,
        )
    )

    assert signal.seen_step is not None
    step = signal.seen_step
    assert step.run_id == "run-eval"
    assert step.step_id == 7
    assert step.agent_type == "chat"
    assert step.thought == "let me try X"
    assert step.tool_name == "research"
    assert step.tool_args_hash == "abc123"
    assert step.tool_args_text == '{"query": "news"}'
    assert step.original_task == "summarize the news"
    # Non-step metadata is preserved in the catch-all bucket.
    assert step.metadata.get("user_id") == "u-1"


async def test_terminal_verdict_resets_run_state() -> None:
    class _Counter:
        name = "counter"
        priority = 10

        def __init__(self) -> None:
            self.seen_counts: list[int] = []

        def __call__(self, step: StepRecord, *, state: dict[str, Any]) -> SignalResult:
            state["n"] = state.get("n", 0) + 1
            self.seen_counts.append(state["n"])
            # Halt on the 2nd call so the hook's reset fires.
            verdict = "halt" if state["n"] == 2 else "ok"
            return SignalResult(name=self.name, verdict=verdict, reason="cnt")  # type: ignore[arg-type]

    counter = _Counter()
    registry = EvaluatorRegistry()
    registry.register(counter)
    registry.init_lock()

    hook = EvaluatorHook(registry=registry)
    await hook(_ctx())  # n=1, ok
    await hook(_ctx())  # n=2, halt → reset_run_state runs
    await hook(_ctx())  # n=1 again because state was reset
    assert counter.seen_counts == [1, 2, 1]
