"""A deterministic handler's answer passes the same PRE_RESPONSE check as a generated one.

Step c of docs/architecture/deterministic-path-parity.md. An intercept hit used to
``break`` out of the pipeline before ``curate``, so a deterministic answer skipped every
response check and was recorded by a second, separate writer. Now stages declare the
turns they serve: a handled turn skips the model path and runs ``guard`` and
``record``, and nothing breaks out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.response_curator import GOVERNANCE_BLOCKED_TEXT
from iris_harness.kernel.governance import (
    GovernanceKernel,
    HookContext,
    HookDecision,
    HookPoint,
)
from iris_harness.kernel.governance.plugins.response_safety import ResponseSafetyHook
from iris_harness.runtime import build_runtime
from iris_harness.runtime.intercepts import InterceptHit, InterceptSpec
from iris_harness.runtime.turn import STAGES, TurnRequest
from iris_harness.runtime.turn.pipeline import serves
from iris_harness.runtime.turn.state import TurnState
from iris_harness.runtime.types import ChatResult

# "what time is it?" is answered by the system plugin's deterministic handler.
DETERMINISTIC = "what time is it?"


@pytest.fixture()
def runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    config_dir = Path(__file__).resolve().parents[5] / "config"
    return build_runtime(
        config_dir=config_dir, data_dir=tmp_path / "data", use_background_scheduler=False
    )


class _Spy:
    """A PRE_RESPONSE hook that records what it is shown."""

    name = "spy"
    hook_point = HookPoint.PRE_RESPONSE
    priority = 5

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def __call__(self, ctx: HookContext) -> HookDecision:
        self.payloads.append(dict(ctx.payload))
        return HookDecision(outcome="allow", reason="seen")


def _spy_kernel(spy: _Spy) -> GovernanceKernel:
    kernel = GovernanceKernel(audit_log=None)
    kernel.register(spy)
    kernel.register(ResponseSafetyHook())
    kernel.init_lock()
    return kernel


def _answer_with(runtime: Any, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """Make the deterministic handler chain answer every message with ``text``."""
    spec = InterceptSpec(name="test_handler", handler="test")
    result = ChatResult(text, "system", "system", ("system",), False, None, {})
    monkeypatch.setattr(
        runtime.intercepts,
        "dispatch",
        lambda *_a, **_k: InterceptHit(spec=spec, result=result),
    )


# -- the audiences --------------------------------------------------------------------


def test_a_handled_turn_skips_the_model_path_and_runs_guard_and_record() -> None:
    state = TurnState(request=TurnRequest("m"))
    state.intercepted = True
    ran = [name for name, _ in STAGES if serves(name, state)]
    assert ran == ["screen", "intercept", "guard", "notice", "record"]


def test_a_generated_turn_runs_the_model_path_and_not_guard() -> None:
    state = TurnState(request=TurnRequest("m"))
    ran = [name for name, _ in STAGES if serves(name, state)]
    assert "guard" not in ran and "curate" in ran and "record" in ran


def test_a_refused_turn_runs_record_alone() -> None:
    state = TurnState(request=TurnRequest("m"))
    state.screened_out = True
    assert [name for name, _ in STAGES if serves(name, state)] == ["record"]


# -- the guard on a deterministic answer ----------------------------------------------


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_an_unsafe_deterministic_answer_is_blocked(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    _answer_with(runtime, monkeypatch, "your SSN is 123-45-6789")
    if entry == "chat":
        result = runtime.chat(DETERMINISTIC, session_id=f"guard-{entry}")
    else:
        events = list(runtime.chat_stream(DETERMINISTIC, session_id=f"guard-{entry}"))
        assert events[-1].kind == "done"
        result = events[-1].result
        # The unsafe text never reached the stream as tokens either.
        assert not any("123-45-6789" in (e.text or "") for e in events)
    assert result.response == GOVERNANCE_BLOCKED_TEXT
    assert result.metadata.get("governance_guard") == "halt"
    assert result.has_errors is True


def test_a_clean_deterministic_answer_passes_unchanged(runtime) -> None:  # type: ignore[no-untyped-def]
    result = runtime.chat(DETERMINISTIC, session_id="guard-clean")
    assert result.metadata.get("deterministic_time_date") is True
    assert result.response.startswith("Current local time")


def test_every_deterministic_turn_fires_pre_response_once_marked(runtime) -> None:  # type: ignore[no-untyped-def]
    spy = _Spy()
    runtime.response_curator._kernel = _spy_kernel(spy)
    runtime.chat(DETERMINISTIC, session_id="guard-marked")
    assert len(spy.payloads) == 1
    assert spy.payloads[0]["deterministic"] is True
    assert spy.payloads[0]["handler"]  # the spec name of the handler that answered


def test_the_kernel_audits_the_deterministic_marker() -> None:
    rows: list[dict[str, Any]] = []

    class _Audit:
        def record(self, **kwargs: Any) -> None:
            rows.append(kwargs)

    kernel = GovernanceKernel(audit_log=_Audit())  # type: ignore[arg-type]
    kernel.register(ResponseSafetyHook())
    kernel.init_lock()
    kernel.fire_sync(
        HookPoint.PRE_RESPONSE,
        HookContext(
            hook_point=HookPoint.PRE_RESPONSE,
            run_id="r",
            agent_type="chat",
            payload={"response": "ok", "deterministic": True, "handler": "time_date"},
        ),
    )
    assert rows[0]["payload"]["deterministic"] is True
    assert rows[0]["payload"]["handler"] == "time_date"


# -- one recorder ---------------------------------------------------------------------


def test_a_deterministic_turn_is_recorded_once(runtime) -> None:  # type: ignore[no-untyped-def]
    """Counted in the session log itself, so a second writer anywhere would show."""
    import json

    from iris_harness.foundation.observability.session_log import session_log_path

    runtime.chat(DETERMINISTIC, session_id="guard-once")
    path = session_log_path("guard-once")
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [e["kind"] for e in events].count("agent_response") == 1
