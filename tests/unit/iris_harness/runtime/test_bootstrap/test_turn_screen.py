"""The turn-level input screen: every turn's raw message passes the kernel's PRE_TURN hooks.

A deterministic handler answers a turn with no model call, and the kernel's input
screens used to fire only where text enters a model (PRE_CLASSIFY), so such a turn
was never screened (docs/architecture/deterministic-path-parity.md, step a).
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
    build_default_kernel,
)
from iris_harness.runtime import build_runtime
from iris_harness.runtime.turn import RESUME_STAGES, TurnRequest
from iris_harness.runtime.turn.stages import screen
from iris_harness.runtime.turn.state import TurnState

# "what time is it?" is answered by the system plugin's deterministic handler.
DETERMINISTIC = "what time is it?"


class _Recorder:
    """A PRE_TURN hook that records what it saw and refuses messages holding a marker."""

    name = "test_recorder"
    hook_point = HookPoint.PRE_TURN
    priority = 50

    def __init__(self, *, deny_marker: str | None = None, rewrite_to: str | None = None) -> None:
        self.seen: list[HookContext] = []
        self._deny_marker = deny_marker
        self._rewrite_to = rewrite_to

    async def __call__(self, ctx: HookContext) -> HookDecision:
        self.seen.append(ctx)
        message = str(ctx.payload.get("message", ""))
        if self._deny_marker and self._deny_marker in message:
            return HookDecision(outcome="deny", reason="test marker", severity="critical")
        if self._rewrite_to is not None:
            return HookDecision(
                outcome="transform",
                reason="test rewrite",
                transformed_payload={"message": self._rewrite_to},
            )
        return HookDecision(outcome="allow", reason="ok")


def _kernel(hook: _Recorder) -> GovernanceKernel:
    kernel = GovernanceKernel(audit_log=None)
    kernel.register(hook)
    kernel.init_lock()
    return kernel


@pytest.fixture()
def runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    config_dir = Path(__file__).resolve().parents[5] / "config"
    return build_runtime(
        config_dir=config_dir, data_dir=tmp_path / "data", use_background_scheduler=False
    )


def test_default_kernel_screens_the_turn_with_the_data_classifier() -> None:
    kernel = build_default_kernel(audit_log=None)
    assert "data_classifier" in kernel.hook_names(HookPoint.PRE_TURN)
    # Still screens what enters a model: PRE_TURN is added, not moved.
    assert "data_classifier" in kernel.hook_names(HookPoint.PRE_CLASSIFY)


def test_register_at_places_a_hook_at_another_point() -> None:
    hook = _Recorder()
    kernel = GovernanceKernel(audit_log=None)
    kernel.register(hook, at=HookPoint.PRE_CLASSIFY)
    kernel.init_lock()
    assert kernel.hook_names(HookPoint.PRE_CLASSIFY) == ("test_recorder",)
    assert kernel.hook_names(HookPoint.PRE_TURN) == ()


def test_build_runtime_hands_the_pipeline_a_kernel(runtime) -> None:  # type: ignore[no-untyped-def]
    assert runtime.governance_kernel is not None
    assert "data_classifier" in runtime.governance_kernel.hook_names(HookPoint.PRE_TURN)


def test_a_deterministic_turn_is_screened(runtime) -> None:  # type: ignore[no-untyped-def]
    hook = _Recorder()
    runtime.governance_kernel = _kernel(hook)
    result = runtime.chat(DETERMINISTIC, session_id="screen-seen")
    assert result.metadata.get("deterministic_time_date") is True
    assert [ctx.payload["message"] for ctx in hook.seen] == [DETERMINISTIC]
    assert hook.seen[0].hook_point is HookPoint.PRE_TURN


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_a_refused_message_never_reaches_the_deterministic_handler(
    runtime: Any, entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The refused message is exactly one the deterministic handler answers (see
    # test_a_deterministic_turn_is_screened), so only the screen stands between them.
    runtime.governance_kernel = _kernel(_Recorder(deny_marker=DETERMINISTIC))
    remembered: list[str] = []
    monkeypatch.setattr(
        runtime.sessions, "record_turn", lambda _sid, user, _reply: remembered.append(user)
    )
    message = DETERMINISTIC
    if entry == "chat":
        result = runtime.chat(message, session_id=f"screen-deny-{entry}")
    else:
        events = list(runtime.chat_stream(message, session_id=f"screen-deny-{entry}"))
        assert events[-1].kind == "done"
        result = events[-1].result
    assert result.response == GOVERNANCE_BLOCKED_TEXT
    assert result.metadata.get("governance_screen") == "deny"
    assert result.metadata.get("deterministic_time_date") is None
    assert result.has_errors is True
    # A refused message must not become context for the next turn's model.
    assert remembered == []


def test_the_screen_records_the_classification_it_inferred(runtime) -> None:  # type: ignore[no-untyped-def]
    state = TurnState(request=TurnRequest(message="my key is AKIAIOSFODNN7EXAMPLE"))
    list(screen.run(runtime, state))
    assert state.classification == "secret"
    assert state.screened_out is False


def test_a_transform_rewrites_the_message_the_turn_sees(runtime) -> None:  # type: ignore[no-untyped-def]
    runtime.governance_kernel = _kernel(_Recorder(rewrite_to="rewritten"))
    state = TurnState(request=TurnRequest(message="original"))
    list(screen.run(runtime, state))
    assert state.message == "rewritten"


def test_no_kernel_means_no_screen(runtime) -> None:  # type: ignore[no-untyped-def]
    runtime.governance_kernel = None
    result = runtime.chat(DETERMINISTIC, session_id="screen-off")
    assert result.metadata.get("deterministic_time_date") is True


def test_a_resume_is_not_screened_again() -> None:
    names = [name for name, _ in RESUME_STAGES]
    assert "screen" not in names and "intercept" not in names
