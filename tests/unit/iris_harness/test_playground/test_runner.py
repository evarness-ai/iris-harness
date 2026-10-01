"""Runner tests with a fake chat_fn — no runtime, no LLM.

The fake emits the same pipeline timeline events the real runtime does, so we
exercise handler-detection, env overrides, setup messages, and error handling
end-to-end without building a runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from iris_harness.foundation.observability import session_log
from iris_harness.playground.models import Scenario, ScenarioExpectation
from iris_harness.playground.runner import PlaygroundRunner


@dataclass
class _FakeChatResult:
    response: str = "ok"
    intent: str = "general"
    agent_type: str = "general"
    sources: tuple[str, ...] = ()
    has_errors: bool = False
    error_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _emit_intercept(phase_name: str) -> None:
    # Mirrors how the runtime tags an intercept that answered a turn.
    session_log.log_timeline_event(
        "pipeline.phase", phase=f"{phase_name}.end", payload={"matched": True}
    )


def test_runner_detects_handler_from_events() -> None:
    def chat(message: str, *, session_id: str, channel: str) -> _FakeChatResult:
        # Events only fan out inside a session_scope.
        with session_log.session_scope(session_id):
            _emit_intercept("dues_request")
        return _FakeChatResult(response="You have 1 insurance due.", intent="finance")

    scenario = Scenario(
        name="dues",
        message="any insurance dues this week?",
        expect=ScenarioExpectation(handler="dues_request", intent="finance"),
    )
    result = PlaygroundRunner(chat).run_scenario(scenario)
    assert result.passed is True
    assert result.handler == "dues_request"


def test_runner_reports_no_handler_for_agent_path() -> None:
    def chat(message: str, *, session_id: str, channel: str) -> _FakeChatResult:
        with session_log.session_scope(session_id):
            pass  # no intercept fired
        return _FakeChatResult(response="hi", intent="general")

    scenario = Scenario(
        name="agent-path", message="tell me a joke", expect=ScenarioExpectation(handler="")
    )
    result = PlaygroundRunner(chat).run_scenario(scenario)
    assert result.passed is True
    assert result.handler is None


def test_runner_applies_env_overrides_during_turn() -> None:
    os.environ.pop("IRIS_PLAYGROUND_TESTFLAG", None)
    seen: dict[str, str | None] = {}

    def chat(message: str, *, session_id: str, channel: str) -> _FakeChatResult:
        seen["value"] = os.environ.get("IRIS_PLAYGROUND_TESTFLAG")
        return _FakeChatResult()

    scenario = Scenario(
        name="env",
        message="x",
        env={"IRIS_PLAYGROUND_TESTFLAG": "1"},
    )
    PlaygroundRunner(chat).run_scenario(scenario)
    assert seen["value"] == "1"
    # Restored after the turn.
    assert os.environ.get("IRIS_PLAYGROUND_TESTFLAG") is None


def test_runner_runs_setup_messages_first() -> None:
    calls: list[str] = []

    def chat(message: str, *, session_id: str, channel: str) -> _FakeChatResult:
        calls.append(message)
        return _FakeChatResult()

    scenario = Scenario(
        name="setup",
        message="the real one",
        setup_messages=("first", "second"),
    )
    PlaygroundRunner(chat).run_scenario(scenario)
    assert calls == ["first", "second", "the real one"]


def test_runner_captures_exception_as_failure() -> None:
    def chat(message: str, *, session_id: str, channel: str) -> _FakeChatResult:
        raise RuntimeError("boom")

    scenario = Scenario(name="err", message="x")
    result = PlaygroundRunner(chat).run_scenario(scenario)
    assert result.passed is False
    assert result.error is not None
    assert "boom" in result.error


def test_subscriber_unsubscribes_after_turn() -> None:
    before = len(session_log._subscribers)

    def chat(message: str, *, session_id: str, channel: str) -> _FakeChatResult:
        return _FakeChatResult()

    PlaygroundRunner(chat).run_scenario(Scenario(name="s", message="x"))
    assert len(session_log._subscribers) == before
