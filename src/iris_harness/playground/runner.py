"""Execute scenarios against a runtime and evaluate their expectations.

The runner is decoupled from ``build_runtime``: it takes a ``chat_fn`` with the
shape of ``IrisRuntime.chat`` (message, session_id, channel) -> ChatResult, so
it is trivially testable with a fake and can later be pointed at an API client.
Handler detection reads the pipeline timeline events the runtime already emits
(``phase="<name>.end"`` with ``matched=True``) via an in-process subscriber, so
no new per-handler instrumentation is required.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol

from iris_harness.foundation.observability import session_log

from .assertions import ObservedTurn, evaluate
from .models import Scenario, ScenarioResult, ScenarioSuite, SuiteResult


class _ChatResultLike(Protocol):
    """What the runner reads off a chat result.

    Read-only properties, not bare attributes. A protocol with mutable attributes
    is invariant, so the real ``ChatResult`` -- a frozen dataclass -- could not
    satisfy it, and `IrisRuntime.chat` did not type-check as a ``ChatFn``. The
    runner only reads.
    """

    @property
    def response(self) -> str: ...

    @property
    def intent(self) -> str: ...

    @property
    def agent_type(self) -> str: ...

    @property
    def sources(self) -> tuple[str, ...]: ...

    @property
    def has_errors(self) -> bool: ...

    @property
    def metadata(self) -> dict[str, Any]: ...


ChatFn = Callable[..., _ChatResultLike]


class PlaygroundRunner:
    """Run scenarios against a ``chat_fn`` and score their expectations."""

    def __init__(self, chat_fn: ChatFn) -> None:
        self._chat = chat_fn

    def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        session_id = f"playground-{scenario.name}-{uuid.uuid4().hex[:8]}"
        start = time.monotonic()
        with _env_overrides(scenario.env):
            try:
                for setup in scenario.setup_messages:
                    self._chat(setup, session_id=session_id, channel=scenario.channel)
                with _capture_events() as events:
                    result = self._chat(
                        scenario.message, session_id=session_id, channel=scenario.channel
                    )
            except Exception as exc:  # noqa: BLE001 — surface as a failed scenario
                return ScenarioResult(
                    scenario_name=scenario.name,
                    passed=False,
                    duration_ms=(time.monotonic() - start) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                )

        handler = _handler_from_events(events)
        turn = ObservedTurn(
            response=result.response,
            intent=result.intent,
            agent_type=result.agent_type,
            handler=handler,
            sources=result.sources,
            metadata=result.metadata,
            has_errors=result.has_errors,
        )
        assertions = tuple(evaluate(scenario.expect, turn))
        return ScenarioResult(
            scenario_name=scenario.name,
            passed=all(a.ok for a in assertions),
            assertions=assertions,
            response=result.response,
            intent=result.intent,
            agent_type=result.agent_type,
            handler=handler,
            sources=tuple(result.sources),
            metadata=dict(result.metadata),
            duration_ms=(time.monotonic() - start) * 1000,
        )

    def run_suite(self, suite: ScenarioSuite) -> SuiteResult:
        return SuiteResult(
            suite_name=suite.name,
            results=tuple(self.run_scenario(s) for s in suite.scenarios),
        )


@contextmanager
def _env_overrides(env: dict[str, str]) -> Iterator[None]:
    """Apply *env* for the duration, restoring the prior values on exit."""
    if not env:
        yield
        return
    sentinel = object()
    prior: dict[str, Any] = {k: os.environ.get(k, sentinel) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, old in prior.items():
            if old is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


@contextmanager
def _capture_events() -> Iterator[list[dict[str, Any]]]:
    """Collect every session_log event emitted while the block runs."""
    events: list[dict[str, Any]] = []
    unsubscribe = session_log.subscribe_events(events.append)
    try:
        yield events
    finally:
        unsubscribe()


def _handler_from_events(events: list[dict[str, Any]]) -> str | None:
    """The intercept that answered, from ``phase="<name>.end"`` + matched=True.

    Returns ``None`` when no intercept fired (the turn reached the agent loop).
    """
    for event in events:
        if event.get("kind") != "pipeline.phase":
            continue
        payload = event.get("payload") or {}
        phase = event.get("phase") or ""
        if payload.get("matched") and phase.endswith(".end"):
            return phase[: -len(".end")]
    return None
