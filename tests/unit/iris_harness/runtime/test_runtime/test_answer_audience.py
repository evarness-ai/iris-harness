"""Who reads the answer reaches PRE_RESPONSE on every path (ADR-0125).

The chat routes take ``audience`` (``/chat`` and ``/chat/stream``), both facades put it on
the turn, the pipeline publishes it for the turn, and the curator reads it when it builds
the PRE_RESPONSE payload -- for a generated answer (``curate``) and a deterministic one
(``guard``). Outside a turn, and by default, the audience is the owner.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from iris_harness.agent.agent_executor import AgentResult
from iris_harness.agent.response_curator import ResponseCurator
from iris_harness.foundation.auth import auth_headers
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.hooks.response_payload import (
    audience_scope,
    current_audience,
)
from iris_harness.runtime import facade as facade_mod
from iris_harness.runtime.facade import IrisRuntime
from iris_harness.runtime.turn import TurnRequest, run_turn
from iris_harness.runtime.turn.state import TurnState
from iris_harness.runtime.types import ChatResult, StreamEvent
from iris_harness.server.iris_api.main import create_app

_RESULT = ChatResult("ok", "general", "general", (), False, None, {})


# -- the pipeline publishes the turn's audience ----------------------------------------


@pytest.mark.parametrize("audience", ["owner", "other"])
def test_the_turn_publishes_its_audience(audience: Any) -> None:
    seen: list[str] = []

    def record(_runtime: Any, state: TurnState) -> Iterator[StreamEvent]:
        seen.append(current_audience())
        state.result = _RESULT
        yield StreamEvent(kind="activity", text="record")

    request = TurnRequest(message="hi", audience=audience)
    events = list(run_turn(None, request, stages=(("record", record),)))  # type: ignore[arg-type]
    assert events[-1].kind == "done"
    assert seen == [audience]
    assert current_audience() == "owner"  # restored after the turn


def test_the_default_turn_is_the_owners() -> None:
    assert TurnRequest(message="hi").audience == "owner"


# -- both facades put it on the turn ---------------------------------------------------


def test_chat_and_chat_stream_carry_the_audience(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[TurnRequest] = []

    def fake_run_turn(_rt: Any, request: TurnRequest, **_kw: Any) -> Iterator[StreamEvent]:
        requests.append(request)
        yield StreamEvent(kind="done", result=_RESULT)

    monkeypatch.setattr(facade_mod, "run_turn", fake_run_turn)
    fake = SimpleNamespace(tracer=None)
    fake._chat_stream_events = lambda request: IrisRuntime._chat_stream_events(fake, request)  # type: ignore[attr-defined]
    IrisRuntime.chat(fake, "hi", audience="other")  # type: ignore[arg-type]
    list(IrisRuntime.chat_stream(fake, "hi", audience="other"))  # type: ignore[arg-type]
    IrisRuntime.chat(fake, "hi")  # type: ignore[arg-type]
    assert [r.audience for r in requests] == ["other", "other", "owner"]


# -- the curator reads it on both answer paths -----------------------------------------


class _Spy:
    name = "spy"
    hook_point = HookPoint.PRE_RESPONSE
    priority = 5

    def __init__(self) -> None:
        self.audiences: list[str] = []

    async def __call__(self, ctx: HookContext) -> HookDecision:
        self.audiences.append(str(ctx.payload.get("audience")))
        return HookDecision(outcome="allow", reason="seen")


def _curator() -> tuple[ResponseCurator, _Spy]:
    spy = _Spy()
    kernel = GovernanceKernel(audit_log=None)
    kernel.register(spy)
    kernel.init_lock()
    return ResponseCurator(kernel=kernel), spy


def test_the_curator_reads_the_turns_audience_on_both_paths() -> None:
    curator, spy = _curator()
    results = [AgentResult(agent_type="system", output="the answer", success=True)]
    curator.curate(results, query="q")
    curator.guard("the answer", handler="h")
    with audience_scope("other"):
        curator.curate(results, query="q")
        curator.guard("the answer", handler="h")
    assert spy.audiences == ["owner", "owner", "other", "other"]


# -- the API takes it on both chat routes ----------------------------------------------


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.sessions = self

    def context_health(self, session_id: str) -> dict[str, Any]:
        return {}

    def chat(self, message: str, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return _RESULT

    def chat_stream(self, message: str, **kwargs: Any) -> Iterator[StreamEvent]:
        self.calls.append(kwargs)
        yield StreamEvent(kind="done", result=_RESULT)


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
def test_both_chat_routes_forward_the_audience(path: str) -> None:
    runtime = _Runtime()
    app = create_app(runtime=runtime)
    with TestClient(app, headers=auth_headers()) as client:
        ok = client.post(path, json={"message": "hi", "audience": "other"})
        ok.read()
        default = client.post(path, json={"message": "hi"})
        default.read()
        bad = client.post(path, json={"message": "hi", "audience": "everyone"})
    assert ok.status_code == 200 and default.status_code == 200
    assert bad.status_code == 422
    assert [c["audience"] for c in runtime.calls] == ["other", "owner"]
