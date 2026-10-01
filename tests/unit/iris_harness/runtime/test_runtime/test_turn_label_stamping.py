"""The turn's data label, stamped on a plugin's code calls (``kernel/governance/turn_label.py``).

The loop feeds its own label into every model call and every tool call it makes. A plugin's
code calls -- ``api.tools`` (``ToolService``) and ``api.capability`` (the registry) -- are
governed as the turn they run in: the harness stamps the turn's label on the call, and the
plugin has no way to pass or override it. The label is a floor: a governed call's
``POST_TOOL_USE`` may raise it for the rest of the turn, never lower it, and the loop reads
the raised label before its next model call. Outside a turn nothing is stamped.
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import pytest

from iris_harness.agent.agentic_core import AgenticCore, AgenticCoreConfig, ToolSpec
from iris_harness.foundation import capabilities as catalogue
from iris_harness.foundation.capabilities import CapabilitySpec, MethodSpec
from iris_harness.foundation.observability.session_log import bind_context
from iris_harness.kernel.governance import (
    GovernanceKernel,
    HookContext,
    HookDecision,
    HookPoint,
)
from iris_harness.kernel.governance.approvals.store import ApprovalItem, ApprovalRow
from iris_harness.kernel.governance.caller_policy import register_caller_policy
from iris_harness.kernel.governance.plugins.caller_policy import CallerPolicyHook
from iris_harness.kernel.governance.turn_label import (
    current_turn_label,
    lift_turn_label,
    turn_label_scope,
)
from iris_harness.runtime.plugin_host.manifest import PluginManifest
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_harness.runtime.tool_access import compile_caller_policy
from iris_harness.runtime.tool_service import ToolService


class Spy:
    """Records the label each context at one hook point carried."""

    name = "spy"
    priority = 99

    def __init__(self, point: HookPoint) -> None:
        self.hook_point = point
        self.labels: list[str | None] = []

    async def __call__(self, ctx: HookContext) -> HookDecision:
        self.labels.append(ctx.classification)
        return HookDecision(outcome="allow", reason="spy")


class ResultLabel:
    """A POST_TOOL_USE classifier: labels the result of the named tools ``label``."""

    name = "result_label"
    hook_point = HookPoint.POST_TOOL_USE
    priority = 10

    def __init__(self, label: str, *, tools: tuple[str, ...] | None = None) -> None:
        self._label = label
        self._tools = tools

    async def __call__(self, ctx: HookContext) -> HookDecision:
        if self._tools is not None and ctx.payload.get("tool_name") not in self._tools:
            return HookDecision(outcome="allow", reason="not labelled")
        return HookDecision(outcome="allow", reason="t", set_classification=self._label)  # type: ignore[arg-type]


def _kernel(*hooks: Any) -> GovernanceKernel:
    kernel = GovernanceKernel(audit_log=None)
    for hook in hooks:
        kernel.register(hook)
    kernel.init_lock()
    return kernel


# ------------------------------------------------------------------ the label itself
def test_outside_a_turn_there_is_no_label_and_a_lift_is_a_no_op() -> None:
    lift_turn_label("secret")
    assert current_turn_label() is None


def test_the_label_only_ever_rises_within_a_turn_and_is_gone_after() -> None:
    with turn_label_scope():
        assert current_turn_label() is None
        lift_turn_label("personal")
        lift_turn_label("internal")  # lower: ignored
        lift_turn_label(None)
        assert current_turn_label() == "personal"
        lift_turn_label("secret")
        assert current_turn_label() == "secret"
    assert current_turn_label() is None


def test_a_nested_turn_starts_fresh_and_restores_the_outer_one() -> None:
    with turn_label_scope():
        lift_turn_label("personal")
        with turn_label_scope():
            assert current_turn_label() is None
            lift_turn_label("secret")
        assert current_turn_label() == "personal"


def test_a_lift_on_a_worker_thread_reaches_the_turn() -> None:
    """The executor runs tasks through ``bind_context`` (a copy of the turn's context)."""
    with turn_label_scope(), ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(bind_context(lift_turn_label), "secret").result()
        assert current_turn_label() == "secret"


# ------------------------------------------------------------------ ToolService calls
def _counter_tool(name: str = "look_up") -> ToolSpec:
    return ToolSpec(name=name, description="d", call=lambda args: "ok", effect="read")


def test_tool_service_stamps_the_turn_label_on_a_code_call() -> None:
    pre = Spy(HookPoint.PRE_TOOL_USE)
    kernel = _kernel(pre)
    tools = ToolService(tools=lambda: [_counter_tool()], kernel=lambda: kernel).for_caller(
        "plugin:p"
    )

    tools.call("look_up", {})
    with turn_label_scope():
        lift_turn_label("personal")
        tools.call("look_up", {})
    assert pre.labels == [None, "personal"]


def test_a_caller_cannot_pass_or_override_the_label() -> None:
    pre = Spy(HookPoint.PRE_TOOL_USE)
    kernel = _kernel(pre)
    tools = ToolService(tools=lambda: [_counter_tool()], kernel=lambda: kernel).for_caller(
        "plugin:p"
    )
    with turn_label_scope():
        lift_turn_label("secret")
        with pytest.raises(TypeError):
            tools.call("look_up", {}, classification="public")  # type: ignore[call-arg]
        tools.call("look_up", {"classification": "public"})  # an argument, not a label
    assert pre.labels == ["secret"]


def test_a_code_calls_result_lifts_the_turn_never_lowers_it() -> None:
    kernel = _kernel(ResultLabel("personal"))
    tools = ToolService(tools=lambda: [_counter_tool()], kernel=lambda: kernel).for_caller(
        "plugin:p"
    )
    with turn_label_scope():
        lift_turn_label("internal")
        tools.call("look_up", {})
        assert current_turn_label() == "personal"
    with turn_label_scope():
        lift_turn_label("secret")
        tools.call("look_up", {})
        assert current_turn_label() == "secret"


def test_the_approved_call_executor_stamps_nothing() -> None:
    """It runs on the owner's answer, outside the turn that queued the call."""
    pre = Spy(HookPoint.PRE_TOOL_USE)
    kernel = _kernel(pre)
    service = ToolService(tools=lambda: [_counter_tool()], kernel=lambda: kernel)
    row = ApprovalRow(
        approval_id="a1",
        run_id="r1",
        checkpoint_id=None,
        signal="s",
        context_summary="c",
        requested_at="2026-09-30T00:00:00+00:00",
        channel="web",
        status="approved",
        responded_at=None,
        response_actor=None,
        timeout_at="2026-10-01T00:00:00+00:00",
        policy_on_timeout="reject",
        items=(ApprovalItem.of("look_up", {}),),
        caller="plugin:p",
    )
    with turn_label_scope():
        lift_turn_label("secret")
        assert service.execute_approved_call(row).status == "ran"
    assert pre.labels == [None]


# ------------------------------------------------------------------ capability calls
@dataclasses.dataclass(frozen=True)
class Note:
    text: str


class Notes(Protocol):
    def get(self, q: str) -> Note: ...
    async def aget(self, q: str) -> Note: ...
    def stream(self, q: str) -> Iterator[Note]: ...
    def astream(self, q: str) -> AsyncIterator[Note]: ...


NOTES = CapabilitySpec(
    name="test.notes",
    protocol=Notes,
    methods={m: MethodSpec(fields=("text",)) for m in ("get", "aget", "stream", "astream")},
)


class Provider:
    def get(self, q: str) -> Note:
        return Note("n")

    async def aget(self, q: str) -> Note:
        return Note("n")

    def stream(self, q: str) -> Iterator[Note]:
        yield Note("n")

    async def astream(self, q: str) -> AsyncIterator[Note]:
        yield Note("n")


@pytest.fixture(autouse=True)
def _no_caller_policy_left_behind() -> Iterator[None]:
    yield
    register_caller_policy(None)


def _notes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *hooks: Any) -> Any:
    """A consumer's facade on ``test.notes``, governed by a kernel with ``hooks``."""
    monkeypatch.setattr(catalogue, "CAPABILITIES", MappingProxyType({"test.notes": NOTES}))
    registry = PluginRegistry()
    for name, caps in (("prov", {"provides": ["test.notes"]}), ("cons", {"uses": ["test.notes"]})):
        registry.add_plugin(
            PluginRecord(
                name=name,
                source="t",
                status=PluginStatus.LOADED,
                manifest=PluginManifest.model_validate({"name": name, "capabilities": caps}),
            )
        )
    assert registry.provide_capability("prov", "test.notes", Provider())
    kernel = _kernel(CallerPolicyHook(), *hooks)
    registry.bind_kernel(lambda: kernel)
    register_caller_policy(compile_caller_policy(registry, config_dir=tmp_path))
    return registry.resolve_capability("cons", "test.notes")


def _drain(facade: Any) -> list[Note]:
    async def drain() -> list[Note]:
        return [n async for n in facade.astream("q")]

    return asyncio.run(drain())


def _all_shapes(facade: Any) -> None:
    facade.get("q")
    asyncio.run(facade.aget("q"))
    list(facade.stream("q"))
    _drain(facade)


def test_every_capability_shape_is_stamped_with_the_turn_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pre = Spy(HookPoint.PRE_TOOL_USE)
    facade = _notes(tmp_path, monkeypatch, pre)
    _all_shapes(facade)  # outside a turn: nothing stamped
    with turn_label_scope():
        lift_turn_label("personal")
        _all_shapes(facade)
    assert pre.labels == [None] * 4 + ["personal"] * 4


@pytest.mark.parametrize("shape", ["get", "aget", "stream", "astream"])
def test_a_capability_result_lifts_the_turn_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    facade = _notes(tmp_path, monkeypatch, ResultLabel("secret"))
    calls = {
        "get": lambda: facade.get("q"),
        "aget": lambda: asyncio.run(facade.aget("q")),
        "stream": lambda: list(facade.stream("q")),
        "astream": lambda: _drain(facade),
    }
    with turn_label_scope():
        lift_turn_label("internal")
        calls[shape]()
        assert current_turn_label() == "secret"


def test_a_capability_result_never_lowers_the_turn_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    facade = _notes(tmp_path, monkeypatch, ResultLabel("public"))
    with turn_label_scope():
        lift_turn_label("personal")
        facade.get("q")
        assert current_turn_label() == "personal"


# ------------------------------------------------------------------ the loop reads the floor
_CALL = 'Thought: look\nAction: summarise\nAction Input: {"q": "x"}'
_DONE = "Thought: done\nFinal Answer: Done."


class _Scripted:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def __call__(self, prompt: str) -> str:
        return self._responses.pop(0) if self._responses else _DONE

    def stream(self, prompt: str) -> Iterator[str]:  # pragma: no cover - not used
        yield self(prompt)


class _QuestionLabel:
    name = "question_label"
    hook_point = HookPoint.PRE_CLASSIFY
    priority = 10

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="t", set_classification="public")


@pytest.mark.parametrize("entry", ["run", "run_stream"])
def test_the_loop_egresses_under_the_label_a_code_call_earned(entry: str) -> None:
    """The loop's own tool ``summarise`` is plugin code that calls ``vault_note`` through
    ``api.tools``. Only ``vault_note``'s result is secret, and the loop never sees that
    call's outcome -- the turn's label is how its next model call learns of it."""
    egress = Spy(HookPoint.PRE_LLM_CALL)
    kernel = _kernel(_QuestionLabel(), ResultLabel("secret", tools=("vault_note",)), egress)
    service = ToolService(tools=lambda: [_counter_tool("vault_note")], kernel=lambda: kernel)
    plugin_tools = service.for_caller("plugin:notes")
    summarise = ToolSpec(
        name="summarise",
        description="d",
        call=lambda args: plugin_tools.call("vault_note", {}).text,
    )
    core = AgenticCore(
        config=AgenticCoreConfig(max_iterations=4),
        llm_call=_Scripted([_CALL, _DONE]),
        tools=[summarise],
        kernel=kernel,
    )
    with turn_label_scope():
        if entry == "run":
            core.run("summarise my note")
        else:
            list(core.run_stream("summarise my note"))
    assert egress.labels[0] == "public" and egress.labels[-1] == "secret", egress.labels


# ------------------------------------------------------------------ chat and chat_stream
class _TurnLabel:
    """PRE_TURN: labels what the user said ``personal``."""

    name = "turn_label"
    hook_point = HookPoint.PRE_TURN
    priority = 10

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="t", set_classification="personal")


@pytest.fixture()
def runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    from iris_harness.runtime import build_runtime

    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    config_dir = Path(__file__).resolve().parents[5] / "config"
    return build_runtime(
        config_dir=config_dir, data_dir=tmp_path / "data", use_background_scheduler=False
    )


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_both_chat_paths_publish_the_turn_label_and_reset_it(
    runtime: Any, entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime.governance_kernel = _kernel(_TurnLabel())
    during: list[str | None] = []
    monkeypatch.setattr(
        runtime.sessions,
        "record_turn",
        lambda *_args, **_kw: during.append(current_turn_label()),
    )
    if entry == "chat":
        runtime.chat("what time is it?", session_id="label-chat")
    else:
        # Stepped the way the server steps it: each ``next`` on a worker thread, in a fresh
        # copy of the caller's context (Starlette's iterate_in_threadpool).
        stream = runtime.chat_stream("what time is it?", session_id="label-stream")
        with ThreadPoolExecutor(max_workers=1) as pool:
            while True:
                try:
                    pool.submit(contextvars.copy_context().run, next, stream).result()
                except StopIteration:
                    break
    assert during == ["personal"]
    assert current_turn_label() is None
