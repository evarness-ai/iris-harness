"""No vault master key, no governed call (the owner's decision, 2026-09-30).

Every governed call's audit rows carry a keyed digest (``kernel/governance/audit/digest.py``).
Without a key there is nothing to key it with, and an unkeyed or empty digest is never
written in its place: the call is refused before it runs. A tool call through the governed
runner comes back ``refused`` (the loop and ``ToolService`` hand on the message), a capability
call raises ``CapabilityDenied``. The tool, the provider, is never invoked.

Also pinned here: ``args_digest`` is of the arguments as the caller wrote them, never a
hook's rewrite (the credential broker resolves a ``vault://`` handle into the secret).
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import pytest

from iris_harness.agent.agentic_core import AgenticCore, ToolSpec
from iris_harness.agent.tool_runner import GovernedToolRunner, ToolCall
from iris_harness.foundation import capabilities as catalogue
from iris_harness.foundation.capabilities import CapabilityDenied, CapabilitySpec, MethodSpec
from iris_harness.kernel.governance import (
    GovernanceKernel,
    HookContext,
    HookDecision,
    HookPoint,
)
from iris_harness.kernel.governance.audit.digest import NO_AUDIT_KEY_MESSAGE, audit_digester
from iris_harness.kernel.governance.caller_policy import register_caller_policy
from iris_harness.kernel.governance.plugins.caller_policy import CallerPolicyHook
from iris_harness.kernel.governance.plugins.credential_broker import CredentialBroker
from iris_harness.runtime.plugin_host.manifest import PluginManifest
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_harness.runtime.tool_access import compile_caller_policy
from iris_harness.runtime.tool_service import ToolService

# A low-entropy canary, never a real secret: what the vault resolves a handle into.
CANARY = "CANARY_VAULT_SECRET_d41f8a27"


class Spy:
    name = "spy"
    priority = 99

    def __init__(self, point: HookPoint) -> None:
        self.hook_point = point
        self.seen: list[HookContext] = []

    async def __call__(self, ctx: HookContext) -> HookDecision:
        self.seen.append(ctx)
        return HookDecision(outcome="allow", reason="spy")


def _kernel(*hooks: Any) -> GovernanceKernel:
    kernel = GovernanceKernel(audit_log=None)
    for hook in hooks:
        kernel.register(hook)
    kernel.init_lock()
    return kernel


class _Counter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, args: dict[str, Any]) -> str:
        self.calls.append(dict(args))
        return "ran"


def _tool(counter: _Counter) -> ToolSpec:
    return ToolSpec(name="look_up", description="d", call=counter, effect="read")


# ------------------------------------------------------------------ plain tool calls
@pytest.mark.usefixtures("no_vault_master_key")
def test_the_runner_refuses_before_the_tool_runs() -> None:
    counter, pre = _Counter(), Spy(HookPoint.PRE_TOOL_USE)
    runner = GovernedToolRunner(kernel=_kernel(pre), agent_type="chat")

    outcome = runner.execute(_tool(counter), {"q": "x"}, ToolCall())

    assert outcome.status == "refused" and NO_AUDIT_KEY_MESSAGE in outcome.text
    assert counter.calls == [] and pre.seen == []  # nothing ran, not even PRE_TOOL_USE


@pytest.mark.usefixtures("no_vault_master_key")
def test_a_destructive_tool_is_refused_before_its_arguments_are_checked() -> None:
    checked: list[dict[str, Any]] = []
    counter = _Counter()
    tool = ToolSpec(
        name="wipe",
        description="d",
        call=counter,
        effect="destructive",
        validate=lambda args: checked.append(args) or None,
    )
    outcome = GovernedToolRunner(kernel=_kernel(), agent_type="chat").execute(
        tool, {"id": 1}, ToolCall()
    )
    assert outcome.status == "refused" and NO_AUDIT_KEY_MESSAGE in outcome.text
    assert checked == [] and counter.calls == []


@pytest.mark.usefixtures("no_vault_master_key")
def test_the_loop_hands_the_model_the_block_message() -> None:
    counter = _Counter()
    core = AgenticCore(tools=[_tool(counter)], kernel=_kernel(), agent_type="chat")

    step = core._execute_tool("look_up", {"q": "x"}, run_id="r")

    assert NO_AUDIT_KEY_MESSAGE in step.observation and counter.calls == []


@pytest.mark.usefixtures("no_vault_master_key")
def test_tool_service_returns_a_refused_result() -> None:
    counter = _Counter()
    kernel = _kernel()
    service = ToolService(tools=lambda: [_tool(counter)], kernel=lambda: kernel)

    result = service.for_caller("plugin:p").call("look_up", {"q": "x"})

    assert not result.ok and result.held and NO_AUDIT_KEY_MESSAGE in result.text
    assert counter.calls == []


@pytest.mark.usefixtures("no_vault_master_key")
def test_without_a_kernel_nothing_is_audited_so_nothing_is_refused() -> None:
    """The no-kernel path (governance off) writes no rows; the key is not its gate."""
    counter = _Counter()
    outcome = GovernedToolRunner(kernel=None, agent_type="chat").execute(
        _tool(counter), {"q": "x"}, ToolCall()
    )
    assert outcome.status == "ran" and counter.calls == [{"q": "x"}]


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
    methods={
        "get": MethodSpec(fields=("text",)),
        "aget": MethodSpec(fields=("text",)),
        "stream": MethodSpec(fields=("text",)),
        "astream": MethodSpec(fields=("text",)),
    },
)


class Provider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, q: str) -> Note:
        self.calls.append("get")
        return Note("n")

    async def aget(self, q: str) -> Note:
        self.calls.append("aget")
        return Note("n")

    def stream(self, q: str) -> Iterator[Note]:
        self.calls.append("stream")
        yield Note("n")

    async def astream(self, q: str) -> AsyncIterator[Note]:
        self.calls.append("astream")
        yield Note("n")


@pytest.fixture()
def notes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Any, Provider]]:
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
    provider = Provider()
    assert registry.provide_capability("prov", "test.notes", provider)
    kernel = _kernel(CallerPolicyHook())
    registry.bind_kernel(lambda: kernel)
    register_caller_policy(compile_caller_policy(registry, config_dir=tmp_path))
    yield registry.resolve_capability("cons", "test.notes"), provider
    register_caller_policy(None)


@pytest.mark.usefixtures("no_vault_master_key")
def test_every_capability_shape_is_denied_before_the_provider_runs(
    notes: tuple[Any, Provider],
) -> None:
    facade, provider = notes
    with pytest.raises(CapabilityDenied, match="no vault master key"):
        facade.get("q")
    with pytest.raises(CapabilityDenied, match="no vault master key"):
        asyncio.run(facade.aget("q"))
    with pytest.raises(CapabilityDenied, match="no vault master key"):
        list(facade.stream("q"))

    async def drain() -> list[Note]:
        return [n async for n in facade.astream("q")]

    with pytest.raises(CapabilityDenied, match="no vault master key"):
        asyncio.run(drain())
    assert provider.calls == []


def test_with_a_key_every_capability_shape_runs(notes: tuple[Any, Provider]) -> None:
    facade, provider = notes
    facade.get("q")
    asyncio.run(facade.aget("q"))
    list(facade.stream("q"))

    async def drain() -> list[Note]:
        return [n async for n in facade.astream("q")]

    asyncio.run(drain())
    assert provider.calls == ["get", "aget", "stream", "astream"]


# ------------------------------------------------------------------ as-written args
class _Vault:
    def get(self, handle: str) -> str | None:
        return {"vault://token": CANARY}.get(handle)


def test_args_digest_is_of_the_arguments_as_written_never_the_resolved_secret() -> None:
    pre = Spy(HookPoint.PRE_TOOL_USE)
    counter = _Counter()
    runner = GovernedToolRunner(
        kernel=_kernel(CredentialBroker(vault=_Vault()), pre), agent_type="chat"
    )

    outcome = runner.execute(_tool(counter), {"key": "vault://token"}, ToolCall())

    assert outcome.status == "ran" and counter.calls == [{"key": CANARY}]  # the tool got it
    (ctx,) = pre.seen
    digester = audit_digester()
    assert ctx.payload["args"] == {"key": CANARY}  # the broker's rewrite ...
    # ... but the fingerprint is of what the caller wrote: the handle, not the secret.
    assert ctx.payload["args_digest"] == digester.digest({"key": "vault://token"})
    assert ctx.payload["args_digest"] != digester.digest({"key": CANARY})
    assert ctx.payload["digest_alg"] == digester.alg
