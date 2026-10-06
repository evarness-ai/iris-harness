"""Tools for code, through governance (docs/architecture/plugin-capabilities.md, step 2).

A plugin, a workflow or the core calls a registered tool through the same governed runner
the model's calls take: PRE_TOOL_USE, the approval rules, POST_TOOL_USE, and the caller
stamped by the harness. The effect rules for a caller that cannot answer an approval are
checked against the real kernel hooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.agentic_core import ToolSpec
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.plugins import DestructiveApprovalHook, ToolPolicyHook
from iris_harness.runtime.harness_services import HarnessServices
from iris_harness.runtime.plugin_host.api import PluginAPI
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_harness.runtime.tool_service import BoundTools, ToolService


class _Spy:
    """Records what each tool hook point was shown."""

    priority = 1

    def __init__(self, point: HookPoint) -> None:
        self.name = f"spy_{point.value}"
        self.hook_point = point
        self.seen: list[HookContext] = []

    async def __call__(self, ctx: HookContext) -> HookDecision:
        self.seen.append(ctx)
        return HookDecision(outcome="allow", reason="seen")


def _tool(
    name: str, effect: str = "read", confirm: str = "never", *, fail: bool = False
) -> ToolSpec:
    def call(args: dict[str, Any]) -> str:
        if fail:
            raise RuntimeError("boom")
        return f"{name} ran with {sorted(args.items())}"

    return ToolSpec(name, f"{name} tool", call, effect=effect, confirm=confirm)


TOOLS = [
    _tool("look_up"),
    _tool("set_pref", "write", "never"),
    _tool("add_thing", "write", "once"),
    _tool("delete_all", "destructive", "approval"),
    _tool("broken", fail=True),
]


@pytest.fixture()
def spies() -> tuple[_Spy, _Spy]:
    return _Spy(HookPoint.PRE_TOOL_USE), _Spy(HookPoint.POST_TOOL_USE)


@pytest.fixture()
def service(spies: tuple[_Spy, _Spy]) -> ToolService:
    kernel = GovernanceKernel(audit_log=None)
    for hook in (*spies, ToolPolicyHook(), DestructiveApprovalHook(approval_queue=None)):
        kernel.register(hook)
    kernel.init_lock()
    return ToolService(tools=lambda: TOOLS, kernel=lambda: kernel)


def test_a_read_tool_runs_through_governance_as_its_caller(
    service: ToolService, spies: tuple[_Spy, _Spy]
) -> None:
    pre, post = spies
    result = service.for_caller("plugin:p").call("look_up", {"q": "x"})
    assert result.ok and not result.held and "look_up ran" in result.text
    assert [c.metadata["caller"] for c in pre.seen] == ["plugin:p"]
    payload = pre.seen[0].payload
    assert (payload["tool_name"], payload["args"]) == ("look_up", {"q": "x"})
    # The row's only trace of the arguments: a keyed digest (kernel/governance/audit/digest.py).
    assert set(payload) == {"tool_name", "tool_plugin", "args", "args_digest", "digest_alg"}
    assert payload["tool_plugin"] == "system"  # the test's tool is not a plugin's
    assert [c.payload["tool_name"] for c in post.seen] == ["look_up"]


def test_a_write_that_never_asks_runs(service: ToolService) -> None:
    assert service.for_caller("plugin:p").call("set_pref", {"v": 1}).ok


def test_a_confirm_once_write_is_held_nobody_was_asked(service: ToolService) -> None:
    result = service.for_caller("plugin:p").call("add_thing")
    assert not result.ok and result.held


def test_a_destructive_tool_is_held_no_caller_can_approve(service: ToolService) -> None:
    result = service.for_caller("plugin:p").call("delete_all")
    assert not result.ok and result.held


def test_a_tool_that_raises_is_reported_not_held(service: ToolService) -> None:
    result = service.for_caller("plugin:p").call("broken")
    assert not result.ok and not result.held and result.text.startswith("Tool error:")


def test_an_unknown_tool_is_reported(service: ToolService) -> None:
    result = service.for_caller("plugin:p").call("nope")
    assert not result.ok and "unknown tool" in result.text


def test_describe_is_the_declared_contract(service: ToolService) -> None:
    infos = {info.name: info for info in service.describe()}
    assert set(infos) == {tool.name for tool in TOOLS}
    assert (infos["add_thing"].effect, infos["add_thing"].confirm) == ("write", "once")
    assert [info.name for info in service.describe("look_up")] == ["look_up"]


def test_a_plugin_is_bound_to_its_own_name_and_cannot_change_it(
    service: ToolService, tmp_path: Path
) -> None:
    registry = PluginRegistry()
    registry.add_plugin(PluginRecord(name="p", source="test", status=PluginStatus.LOADED))
    services = HarnessServices(
        config_dir=tmp_path,
        data_dir=tmp_path,
        tier_router=None,  # type: ignore[arg-type]
        agent_executor=None,  # type: ignore[arg-type]
        heartbeats=None,  # type: ignore[arg-type]
        channels=None,  # type: ignore[arg-type]
        deterministic_reply=lambda **kw: None,
        tools=service,
    )
    tools = PluginAPI(plugin="p", services=services, registry=registry).tools
    assert isinstance(tools, BoundTools) and tools.caller == "plugin:p"
    with pytest.raises(AttributeError):
        tools.caller = "core:anything"  # type: ignore[misc]
