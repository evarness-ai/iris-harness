"""The permission contract (docs/architecture/plugin-capabilities.md §4, step 3a).

A plugin cannot call another plugin's tool unless the governance contract allows it: its
own tools, plus the ones its manifest lists under ``uses: tools``, minus what the operator
takes away in ``tool-access.yaml``. The kernel's CallerPolicyHook enforces it on every
PRE_TOOL_USE, and fails closed when no policy is loaded.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from iris_harness.agent.agentic_core import ToolSpec
from iris_harness.kernel.governance import (
    GovernanceKernel,
    HookContext,
    HookPoint,
    build_default_kernel,
)
from iris_harness.kernel.governance.caller_policy import caller_policy, register_caller_policy
from iris_harness.kernel.governance.plugins.caller_policy import CallerPolicyHook
from iris_harness.runtime.harness_services import HarnessServices
from iris_harness.runtime.plugin_host.api import PluginAPI
from iris_harness.runtime.plugin_host.manifest import PluginManifest
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_harness.runtime.tool_access import compile_caller_policy
from iris_harness.runtime.tool_service import ToolService


@pytest.fixture(autouse=True)
def _no_policy_leaks() -> Iterator[None]:
    register_caller_policy(None)
    yield
    register_caller_policy(None)


def _ctx(caller: str, tool: str = "search") -> HookContext:
    return HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="r",
        agent_type="t",
        payload={"tool_name": tool, "args": {}},
        metadata={"caller": caller},
    )


# -- the hook ---------------------------------------------------------------------------


@pytest.mark.parametrize("caller", ["model:email", "core:digest", ""])
async def test_the_model_and_the_core_keep_their_access(caller: str) -> None:
    assert (await CallerPolicyHook()(_ctx(caller))).outcome == "allow"


async def test_a_plugin_call_fails_closed_with_no_policy_loaded() -> None:
    decision = await CallerPolicyHook()(_ctx("plugin:finance"))
    assert decision.outcome == "deny" and "no permission contract" in decision.reason


async def test_the_hook_enforces_the_registered_policy() -> None:
    register_caller_policy(lambda caller, tool: None if tool == "search" else "not allowed")
    assert (await CallerPolicyHook()(_ctx("plugin:finance", "search"))).outcome == "allow"
    denied = await CallerPolicyHook()(_ctx("plugin:finance", "trash"))
    assert denied.outcome == "deny" and "not allowed" in denied.reason


def test_the_default_kernel_enforces_it_at_pre_tool_use() -> None:
    kernel = build_default_kernel(audit_log=None)
    assert "caller_policy" in kernel.hook_names(HookPoint.PRE_TOOL_USE)


# -- the grants: own tools, plus `uses: tools` -------------------------------------------


def _manifest(name: str, *, uses: tuple[str, ...] = ()) -> PluginManifest:
    return PluginManifest.model_validate({"name": name, "uses": {"tools": list(uses)}})


def _tool(name: str) -> ToolSpec:
    return ToolSpec(name, name, lambda args: f"{name} ran")


def _registry(*, finance_uses: tuple[str, ...] = ()) -> PluginRegistry:
    registry = PluginRegistry()
    registry.add_plugin(
        PluginRecord("email", "test", PluginStatus.LOADED, manifest=_manifest("email"))
    )
    registry.add_plugin(
        PluginRecord(
            "finance", "test", PluginStatus.LOADED, manifest=_manifest("finance", uses=finance_uses)
        )
    )
    registry.add_tool("email", _tool("search_inbox"))
    registry.add_tool("finance", _tool("finance_lookup"))
    return registry


def test_a_plugin_may_call_its_own_tools() -> None:
    assert _registry().caller_denial("plugin:finance", "finance_lookup") is None


def test_another_plugins_tool_needs_a_uses_grant() -> None:
    assert "uses: tools" in (_registry().caller_denial("plugin:finance", "search_inbox") or "")
    granted = _registry(finance_uses=("search_inbox",))
    assert granted.caller_denial("plugin:finance", "search_inbox") is None


def test_an_unmounted_caller_is_denied() -> None:
    assert "not a mounted plugin" in (_registry().caller_denial("plugin:ghost", "x") or "")


def test_the_manifest_takes_only_what_is_enforced() -> None:
    assert _manifest("p", uses=("a",)).uses.tools == ("a",)
    with pytest.raises(ValidationError):
        PluginManifest.model_validate({"name": "p", "uses": {"agents": ["x"]}})


# -- the operator can only narrow --------------------------------------------------------


def _access(tmp_path: Path, text: str) -> Path:
    (tmp_path / "governance").mkdir(parents=True, exist_ok=True)
    (tmp_path / "governance" / "tool-access.yaml").write_text(text)
    return tmp_path


def test_the_operator_takes_a_tool_away_even_from_its_own_plugin(tmp_path: Path) -> None:
    config = _access(tmp_path, "deny:\n  finance: [finance_lookup]\n")
    policy = compile_caller_policy(_registry(), config_dir=config)
    assert "operator took" in (policy("plugin:finance", "finance_lookup") or "")


def test_no_operator_file_means_the_manifests_decide(tmp_path: Path) -> None:
    policy = compile_caller_policy(_registry(), config_dir=tmp_path)
    assert policy("plugin:finance", "finance_lookup") is None


def test_an_unreadable_operator_file_denies_every_plugin_call(tmp_path: Path) -> None:
    config = _access(tmp_path, "deny: [not, a, mapping]\n")
    policy = compile_caller_policy(_registry(), config_dir=config)
    assert "unreadable" in (policy("plugin:finance", "finance_lookup") or "")


# -- end to end: one plugin calling another's tool through api.tools ---------------------


def _finance_tools(registry: PluginRegistry, tmp_path: Path) -> Any:
    kernel = GovernanceKernel(audit_log=None)
    kernel.register(CallerPolicyHook())
    kernel.init_lock()
    register_caller_policy(compile_caller_policy(registry, config_dir=tmp_path))
    services = HarnessServices(
        config_dir=tmp_path,
        data_dir=tmp_path,
        tier_router=None,  # type: ignore[arg-type]
        agent_executor=None,  # type: ignore[arg-type]
        heartbeats=None,  # type: ignore[arg-type]
        channels=None,  # type: ignore[arg-type]
        deterministic_reply=lambda **kw: None,
        tools=ToolService(tools=registry.tools, kernel=lambda: kernel),
    )
    return PluginAPI(plugin="finance", services=services, registry=registry).tools


def test_without_a_grant_the_call_is_held_by_the_contract(tmp_path: Path) -> None:
    result = _finance_tools(_registry(), tmp_path).call("search_inbox")
    assert not result.ok and result.held and "uses: tools" in result.text


def test_with_a_grant_the_call_runs(tmp_path: Path) -> None:
    tools = _finance_tools(_registry(finance_uses=("search_inbox",)), tmp_path)
    assert tools.call("search_inbox").ok


def test_build_runtime_loads_the_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from iris_harness.runtime import build_runtime

    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    config_dir = Path(__file__).resolve().parents[5] / "config"
    build_runtime(config_dir=config_dir, data_dir=tmp_path / "data", use_background_scheduler=False)
    assert caller_policy() is not None
