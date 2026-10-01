"""code_exec as a reference plugin, pinned from both sides (OSS plan M4.6).

The sandbox RUNTIME is core: it is a governed capability with an egress proxy and
a runtime policy, and the eval sandbox uses it too. What left is the *agent* that
drives it — the bounded LLM ↔ sandbox loop.

The mount is conditional on the sandbox being reachable, which is the interesting
half: the capability decides whether it can serve, instead of the composition root
knowing how to ask. With Docker down nothing registers, so routing falls back
exactly as it did — rather than offering a tool whose every call answers
"unavailable".
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from iris_harness.runtime.plugin_host.api import HarnessServices, PluginAPI
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus


class _Executor:
    def __init__(self) -> None:
        self.agents: dict[str, object] = {}
        self.streams: dict[str, object] = {}

    def register(self, agent_type: str, handler: object) -> None:
        self.agents[agent_type] = handler

    def register_stream(self, agent_type: str, handler: object) -> None:
        self.streams[agent_type] = handler


class _TierRouter:
    def get_llm_config(self, intent: str) -> Any:
        from iris_harness.llm.client import CodingLLMConfig

        return CodingLLMConfig(provider="ollama", model="test-model")

    def get_tier(self, intent: str) -> Any:
        return None


def _api(executor: _Executor) -> PluginAPI:
    registry = PluginRegistry()
    registry.add_plugin(
        PluginRecord(name="code_exec", source="builtin", status=PluginStatus.LOADED)
    )
    services = HarnessServices(
        config_dir=Path("config"),
        data_dir=Path("/nonexistent"),
        tier_router=_TierRouter(),
        agent_executor=executor,
        heartbeats=None,
        channels=None,
        deterministic_reply=lambda **kw: None,
    )
    return PluginAPI(plugin="code_exec", services=services, registry=registry)


# ─── It loads through the public API only ────────────────────────────────────


def test_plugin_imports_no_runtime_internals() -> None:
    """Release gate 2: a reference plugin may not reach into the runtime.

    ``runtime.tool_call_parsing`` is the exception the harness publishes on purpose
    — the JSON-tool-call parser is a shared utility, not runtime state — so the
    check is on ``runtime.bootstrap``, the composition root.
    """
    from iris_harness.plugins_builtin.code_exec import handler, plugin

    for module in (plugin, handler):
        source = inspect.getsource(module)
        assert "runtime.bootstrap" not in source, f"{module.__name__} imports the composition root"


def test_setup_registers_the_tool_and_the_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_harness.plugins_builtin.code_exec import plugin

    monkeypatch.setattr(plugin, "_is_docker_available", lambda: True)
    executor = _Executor()
    api = _api(executor)

    plugin.setup(api)

    assert [t.name for t in api._registry.tools()] == ["code_exec"]
    assert sorted(executor.agents) == ["code_exec"]


def test_nothing_mounts_without_a_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Docker → no tool and no agent, rather than a tool that always fails."""
    from iris_harness.plugins_builtin.code_exec import plugin

    monkeypatch.setattr(plugin, "_is_docker_available", lambda: False)
    executor = _Executor()
    api = _api(executor)

    plugin.setup(api)

    assert api._registry.tools() == []
    assert executor.agents == {}


def test_the_tool_rejects_a_missing_task(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_harness.plugins_builtin.code_exec import plugin

    monkeypatch.setattr(plugin, "_is_docker_available", lambda: True)
    api = _api(_Executor())
    plugin.setup(api)

    (tool,) = api._registry.tools()
    assert "requires a 'task'" in tool.call({})


# ─── The core built no code_exec of its own ──────────────────────────────────


def test_the_core_builtin_pool_has_no_code_exec_tool() -> None:
    from iris_harness.runtime.react_tools import builtin_react_tools

    specs = builtin_react_tools(semantic_index=None, wiki=None, repo_root=Path("/tmp"))
    assert "code_exec" not in {s.name for s in specs}


def test_the_sandbox_runtime_stays_core() -> None:
    """The governed sandbox is a harness capability; only the agent moved."""
    from iris_harness.tools.sandbox import SandboxConfig

    assert SandboxConfig.default() is not None
