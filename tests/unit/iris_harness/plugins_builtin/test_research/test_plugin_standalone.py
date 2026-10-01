"""Research as a reference plugin, pinned from both sides (OSS plan M4.7).

Release gate 2 names `research` as one of the three reference plugins that must
load "only through the public API + profile". What that means concretely, and
what these pin:

* the plugin registers through ``PluginAPI`` and imports no runtime internals;
* its egress guards travel with it — a personal-finance query is refused and the
  user's identifiers are stripped before any query leaves the machine, on every
  lane that reaches the tool;
* the core no longer builds a `research` tool of its own.
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


def _api(executor: _Executor, *, react: Any = None, embed: Any = None) -> PluginAPI:
    registry = PluginRegistry()
    registry.add_plugin(PluginRecord(name="research", source="builtin", status=PluginStatus.LOADED))
    services = HarnessServices(
        config_dir=Path("/nonexistent"),
        data_dir=Path("/nonexistent"),
        tier_router=None,
        agent_executor=executor,
        heartbeats=None,
        channels=None,
        deterministic_reply=lambda **kw: None,
        react_handler=react,
        react_stream_handler=react,
        embed=embed,
    )
    return PluginAPI(plugin="research", services=services, registry=registry)


# ─── It loads through the public API only ────────────────────────────────────


def test_plugin_imports_no_runtime_internals() -> None:
    """Release gate 2: a reference plugin may not reach into the runtime."""
    from iris_harness.plugins_builtin.research import guard, plugin, tools

    for module in (plugin, tools, guard):
        source = inspect.getsource(module)
        assert "iris_harness.runtime" not in source, f"{module.__name__} imports runtime internals"


def test_setup_registers_the_research_tool() -> None:
    from iris_harness.plugins_builtin.research import plugin

    api = _api(_Executor())
    plugin.setup(api)
    assert [t.name for t in api._registry.tools()] == ["research"]


def test_setup_registers_the_builtin_providers_through_the_public_seam() -> None:
    """The built-ins take the path any plugin's provider takes: no special case."""
    from iris_harness.plugins_builtin.research import plugin

    api = _api(_Executor())
    plugin.setup(api)
    assert api._registry.seams() == [
        ("research", "search_provider", name)
        for name in ("searxng", "tavily", "exa", "brave", "ddg")
    ]


def test_the_manifest_declares_every_builtin_provider_it_registers() -> None:
    """Mounted with its shipped manifest, all five are accepted and nothing is charged."""
    from iris_harness.plugins_builtin.research import plugin
    from iris_harness.plugins_builtin.research.providers import BUILTIN_PROVIDERS
    from iris_harness.runtime.plugin_host.manifest import load_manifest

    manifest = load_manifest(Path(plugin.__file__).with_name("manifest.yaml"))
    assert manifest.search_providers == tuple(p.name for p in BUILTIN_PROVIDERS)
    api = _api(_Executor())
    record = api._registry.get("research")
    assert record is not None
    record.manifest = manifest
    plugin.setup(api)
    assert record.failure_count == 0 and record.status is PluginStatus.LOADED
    assert [key for _, seam, key in api._registry.seams() if seam == "search_provider"] == list(
        manifest.search_providers
    )


def test_the_persona_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_harness.plugins_builtin.research import plugin

    monkeypatch.delenv("IRIS_RESEARCH_AGENT", raising=False)
    executor = _Executor()
    plugin.setup(_api(executor, react=lambda task: ("x", {})))
    assert executor.agents == {}


def test_the_persona_mounts_the_harness_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A persona plugin supplies no handler — it mounts the harness's own loop."""
    from iris_harness.plugins_builtin.research import plugin

    monkeypatch.setenv("IRIS_RESEARCH_AGENT", "1")
    executor = _Executor()
    plugin.setup(_api(executor, react=lambda task: ("x", {})))
    assert sorted(executor.agents) == ["research"]


def test_the_persona_does_not_mount_when_the_loop_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_harness.plugins_builtin.research import plugin

    monkeypatch.setenv("IRIS_RESEARCH_AGENT", "1")
    executor = _Executor()
    plugin.setup(_api(executor, react=None))
    assert executor.agents == {}


# ─── The guards travel with the tool ─────────────────────────────────────────


def test_the_tool_refuses_a_personal_finance_query(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_harness.plugins_builtin.research import tools

    reached: list[dict[str, object]] = []
    monkeypatch.setattr(
        "iris_harness.plugins_builtin.research.tool.run_research",
        lambda args, **_: reached.append(args) or "web results",
    )
    call = tools.build_research_call()

    out = call({"query": "what are my insurance dues"})

    assert reached == []
    assert "won't web-search your personal finances" in out


def test_the_tool_still_answers_a_general_question(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_harness.plugins_builtin.research import tools

    monkeypatch.setattr(
        "iris_harness.plugins_builtin.research.tool.run_research",
        lambda args, **_: f"results for {args.get('query')}",
    )
    call = tools.build_research_call()

    assert call({"query": "best health insurance plans in India"}).startswith("results for")


def test_the_tool_strips_an_email_before_egress(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_harness.plugins_builtin.research import tools

    seen: list[str] = []
    monkeypatch.setattr(
        "iris_harness.plugins_builtin.research.tool.run_research",
        lambda args, **_: seen.append(str(args.get("query"))) or "ok",
    )
    call = tools.build_research_call()

    call({"query": "who is someone@example.com on github"})

    assert seen and "someone@example.com" not in seen[0]


# ─── The core built no research tool of its own ──────────────────────────────


def test_the_core_builtin_pool_has_no_research_tool() -> None:
    from iris_harness.runtime.react_tools import builtin_react_tools

    specs = builtin_react_tools(semantic_index=None, wiki=None, repo_root=Path("/tmp"))
    assert "research" not in {s.name for s in specs}
