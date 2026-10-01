"""The read-only plugin inventory: summaries, detail, drift, YAML files, agent owners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.agent_executor import AgentExecutor
from iris_harness.runtime.plugin_host import (
    PluginRef,
    PluginRegistry,
    load_plugins,
    load_profile,
)
from iris_harness.runtime.plugin_host.inventory import (
    MAX_YAML_BYTES,
    agent_plugins,
    plugin_detail,
    plugins_inventory,
)
from iris_harness.sdk import HarnessServices
from iris_harness.services.heartbeat import HeartbeatScheduler


class _Gateway:
    def register(self, connector: Any) -> None:
        pass


@pytest.fixture()
def services(tmp_path: Path) -> HarnessServices:
    return HarnessServices(
        config_dir=tmp_path / "config",
        data_dir=tmp_path / "data",
        tier_router=None,
        agent_executor=AgentExecutor(),
        heartbeats=HeartbeatScheduler(),
        channels=_Gateway(),
        deterministic_reply=lambda **kw: kw,
    )


def _home_plugin(home: Path, name: str, body: str, manifest_extra: str = "") -> Path:
    d = home / "plugins" / name
    d.mkdir(parents=True)
    (d / "manifest.yaml").write_text(
        f"name: {name}\nversion: 1.2.3\ndescription: the {name} plugin\n{manifest_extra}",
        encoding="utf-8",
    )
    (d / "plugin.py").write_text(body, encoding="utf-8")
    return d


_MINE_BODY = (
    "def setup(api):\n"
    "    api.register_tool('mine_tool', 'd', lambda args: 'ok')\n"
    "    api.register_intent_handler('mine_agent', lambda task: 'x')\n"
)
_MINE_MANIFEST = (
    "provides: [tool, intent_handler, heartbeat]\n"
    "tools:\n"
    "  mine_tool:\n"
    "    effect: read\n"
    "    guidance: use it\n"
    "  mine_write:\n"
    "    effect: write\n"
)


@pytest.fixture()
def loaded(
    tmp_path: Path, services: HarnessServices, monkeypatch: pytest.MonkeyPatch
) -> tuple[PluginRegistry, Any, Path]:
    home = tmp_path / "home"
    monkeypatch.setenv("IRIS_HOME", str(home))
    mine = _home_plugin(home, "mine", _MINE_BODY, _MINE_MANIFEST)
    (mine / "nlu.yaml").write_text("vocab: [a, b]\n", encoding="utf-8")
    (mine / "prompts").mkdir()
    (mine / "prompts" / "extract.yml").write_text("prompt: hi\n", encoding="utf-8")
    (mine / "__pycache__").mkdir()
    (mine / "__pycache__" / "stale.yaml").write_text("x: 1\n", encoding="utf-8")
    (mine / "big.yaml").write_text("x: " + "a" * (MAX_YAML_BYTES + 1), encoding="utf-8")
    _home_plugin(home, "sleepy", "def setup(api):\n    raise AssertionError('disabled')\n")

    config = tmp_path / "config"
    (config / "profiles").mkdir(parents=True)
    (config / "profiles" / "x.yaml").write_text(
        "name: x\ndescription: test profile\nplugins:\n  - name: mine\n", encoding="utf-8"
    )
    prof = load_profile(config, "x", home_dir=home, env={})
    prof.plugins = [
        PluginRef(name="mine"),
        PluginRef(name="sleepy", enabled=False),
        PluginRef(name="ghost"),
    ]
    registry = PluginRegistry()
    load_plugins(prof, services=services, registry=registry, home_dir=home, skip_entry_points=True)
    return registry, prof, config


def test_inventory_lists_every_profile_plugin_with_status(loaded: Any) -> None:
    registry, prof, config = loaded
    inv = plugins_inventory(registry, prof, config_dir=config)
    by_name = {p["name"]: p for p in inv["plugins"]}
    assert [p["name"] for p in inv["plugins"]] == ["mine", "sleepy", "ghost"]
    assert inv["totals"]["loaded"] == 1
    assert inv["totals"]["disabled"] == 1
    assert inv["totals"]["failed"] == 1

    mine = by_name["mine"]
    assert mine["status"] == "loaded"
    assert mine["version"] == "1.2.3"
    assert mine["agents"] == ["mine_agent"]
    assert mine["registration_counts"] == {"tool": 1, "intent_handler": 1}
    assert mine["declared_tools"] == 2
    assert mine["enabled"] is True

    # A disabled plugin never loaded, but its manifest is still read for display.
    assert by_name["sleepy"]["status"] == "disabled"
    assert by_name["sleepy"]["description"] == "the sleepy plugin"
    assert by_name["sleepy"]["enabled"] is False
    assert "not found" in by_name["ghost"]["load_error"]


def test_inventory_carries_the_profile_and_its_yaml(loaded: Any) -> None:
    registry, prof, config = loaded
    profile = plugins_inventory(registry, prof, config_dir=config)["profile"]
    assert profile["name"] == "x"
    assert profile["available_profiles"] == ["x"]
    assert len(profile["files"]) == 1
    assert "description: test profile" in profile["files"][0]["content"]


def test_detail_reports_tools_drift_and_yaml_files(loaded: Any) -> None:
    registry, prof, _config = loaded
    detail = plugin_detail(registry, prof, "mine")
    assert detail is not None
    tools = {t["name"]: t for t in detail["tools"]}
    assert tools["mine_tool"] == {
        "name": "mine_tool",
        "effect": "read",
        "confirm": "never",
        "pinned": False,
        "answers_directly": False,
        "undo": None,
        "undo_window_days": None,
        "content": "internal",
        "sends_to": None,
        "executes_code": False,
        "verify": None,
        "guidance": "use it",
        "registered": True,
    }
    assert tools["mine_write"]["confirm"] == "once"
    assert tools["mine_write"]["registered"] is False
    assert detail["drift"]["tools_declared_not_registered"] == ["mine_write"]
    assert detail["drift"]["provides_not_registered"] == ["heartbeat"]
    assert detail["drift"]["registered_not_provided"] == []
    assert detail["manifest"]["name"] == "mine"

    files = {f["path"]: f for f in detail["files"]}
    assert list(files)[0] == "manifest.yaml"
    assert set(files) == {"manifest.yaml", "big.yaml", "nlu.yaml", "prompts/extract.yml"}
    assert files["nlu.yaml"]["content"] == "vocab: [a, b]\n"
    assert files["big.yaml"]["truncated"] is True and files["big.yaml"]["content"] is None


def test_detail_for_unloaded_plugins_has_no_drift(loaded: Any) -> None:
    registry, prof, _config = loaded
    sleepy = plugin_detail(registry, prof, "sleepy")
    assert sleepy is not None
    assert all(v == [] for v in sleepy["drift"].values())
    assert [f["path"] for f in sleepy["files"]] == ["manifest.yaml"]
    ghost = plugin_detail(registry, prof, "ghost")
    assert ghost is not None and ghost["manifest"] is None and ghost["files"] == []


def test_detail_unknown_plugin_is_none(loaded: Any) -> None:
    registry, prof, _config = loaded
    assert plugin_detail(registry, prof, "nope") is None
    assert plugin_detail(None, prof, "mine") is None


def test_agent_plugins_maps_agents_to_their_plugin(loaded: Any) -> None:
    registry, _prof, _config = loaded
    assert agent_plugins(registry) == {"mine_agent": "mine"}
    assert agent_plugins(None) == {}


def test_yaml_outside_a_manifest_directory_is_not_scanned(
    tmp_path: Path, services: HarnessServices
) -> None:
    from iris_harness.runtime.plugin_host.inventory import _yaml_files

    (tmp_path / "loose.yaml").write_text("a: 1\n", encoding="utf-8")
    assert _yaml_files(tmp_path) == []


def test_subscriptions_and_seams_are_listed_per_plugin() -> None:
    """What `--dump-config` cannot see (it never runs setup): each plugin's bus
    subscriptions, with the bus they landed on, and the core seams it filled."""
    from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginStatus

    registry = PluginRegistry()
    for name in ("a", "b"):
        registry.add_plugin(PluginRecord(name=name, source="test", status=PluginStatus.LOADED))
    registry.add_subscription("a", "email.classified", lambda _p: None, scope="process")
    registry.add_seam("a", "api_router", "things", lambda: None)
    registry.declare_seam("a", "public_callback", "/api/v1/things/callback")
    registry.add_seam("b", "learned_source", "b_source", lambda _s, _e: [])

    detail = plugin_detail(registry, None, "a")
    assert detail is not None
    assert detail["subscriptions"] == [{"topic": "email.classified", "scope": "process"}]
    assert detail["seams"] == [
        {"seam": "api_router", "key": "things"},
        {"seam": "public_callback", "key": "/api/v1/things/callback"},
    ]
    by_name = {p["name"]: p for p in plugins_inventory(registry, None)["plugins"]}
    assert (by_name["a"]["subscription_count"], by_name["a"]["seam_count"]) == (1, 2)
    assert (by_name["b"]["subscription_count"], by_name["b"]["seam_count"]) == (0, 1)


def test_declared_search_providers_and_their_drift_are_shown(services: HarnessServices) -> None:
    from iris_harness.foundation.process_state import (
        restore_process_state,
        snapshot_process_state,
    )
    from iris_harness.runtime.plugin_host.api import PluginAPI
    from iris_harness.runtime.plugin_host.manifest import PluginManifest
    from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginStatus

    class _Provider:
        def is_available(self) -> bool:
            return False

        def search(self, query: str, *, max_results: int, **_: object) -> list[object]:
            return []

    snapshot = snapshot_process_state()  # the chain is process-wide
    try:
        registry = PluginRegistry()
        manifest = PluginManifest.model_validate(
            {"name": "finder", "search_providers": ["finder", "finder_news"]}
        )
        registry.add_plugin(
            PluginRecord(
                name="finder", source="test", status=PluginStatus.LOADED, manifest=manifest
            )
        )
        PluginAPI(plugin="finder", services=services, registry=registry).register_search_provider(
            "finder", _Provider()  # type: ignore[arg-type]
        )

        detail = plugin_detail(registry, None, "finder")
        assert detail is not None
        assert detail["search_providers"] == [
            {"name": "finder", "registered": True},
            {"name": "finder_news", "registered": False},
        ]
        assert detail["drift"]["search_providers_declared_not_registered"] == ["finder_news"]
        (row,) = plugins_inventory(registry, None)["plugins"]
        assert row["search_providers"] == ["finder", "finder_news"]
    finally:
        restore_process_state(snapshot)
