"""GET /plugins and /plugins/{name}: thin reads over the live plugin registry."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.runtime.plugin_host.manifest import PluginManifest, RegistrationKind
from iris_harness.runtime.plugin_host.registry import (
    PluginRecord,
    PluginRegistry,
    PluginStatus,
    Registration,
)
from iris_harness.server.iris_api.main import create_app


def _registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.add_plugin(
        PluginRecord(
            name="finance_workflows",
            source="entry_point:x",
            status=PluginStatus.LOADED,
            manifest=PluginManifest(name="finance_workflows", version="0.1.0"),
            registrations=[
                Registration("finance_workflows", RegistrationKind.INTENT_HANDLER, "finance")
            ],
        )
    )
    return registry


def _client(*agents: str, registry: PluginRegistry | None = None) -> TestClient:
    runtime = SimpleNamespace(
        agent_executor=SimpleNamespace(registered_agents=lambda: frozenset(agents)),
        plugin_registry=registry,
        profile=None,
    )
    return TestClient(create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers())


def test_agents_name_the_plugin_that_registered_them() -> None:
    body = _client("finance", "clarify", registry=_registry()).get("/agents").json()
    owners = {a["name"]: a["plugin"] for a in body["agents"]}
    assert owners == {"clarify": None, "finance": "finance_workflows"}


def test_agent_dashboard_names_its_plugin(tmp_path) -> None:  # type: ignore[no-untyped-def]
    runtime = SimpleNamespace(
        agent_executor=SimpleNamespace(registered_agents=lambda: frozenset({"finance", "clarify"})),
        plugin_registry=_registry(),
        profile=None,
        data_dir=tmp_path,
        config_dir=tmp_path,
        heartbeats=SimpleNamespace(runs=lambda: [], list_definitions=lambda: []),
        tier_router=SimpleNamespace(
            get_tier=lambda intent: SimpleNamespace(name="A", model="m", provider="p")
        ),
    )
    client = TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    )
    assert client.get("/agents/finance").json()["plugin"] == "finance_workflows"
    assert client.get("/agents/clarify").json()["plugin"] is None


def test_list_plugins() -> None:
    body = _client(registry=_registry()).get("/plugins").json()
    assert body["count"] == 1
    assert body["plugins"][0]["name"] == "finance_workflows"
    assert body["plugins"][0]["agents"] == ["finance"]
    assert body["profile"] is None


def test_list_plugins_without_a_registry_is_empty() -> None:
    body = _client().get("/plugins").json()
    assert body["count"] == 0 and body["plugins"] == []


def test_get_plugin_and_404() -> None:
    client = _client(registry=_registry())
    detail = client.get("/plugins/finance_workflows").json()
    assert detail["name"] == "finance_workflows"
    assert detail["manifest"]["version"] == "0.1.0"
    assert detail["registrations"] == [{"kind": "intent_handler", "name": "finance", "detail": ""}]
    assert client.get("/plugins/nope").status_code == 404


def test_web_nav_shows_a_mounted_plugins_screens_and_drops_a_failed_ones() -> None:
    """GET /api/v1/webui/nav (OSS plan R17): the console draws only what this returns."""
    registry = PluginRegistry()
    for name, status, route in (
        ("notes", PluginStatus.LOADED, "/notes"),
        ("ledger", PluginStatus.FAILED, "/ledger"),
    ):
        manifest = PluginManifest.model_validate(
            {"name": name, "webui": {"screens": [{"id": name, "label": name, "route": route}]}}
        )
        registry.add_plugin(
            PluginRecord(name=name, source=f"home:{name}", status=status, manifest=manifest)
        )
    body = _client(registry=registry).get("/api/v1/webui/nav").json()
    routes = [item["route"] for group in body["groups"] for item in group["items"]]
    assert routes[0] == "/chat"
    assert "/notes" in routes and "/ledger" not in routes
    assert "/settings" in routes


def test_web_nav_without_a_registry_is_the_core_nav() -> None:
    body = _client().get("/api/v1/webui/nav").json()
    routes = [item["route"] for group in body["groups"] for item in group["items"]]
    assert "/chat" in routes and "/inbox" not in routes and "/portfolio" not in routes
