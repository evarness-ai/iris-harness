"""PluginAPI's keyed core seams (core/SDK boundary plan, PR 3c-2).

A plugin fills the API service's routers, the agent dashboard's panels and the digest's
"learned yesterday" line through ``api.register_*`` instead of importing the core
registries. Each forwards to the same keyed registry the core reads, records the entry
for ``--dump-config``, and guards the callable: a failure is charged to the plugin
(System Health) and re-raised, so the consumer's own "skip a broken entry" still decides
what the owner sees.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from iris_harness.runtime import agent_panels, api_routes
from iris_harness.runtime.plugin_host.api import HarnessServices, PluginAPI
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_harness.services.digest import footer, learned


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    api_routes.clear_api_routers()
    agent_panels.clear_agent_panels()
    yield
    api_routes.clear_api_routers()
    agent_panels.clear_agent_panels()
    learned.unregister_learned_source("seam-test")
    footer.unregister_footer_line("seam-test")


def _api(registry: PluginRegistry, name: str = "p") -> PluginAPI:
    registry.add_plugin(PluginRecord(name=name, source="test", status=PluginStatus.LOADED))
    services = HarnessServices(
        config_dir=Path("/nonexistent"),
        data_dir=Path("/nonexistent"),
        tier_router=None,
        agent_executor=None,
        heartbeats=None,
        channels=None,
        deterministic_reply=lambda **kw: None,
    )
    return PluginAPI(plugin=name, services=services, registry=registry)


def test_an_api_router_reaches_the_core_registry_and_is_recorded() -> None:
    registry = PluginRegistry()
    _api(registry).register_api_router("things", lambda: "router")
    assert api_routes.registered_api_routers()["things"]() == "router"
    assert registry.describe()["seams"] == ["p:api_router:things"]


def test_a_failing_router_factory_is_charged_to_the_plugin_and_still_raises() -> None:
    registry = PluginRegistry()

    def boom() -> object:
        raise RuntimeError("no router")

    _api(registry).register_api_router("things", boom)
    with pytest.raises(RuntimeError):
        api_routes.registered_api_routers()["things"]()
    rec = registry.get("p")
    assert rec is not None and rec.status is PluginStatus.DEGRADED
    assert rec.last_error is not None and "api_router:things" in rec.last_error


def test_a_public_callback_is_declared_and_still_validated() -> None:
    registry = PluginRegistry()
    api = _api(registry)
    api.register_public_callback("/api/v1/things/callback")
    assert api_routes.is_public_callback("GET", "/api/v1/things/callback")
    assert registry.seams() == [("p", "public_callback", "/api/v1/things/callback")]
    with pytest.raises(ValueError):
        api.register_public_callback("/memory")


def test_an_agent_panel_is_served_by_agent_name() -> None:
    registry = PluginRegistry()
    _api(registry).register_agent_panel("finance", lambda: {"stores": 2})
    assert agent_panels.agent_panel("finance") == {"stores": 2}
    assert registry.seams() == [("p", "agent_panel", "finance")]


def test_a_failing_learned_source_is_skipped_and_charged_to_the_plugin() -> None:
    registry = PluginRegistry()

    def boom(_start: datetime, _end: datetime) -> list[str]:
        raise RuntimeError("store gone")

    _api(registry).register_learned_source("seam-test", boom)
    start = datetime(2026, 9, 26, tzinfo=UTC)
    assert learned.learned_between(start, start) == []
    rec = registry.get("p")
    assert rec is not None and rec.status is PluginStatus.DEGRADED


def test_a_learned_source_feeds_the_footer() -> None:
    registry = PluginRegistry()
    _api(registry).register_learned_source("seam-test", lambda _s, _e: ["hid 2 senders"])
    start = datetime(2026, 9, 26, tzinfo=UTC)
    assert "hid 2 senders" in learned.learned_between(start, start)


def test_a_footer_line_reaches_the_digest_footer_and_is_recorded() -> None:
    registry = PluginRegistry()
    _api(registry).register_footer_line("seam-test", lambda _s, _e: "Email jobs: 3/3 judged")
    now = datetime(2026, 9, 27, 12, tzinfo=UTC)
    assert "Email jobs: 3/3 judged" in footer.footer_lines(now, ZoneInfo("UTC"))
    assert ("p", "footer_line", "seam-test") in registry.seams()


def test_a_failing_footer_line_is_skipped_and_charged_to_the_plugin() -> None:
    registry = PluginRegistry()

    def boom(_start: datetime, _end: datetime) -> str | None:
        raise RuntimeError("store gone")

    _api(registry).register_footer_line("seam-test", boom)
    now = datetime(2026, 9, 27, 12, tzinfo=UTC)
    assert footer.footer_lines(now, UTC) == []  # type: ignore[arg-type]
    rec = registry.get("p")
    assert rec is not None and rec.status is PluginStatus.DEGRADED


def test_a_loop_intent_is_recorded_with_its_fallback_and_touches_no_executor() -> None:
    """The plugin claims the intent; the harness decides after mounting whether the loop
    answers it (`_reassert_loop_intents`), so registering puts nothing on the executor
    (these services have none)."""
    registry = PluginRegistry()
    _api(registry).register_loop_intent("things", fallback=lambda _task: "the floor")
    assert registry.loop_intents()["things"](object()) == "the floor"
    assert registry.describe()["seams"] == ["p:loop_intent:things"]
    assert registry.intent_handlers() == {}


def test_a_failing_fallback_is_charged_to_the_plugin_and_answers_an_apology() -> None:
    registry = PluginRegistry()

    def boom(_task: object) -> str:
        raise RuntimeError("store gone")

    _api(registry).register_loop_intent("things", fallback=boom)
    assert "things capability is unavailable" in registry.loop_intents()["things"](object())
    rec = registry.get("p")
    assert rec is not None and rec.status is PluginStatus.DEGRADED
