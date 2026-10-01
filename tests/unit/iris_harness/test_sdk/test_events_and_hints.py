"""The two seams M2 added without a seventh registration kind.

1. **Events.** Async work that finishes outside a chat turn — an Activity
   completing — reaches a plugin as a bus subscription, which is a harness service
   it consumes, not a capability it provides. The tests pin what that rests on: the
   subscriber is inside the fault boundary, one plugin's bad subscriber does not
   stop the others, and subscribing never shows up in ``kinds()``.
2. **Activity hints.** An intercept that scans a folder for seconds ships the
   "working" line to stream before it runs, as a property of that intercept.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.foundation.eventbus import EventBus
from iris_harness.runtime.plugin_host.api import HarnessServices, PluginAPI
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus


def _api(bus: EventBus | None, registry: PluginRegistry, name: str = "p") -> PluginAPI:
    registry.add_plugin(PluginRecord(name=name, source="test", status=PluginStatus.LOADED))
    services = HarnessServices(
        config_dir=Path("/nonexistent"),
        data_dir=Path("/nonexistent"),
        tier_router=None,
        agent_executor=None,
        heartbeats=None,
        channels=None,
        deterministic_reply=lambda **kw: None,
        events=bus,
    )
    return PluginAPI(plugin=name, services=services, registry=registry)


def test_subscriber_receives_the_payload() -> None:
    bus, registry = EventBus(), PluginRegistry()
    seen: list[object] = []
    _api(bus, registry).subscribe("activity.completed", seen.append)
    bus.emit_sync("activity.completed", {"activity_id": "a1"})
    assert seen == [{"activity_id": "a1"}]


def test_subscribing_is_not_a_registration_kind() -> None:
    """It is a consumed service: it must not appear in the plugin's ``kinds()``."""
    bus, registry = EventBus(), PluginRegistry()
    api = _api(bus, registry)
    api.subscribe("activity.completed", lambda payload: None)
    assert api.kinds() == ()
    assert registry.subscriptions() == [("p", "activity.completed", "runtime")]
    assert registry.describe()["subscriptions"] == ["p:activity.completed@runtime"]


def test_raising_subscriber_degrades_the_plugin_and_spares_the_others() -> None:
    bus, registry = EventBus(), PluginRegistry()

    def boom(payload: object) -> None:
        raise RuntimeError("kaboom")

    delivered: list[object] = []
    _api(bus, registry, "bad").subscribe("activity.failed", boom)
    _api(bus, registry, "good").subscribe("activity.failed", delivered.append)

    bus.emit_sync("activity.failed", "payload")

    assert delivered == ["payload"]  # the good subscriber still ran
    bad = registry.get("bad")
    assert bad is not None
    assert bad.status is PluginStatus.DEGRADED
    assert bad.failure_count == 1
    assert "subscription:activity.failed" in (bad.last_error or "")
    assert "kaboom" in (bad.last_error or "")
    good = registry.get("good")
    assert good is not None
    assert good.status is PluginStatus.LOADED


def test_publish_reaches_core_subscribers() -> None:
    bus, registry = EventBus(), PluginRegistry()
    seen: list[object] = []
    bus.on("filemanager.organized", seen.append)
    _api(bus, registry).publish("filemanager.organized", {"moved": 3})
    assert seen == [{"moved": 3}]


@pytest.mark.parametrize("call", ["subscribe", "publish"])
def test_bus_absent_is_a_clear_error(call: str) -> None:
    api = _api(None, PluginRegistry())
    with pytest.raises(RuntimeError, match="event bus"):
        if call == "subscribe":
            api.subscribe("t", lambda payload: None)
        else:
            api.publish("t", None)


def test_intercept_activity_hint_is_guarded_and_not_a_kind() -> None:
    """A slow intercept ships its own "working" line; a raising hint degrades to None."""
    from iris_harness.runtime.intercepts import InterceptSpec

    registry = PluginRegistry()
    registry.add_plugin(PluginRecord(name="fm", source="test", status=PluginStatus.LOADED))

    def hint(message: str) -> str | None:
        if "photos" in message:
            return "scanning the folder…"
        raise RuntimeError("hint blew up")

    registry.add_intercept("fm", InterceptSpec("organize", "plugin:fm"), lambda *a, **k: None, hint)
    reg = registry.intercept("organize")
    assert reg is not None
    assert reg.activity_hint is not None
    assert reg.activity_hint("organize my photos") == "scanning the folder…"

    assert reg.activity_hint("something else") is None  # degraded, not raised
    rec = registry.get("fm")
    assert rec is not None
    assert rec.status is PluginStatus.DEGRADED
    assert "intercept_hint:organize" in (rec.last_error or "")
    # One registration, of kind intercept — the hint is a property of it, not its own.
    assert [r.kind.value for r in rec.registrations] == ["intercept"]


def test_intercept_without_a_hint_has_none() -> None:
    from iris_harness.runtime.intercepts import InterceptSpec

    registry = PluginRegistry()
    registry.add_plugin(PluginRecord(name="fm", source="test", status=PluginStatus.LOADED))
    registry.add_intercept("fm", InterceptSpec("x", "plugin:fm"), lambda *a, **k: None)
    reg = registry.intercept("x")
    assert reg is not None and reg.activity_hint is None


# ─── Bus scope (OSS plan M3.2) ───────────────────────────────────────────────
#
# IRIS runs two buses: the runtime-private one (activities, so completion
# subscribers don't cross-talk between runtimes) and the process-global
# singleton (the email chain, which `iris email recategorize` drives with no
# runtime built at all). Subscribing on the wrong one fails SILENTLY — the
# handler simply never fires — so these pin the selector rather than leaving it
# to be discovered by a dead heartbeat in production.


def test_process_scope_subscribes_to_the_global_bus() -> None:
    from iris_harness.foundation.eventbus import get_default_bus, reset_default_bus

    reset_default_bus()
    runtime_bus, registry = EventBus(), PluginRegistry()
    seen: list[object] = []
    _api(runtime_bus, registry).subscribe("email.new_arrived", seen.append, scope="process")

    # The runtime bus must NOT carry it...
    runtime_bus.emit_sync("email.new_arrived", {"id": "wrong-bus"})
    assert seen == []

    # ...the process bus must.
    get_default_bus().emit_sync("email.new_arrived", {"id": "m1"})
    assert seen == [{"id": "m1"}]
    reset_default_bus()


def test_process_scope_is_recorded_for_dump_config() -> None:
    from iris_harness.foundation.eventbus import reset_default_bus

    reset_default_bus()
    registry = PluginRegistry()
    _api(EventBus(), registry).subscribe("email.classified", lambda p: None, scope="process")
    assert registry.subscriptions() == [("p", "email.classified", "process")]
    assert registry.describe()["subscriptions"] == ["p:email.classified@process"]
    reset_default_bus()


def test_process_scope_subscriber_is_still_inside_the_fault_boundary() -> None:
    """The guard is the point of going through the API rather than get_default_bus()."""
    from iris_harness.foundation.eventbus import get_default_bus, reset_default_bus

    reset_default_bus()
    registry = PluginRegistry()
    survivor: list[object] = []
    api = _api(EventBus(), registry)

    def boom(payload: object) -> None:
        raise RuntimeError("kaboom")

    api.subscribe("email.new_arrived", boom, scope="process")
    api.subscribe("email.new_arrived", survivor.append, scope="process")
    get_default_bus().emit_sync("email.new_arrived", {"id": "m2"})

    assert survivor == [{"id": "m2"}]  # the raiser did not stop the other
    record = next(r for r in registry.describe()["plugins"] if r["name"] == "p")
    assert record["status"] == "degraded"
    reset_default_bus()


def test_unknown_scope_raises_rather_than_subscribing_nowhere() -> None:
    registry = PluginRegistry()
    with pytest.raises(ValueError, match="unknown bus scope"):
        _api(EventBus(), registry).subscribe("x", lambda p: None, scope="glonal")


def test_publish_honours_scope() -> None:
    from iris_harness.foundation.eventbus import get_default_bus, reset_default_bus

    reset_default_bus()
    seen: list[object] = []
    get_default_bus().on("email.classified", seen.append)
    _api(EventBus(), PluginRegistry()).publish("email.classified", {"id": "m3"}, scope="process")
    assert seen == [{"id": "m3"}]
    reset_default_bus()
