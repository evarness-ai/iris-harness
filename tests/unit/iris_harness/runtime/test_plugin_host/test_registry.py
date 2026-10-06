"""PluginRegistry: fault boundary per kind, duplicate detection, health projection."""

from __future__ import annotations

import pytest

from iris_harness.agent.agentic_core import ToolSpec
from iris_harness.agent.tool_runner import ToolUnavailable
from iris_harness.runtime.intercepts import InterceptSpec
from iris_harness.runtime.plugin_host.manifest import RegistrationKind
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_harness.services.health.models import CheckKind, HealthState
from iris_harness.services.heartbeat.models import HeartbeatDefinition, HeartbeatStatus


def _loaded(registry: PluginRegistry, name: str = "p") -> PluginRecord:
    return registry.add_plugin(PluginRecord(name=name, source="test", status=PluginStatus.LOADED))


def test_intercept_failure_falls_through_and_marks_degraded() -> None:
    registry = PluginRegistry()
    _loaded(registry)

    def boom(message: str, *, session_id: str, span: object = None) -> object:
        raise RuntimeError("kaboom")

    registry.add_intercept("p", InterceptSpec("x", "plugin:p"), boom)
    reg = registry.intercept("x")
    assert reg is not None
    assert reg.handler("hi", session_id="s") is None  # degrade → fall through
    rec = registry.get("p")
    assert rec is not None
    assert rec.status is PluginStatus.DEGRADED
    assert rec.failure_count == 1
    assert "intercept:x" in (rec.last_error or "")
    assert "kaboom" in (rec.last_error or "")


def test_tool_failure_is_recorded_and_raises_tool_unavailable() -> None:
    """The boundary records the failure against the plugin and raises a typed error
    carrying what the caller is told, so the runner counts the call as failed rather
    than as one that ran (an approved call that raised used to settle as ``ran``)."""
    registry = PluginRegistry()
    record = _loaded(registry)
    registry.add_tool("p", ToolSpec("t", "desc", lambda args: 1 / 0))
    (tool,) = registry.tools()
    with pytest.raises(ToolUnavailable, match=r"^t is unavailable \(plugin 'p' raised Zero"):
        tool.call({})
    assert record.status is PluginStatus.DEGRADED and record.failure_count == 1


def test_intent_handler_failure_returns_apology_and_stream_reraises() -> None:
    registry = PluginRegistry()
    _loaded(registry)

    def sync(task: object) -> str:
        raise ValueError("nope")

    def stream(task: object):  # type: ignore[no-untyped-def]
        yield "a"
        raise ValueError("mid-stream")

    guarded, guarded_stream = registry.add_intent_handler("p", "agent_x", sync, stream)
    assert "agent_x capability is unavailable" in guarded(None)
    assert guarded_stream is not None
    it = guarded_stream(None)
    assert next(it) == "a"
    with pytest.raises(ValueError):
        next(it)
    assert registry.get("p").failure_count == 2  # type: ignore[union-attr]


def test_heartbeat_failure_becomes_failed_run() -> None:
    registry = PluginRegistry()
    _loaded(registry)
    definition = HeartbeatDefinition(name="tick", handler="tick", schedule="interval:60")

    def handler(d: HeartbeatDefinition) -> object:
        raise OSError("disk")

    guarded = registry.add_heartbeat("p", "tick", handler, definition)
    run = guarded(definition)
    assert run.status is HeartbeatStatus.FAILED
    assert "OSError" in run.error
    assert registry.heartbeats()[0][2] is definition


def test_confirmation_executor_and_channel_reraise_after_recording() -> None:
    registry = PluginRegistry()
    _loaded(registry)

    def executor(**kwargs: object) -> object:
        raise PermissionError("denied")

    registry.add_confirmation_executor("p", "thing", executor)
    with pytest.raises(PermissionError):
        registry.confirmation_executors()["thing"]()

    class Connector:
        name = "chan"

        def send(self, message: object) -> object:
            raise ConnectionError("down")

        def healthy(self) -> bool:
            return True

    connector = registry.add_channel("p", Connector())
    with pytest.raises(ConnectionError):
        connector.send("m")
    assert registry.get("p").failure_count == 2  # type: ignore[union-attr]


def test_duplicate_registrations_rejected() -> None:
    registry = PluginRegistry()
    _loaded(registry, "a")
    _loaded(registry, "b")
    registry.add_intercept("a", InterceptSpec("dup", "plugin:a"), lambda *a, **k: None)
    with pytest.raises(ValueError, match="already registered by plugin 'a'"):
        registry.add_intercept("b", InterceptSpec("dup", "plugin:b"), lambda *a, **k: None)
    registry.add_tool("a", ToolSpec("tool", "d", lambda args: ""))
    with pytest.raises(ValueError, match="already registered"):
        registry.add_tool("b", ToolSpec("tool", "d", lambda args: ""))
    registry.add_confirmation_executor("a", "k", lambda **kw: None)
    with pytest.raises(ValueError):
        registry.add_confirmation_executor("b", "k", lambda **kw: None)


def test_health_checks_project_every_status() -> None:
    registry = PluginRegistry()
    _loaded(registry, "ok")
    registry.add_tool("ok", ToolSpec("t", "d", lambda args: ""))
    registry.add_plugin(
        PluginRecord(name="broken", source="profile", status=PluginStatus.FAILED, load_error="boom")
    )
    registry.add_plugin(PluginRecord(name="off", source="profile", status=PluginStatus.DISABLED))
    degraded = _loaded(registry, "shaky")
    registry.record_failure("shaky", where="tool:x", exc=RuntimeError("meh"))
    assert degraded.status is PluginStatus.DEGRADED

    by_target = {c.target: c for c in registry.health_checks()}
    assert all(c.kind is CheckKind.PLUGIN for c in by_target.values())
    assert by_target["plugin:ok"].state is HealthState.GREEN
    assert "1 registration" in by_target["plugin:ok"].detail
    assert by_target["plugin:broken"].state is HealthState.RED
    assert by_target["plugin:broken"].action == "iris plugins show broken"
    assert by_target["plugin:off"].state is HealthState.GREY
    assert by_target["plugin:shaky"].state is HealthState.YELLOW
    assert "meh" in by_target["plugin:shaky"].detail


def test_describe_lists_registrations_by_kind() -> None:
    registry = PluginRegistry()
    _loaded(registry)
    registry.add_intercept("p", InterceptSpec("i", "plugin:p"), lambda *a, **k: None)
    registry.add_tool("p", ToolSpec("t", "d", lambda args: ""))
    registry.add_confirmation_executor("p", "c", lambda **kw: None)
    tree = registry.describe()
    assert tree["intercepts"] == ["i"] and tree["tools"] == ["t"]
    assert tree["confirmation_executors"] == ["c"]
    kinds = {r["kind"] for r in tree["plugins"][0]["registrations"]}
    assert kinds == {
        RegistrationKind.INTERCEPT.value,
        RegistrationKind.TOOL.value,
        RegistrationKind.CONFIRMATION_EXECUTOR.value,
    }
