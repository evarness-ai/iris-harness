"""Runtime ↔ plugin wiring: the `system` tracer bullet answers through the public
API, plugin intercepts merge into the declared chain, failures degrade, and the
health snapshot carries a per-plugin verdict (OSS plan M1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.runtime import build_runtime
from iris_harness.runtime.intercepts import InterceptSpec
from iris_harness.runtime.plugin_host import PluginStatus
from iris_harness.runtime.types import ChatResult
from iris_harness.services.health.service import clear_check_providers, refresh


@pytest.fixture()
def runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    # Named, not unset: an unnamed run picks `email` where the email plugins are
    # installed (default.yaml, prefer_when_installed); these tests are about `default`.
    monkeypatch.setenv("IRIS_PROFILE", "default")
    clear_check_providers()
    config_dir = Path(__file__).resolve().parents[5] / "config"
    rt = build_runtime(
        config_dir=config_dir, data_dir=tmp_path / "data", use_background_scheduler=False
    )
    yield rt
    clear_check_providers()


def test_system_plugin_mounted_from_default_profile(runtime) -> None:  # type: ignore[no-untyped-def]
    assert runtime.profile is not None and runtime.profile.name == "default"
    rec = runtime.plugin_registry.get("system")
    assert rec is not None and rec.status is PluginStatus.LOADED
    assert rec.source == "builtin:iris_harness.plugins_builtin.system"
    kinds = {r.kind.value for r in rec.registrations}
    assert kinds == {"intercept", "tool", "heartbeat"}
    # The moved pieces are gone from the core and present via the plugin.
    assert not hasattr(runtime, "_handle_time_date_turn")
    # The registry holds every mounted plugin's tools; `system` contributes two: system_health and search_docs.
    assert "system_health" in [t.name for t in runtime.plugin_registry.tools()]
    assert [r.name for r in rec.registrations if r.kind.value == "tool"] == [
        "system_health",
        "search_docs",
    ]
    assert runtime.heartbeats.has_handler("health_tick")


def test_search_docs_is_an_internal_read_tool_on_every_tool_surface(runtime) -> None:  # type: ignore[no-untyped-def]
    """The registry is what the ReAct loop, ``ToolService`` and ``iris mcp serve`` all read,
    so one registration reaches the three. It is a read over IRIS's own docs: internal
    content (not scanned as third-party text), and `iris mcp serve` serves it unnamed."""
    from iris_harness.runtime.mcp_serve import contained

    spec = next(t for t in runtime.plugin_registry.tools() if t.name == "search_docs")
    assert (spec.effect, spec.content) == ("read", "internal")
    assert spec.sends_to is None and not spec.executes_code
    assert contained(spec)
    assert [i.name for i in runtime.tool_service.describe("search_docs")] == ["search_docs"]
    result = runtime.tool_service.for_caller("core:test").call("search_docs", {"query": "tier"})
    assert result.ok and "search_docs:" in result.text


# The declared ORDER of the chain (time_date between the calendar's meeting_creation and
# the planner's brief_request) needs the domains mounted, so it is checked on the
# personal profile: tests/unit/iris_personal/plugins/test_personal_profile/
# test_intercept_order.py.


def test_time_date_intercept_served_by_plugin_on_both_paths(runtime) -> None:  # type: ignore[no-untyped-def]
    chain = runtime.intercepts.effective_chain()
    names = [spec.name for spec, _h in chain]
    assert "time_date" in names
    spec = next(s for s, _h in chain if s.name == "time_date")
    assert spec.handler == "plugin:system"

    out = runtime.chat("what time is it?", session_id="plug-sync")
    assert isinstance(out, ChatResult) and out.metadata.get("deterministic_time_date") is True
    events = list(runtime.chat_stream("what time is it?", session_id="plug-stream"))
    done = next(e.result for e in events if e.kind == "done")
    assert done is not None and done.metadata.get("deterministic_time_date") is True


def test_undeclared_plugin_intercept_runs_after_declared_ones(runtime) -> None:  # type: ignore[no-untyped-def]
    runtime.plugin_registry.add_plugin(
        __import__(
            "iris_harness.runtime.plugin_host.registry", fromlist=["PluginRecord"]
        ).PluginRecord(name="extra", source="test", status=PluginStatus.LOADED)
    )
    runtime.plugin_registry.add_intercept(
        "extra",
        InterceptSpec("extra_hit", "plugin:extra"),
        lambda m, *, session_id, span=None: None,
    )
    names = [s.name for s, _h in runtime.intercepts.effective_chain()]
    assert names[-1] == "extra_hit"
    # A profile intercept_order pulls it to the front.
    runtime.profile.intercept_order = ["extra_hit"]
    names = [s.name for s, _h in runtime.intercepts.effective_chain()]
    assert names[0] == "extra_hit"


def test_failing_plugin_intercept_falls_through(runtime) -> None:  # type: ignore[no-untyped-def]
    from iris_harness.runtime.plugin_host.registry import PluginRecord

    runtime.plugin_registry.add_plugin(
        PluginRecord(name="flaky", source="test", status=PluginStatus.LOADED)
    )

    def boom(message: str, *, session_id: str, span: object = None) -> object:
        raise RuntimeError("flaky plugin")

    runtime.plugin_registry.add_intercept("flaky", InterceptSpec("flaky_hit", "plugin:flaky"), boom)
    runtime.profile.intercept_order = ["flaky_hit"]  # run it first
    out = runtime.chat("what time is it?", session_id="flaky")
    # The turn still completes through the next intercept (time_date).
    assert out.metadata.get("deterministic_time_date") is True
    rec = runtime.plugin_registry.get("flaky")
    assert rec is not None and rec.status is PluginStatus.DEGRADED
    assert "flaky plugin" in (rec.last_error or "")


def test_declared_plugin_row_without_registration_is_skipped(runtime, caplog) -> None:  # type: ignore[no-untyped-def]
    runtime.intercept_chain = (InterceptSpec("phantom", "plugin:nobody"),) + tuple(
        runtime.intercept_chain
    )
    names = [s.name for s, _h in runtime.intercepts.effective_chain()]
    assert "phantom" not in names and "time_date" in names


@pytest.mark.usefixtures("offline_services")  # the probes would ask the live stack
def test_health_snapshot_carries_plugin_verdict(runtime) -> None:  # type: ignore[no-untyped-def]
    snapshot = refresh(net_probe=False)
    plugin_checks = {c.target: c for c in snapshot.checks if c.kind.value == "plugin"}
    assert plugin_checks["plugin:system"].state.value == "green"
    runtime.plugin_registry.record_failure("system", where="tool:x", exc=RuntimeError("x"))
    snapshot = refresh(net_probe=False)
    plugin_checks = {c.target: c for c in snapshot.checks if c.kind.value == "plugin"}
    assert plugin_checks["plugin:system"].state.value == "yellow"


def test_drift_panel_sees_plugin_intercept_as_wired(runtime) -> None:  # type: ignore[no-untyped-def]
    from iris_harness.playground.drift import build_drift_report

    report = build_drift_report(runtime, config_dir=runtime.config_dir)
    intercepts = next(s for s in report.surfaces if s.surface == "intercepts")
    assert "time_date" in intercepts.in_sync
    assert "time_date" not in intercepts.declared_only
