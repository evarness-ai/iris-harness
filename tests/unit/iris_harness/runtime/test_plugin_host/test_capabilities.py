"""Declared capabilities (plugin-capabilities §2, step 3b).

Each rule has a test: the manifest block, provide / capability and their refusals, the
mount rules (requires, uses, providers before consumers), the fault boundary attributing a
provider's failure to the provider, fan-out, drift, and the views that print the graph.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Protocol

import pytest

from iris_harness.agent.agent_executor import AgentExecutor
from iris_harness.foundation import capabilities as catalogue
from iris_harness.foundation.capabilities import (
    CapabilitySpec,
    CapabilityUnavailable,
    MethodSpec,
)
from iris_harness.kernel.governance import GovernanceKernel
from iris_harness.kernel.governance.caller_policy import register_caller_policy
from iris_harness.kernel.governance.plugins.caller_policy import CallerPolicyHook
from iris_harness.kernel.governance.plugins.capability_redaction import CapabilityRedactionHook
from iris_harness.kernel.governance.plugins.tool_policy import ToolPolicyHook
from iris_harness.playground.drift import build_drift_report
from iris_harness.runtime.plugin_host import PluginRef, PluginRegistry, PluginStatus
from iris_harness.runtime.plugin_host.dump import capability_graph, render_text
from iris_harness.runtime.plugin_host.inventory import plugin_detail
from iris_harness.runtime.plugin_host.loader import describe_sources, load_plugins
from iris_harness.runtime.plugin_host.manifest import PluginManifest
from iris_harness.runtime.plugin_host.profile import EffectiveProfile
from iris_harness.runtime.plugin_host.registry import PluginRecord
from iris_harness.runtime.tool_access import compile_caller_policy
from iris_harness.sdk import HarnessServices
from iris_harness.sdk import capabilities as sdk_capabilities
from iris_harness.services.health.models import HealthState
from iris_harness.services.heartbeat import HeartbeatScheduler


class Ping(Protocol):
    def ping(self) -> str: ...


class Names(Protocol):
    def names(self) -> list[str]: ...


class _Fanned:
    """A fan-out for ``test.names``: one implementation over every provider."""

    def __init__(self, impls: Sequence[Any]) -> None:
        self.impls = list(impls)

    def names(self) -> list[str]:
        return [n for impl in self.impls for n in impl.names()]


class Rich(Protocol):
    """Every method shape the fault boundary must cover."""

    def ping(self) -> str: ...
    def boom(self) -> str: ...
    async def aping(self) -> str: ...
    async def aboom(self) -> str: ...
    def agen(self) -> AsyncIterator[int]: ...
    def gen(self) -> Iterator[int]: ...


TEXT = MethodSpec(fields=("",))
PING = CapabilitySpec(name="test.ping", protocol=Ping, methods={"ping": TEXT})
NAMES = CapabilitySpec(
    name="test.names",
    protocol=Names,
    methods={"names": MethodSpec(fields=("[]",))},
    fan_out=_Fanned,
)
RICH = CapabilitySpec(
    name="test.rich",
    protocol=Rich,
    methods={
        "ping": TEXT,
        "boom": TEXT,
        "aping": TEXT,
        "aboom": TEXT,
        "agen": MethodSpec(),
        "gen": MethodSpec(),
    },
)


@pytest.fixture(autouse=True)
def published(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Test capabilities in the catalogue for the test only (the SDK ships none before step 4).

    The catalogue is read-only, so the test swaps the module's map; the SDK facade was
    imported (above) before any test, so it still holds the real one.
    """
    monkeypatch.setattr(
        catalogue,
        "CAPABILITIES",
        MappingProxyType({s.name: s for s in (PING, NAMES, RICH)}),
    )
    yield
    register_caller_policy(None)


def _new_registry() -> PluginRegistry:
    """A registry whose capability calls are governed by a real kernel and caller policy."""
    registry = PluginRegistry()
    kernel = GovernanceKernel(audit_log=None)
    for hook in (CallerPolicyHook(), ToolPolicyHook(), CapabilityRedactionHook()):
        kernel.register(hook)
    kernel.init_lock()
    registry.bind_kernel(lambda: kernel)
    register_caller_policy(compile_caller_policy(registry, config_dir=Path("/nonexistent")))
    return registry


class _Gateway:
    def register(self, connector: Any) -> None:  # pragma: no cover - unused
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


PROVIDER = """
class Impl:
    secret = "not part of the interface"

    def ping(self):
        return "pong"

def setup(api):
    api.provide("test.ping", Impl())
"""

RICH_PROVIDER = """
import asyncio

class Impl:
    def ping(self):
        return "pong"

    def boom(self):
        raise RuntimeError("provider broke")

    async def aping(self):
        return "apong"

    async def aboom(self):
        raise RuntimeError("async provider broke")

    async def agen(self):
        yield 1
        raise RuntimeError("async generator broke")

    def gen(self):
        yield 1
        raise RuntimeError("generator broke")

def setup(api):
    api.provide("test.rich", Impl())
"""

RICH_CONSUMER = """
SEEN = {}

def setup(api):
    SEEN["cap"] = api.capability("test.rich")
"""

CONSUMER = """
SEEN = {}

def setup(api):
    SEEN["cap"] = api.capability("test.ping")
    SEEN["api"] = api
"""


def _plugin(home: Path, name: str, body: str, capabilities: str) -> None:
    d = home / "plugins" / name
    d.mkdir(parents=True)
    (d / "manifest.yaml").write_text(
        f"name: {name}\nversion: 1.0.0\ncapabilities:\n{capabilities}", encoding="utf-8"
    )
    (d / "plugin.py").write_text(body, encoding="utf-8")


def _load(
    home: Path, names: list[str], services: HarnessServices
) -> tuple[PluginRegistry, dict[str, PluginRecord]]:
    profile = EffectiveProfile(
        name="t", description="", plugins=[PluginRef(name=n) for n in names], intercept_order=[]
    )
    registry = _new_registry()
    records = load_plugins(
        profile, services=services, registry=registry, home_dir=home, skip_entry_points=True
    )
    return registry, {r.name: r for r in records}


def _seen(name: str) -> dict[str, Any]:
    seen: dict[str, Any] = sys.modules[f"iris_plugin_{name}_plugin"].SEEN
    return seen


def _record(registry: PluginRegistry, name: str, **caps: list[str]) -> PluginRecord:
    manifest = PluginManifest.model_validate({"name": name, "capabilities": caps})
    return registry.add_plugin(
        PluginRecord(name=name, source="test", status=PluginStatus.LOADED, manifest=manifest)
    )


# ------------------------------------------------------------------- the manifest block
def test_manifest_declares_provides_uses_requires() -> None:
    m = PluginManifest.model_validate(
        {
            "name": "p",
            "provides": ["tool"],
            "capabilities": {"provides": ["a.read"], "uses": ["b.read"], "requires": ["c.read"]},
        }
    )
    assert m.capabilities.provides == ("a.read",)
    assert m.capabilities.consumes == ("b.read", "c.read")
    # the top-level `provides:` keeps meaning registration kinds
    assert [k.value for k in m.provides] == ["tool"]


@pytest.mark.parametrize(
    "caps",
    [
        {"uses": ["mailread"]},  # not domain.verb
        {"uses": ["Mail.Read"]},
        {"provides": ["a.read"], "uses": ["a.read"]},  # consumes its own
        {"uses": ["a.read"], "requires": ["a.read"]},  # optional and required
        {"needs": ["a.read"]},  # unknown key
    ],
)
def test_manifest_refuses_a_bad_capabilities_block(caps: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        PluginManifest.model_validate({"name": "p", "capabilities": caps})


def test_a_capability_is_not_a_registration_kind() -> None:
    with pytest.raises(ValueError):
        PluginManifest.model_validate({"name": "p", "provides": ["test.ping"]})


# ------------------------------------------------------------- provide / capability
def test_declared_provider_reaches_declared_consumer() -> None:
    registry = _new_registry()
    _record(registry, "prov", provides=["test.ping"])
    _record(registry, "cons", uses=["test.ping"])

    class Impl:
        def ping(self) -> str:
            return "pong"

    assert registry.provide_capability("prov", "test.ping", Impl()) is True
    impl = registry.resolve_capability("cons", "test.ping")
    assert impl.ping() == "pong"
    assert registry.capability_providers("test.ping") == ("prov",)
    assert registry.describe()["capabilities"] == {"test.ping": ["prov"]}


async def test_weather_forecast_mounts_through_the_registry_and_resolves() -> None:
    """The published capability, end to end: provide, resolve, governed call, tool name."""
    from datetime import UTC, datetime

    real = {s.name: s for s in (catalogue.WEATHER_FORECAST,)}
    registry = _new_registry()
    _record(registry, "weather", provides=["weather.forecast"])
    _record(registry, "trip", uses=["weather.forecast"])
    now = datetime(2026, 10, 5, tzinfo=UTC)

    class Impl:
        async def forecast(self, location: str, days: int = 3) -> catalogue.Forecast:
            return catalogue.Forecast(location, now, ())

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(catalogue, "CAPABILITIES", MappingProxyType(real))
        assert registry.provide_capability("weather", "weather.forecast", Impl()) is True
        assert registry.capability_providers("weather.forecast") == ("weather",)
        impl = registry.resolve_capability("trip", "weather.forecast")
        assert impl is not None
        result = await impl.forecast("Lisbon")
        assert result.location == "Lisbon"

    tool = catalogue.capability_tool_name("weather.forecast", "forecast")
    assert tool == "capability:weather.forecast.forecast"
    assert catalogue.split_capability_tool(tool) == ("weather.forecast", "forecast")
    assert registry.caller_denial("plugin:trip", tool) is None
    assert registry.caller_denial("plugin:weather", tool) is not None  # provider does not consume


def test_undeclared_provide_is_refused_and_recorded() -> None:
    registry = _new_registry()
    rec = _record(registry, "prov")
    assert registry.provide_capability("prov", "test.ping", SimpleNamespace(ping=str)) is False
    assert registry.capability_providers("test.ping") == ()
    assert rec.status is PluginStatus.DEGRADED
    assert "capabilities: provides" in (rec.last_error or "")


def test_provide_of_an_unpublished_capability_is_refused() -> None:
    registry = _new_registry()
    rec = _record(registry, "prov", provides=["mail.read"])
    assert registry.provide_capability("prov", "mail.read", object()) is False
    assert "not published" in (rec.last_error or "")


def test_provide_missing_part_of_the_interface_is_refused() -> None:
    registry = _new_registry()
    rec = _record(registry, "prov", provides=["test.ping"])
    assert registry.provide_capability("prov", "test.ping", object()) is False
    assert "lacks ping" in (rec.last_error or "")
    # a member that is there but is not a method does not conform either
    assert registry.provide_capability("prov", "test.ping", SimpleNamespace(ping="x")) is False


def test_undeclared_use_is_refused_and_returns_none() -> None:
    registry = _new_registry()
    _record(registry, "prov", provides=["test.ping"])
    cons = _record(registry, "cons")
    registry.provide_capability("prov", "test.ping", SimpleNamespace(ping=lambda: "pong"))
    assert registry.resolve_capability("cons", "test.ping") is None
    assert cons.status is PluginStatus.DEGRADED
    assert "capabilities: uses" in (cons.last_error or "")


def test_a_second_provider_without_fan_out_is_refused() -> None:
    registry = _new_registry()
    _record(registry, "one", provides=["test.ping"])
    two = _record(registry, "two", provides=["test.ping"])
    assert registry.provide_capability("one", "test.ping", SimpleNamespace(ping=str))
    assert not registry.provide_capability("two", "test.ping", SimpleNamespace(ping=str))
    assert "one provider" in (two.last_error or "")


def test_several_providers_fan_out_into_one_implementation() -> None:
    registry = _new_registry()
    _record(registry, "gmail", provides=["test.names"])
    _record(registry, "imap", provides=["test.names"])
    _record(registry, "cons", requires=["test.names"])
    registry.provide_capability("gmail", "test.names", SimpleNamespace(names=lambda: ["a"]))
    registry.provide_capability("imap", "test.names", SimpleNamespace(names=lambda: ["b"]))
    impl = registry.resolve_capability("cons", "test.names")
    assert isinstance(impl, _Fanned)
    assert impl.names() == ["a", "b"]


def test_a_provider_that_is_not_mounted_provides_nothing() -> None:
    registry = _new_registry()
    rec = _record(registry, "prov", provides=["test.ping"])
    _record(registry, "cons", uses=["test.ping"])
    registry.provide_capability("prov", "test.ping", SimpleNamespace(ping=str))
    rec.status = PluginStatus.FAILED  # its setup raised after it provided
    assert registry.resolve_capability("cons", "test.ping") is None
    assert registry.provided_capabilities() == {}


# ----------------------------------------------------------------- the fault boundary
async def _drain_agen(agen: AsyncIterator[int]) -> list[int]:
    return [item async for item in agen]


# Each method shape, and how a consumer drives it to the provider's failure.
_SHAPES: dict[str, Any] = {
    "boom": lambda impl: impl.boom(),
    "aboom": lambda impl: asyncio.run(impl.aboom()),
    "agen": lambda impl: asyncio.run(_drain_agen(impl.agen())),
    "gen": lambda impl: list(impl.gen()),
}


@pytest.fixture()
def rich(tmp_path: Path, services: HarnessServices) -> tuple[Any, dict[str, PluginRecord]]:
    home = tmp_path / "home"
    _plugin(home, "rprov", RICH_PROVIDER, "  provides: [test.rich]\n")
    _plugin(home, "rcons", RICH_CONSUMER, "  uses: [test.rich]\n")
    _registry, recs = _load(home, ["rprov", "rcons"], services)
    return _seen("rcons")["cap"], recs


@pytest.mark.parametrize("method", sorted(_SHAPES))
def test_a_provider_failure_is_attributed_to_the_provider(
    rich: tuple[Any, dict[str, PluginRecord]], method: str
) -> None:
    impl, recs = rich
    with pytest.raises(RuntimeError, match="broke"):
        _SHAPES[method](impl)
    assert recs["rprov"].status is PluginStatus.DEGRADED
    assert recs["rprov"].failure_count == 1
    assert (recs["rprov"].last_error or "").startswith(f"capability:test.rich.{method}:")
    assert recs["rcons"].status is PluginStatus.LOADED


def test_working_calls_pass_through_the_boundary(
    rich: tuple[Any, dict[str, PluginRecord]],
) -> None:
    impl, recs = rich
    assert impl.ping() == "pong"
    assert asyncio.run(impl.aping()) == "apong"
    assert recs["rprov"].failure_count == 0


def test_the_proxy_exposes_only_the_protocol(tmp_path: Path, services: HarnessServices) -> None:
    home = tmp_path / "home"
    _plugin(home, "prov", PROVIDER, "  provides: [test.ping]\n")
    _plugin(home, "cons", CONSUMER, "  uses: [test.ping]\n")
    _load(home, ["prov", "cons"], services)
    impl = _seen("cons")["cap"]
    assert impl.ping() == "pong"
    for attr in ("secret", "_impl", "__dict__"):
        with pytest.raises(AttributeError):
            getattr(impl, attr)


def test_a_stale_proxy_stops_calling_its_provider(
    rich: tuple[Any, dict[str, PluginRecord]],
) -> None:
    impl, recs = rich
    assert impl.ping() == "pong"
    recs["rprov"].status = PluginStatus.FAILED  # the provider went away after resolution
    with pytest.raises(CapabilityUnavailable, match="rprov"):
        impl.ping()
    assert recs["rprov"].failure_count == 0  # refused before the provider ran


# ------------------------------------------------------------------------ mount rules
def test_providers_mount_before_consumers(tmp_path: Path, services: HarnessServices) -> None:
    home = tmp_path / "home"
    _plugin(home, "cons", CONSUMER, "  requires: [test.ping]\n")
    _plugin(home, "prov", PROVIDER, "  provides: [test.ping]\n")
    registry, recs = _load(home, ["cons", "prov"], services)  # consumer listed first
    assert [r.name for r in registry.plugins()] == ["prov", "cons"]
    assert recs["cons"].status is PluginStatus.LOADED
    assert _seen("cons")["cap"].ping() == "pong"  # resolved inside the consumer's setup


def test_missing_requires_keeps_the_plugin_unloaded_with_the_reason_in_health(
    tmp_path: Path, services: HarnessServices
) -> None:
    home = tmp_path / "home"
    _plugin(home, "cons", CONSUMER, "  requires: [test.ping]\n")
    registry, recs = _load(home, ["cons"], services)
    assert recs["cons"].status is PluginStatus.FAILED
    assert "required capability not provided: test.ping" in (recs["cons"].load_error or "")
    (check,) = registry.health_checks()
    assert check.state is HealthState.RED
    assert "test.ping" in check.detail


def test_missing_uses_loads_and_capability_is_none(
    tmp_path: Path, services: HarnessServices
) -> None:
    home = tmp_path / "home"
    _plugin(home, "cons", CONSUMER, "  uses: [test.ping]\n")
    _registry, recs = _load(home, ["cons"], services)
    assert recs["cons"].status is PluginStatus.LOADED
    assert _seen("cons")["cap"] is None


def test_missing_uses_is_yellow_and_names_the_capability_until_a_provider_mounts() -> None:
    registry = _new_registry()
    _record(registry, "prov", provides=["test.ping"])
    _record(registry, "cons", uses=["test.ping"])

    def line(name: str) -> Any:
        return next(c for c in registry.health_checks() if c.target == f"plugin:{name}")

    assert line("cons").state is HealthState.YELLOW
    assert "test.ping" in line("cons").detail
    assert "unavailable (degraded)" in line("cons").detail
    assert registry.unavailable_optional_capabilities("cons") == ("test.ping",)
    assert line("prov").state is HealthState.GREEN  # the provider has nothing missing

    registry.provide_capability("prov", "test.ping", SimpleNamespace(ping=str))  # late mount
    assert line("cons").state is HealthState.GREEN
    assert registry.degraded_reason("cons") is None

    registry.plugins()[0].status = PluginStatus.FAILED  # the provider goes away again
    assert line("cons").state is HealthState.YELLOW
    assert "test.ping" in line("cons").detail


def test_a_failing_consumer_with_a_missing_uses_names_both_causes() -> None:
    registry = _new_registry()
    rec = _record(registry, "cons", uses=["test.ping"])
    rec.status = PluginStatus.DEGRADED
    rec.failure_count, rec.last_error = 2, "ValueError: boom"
    reason = registry.degraded_reason("cons") or ""
    assert "2 failure(s); last: ValueError: boom" in reason
    assert "optional capability test.ping unavailable" in reason


def test_a_capability_cycle_still_mounts_in_profile_order(
    tmp_path: Path, services: HarnessServices
) -> None:
    home = tmp_path / "home"
    body = "def setup(api):\n    pass\n"
    _plugin(home, "a", body, "  provides: [test.ping]\n  uses: [test.names]\n")
    _plugin(home, "b", body, "  provides: [test.names]\n  uses: [test.ping]\n")
    registry, recs = _load(home, ["a", "b"], services)
    assert [r.name for r in registry.plugins()] == ["a", "b"]
    assert all(r.status is PluginStatus.LOADED for r in recs.values())


# ----------------------------------------------------------------------------- drift
def test_declared_but_unprovided_capability_is_drift() -> None:
    registry = _new_registry()
    _record(registry, "prov", provides=["test.ping", "test.names"])
    _record(registry, "cons", uses=["test.names"])
    registry.provide_capability("prov", "test.ping", SimpleNamespace(ping=str))
    surfaces = {
        s.surface: s for s in build_drift_report(SimpleNamespace(plugin_registry=registry)).surfaces
    }
    provided = surfaces["plugin_capabilities"]
    assert provided.declared_only == ("prov:test.names",)
    assert provided.in_sync == ("prov:test.ping",)
    assert surfaces["capability_uses"].declared_only == ("test.names",)


def test_a_capability_used_and_provided_is_in_sync() -> None:
    registry = _new_registry()
    _record(registry, "prov", provides=["test.ping"])
    _record(registry, "cons", uses=["test.ping"])
    registry.provide_capability("prov", "test.ping", SimpleNamespace(ping=str))
    surfaces = {
        s.surface: s for s in build_drift_report(SimpleNamespace(plugin_registry=registry)).surfaces
    }
    assert surfaces["plugin_capabilities"].ok and surfaces["capability_uses"].ok


# ----------------------------------------------------------------------------- views
def test_dump_config_prints_who_provides_and_uses_what(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _plugin(home, "prov", PROVIDER, "  provides: [test.ping]\n")
    _plugin(home, "cons", CONSUMER, "  uses: [test.ping]\n  requires: [test.names]\n")
    profile = EffectiveProfile(
        name="t",
        description="",
        plugins=[PluginRef(name="prov"), PluginRef(name="cons")],
        intercept_order=[],
    )
    rows = describe_sources(profile, home_dir=home)
    graph = capability_graph(rows)
    assert graph == {
        "test.names": {"provides": [], "uses": [], "requires": ["cons"]},
        "test.ping": {"provides": ["prov"], "uses": ["cons"], "requires": []},
    }
    text = render_text(
        {
            "profile": {"name": "t", "layers": [], "intercept_order": []},
            "available_profiles": [],
            "plugins": rows,
        }
    )
    assert "test.ping" in text and "provided by prov" in text and "used by cons" in text
    assert "provided by -" in text  # test.names: required, nobody provides it


def test_plugin_detail_shows_the_capabilities_from_each_side() -> None:
    registry = _new_registry()
    _record(registry, "prov", provides=["test.ping", "test.names"])
    _record(registry, "cons", uses=["test.ping"])
    registry.provide_capability("prov", "test.ping", SimpleNamespace(ping=str))
    prov = plugin_detail(registry, None, "prov")
    cons = plugin_detail(registry, None, "cons")
    assert prov is not None and cons is not None
    assert prov["capabilities"]["provides"] == [
        {"name": "test.ping", "provided": True, "used_by": ["cons"]},
        {"name": "test.names", "provided": False, "used_by": []},
    ]
    assert prov["drift"]["capabilities_declared_not_provided"] == ["test.names"]
    assert cons["capabilities"]["uses"] == [{"name": "test.ping", "providers": ["prov"]}]


# ------------------------------------------------------------------------ the catalogue
def test_a_spec_needs_a_domain_verb_name() -> None:
    with pytest.raises(ValueError):
        CapabilitySpec(name="ping", protocol=Ping, methods={"ping": TEXT})


class _WithData(Protocol):
    size: int

    def ping(self) -> str: ...


class _WithProperty(Protocol):
    @property
    def size(self) -> int: ...


@pytest.mark.parametrize("protocol", [_WithData, _WithProperty])
def test_protocol_members_must_be_methods(protocol: type[Any]) -> None:
    with pytest.raises(ValueError, match="must be methods"):
        CapabilitySpec(name="test.data", protocol=protocol, methods={})


def test_the_sdk_module_is_a_facade_over_the_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.undo()  # the real catalogue, not the test one
    assert sdk_capabilities.CAPABILITIES is catalogue.CAPABILITIES
    assert sdk_capabilities.CapabilitySpec is CapabilitySpec
    assert sdk_capabilities.CapabilityUnavailable is CapabilityUnavailable
    assert "is_capability_name" not in sdk_capabilities.__all__


_NO_SDK = """
import sys
from typing import Protocol
from iris_harness.foundation import capabilities as catalogue
from iris_harness.kernel.governance import GovernanceKernel
from iris_harness.kernel.governance.caller_policy import register_caller_policy
from iris_harness.kernel.governance.plugins.caller_policy import CallerPolicyHook
from iris_harness.runtime.plugin_host.manifest import PluginManifest
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_harness.runtime.tool_access import compile_caller_policy

class Ping(Protocol):
    def ping(self) -> str: ...

class Impl:
    def ping(self) -> str:
        return "pong"

catalogue.CAPABILITIES = {
    "test.ping": catalogue.CapabilitySpec(
        "test.ping", Ping, {"ping": catalogue.MethodSpec(fields=("",))}
    )
}
registry = PluginRegistry()
kernel = GovernanceKernel(audit_log=None)
kernel.register(CallerPolicyHook())
kernel.init_lock()
registry.bind_kernel(lambda: kernel)
register_caller_policy(compile_caller_policy(registry, config_dir=__import__("pathlib").Path("/x")))
for name, caps in (("prov", {"provides": ["test.ping"]}), ("cons", {"uses": ["test.ping"]})):
    registry.add_plugin(PluginRecord(
        name=name, source="t", status=PluginStatus.LOADED,
        manifest=PluginManifest.model_validate({"name": name, "capabilities": caps}),
    ))
assert registry.provide_capability("prov", "test.ping", Impl())
assert registry.resolve_capability("cons", "test.ping").ping() == "pong"
assert "iris_harness.sdk" not in sys.modules, "the host imported the SDK"
print("ok")
"""


def test_provide_works_in_a_process_that_never_imported_the_sdk(tmp_path: Path) -> None:
    import os
    import subprocess

    env = dict(os.environ, IRIS_HOME=str(tmp_path / "home"), IRIS_DATA_DIR=str(tmp_path))
    src = str(Path(__file__).resolve().parents[5] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    # A child process sees the refusing keyring (tests/conftest.py), not this test's
    # in-memory one, and a governed call without a vault master key is refused. Its own.
    from cryptography.fernet import Fernet

    env["IRIS_VAULT_MASTER_KEY"] = Fernet.generate_key().decode("utf-8")
    proc = subprocess.run(  # noqa: S603 -- our own interpreter, a fixed script
        [sys.executable, "-c", _NO_SDK], env=env, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("ok")


# --------------------------------------------- identities stay the harness's stamp
def test_a_plugin_cannot_reach_the_core_or_another_plugins_identity(
    services: HarnessServices,
) -> None:
    import dataclasses

    from iris_harness.runtime.plugin_host.api import PluginAPI
    from iris_harness.runtime.tool_service import ToolCatalogue, ToolService

    registry = _new_registry()
    tool_service = ToolService(tools=registry.tools, kernel=lambda: None)
    core_services = dataclasses.replace(services, tools=tool_service)
    api = PluginAPI(plugin="cons", services=core_services, registry=registry)

    # The registry (capability_for_core, resolve_capability for anyone) is not public ...
    assert not hasattr(api, "registry")
    # ... nor is the tool service (for_caller binds any caller): plugins see a catalogue.
    assert isinstance(api.services.tools, ToolCatalogue)
    assert not hasattr(api.services.tools, "for_caller")
    public = [getattr(api, n) for n in dir(api) if not n.startswith("_")]
    public += [getattr(api.services, f.name) for f in dataclasses.fields(api.services)]
    assert not any(isinstance(v, PluginRegistry | ToolService) for v in public)
    # The bound entries are this plugin's, and the core keeps its own service.
    assert api.tools is not None and api.tools.caller == "plugin:cons"
    assert core_services.tools is tool_service
