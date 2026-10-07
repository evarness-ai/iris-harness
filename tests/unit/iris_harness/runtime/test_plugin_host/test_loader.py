"""Discovery + loading: builtin, home-dir, not found, unsupported trust, bad setup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.agent_executor import AgentExecutor
from iris_harness.runtime.plugin_host import (
    PluginRef,
    PluginRegistry,
    PluginStatus,
    discover_plugin,
    load_plugins,
    load_profile,
)
from iris_harness.runtime.plugin_host.loader import (
    add_in_process,
    describe_sources,
    in_process_plugin,
    load_plugin,
)
from iris_harness.sdk import HarnessServices
from iris_harness.services.heartbeat import HeartbeatScheduler


class _Gateway:
    def __init__(self) -> None:
        self.registered: list[Any] = []

    def register(self, connector: Any) -> None:
        self.registered.append(connector)


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
        f"name: {name}\nversion: 1.2.3\n{manifest_extra}", encoding="utf-8"
    )
    (d / "plugin.py").write_text(body, encoding="utf-8")
    return d


def test_builtin_system_is_discoverable() -> None:
    src = discover_plugin("system", skip_entry_points=True)
    assert src is not None
    assert src.kind == "builtin" and src.manifest.name == "system"
    assert {k.value for k in src.manifest.provides} == {"intercept", "tool", "heartbeat"}


def test_home_plugin_loads_and_registers(tmp_path: Path, services: HarnessServices) -> None:
    home = tmp_path / "home"
    _home_plugin(
        home,
        "mine",
        "def setup(api):\n"
        "    api.register_tool('mine_tool', 'd', lambda args: 'ok')\n"
        "    api.register_intercept('mine_hit', lambda m, *, session_id, span=None: None)\n"
        "    api.register_intent_handler('mine_agent', lambda task: 'x')\n"
        "    api.register_heartbeat('mine_tick', lambda d: None, schedule='interval:60')\n"
        "    api.register_confirmation_executor('mine_kind', lambda **kw: None)\n",
        manifest_extra="provides: [tool]\ntools:\n  mine_tool:\n    effect: read\n",
    )
    registry = PluginRegistry()
    rec = load_plugin(
        PluginRef(name="mine"),
        services=services,
        registry=registry,
        home_dir=home,
        skip_entry_points=True,
    )
    assert rec.status is PluginStatus.LOADED
    assert rec.source == f"home:{home / 'plugins' / 'mine'}"
    assert rec.manifest is not None and rec.manifest.version == "1.2.3"
    assert [t.name for t in registry.tools()] == ["mine_tool"]
    assert registry.intercept("mine_hit") is not None
    assert "mine_agent" in services.agent_executor.registered_agents()
    assert services.heartbeats.has_handler("mine_tick")
    assert "mine_kind" in registry.confirmation_executors()
    kinds = {r.kind.value for r in rec.registrations}
    assert kinds == {"tool", "intercept", "intent_handler", "heartbeat", "confirmation_executor"}


_NOTICE = "does not declare `party`"
_LOADER_LOG = "iris_harness.runtime.plugin_host.loader"


def _notices(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if _NOTICE in r.getMessage()]


@pytest.mark.parametrize(
    ("extra", "notice"),
    [("", True), ("party: untrusted\n", False), ("party: first-party\n", False)],
)
def test_an_undeclared_party_is_noticed_once_at_mount(
    tmp_path: Path,
    services: HarnessServices,
    caplog: pytest.LogCaptureFixture,
    extra: str,
    notice: bool,
) -> None:
    """Issue #97: a manifest that says nothing about `party` is untrusted, and says so."""
    home = tmp_path / "home"
    _home_plugin(home, "mine", "def setup(api):\n    return None\n", manifest_extra=extra)
    with caplog.at_level("WARNING", logger=_LOADER_LOG):
        rec = load_plugin(
            PluginRef(name="mine"),
            services=services,
            registry=PluginRegistry(),
            home_dir=home,
            skip_entry_points=True,
        )
    assert rec.status is PluginStatus.LOADED
    found = _notices(caplog)
    assert len(found) == (1 if notice else 0)
    if notice:
        assert "'mine'" in found[0] and "party: first-party | trusted-third-party" in found[0]
    assert rec.manifest is not None
    assert rec.manifest.party == (extra.split()[1] if extra else "untrusted")


def test_a_manifestless_entry_point_is_noticed(
    tmp_path: Path,
    services: HarnessServices,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loader-synthesised manifest never sets `party`: the third-party case is covered."""
    pkg = tmp_path / "bare_party_plugin"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def setup(api):\n    return None\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    class _EP:
        name = "bare_party"
        value = "bare_party_plugin:setup"
        dist = None

        def load(self) -> Any:
            import importlib

            return importlib.import_module("bare_party_plugin").setup

    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None: [_EP()])
    with caplog.at_level("WARNING", logger=_LOADER_LOG):
        rec = load_plugin(
            PluginRef(name="bare_party"),
            services=services,
            registry=PluginRegistry(),
            home_dir=tmp_path / "nohome",
        )
    assert rec.status is PluginStatus.LOADED
    assert len(_notices(caplog)) == 1
    assert rec.manifest is not None and rec.manifest.party == "untrusted"


def test_a_plugin_supplied_from_code_is_not_noticed(
    tmp_path: Path, services: HarnessServices, caplog: pytest.LogCaptureFixture
) -> None:
    """In-process plugins (the testing harness, `add_in_process`) are the caller's own."""
    supplied = in_process_plugin(lambda api: None, name="from_code")
    profile = add_in_process(
        load_profile(tmp_path / "config", "x", home_dir=tmp_path / "home", env={}), [supplied]
    )
    with caplog.at_level("WARNING", logger=_LOADER_LOG):
        records = load_plugins(
            profile,
            services=services,
            registry=PluginRegistry(),
            skip_entry_points=True,
            in_process=[supplied],
        )
    mounted = next(r for r in records if r.name == "from_code")
    assert mounted.status is PluginStatus.LOADED
    assert not [m for m in _notices(caplog) if "from_code" in m]


def test_not_found_and_disabled_and_mcp(tmp_path: Path, services: HarnessServices) -> None:
    home = tmp_path / "home"
    _home_plugin(home, "remote", "def setup(api):\n    pass\n", "trust: mcp\n")
    registry = PluginRegistry()
    prof = load_profile(tmp_path / "config", "x", home_dir=home, env={})
    prof.plugins = [
        PluginRef(name="ghost"),
        PluginRef(name="system", enabled=False),
        PluginRef(name="remote"),
    ]
    records = {
        r.name: r for r in load_plugins(prof, services=services, registry=registry, home_dir=home)
    }
    assert records["ghost"].status is PluginStatus.FAILED
    assert "not found" in (records["ghost"].load_error or "")
    assert records["system"].status is PluginStatus.DISABLED
    assert records["remote"].status is PluginStatus.UNSUPPORTED
    assert registry.tools() == []


def test_setup_exception_marks_failed_but_boot_continues(
    tmp_path: Path, services: HarnessServices
) -> None:
    home = tmp_path / "home"
    _home_plugin(home, "bad", "def setup(api):\n    raise RuntimeError('no')\n")
    _home_plugin(
        home,
        "good",
        "def setup(api):\n    api.register_tool('g', 'd', lambda a: '')\n",
        manifest_extra="provides: [tool]\ntools:\n  g:\n    effect: read\n",
    )
    registry = PluginRegistry()
    prof = load_profile(tmp_path / "config", "x", home_dir=home, env={})
    prof.plugins = [PluginRef(name="bad"), PluginRef(name="good")]
    load_plugins(prof, services=services, registry=registry, home_dir=home, skip_entry_points=True)
    assert registry.get("bad").status is PluginStatus.FAILED  # type: ignore[union-attr]
    assert "setup failed: RuntimeError: no" in (registry.get("bad").load_error or "")  # type: ignore[union-attr]
    assert registry.get("good").status is PluginStatus.LOADED  # type: ignore[union-attr]


def test_unmet_requirements_block_setup(tmp_path: Path, services: HarnessServices) -> None:
    home = tmp_path / "home"
    _home_plugin(
        home,
        "needy",
        "def setup(api):\n    raise AssertionError('must not run')\n",
        "requires:\n  packages: [definitely-not-a-real-package-xyz]\n",
    )
    registry = PluginRegistry()
    rec = load_plugin(PluginRef(name="needy"), services=services, registry=registry, home_dir=home)
    assert rec.status is PluginStatus.FAILED
    assert "required package not installed" in (rec.load_error or "")


def test_bad_manifest_is_reported(tmp_path: Path, services: HarnessServices) -> None:
    home = tmp_path / "home"
    d = home / "plugins" / "broken"
    d.mkdir(parents=True)
    (d / "manifest.yaml").write_text("name: broken\nunknown_key: 1\n", encoding="utf-8")
    rec = load_plugin(
        PluginRef(name="broken"), services=services, registry=PluginRegistry(), home_dir=home
    )
    assert rec.status is PluginStatus.FAILED
    assert "manifest invalid" in (rec.load_error or "")


def test_describe_sources_is_discovery_only(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _home_plugin(home, "mine", "def setup(api):\n    raise AssertionError('not called')\n")
    prof = load_profile(tmp_path / "config", "x", home_dir=home, env={})
    prof.plugins = [PluginRef(name="mine"), PluginRef(name="ghost")]
    rows = {r["name"]: r for r in describe_sources(prof, home_dir=home)}
    assert rows["mine"]["source"].startswith("home:") and rows["mine"]["version"] == "1.2.3"
    assert rows["ghost"]["source"] is None and rows["ghost"]["error"] == "not found"


# --- ADR-0110: register_tool takes the manifest's declaration into account -------------


def test_undeclared_tool_is_refused_and_recorded(tmp_path: Path, services: HarnessServices) -> None:
    home = tmp_path / "home"
    _home_plugin(
        home,
        "sneaky",
        "def setup(api):\n"
        "    api.register_tool('declared', 'd', lambda a: 'ok')\n"
        "    api.register_tool('undeclared_write', 'd', lambda a: 'wrote')\n",
        manifest_extra="provides: [tool]\ntools:\n  declared:\n    effect: read\n",
    )
    registry = PluginRegistry()
    rec = load_plugin(
        PluginRef(name="sneaky"),
        services=services,
        registry=registry,
        home_dir=home,
        skip_entry_points=True,
    )
    assert [t.name for t in registry.tools()] == ["declared"]
    assert rec.status is PluginStatus.DEGRADED
    assert "undeclared_write" in (rec.last_error or "") and "ADR-0110" in (rec.last_error or "")


def test_declared_effect_confirm_and_guidance_ride_on_the_tool(
    tmp_path: Path, services: HarnessServices
) -> None:
    home = tmp_path / "home"
    _home_plugin(
        home,
        "writer",
        "def setup(api):\n"
        "    api.register_tool('note_add', 'd', lambda a: 'ok')\n"
        "    api.register_tool('note_find', 'd', lambda a: 'ok')\n"
        "    api.register_tool('note_clear', 'd', lambda a: 'ok')\n",
        manifest_extra=(
            "provides: [tool]\n"
            "tools:\n"
            "  note_add:\n    effect: write\n"
            "  note_find:\n    effect: read\n    guidance: Use note_find before note_add.\n"
            "    pinned: true\n"
            "  note_clear:\n    effect: write\n    confirm: never\n"
        ),
    )
    registry = PluginRegistry()
    rec = load_plugin(
        PluginRef(name="writer"),
        services=services,
        registry=registry,
        home_dir=home,
        skip_entry_points=True,
    )
    assert rec.status is PluginStatus.LOADED
    tools = {t.name: t for t in registry.tools()}
    assert (tools["note_add"].effect, tools["note_add"].confirm) == ("write", "once")  # default
    assert (tools["note_find"].effect, tools["note_find"].confirm) == ("read", "never")
    assert tools["note_find"].guidance == "Use note_find before note_add."
    assert tools["note_find"].pinned is True and tools["note_add"].pinned is False
    assert (tools["note_clear"].effect, tools["note_clear"].confirm) == ("write", "never")


def test_a_plugin_without_a_manifest_registers_read_tools(tmp_path: Path) -> None:
    """Tests and probes mount with a bare PluginRecord: the defaults apply, nothing is refused."""
    from iris_harness.runtime.plugin_host.api import PluginAPI
    from iris_harness.runtime.plugin_host.registry import PluginRecord

    registry = PluginRegistry()
    registry.add_plugin(PluginRecord(name="bare", source="test", status=PluginStatus.LOADED))
    api = PluginAPI(
        plugin="bare",
        services=HarnessServices(
            config_dir=tmp_path,
            data_dir=tmp_path,
            tier_router=None,
            agent_executor=None,
            heartbeats=None,
            channels=None,
            deterministic_reply=lambda **kw: None,
        ),
        registry=registry,
    )
    api.register_tool("anything", "d", lambda a: "ok")
    (tool,) = registry.tools()
    assert (tool.effect, tool.confirm, tool.guidance) == ("read", "never", "")


def test_manifest_tool_declarations_are_validated(
    tmp_path: Path, services: HarnessServices
) -> None:
    import pytest

    from iris_harness.runtime.plugin_host.manifest import PluginManifest

    with pytest.raises(ValueError, match="confirm"):
        PluginManifest.model_validate(
            {
                "name": "x",
                "provides": ["tool"],
                "tools": {"t": {"effect": "read", "confirm": "once"}},
            }
        )
    with pytest.raises(ValueError, match="provides"):
        PluginManifest.model_validate({"name": "x", "tools": {"t": {"effect": "read"}}})
    with pytest.raises(ValueError, match="tool name"):
        PluginManifest.model_validate(
            {"name": "x", "provides": ["tool"], "tools": {"Bad-Name": {"effect": "read"}}}
        )
    with pytest.raises(ValueError):
        PluginManifest.model_validate(
            {"name": "x", "provides": ["tool"], "tools": {"t": {"effect": "delete"}}}
        )
    ok = PluginManifest.model_validate(
        {"name": "x", "provides": ["tool"], "tools": {"t": {"effect": "write"}}}
    )
    assert ok.tools["t"].confirm_mode == "once"


def test_answers_directly_rides_on_registered_and_declared_tools(
    tmp_path: Path, services: HarnessServices
) -> None:
    """``register_tool`` and ``declare_tool`` hand out the same declaration: a plugin agent
    running its own loop over its own tools gets what the shared pool gets."""
    from iris_harness.agent.agentic_core import ToolSpec
    from iris_harness.runtime.plugin_host.api import PluginAPI

    home = tmp_path / "home"
    _home_plugin(
        home,
        "digester",
        "def setup(api):\n    api.register_tool('digest', 'd', lambda a: 'ok')\n",
        manifest_extra=(
            "provides: [tool]\ntools:\n  digest:\n    effect: read\n    answers_directly: true\n"
        ),
    )
    registry = PluginRegistry()
    rec = load_plugin(
        PluginRef(name="digester"),
        services=services,
        registry=registry,
        home_dir=home,
        skip_entry_points=True,
    )
    assert rec.status is PluginStatus.LOADED
    (pooled,) = registry.tools()
    assert pooled.answers_directly is True

    api = PluginAPI(plugin="digester", services=services, registry=registry)
    own = api.declare_tool(ToolSpec(name="digest", description="d", call=lambda a: "ok"))
    assert own is not None and own.answers_directly is True and own.effect == "read"
    assert api.declare_tool(ToolSpec(name="stray", description="d", call=lambda a: "")) is None
    assert "stray" in (registry.get("digester").last_error or "")  # type: ignore[union-attr]


def test_answers_directly_is_refused_on_a_write_tool() -> None:
    import pytest

    from iris_harness.runtime.plugin_host.manifest import ToolDeclaration

    assert ToolDeclaration().answers_directly is False
    with pytest.raises(ValueError, match="answers_directly"):
        ToolDeclaration(effect="write", answers_directly=True)  # confirm once by default


def test_answers_directly_is_allowed_on_a_write_that_never_asks() -> None:
    """ADR-0110 amendment (2026-09-22): a write with ``confirm: never`` whose output is
    the full account ends the run on it; anything that asks or waits on a card cannot."""
    import pytest

    from iris_harness.runtime.plugin_host.manifest import ToolDeclaration

    ok = ToolDeclaration(effect="write", confirm="never", answers_directly=True)
    assert ok.answers_directly is True
    for held in (
        {"effect": "write", "confirm": "once"},
        {"effect": "write", "approval": "pinned"},
        {"effect": "destructive"},
    ):
        with pytest.raises(ValueError, match="answers_directly"):
            ToolDeclaration(**held, answers_directly=True)


# --- ADR-0118 build step 1: the destructive effect and its undo ----------------------


def _manifest(tools: dict[str, dict[str, object]]) -> object:
    from iris_harness.runtime.plugin_host.manifest import PluginManifest

    return PluginManifest.model_validate({"name": "x", "provides": ["tool"], "tools": tools})


_RESTORE = {"effect": "write", "confirm": "never"}


def test_a_destructive_tool_is_approved_not_confirmed() -> None:
    import pytest

    ok = _manifest({"trash": {"effect": "destructive"}})
    assert ok.tools["trash"].confirm_mode == "approval"  # type: ignore[attr-defined]
    # No way to declare `confirm: never` (or once) on a delete.
    for confirm in ("never", "once"):
        with pytest.raises(ValueError, match="confirm"):
            _manifest({"trash": {"effect": "destructive", "confirm": confirm}})
    with pytest.raises(ValueError, match="answers_directly"):
        _manifest({"trash": {"effect": "destructive", "answers_directly": True}})


def test_undo_names_a_declared_restoring_write() -> None:
    import pytest

    ok = _manifest({"trash": {"effect": "destructive", "undo": "restore"}, "restore": _RESTORE})
    assert ok.tools["trash"].undo == "restore"  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="not declared"):
        _manifest({"trash": {"effect": "destructive", "undo": "restore"}})
    # The undo must itself be a write that does not ask (ADR-0118 decision 3).
    for bad in ({"effect": "read"}, {"effect": "write"}, {"effect": "destructive"}):
        with pytest.raises(ValueError, match="confirm: never"):
            _manifest({"trash": {"effect": "destructive", "undo": "restore"}, "restore": bad})


def test_undo_only_applies_to_a_destructive_tool() -> None:
    import pytest

    for effect in ("read", "write"):
        with pytest.raises(ValueError, match="'undo' only applies"):
            _manifest({"t": {"effect": effect, "undo": "restore"}, "restore": _RESTORE})


def test_a_declared_destructive_tool_registers_with_the_approval_gate(
    tmp_path: Path, services: HarnessServices
) -> None:
    home = tmp_path / "home"
    _home_plugin(
        home,
        "mailbox",
        "def setup(api):\n"
        "    api.register_tool('trash_email', 'd', lambda a: 'trashed')\n"
        "    api.register_tool('restore_email', 'd', lambda a: 'restored')\n",
        manifest_extra=(
            "provides: [tool]\n"
            "tools:\n"
            "  trash_email:\n    effect: destructive\n    undo: restore_email\n"
            "  restore_email:\n    effect: write\n    confirm: never\n"
        ),
    )
    registry = PluginRegistry()
    rec = load_plugin(
        PluginRef(name="mailbox"),
        services=services,
        registry=registry,
        home_dir=home,
        skip_entry_points=True,
    )
    assert rec.status is PluginStatus.LOADED
    tools = {t.name: t for t in registry.tools()}
    assert (tools["trash_email"].effect, tools["trash_email"].confirm) == (
        "destructive",
        "approval",
    )
    assert (tools["restore_email"].effect, tools["restore_email"].confirm) == ("write", "never")


def test_undo_window_days_needs_an_undo_and_a_positive_number() -> None:
    import pytest

    ok = _manifest(
        {
            "trash": {"effect": "destructive", "undo": "restore", "undo_window_days": 30},
            "restore": _RESTORE,
        }
    )
    assert ok.tools["trash"].undo_window_days == 30  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="needs an 'undo'"):
        _manifest({"trash": {"effect": "destructive", "undo_window_days": 30}})
    with pytest.raises(ValueError):
        _manifest(
            {
                "trash": {"effect": "destructive", "undo": "restore", "undo_window_days": 0},
                "restore": _RESTORE,
            }
        )


def test_describe_and_the_declared_undo_ride_on_the_registered_tool(
    tmp_path: Path, services: HarnessServices
) -> None:
    """A plugin passes describe= to register_tool; the manifest supplies undo and its
    window; the pooled tool the loop reads carries all three (ADR-0118 step 4)."""
    home = tmp_path / "home"
    _home_plugin(
        home,
        "mailbox",
        "from iris_harness.sdk.types import ToolDescription\n"
        "def _describe(args):\n"
        "    return ToolDescription(title=f\"Trash {len(args['ids'])} emails\", lines=tuple(args['ids']))\n"
        "def setup(api):\n"
        "    api.register_tool('trash_email', 'd', lambda a: 'trashed', describe=_describe)\n"
        "    api.register_tool('restore_email', 'd', lambda a: 'restored')\n",
        manifest_extra=(
            "provides: [tool]\n"
            "tools:\n"
            "  trash_email:\n    effect: destructive\n    undo: restore_email\n"
            "    undo_window_days: 30\n"
            "  restore_email:\n    effect: write\n    confirm: never\n"
        ),
    )
    registry = PluginRegistry()
    rec = load_plugin(
        PluginRef(name="mailbox"),
        services=services,
        registry=registry,
        home_dir=home,
        skip_entry_points=True,
    )
    assert rec.status is PluginStatus.LOADED
    trash = next(t for t in registry.tools() if t.name == "trash_email")
    assert (trash.undo, trash.undo_window_days) == ("restore_email", 30)
    assert trash.describe is not None
    assert trash.describe({"ids": ["a", "b"]}).title == "Trash 2 emails"


# --- the tool-hook payload contract: content and verify are declarations ----------------


def test_content_and_verify_are_declared_and_default_to_internal_and_none() -> None:
    import pytest

    plain = _manifest({"t": {"effect": "read"}})
    assert plain.tools["t"].content == "internal"  # type: ignore[attr-defined]
    assert plain.tools["t"].verify is None  # type: ignore[attr-defined]
    ok = _manifest(
        {
            "fetch": {"effect": "read", "content": "external"},
            "save": {"effect": "write", "confirm": "never", "verify": "write_file"},
        }
    )
    assert ok.tools["fetch"].content == "external"  # type: ignore[attr-defined]
    assert ok.tools["save"].verify == "write_file"  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="content"):
        _manifest({"t": {"effect": "read", "content": "third-party"}})
    # A probe the ledger cannot run, and a probe on a read, are refused at load.
    with pytest.raises(ValueError, match="no registered side-effect probe"):
        _manifest({"t": {"effect": "write", "verify": "no_such_probe"}})
    with pytest.raises(ValueError, match="'verify' only applies"):
        _manifest({"t": {"effect": "read", "verify": "write_file"}})


def test_declared_content_and_verify_ride_on_the_tool(
    tmp_path: Path, services: HarnessServices
) -> None:
    home = tmp_path / "home"
    _home_plugin(
        home,
        "fetcher",
        "def setup(api):\n"
        "    api.register_tool('page_fetch', 'd', lambda a: 'ok')\n"
        "    api.register_tool('page_save', 'd', lambda a: 'ok')\n",
        manifest_extra=(
            "provides: [tool]\n"
            "tools:\n"
            "  page_fetch:\n    effect: read\n    content: external\n"
            "  page_save:\n    effect: write\n    confirm: never\n    verify: write_file\n"
        ),
    )
    registry = PluginRegistry()
    rec = load_plugin(
        PluginRef(name="fetcher"),
        services=services,
        registry=registry,
        home_dir=home,
        skip_entry_points=True,
    )
    assert rec.status is PluginStatus.LOADED
    tools = {t.name: t for t in registry.tools()}
    assert (tools["page_fetch"].content, tools["page_fetch"].verify) == ("external", None)
    assert (tools["page_save"].content, tools["page_save"].verify) == ("internal", "write_file")


# ------------------------------------------------- a reserved name is the harness's (#139)
def test_a_plugin_from_code_cannot_take_the_name_the_core_stamps_on_its_own_tools(
    tmp_path: Path, services: HarnessServices
) -> None:
    """The allow-list reads the ``system`` stamp as the core (``core:<tool>``). An in-process
    plugin is looked up before discovery, so one named ``system`` would shadow the builtin
    and be judged as the core: refused, and its ``setup`` never runs."""
    ran: list[str] = []

    def setup(api: Any) -> None:
        ran.append("setup")
        api.register_tool("impostor", "d", lambda args: "x")

    impostor = in_process_plugin(
        setup,
        name="system",
        manifest={"provides": ["tool"], "tools": {"impostor": {"effect": "read"}}},
    )
    prof = load_profile(tmp_path / "config", "x", home_dir=tmp_path / "home", env={})
    prof.plugins = [PluginRef(name="system")]
    registry = PluginRegistry()

    records = load_plugins(
        prof,
        services=services,
        registry=registry,
        skip_entry_points=True,
        in_process=[impostor],
    )

    (record,) = records
    assert record.status is PluginStatus.FAILED and "reserved" in (record.load_error or "")
    assert ran == [] and registry.tools() == []


def test_the_builtin_system_plugin_still_mounts_under_its_reserved_name(
    tmp_path: Path, services: HarnessServices
) -> None:
    prof = load_profile(tmp_path / "config", "x", home_dir=tmp_path / "home", env={})
    prof.plugins = [PluginRef(name="system")]
    (record,) = load_plugins(
        prof, services=services, registry=PluginRegistry(), skip_entry_points=True
    )
    assert record.status is PluginStatus.LOADED


def test_the_reserved_name_is_the_stamp_the_allow_list_reads_as_the_core() -> None:
    """Three places spell the core's stamp; they must not drift apart."""
    from iris_harness.agent.agentic_core import ToolSpec
    from iris_harness.kernel.governance import external_content_allow as allow
    from iris_harness.runtime.plugin_host.loader import RESERVED_PLUGIN_NAMES

    default_stamp = ToolSpec("t", "d", lambda a: "").plugin
    assert default_stamp == allow._SYSTEM
    assert default_stamp in RESERVED_PLUGIN_NAMES
    assert allow.scope_source(default_stamp, "wiki_search") == "core:wiki_search"
