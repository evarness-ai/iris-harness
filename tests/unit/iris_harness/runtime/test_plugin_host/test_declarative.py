"""``flavor: declarative``: a plugin that is only its manifest (OSS plan decision 5).

Each tool names its function (``impl``), description and typed ``args``; the loader binds
them with no ``setup()`` of the plugin's, the arguments are checked before every call,
and the tools register through ``register_tool`` -- so the manifest's effect and gate
apply and the call goes through the governed runner like any tool's.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.agent_executor import AgentExecutor
from iris_harness.agent.tool_runner import ToolUnavailable
from iris_harness.runtime.plugin_host import PluginRef, PluginRegistry, PluginStatus
from iris_harness.runtime.plugin_host.loader import in_process_plugin, load_plugin
from iris_harness.runtime.plugin_host.manifest import PluginManifest
from iris_harness.sdk import HarnessServices
from iris_harness.services.heartbeat import HeartbeatScheduler

_HERE = __name__
CALLS: list[dict[str, Any]] = []


def add(a: int, b: int = 0, label: str = "sum") -> dict[str, Any]:
    CALLS.append({"a": a, "b": b, "label": label})
    return {"label": label, "value": a + b}


def shout(text: str) -> str:
    return text.upper()


def _manifest(**overrides: Any) -> dict[str, Any]:
    tools = {
        "calc_add": {
            "effect": "read",
            "description": "Add two integers.",
            "impl": f"{_HERE}:add",
            "args": {
                "a": {"type": "integer"},
                "b": {"type": "integer", "required": False, "default": 2},
                "label": {"type": "enum", "options": ["sum", "total"], "required": False},
            },
        },
        "calc_shout": {
            "effect": "write",
            "description": "Upper-case a note.",
            "impl": f"{_HERE}:shout",
            "args": {"text": {"type": "string"}},
        },
    }
    raw: dict[str, Any] = {
        "name": "calc",
        "flavor": "declarative",
        "provides": ["tool"],
        "tools": tools,
    }
    raw.update(overrides)
    return raw


@pytest.fixture()
def services(tmp_path: Path) -> HarnessServices:
    return HarnessServices(
        config_dir=tmp_path / "config",
        data_dir=tmp_path / "data",
        tier_router=None,
        agent_executor=AgentExecutor(),
        heartbeats=HeartbeatScheduler(),
        channels=None,
        deterministic_reply=lambda **kw: kw,
    )


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    CALLS.clear()


def _home(home: Path, raw: dict[str, Any]) -> None:
    import yaml

    d = home / "plugins" / raw["name"]
    d.mkdir(parents=True)
    (d / "manifest.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")


def _load(tmp_path: Path, services: HarnessServices, raw: dict[str, Any]) -> Any:
    home = tmp_path / "home"
    _home(home, raw)
    registry = PluginRegistry()
    record = load_plugin(
        PluginRef(name=raw["name"]),
        services=services,
        registry=registry,
        home_dir=home,
        skip_entry_points=True,
    )
    return record, registry


# ─── Mounting ────────────────────────────────────────────────────────────────


def test_a_manifest_only_plugin_mounts_and_its_tools_carry_the_declaration(
    tmp_path: Path, services: HarnessServices
) -> None:
    record, registry = _load(tmp_path, services, _manifest())  # no plugin.py anywhere

    assert record.status is PluginStatus.LOADED, record.load_error
    tools = {t.name: t for t in registry.tools()}
    assert set(tools) == {"calc_add", "calc_shout"}
    # The manifest's effect and gate ride on the tool, as for register_tool.
    assert (tools["calc_add"].effect, tools["calc_add"].confirm) == ("read", "never")
    assert (tools["calc_shout"].effect, tools["calc_shout"].confirm) == ("write", "once")
    # What the model reads: the description, then the arguments.
    assert tools["calc_add"].description == (
        'Add two integers. Args: {"a": integer, "b": integer, optional, default 2, '
        '"label": one of sum|total, optional}.'
    )
    assert {r.kind.value for r in record.registrations} == {"tool"}


def test_a_call_runs_the_function_with_defaults_and_encodes_the_result(
    tmp_path: Path, services: HarnessServices
) -> None:
    _record, registry = _load(tmp_path, services, _manifest())
    tools = {t.name: t for t in registry.tools()}

    assert tools["calc_add"].call({"a": 3}) == '{"label": "sum", "value": 5}'
    assert CALLS == [{"a": 3, "b": 2, "label": "sum"}]  # b from the manifest, label the fn's
    assert tools["calc_shout"].call({"text": "hi"}) == "HI"  # a string is the observation


@pytest.mark.parametrize(
    ("args", "problem"),
    [
        ({}, "missing argument 'a'"),
        ({"a": 1, "c": 2}, "unknown argument(s): c"),
        ({"a": "1"}, "argument 'a' must be an integer"),
        ({"a": True}, "argument 'a' must be an integer"),
        ({"a": 1.5}, "argument 'a' must be an integer"),
        ({"a": 1, "label": "avg"}, "argument 'label' must be one of sum, total"),
    ],
)
def test_bad_arguments_are_refused_before_the_function_runs(
    tmp_path: Path, services: HarnessServices, args: dict[str, Any], problem: str
) -> None:
    _record, registry = _load(tmp_path, services, _manifest())
    (tool,) = [t for t in registry.tools() if t.name == "calc_add"]

    assert tool.validate is not None and tool.validate(args) == problem
    assert tool.call(args) == f"error: {problem}"
    assert CALLS == []


def test_a_raising_function_is_charged_to_the_plugin(
    tmp_path: Path, services: HarnessServices
) -> None:
    raw = _manifest()
    raw["tools"]["calc_shout"]["args"]["text"] = {"type": "number"}  # str.upper on a float
    raw["tools"]["calc_shout"]["impl"] = f"{_HERE}:shout"
    record, registry = _load(tmp_path, services, raw)
    (tool,) = [t for t in registry.tools() if t.name == "calc_shout"]

    with pytest.raises(ToolUnavailable, match="AttributeError"):
        tool.call({"text": 1.5})

    assert record.failure_count == 1


@pytest.mark.parametrize(
    ("impl", "reason"),
    [
        ("no_such_module_xyz:add", "No module named"),
        (f"{_HERE}:missing", "no callable 'missing'"),
        (f"{_HERE}:shout", "cannot take the declared args"),  # shout has no 'a'
    ],
)
def test_an_impl_that_does_not_bind_fails_the_plugin(
    tmp_path: Path, services: HarnessServices, impl: str, reason: str
) -> None:
    raw = _manifest()
    raw["tools"]["calc_add"]["impl"] = impl
    record, registry = _load(tmp_path, services, raw)

    assert record.status is PluginStatus.FAILED
    assert reason in (record.load_error or "")
    assert registry.tools() == []  # nothing half-registered


def test_an_in_process_declarative_plugin_takes_no_setup() -> None:
    plugin = in_process_plugin(manifest=_manifest())
    assert plugin.setup is None and plugin.manifest.flavor == "declarative"
    with pytest.raises(ValueError, match="takes no setup"):
        in_process_plugin(lambda api: None, manifest=_manifest())
    with pytest.raises(TypeError, match="needs a setup"):
        in_process_plugin(manifest={"name": "plain"})


# ─── The manifest is checked ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw["tools"]["calc_add"].pop("impl"), "needs 'description' and 'impl'"),
        (lambda raw: raw["tools"]["calc_add"].pop("description"), "needs 'description'"),
        (lambda raw: raw.update(entrypoint="plugin:setup"), "has no 'entrypoint'"),
        (lambda raw: raw.update(cli="cli:register"), "has no 'cli'"),
        (lambda raw: raw.update(provides=["tool", "heartbeat"]), "provides only tools"),
        # No code runs at mount, so there is nothing to register a search provider.
        (lambda raw: raw.update(search_providers=["web"]), "not search_providers"),
        (lambda raw: raw.update(tools={}, provides=[]), "at least one tool"),
        (
            lambda raw: raw["tools"]["calc_add"].update(impl="not a path"),
            "String should match pattern",
        ),
        (
            lambda raw: raw["tools"]["calc_add"]["args"].update({"Bad": {"type": "string"}}),
            "not an argument name",
        ),
        (
            lambda raw: raw["tools"]["calc_add"]["args"].update({"u": {"type": "enum"}}),
            "lists its 'options'",
        ),
        (
            lambda raw: raw["tools"]["calc_add"]["args"].update(
                {"u": {"type": "string", "options": ["x"]}}
            ),
            "only applies to an enum",
        ),
        (
            lambda raw: raw["tools"]["calc_add"]["args"].update(
                {"u": {"type": "integer", "default": 1}}
            ),
            "required: false",
        ),
        (
            lambda raw: raw["tools"]["calc_add"]["args"].update(
                {"u": {"type": "integer", "required": False, "default": "one"}}
            ),
            "must be an integer",
        ),
    ],
)
def test_a_bad_declarative_manifest_is_refused(change: Any, message: str) -> None:
    raw = _manifest()
    change(raw)
    with pytest.raises(ValueError, match=message):
        PluginManifest.model_validate(raw)


def test_a_python_plugin_may_not_carry_a_binding() -> None:
    raw = _manifest(flavor="python")
    with pytest.raises(ValueError, match="only a 'flavor: declarative' plugin does"):
        PluginManifest.model_validate(raw)


def test_a_refused_manifest_leaves_the_plugin_failed(
    tmp_path: Path, services: HarnessServices
) -> None:
    raw = _manifest()
    raw["tools"]["calc_add"].pop("impl")
    record, registry = _load(tmp_path, services, raw)
    assert record.status is PluginStatus.FAILED
    assert "manifest invalid" in (record.load_error or "")


# ─── Installed: the entry point names the package ────────────────────────────


@dataclass
class _FakeEntryPoint:
    name: str
    value: str
    dist: object | None = None


def test_an_installed_declarative_plugin_mounts_from_its_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, services: HarnessServices
) -> None:
    import yaml

    pkg = tmp_path / "acme_calc"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "functions.py").write_text("def double(n):\n    return n * 2\n")
    raw = {
        "name": "acme-calc",
        "flavor": "declarative",
        "provides": ["tool"],
        "tools": {
            "acme_double": {
                "description": "Double a number.",
                "impl": "acme_calc.functions:double",
                "args": {"n": {"type": "number"}},
            }
        },
    }
    (pkg / "manifest.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    ep = _FakeEntryPoint("acme-calc", "acme_calc")
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None: [ep])
    registry = PluginRegistry()
    try:
        record = load_plugin(PluginRef(name="acme-calc"), services=services, registry=registry)
        assert record.status is PluginStatus.LOADED, record.load_error
        assert record.source == "entry_point:acme_calc"
        (tool,) = registry.tools()
        assert tool.call({"n": 4}) == "8"
    finally:
        for mod in [m for m in sys.modules if m.startswith("acme_calc")]:
            del sys.modules[mod]


# ─── Governed: a model's call goes through the kernel ────────────────────────


def test_a_model_call_to_a_declarative_tool_is_governed_and_audited() -> None:
    from iris_harness.testing import harness, plugin

    script = {
        "rules": [
            {
                "name": "answer",
                "match": {"user": '(?s)Observation:.*"value": 7'},
                "reply": {"content": "Thought: done.\nFinal Answer: It is 7."},
            },
            {
                "name": "call",
                "match": {"user": "User: add five and two"},
                "reply": {
                    "content": 'Thought: add.\nAction: calc_add\nAction Input: {"a": 5, "b": 2}'
                },
            },
        ]
    }
    with harness(plugins=[plugin(manifest=_manifest())], fake_model=script) as h:
        assert h.plugin_loaded("calc"), h.plugins()["calc"]
        result = h.chat("add five and two")
        assert result.text == "It is 7.", result.error
        assert CALLS == [{"a": 5, "b": 2, "label": "sum"}]
        assert {row.tool for row in h.audit_rows(hook_point="pre_tool_use")} == {"calc_add"}
        assert {row.tool for row in h.audit_rows(hook_point="post_tool_use")} == {"calc_add"}
        assert h.audit_gaps() == []
