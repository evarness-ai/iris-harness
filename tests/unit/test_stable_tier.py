"""The stable tier (OSS plan R16): declared once, and the tree held to it.

``iris_harness/sdk/stable_tier.yaml`` declares what a plugin, an example and the
``iris plugin new`` scaffold may import and rely on. import-linter cannot hold the
examples to it -- ``examples/00-quickstart`` is not an importable package name, and a
forbidden contract cannot say "every module but these names" -- so this test does, with
``iris_harness.testing.check_stable_imports`` (an AST walk of every import).

Beside the import check, the declaration's non-import promises are pinned to the code:
the registration kinds, the entry-point group, the manifest fields, the quickstart CLI,
and a snapshot of every stable name, so removing one fails here (deprecate it first)
and adding one is a deliberate, reviewed promise.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

from iris_harness.testing import check_stable_imports, stable_tier

_ROOT = Path(__file__).resolve()
while not (_ROOT / "src" / "iris_harness").is_dir():
    if _ROOT == _ROOT.parent:
        raise RuntimeError("could not locate repo root (src/iris_harness)")
    _ROOT = _ROOT.parent

_SNAPSHOT = _ROOT / "tests" / "fixtures" / "stable_tier" / "names.txt"


def test_every_enforced_tree_imports_only_the_stable_tier() -> None:
    tier = stable_tier()
    assert tier.enforced_roots, "the declaration names the trees it holds"
    roots = [_ROOT / root for root in tier.enforced_roots]
    for root in roots:
        assert root.is_dir(), f"enforced root {root} is missing"
    violations = check_stable_imports(roots)
    assert violations == [], "\n".join(str(v) for v in violations)


_EXAMPLE = """\
import json
from pathlib import Path

import iris_harness.testing
from iris_harness.sdk import PluginAPI, llm
from iris_harness.sdk.llm import TierRouter
from iris_harness.testing import harness
from iris_personal.email.provider_api import MailProvider

from .helpers import local_thing

from iris_harness.runtime.bootstrap import build_runtime
import memris
from iris_harness.sdk.llm import not_a_stable_name
from iris_harness.sdk import *
import iris_personal.email.providers
from iris_personal.email.store import EmailStore
import iris_harness
"""


def test_a_deliberate_violation_is_caught(tmp_path: Path) -> None:
    example = tmp_path / "examples" / "99-bad" / "plugin.py"
    example.parent.mkdir(parents=True)
    example.write_text(_EXAMPLE, encoding="utf-8")

    found = {(v.line, v.module, v.name) for v in check_stable_imports([tmp_path / "examples"])}

    assert found == {
        (12, "iris_harness.runtime.bootstrap", "build_runtime"),
        (13, "memris", None),
        (14, "iris_harness.sdk.llm", "not_a_stable_name"),
        (15, "iris_harness.sdk", "*"),
        (16, "iris_personal.email.providers", None),
        (17, "iris_personal.email.store", "EmailStore"),
        (18, "iris_harness", None),
    }


def test_the_registration_kinds_are_the_declared_ones() -> None:
    from iris_harness.sdk import RegistrationKind

    assert tuple(kind.value for kind in RegistrationKind) == stable_tier().registration_kinds


def test_the_entry_point_group_is_the_declared_one() -> None:
    from iris_harness.runtime.plugin_host import ENTRY_POINT_GROUP

    assert ENTRY_POINT_GROUP == stable_tier().entry_point_group


def test_the_manifest_fields_are_the_declared_ones() -> None:
    from iris_harness.sdk import PluginManifest

    assert tuple(PluginManifest.model_fields) == stable_tier().manifest_fields


def test_the_manifest_flavors_and_declarative_keys_are_the_declared_ones() -> None:
    from typing import get_args

    from iris_harness.runtime.plugin_host.manifest import PluginFlavor, ToolArg, ToolDeclaration

    tier = stable_tier()
    assert get_args(PluginFlavor) == tier.manifest_flavors
    declared = tuple(ToolDeclaration.model_fields)
    assert declared[-len(tier.declarative_tool_fields) :] == tier.declarative_tool_fields
    assert tuple(ToolArg.model_fields) == tier.declarative_arg_fields


def _stable_names() -> set[str]:
    names: set[str] = set()
    for package in stable_tier().packages:
        root = importlib.import_module(package)
        modules = [package] + [
            info.name for info in pkgutil.walk_packages(root.__path__, prefix=f"{package}.")
        ]
        for module in modules:
            mod = importlib.import_module(module)
            exported = getattr(mod, "__all__", None)
            assert exported is not None, f"stable module {module} has no __all__"
            for name in exported:
                assert hasattr(mod, name), f"{module}.__all__ lists missing {name!r}"
                names.add(f"{module}:{name}")
    return names


def test_no_stable_name_disappears_and_none_appears_unreviewed() -> None:
    snapshot = {
        line.strip()
        for line in _SNAPSHOT.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    current = _stable_names()
    removed = sorted(snapshot - current)
    added = sorted(current - snapshot)
    assert not removed, (
        "stable names removed without a deprecation cycle (docs/reference/stable-api.md):\n"
        + "\n".join(removed)
    )
    assert not added, (
        f"new stable names -- each is a promise; add them to {_SNAPSHOT.relative_to(_ROOT)}:\n"
        + "\n".join(added)
    )


@pytest.mark.parametrize(
    "command", [cmd for cmd in stable_tier().quickstart_cli if cmd[0] != "email"]
)
def test_the_core_quickstart_commands_exist(command: tuple[str, ...]) -> None:
    from typer.testing import CliRunner

    from iris_harness.main import app

    result = CliRunner().invoke(app, [*command, "--help"])
    assert result.exit_code == 0, result.output
