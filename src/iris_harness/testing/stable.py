"""The stable tier as data, and the check that code imports nothing else (OSS plan R16).

The declaration is ``iris_harness/sdk/stable_tier.yaml``: whole packages (every module,
each module's ``__all__``), single names at their home module, and the non-import
promises -- registration kinds, the entry-point group, the manifest fields, the
quickstart CLI, the stable settings. :func:`check_stable_imports` walks Python files and reports every import
from the tree's own packages that is not in the tier. The repo's examples and the
``iris plugin new`` scaffold are held to it by a test; a plugin's own CI can run it over
the plugin the same way.

Static by design: it reads ``import`` statements (the AST), so it sees what a reader
sees. ``importlib.import_module("...")`` with a computed name is out of its reach.
"""

from __future__ import annotations

import ast
import importlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

_DECLARATION = "stable_tier.yaml"


@dataclass(frozen=True)
class StableTier:
    """The parsed declaration."""

    version: str
    packages: tuple[str, ...]
    names: dict[str, frozenset[str]]
    project_roots: frozenset[str]
    registration_kinds: tuple[str, ...]
    entry_point_group: str
    manifest_fields: tuple[str, ...]
    quickstart_cli: tuple[tuple[str, ...], ...]
    enforced_roots: tuple[str, ...]
    # The manifest's flavors, and what a declarative tool entry and each of its args carry.
    manifest_flavors: tuple[str, ...] = ()
    declarative_tool_fields: tuple[str, ...] = ()
    declarative_arg_fields: tuple[str, ...] = ()
    # The settings (environment variables) a release-1 email install relies on.
    settings: tuple[str, ...] = ()

    def in_package(self, module: str) -> bool:
        """``module`` is one of the whole stable packages or a module under one."""
        return any(module == pkg or module.startswith(pkg + ".") for pkg in self.packages)


@dataclass(frozen=True)
class StableImportViolation:
    """One import from the tree that the stable tier does not cover."""

    path: Path
    line: int
    module: str
    name: str | None
    reason: str

    def __str__(self) -> str:
        what = f"{self.module}.{self.name}" if self.name else self.module
        return f"{self.path}:{self.line}: {what}: {self.reason}"


def declaration_path() -> Path:
    import iris_harness.sdk as sdk

    return Path(sdk.__file__).with_name(_DECLARATION)


@cache
def stable_tier() -> StableTier:
    """The stable tier ``stable_tier.yaml`` declares."""
    raw: dict[str, Any] = yaml.safe_load(declaration_path().read_text(encoding="utf-8"))
    return StableTier(
        version=str(raw["version"]),
        packages=tuple(raw["packages"]),
        names={module: frozenset(names) for module, names in raw["names"].items()},
        project_roots=frozenset(raw["project_roots"]),
        registration_kinds=tuple(raw["registration_kinds"]),
        entry_point_group=str(raw["entry_point_group"]),
        manifest_fields=tuple(raw["manifest_fields"]),
        quickstart_cli=tuple(tuple(cmd) for cmd in raw["quickstart_cli"]),
        enforced_roots=tuple(raw["enforced_roots"]),
        manifest_flavors=tuple(raw.get("manifest_flavors") or ()),
        declarative_tool_fields=tuple(raw.get("declarative_tool_fields") or ()),
        declarative_arg_fields=tuple(raw.get("declarative_arg_fields") or ()),
        settings=tuple(raw.get("settings") or ()),
    )


@cache
def _exported(module: str) -> frozenset[str] | None:
    """``module.__all__``; ``None`` when the module cannot be imported."""
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return None
    return frozenset(getattr(mod, "__all__", ()))


def _is_module(module: str) -> bool:
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True


def _check_import(tier: StableTier, module: str) -> str | None:
    """Why ``import module`` is outside the tier, or ``None``."""
    if tier.in_package(module):
        return None if _is_module(module) else "no such stable module"
    if module in tier.names:
        return "only some names of this module are stable; import them by name"
    return "internal: not in the stable tier (iris_harness/sdk/stable_tier.yaml)"


def _check_from(tier: StableTier, module: str, name: str) -> str | None:
    """Why ``from module import name`` is outside the tier, or ``None``."""
    if tier.in_package(module):
        if name == "*":
            return "a star import takes internal names too; import the names you use"
        if _is_module(f"{module}.{name}"):
            return None  # a stable submodule
        exported = _exported(module)
        if exported is None:
            return "no such stable module"
        return None if name in exported else f"not in {module}.__all__"
    declared = tier.names.get(module)
    if declared is not None:
        return None if name in declared else "internal: not a stable name of this module"
    return "internal: not in the stable tier (iris_harness/sdk/stable_tier.yaml)"


def _imports(tree: ast.AST) -> Iterator[tuple[int, str, str | None]]:
    """``(line, module, name)`` for every absolute import; ``name`` is ``None`` for
    ``import module``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, None
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                yield node.lineno, node.module, alias.name


def check_file(path: Path, tier: StableTier | None = None) -> list[StableImportViolation]:
    """Every import in ``path`` from the tree's packages that the tier does not cover."""
    tier = tier or stable_tier()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[StableImportViolation] = []
    for line, module, name in _imports(tree):
        if module.split(".", 1)[0] not in tier.project_roots:
            continue
        reason = _check_import(tier, module) if name is None else _check_from(tier, module, name)
        if reason is not None:
            violations.append(StableImportViolation(path, line, module, name, reason))
    return violations


def check_stable_imports(paths: Iterable[Path]) -> list[StableImportViolation]:
    """Check every ``.py`` file under ``paths`` (files or directories)."""
    tier = stable_tier()
    violations: list[StableImportViolation] = []
    for root in paths:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for file in files:
            violations.extend(check_file(file, tier))
    return violations


__all__ = [
    "StableImportViolation",
    "StableTier",
    "check_file",
    "check_stable_imports",
    "declaration_path",
    "stable_tier",
]
