"""``iris plugin new``: a standalone, installable plugin package, rendered from templates.

OSS plan R10 (plugins are the contribution unit) and the L2 exit criterion: what this
writes passes its own tests out of the box. Each kind is a template tree under
``templates/plugin/<kind>/``, laid over ``templates/plugin/_common/``:

* ``pyproject.toml`` -- the ``iris_harness.plugins`` entry point, the package data (the
  manifest), pytest pointed at ``src/``;
* ``src/<package>/`` -- ``plugin.py`` with ``setup(api)`` and ``manifest.yaml`` declaring
  what the kind registers (tools with their effect and content, the channel, the mail
  provider's ``identity: provides``...); a ``skill`` is declarative -- its manifest binds
  each tool to a function, and it has no ``plugin.py``;
* ``tests/`` -- tests that import only the stable tier (``iris_harness.sdk``,
  ``iris_harness.testing``, the mail-provider facade) and run the plugin in a real,
  governed harness on the scripted fake model, offline.

The templates are Python files the stable-tier test reads as they are (``enforced_roots``
in ``iris_harness/sdk/stable_tier.yaml``), so a placeholder is spelled as an identifier:
``__tmpl_package`` becomes the plugin's package name. A file whose name ends ``.tmpl`` loses
the suffix (``pyproject.toml.tmpl``: a real ``pyproject.toml`` in the package would be read
by the tools that walk the tree).

This module is the logic; ``iris_harness.cli.plugins`` renders it.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any

import yaml

from iris_harness.foundation.plugin_dirs import MANIFEST_FILENAME

if TYPE_CHECKING:
    # Imported when a plugin is written, not when `iris` starts: the CLI imports this
    # module for the kinds its --help lists, and the manifest schema and the testing
    # package pull in most of the harness.
    from iris_harness.runtime.plugin_host.manifest import PluginManifest
    from iris_harness.testing.stable import StableTier

TEMPLATE_ROOT = Path(__file__).with_name("templates") / "plugin"
_COMMON = "_common"
_TMPL_SUFFIX = ".tmpl"
# Lower-case words joined by single hyphens or underscores: a valid manifest name
# (``[a-z][a-z0-9_-]*``) that also maps to a clean package and tool name.
_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")
_MAX_NAME = 40
_SKIPPED = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".DS_Store"}


@dataclass(frozen=True)
class PluginKind:
    """One ``--kind``, as ``templates/plugin/kinds.yaml`` declares it."""

    name: str
    summary: str
    distribution: str
    dependencies: tuple[str, ...] = ()
    # What the entry point names; ``{package}`` is the import package.
    entry_point: str = "{package}.plugin:setup"


def _load_kinds() -> dict[str, PluginKind]:
    raw: dict[str, dict[str, Any]] = yaml.safe_load(
        (TEMPLATE_ROOT / "kinds.yaml").read_text(encoding="utf-8")
    )
    return {
        name: PluginKind(
            name=name,
            summary=str(spec["summary"]),
            distribution=str(spec["distribution"]),
            dependencies=tuple(str(d) for d in spec.get("dependencies") or ()),
            entry_point=str(spec.get("entry_point") or PluginKind.entry_point),
        )
        for name, spec in raw.items()
    }


PLUGIN_KINDS: dict[str, PluginKind] = _load_kinds()
KINDS: tuple[str, ...] = tuple(PLUGIN_KINDS)


class ScaffoldError(ValueError):
    """The plugin cannot be generated as asked; the message says why and what to do."""


class _Placeholders(Template):
    """``__tmpl_<var>``: a placeholder that is itself a Python identifier, so a template
    ``.py`` file parses (and is import-checked) before it is rendered."""

    delimiter = "__tmpl_"
    idpattern = r"[a-z][a-z0-9_]*"
    flags = re.ASCII  # case-sensitive: ``__tmpl_package``, never ``__TMPL_PACKAGE``


@dataclass(frozen=True)
class PluginNames:
    """Every name the templates use, derived from the one the author gave."""

    name: str  # the plugin (manifest ``name``, entry-point name, profile row)
    package: str  # the import package
    tool: str  # the snake-case stem of the plugin's tool / channel / provider names
    dist: str  # the distribution (``pip install <dist>``)
    title: str  # for prose

    def variables(self, kind: PluginKind, tier: StableTier) -> dict[str, str]:
        major, minor = (tier.version.split(".") + ["0", "0"])[:2]
        return {
            "name": self.name,
            "package": self.package,
            "tool": self.tool,
            "dist": self.dist,
            "title": self.title,
            "kind": kind.name,
            "summary": kind.summary,
            "distribution": kind.distribution,
            "extra_dependencies": "".join(f', "{dep}"' for dep in kind.dependencies),
            # The stable tier's minor series: a 0.x minor may carry a deprecation's removal
            # (docs/reference/stable-api.md), so a plugin pins the series it was built on.
            "harness_requirement": f">={major}.{minor},<{major}.{int(minor) + 1}",
            "entry_point_group": tier.entry_point_group,
            "entry_point": kind.entry_point.format(package=self.package),
        }


@dataclass(frozen=True)
class ScaffoldResult:
    """What :func:`new_plugin` wrote."""

    root: Path
    kind: str
    names: PluginNames
    files: tuple[Path, ...]
    manifest: PluginManifest

    @property
    def package_dir(self) -> Path:
        return self.root / "src" / self.names.package


def plugin_names(name: str) -> PluginNames:
    """Derive the package, tool and distribution names; raise :class:`ScaffoldError` for a
    name that cannot be all of them."""
    if not _NAME.match(name):
        raise ScaffoldError(
            f"plugin name {name!r} is not valid: use lower-case letters and digits, starting "
            "with a letter, words joined by single '-' or '_' (e.g. 'weather-now')"
        )
    if len(name) > _MAX_NAME:
        raise ScaffoldError(f"plugin name {name!r} is longer than {_MAX_NAME} characters")
    snake = name.replace("-", "_")
    if keyword.iskeyword(snake):
        raise ScaffoldError(f"plugin name {name!r} is a Python keyword; pick another")
    return PluginNames(
        name=name,
        # Prefixed, so a plugin called ``email`` or ``json`` never shadows a module of
        # that name; the entry-point name is the plugin name, unprefixed.
        package=f"iris_plugin_{snake}",
        tool=snake,
        dist=f"iris-plugin-{snake.replace('_', '-')}",
        title=" ".join(word.capitalize() for word in re.split(r"[-_]", name)),
    )


def _template_files(kind: str) -> dict[Path, Path]:
    """``relative output path (still with placeholders) -> template file``; the kind's own
    files replace the common ones of the same path."""
    files: dict[Path, Path] = {}
    for layer in (_COMMON, kind):
        base = TEMPLATE_ROOT / layer
        for path in sorted(base.rglob("*")):
            if path.is_dir() or _SKIPPED & set(path.relative_to(base).parts):
                continue
            files[path.relative_to(base)] = path
    return files


def _output_path(relative: Path, variables: dict[str, str]) -> Path:
    parts = [_Placeholders(part).substitute(variables) for part in relative.parts]
    if parts[-1].endswith(_TMPL_SUFFIX):
        parts[-1] = parts[-1][: -len(_TMPL_SUFFIX)]
    return Path(*parts)


def new_plugin(
    name: str,
    kind: str,
    *,
    parent: Path,
    force: bool = False,
) -> ScaffoldResult:
    """Write the ``kind`` plugin ``name`` into ``parent/<name>`` and return what was written.

    Refuses a directory that already holds files unless ``force`` (which overwrites the
    files the scaffold writes and leaves every other file alone). The rendered manifest is
    validated against the manifest schema before returning.
    """
    from iris_harness.runtime.plugin_host.manifest import load_manifest
    from iris_harness.testing.stable import stable_tier

    if kind not in KINDS:
        raise ScaffoldError(f"unknown kind {kind!r}: one of {', '.join(KINDS)}")
    names = plugin_names(name)
    root = (parent / name).resolve()
    if root.exists() and not root.is_dir():
        raise ScaffoldError(f"{root} exists and is not a directory")
    if root.is_dir() and any(root.iterdir()) and not force:
        raise ScaffoldError(f"{root} already exists and is not empty; pass --force to overwrite")
    variables = names.variables(PLUGIN_KINDS[kind], stable_tier())

    rendered: list[tuple[Path, str]] = []
    for relative, template in _template_files(kind).items():
        text = template.read_text(encoding="utf-8")
        try:
            body = _Placeholders(text).substitute(variables)
        except (KeyError, ValueError) as exc:  # a template bug, never the author's
            raise RuntimeError(f"template {template} has a bad placeholder: {exc}") from exc
        rendered.append((root / _output_path(relative, variables), body))

    written: list[Path] = []
    for path, body in rendered:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8", newline="\n")
        written.append(path)

    manifest = load_manifest(root / "src" / names.package / MANIFEST_FILENAME)
    if manifest.name != name:  # pragma: no cover - a template bug
        raise RuntimeError(f"rendered manifest names {manifest.name!r}, not {name!r}")
    return ScaffoldResult(
        root=root, kind=kind, names=names, files=tuple(sorted(written)), manifest=manifest
    )


__all__ = [
    "KINDS",
    "PLUGIN_KINDS",
    "PluginKind",
    "TEMPLATE_ROOT",
    "PluginNames",
    "ScaffoldError",
    "ScaffoldResult",
    "new_plugin",
    "plugin_names",
]
