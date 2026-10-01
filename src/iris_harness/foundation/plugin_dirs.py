"""Where installed plugins live on disk — found without importing or loading them.

The plugin host (``runtime/plugin_host/loader.py``) loads the plugins a profile names.
Some things a plugin ships are needed without a runtime and whether or not the plugin is
mounted: its memory vocabulary (ADR-0115 — what was stored with a plugin's terms must
stay readable when the plugin is switched off). This module answers only "which plugin
directories exist", from the same three places the loader looks, in the same order:

1. **builtin** — each package under ``iris_harness.plugins_builtin`` with a manifest;
2. **entry point** — each ``iris_harness.plugins`` entry point, its package directory;
3. **home** — ``$IRIS_HOME/plugins/<name>/`` with a manifest.

A name found earlier wins, as it does when loading. Nothing here imports a plugin's
module: packages are located with ``importlib.util.find_spec`` on the package, which
runs only its (and its parents') ``__init__``.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "iris_harness.plugins"
BUILTIN_PACKAGE = "iris_harness.plugins_builtin"
MANIFEST_FILENAME = "manifest.yaml"


def iris_home() -> Path:
    """``$IRIS_HOME`` (tests relocate it) else ``~/.iris``."""
    home = os.environ.get("IRIS_HOME")
    return Path(home).expanduser() if home else Path.home() / ".iris"


def _package_dir(package: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    locations = list(spec.submodule_search_locations or [])
    if locations:
        return Path(locations[0])
    return Path(spec.origin).parent if spec.origin else None


def _builtin_dirs() -> list[tuple[str, Path]]:
    root = _package_dir(BUILTIN_PACKAGE)
    if root is None:
        return []
    return [
        (child.name, child)
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / MANIFEST_FILENAME).is_file()
    ]


def _entry_point_dirs() -> list[tuple[str, Path]]:
    try:
        eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 — metadata scan is best-effort
        return []
    found: list[tuple[str, Path]] = []
    for ep in sorted(eps, key=lambda e: e.name):
        module = ep.value.split(":", 1)[0].strip()
        package = module.rpartition(".")[0] or module
        directory = _package_dir(package)
        if directory is not None:
            found.append((ep.name, directory))
    return found


def _home_dirs(home: Path) -> list[tuple[str, Path]]:
    plugins = home / "plugins"
    if not plugins.is_dir():
        return []
    return [
        (child.name, child)
        for child in sorted(plugins.iterdir())
        if child.is_dir() and (child / MANIFEST_FILENAME).is_file()
    ]


def installed_plugin_dirs(home: Path | None = None) -> list[tuple[str, Path]]:
    """``(name, directory)`` of every installed plugin, first place found wins."""
    seen: dict[str, Path] = {}
    for name, directory in (
        *_builtin_dirs(),
        *_entry_point_dirs(),
        *_home_dirs(home or iris_home()),
    ):
        seen.setdefault(name, directory)
    return list(seen.items())


__all__ = [
    "BUILTIN_PACKAGE",
    "ENTRY_POINT_GROUP",
    "MANIFEST_FILENAME",
    "installed_plugin_dirs",
    "iris_home",
]
