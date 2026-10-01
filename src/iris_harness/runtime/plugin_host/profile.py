"""Profiles: which plugins mount, in what order, with what trust (decision 4).

A profile is YAML::

    name: default
    description: The harness with its reference plugins.
    plugins:
      - name: system            # required
        enabled: true           # default true
        trust: in-process       # in-process | mcp (default from the manifest)
        optional: false         # true: mount only if installed here (below)
    intercept_order: []         # optional: intercept names to run first, in this order

    prefer_when_installed: []   # optional: profiles to run instead when unnamed (below)
    requires_one_of: []         # optional: [[a, b], ...] -- see "when none is named"

Layers apply in a fixed order, later wins:

1. the shipped profile ``config/profiles/<name>.yaml`` (``IRIS_PROFILE``, default ``default``)
2. the user's ``$IRIS_HOME/profile.yaml`` (plugins merge by name; ``intercept_order`` replaces)
3. environment: ``IRIS_PLUGINS_DISABLE=a,b`` / ``IRIS_PLUGINS_ENABLE=c,d``

``iris --dump-config`` prints the effective result and which layer set each value.

**Which profile, when none is named.** With no ``name`` and no ``IRIS_PROFILE``, the
``default`` profile's ``prefer_when_installed`` list is tried in order: the first
profile whose every enabled plugin is installed and loadable here (found, and its
manifest's ``requires`` met) runs instead. That is how an install carrying the email
slice comes up as the email assistant with no setting at all (OSS plan R2/R6): Gmail
when the ``email`` extra's packages are there, IMAP (stdlib only) either way; a tree
without the email plugins keeps ``default``. The names live in
YAML, not here: the core knows no domain plugin's name. A named profile, from either
place, is always taken as named.

**Optional entries.** A plugin row with ``optional: true`` mounts when it is installed
and loadable here, and is otherwise left out of the effective profile (a line in
``layers`` says so) instead of loading as ``failed``. That is how one profile serves
installs that carry different providers: ``email`` lists Gmail and IMAP as optional, so
an install without the Google packages is still the email assistant, over IMAP. The
check is the same one the preference uses, and it happens here, once, so the loader,
the CLI registration and the web nav all see the same list.

For the preference, optional rows are not required; instead ``requires_one_of`` names
groups of which at least one plugin must be available (``[[gmail, imap]]``: the email
assistant needs *a* mailbox provider, whichever is installed).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

DEFAULT_PROFILE_NAME = "default"
PROFILE_ENV = "IRIS_PROFILE"
DISABLE_ENV = "IRIS_PLUGINS_DISABLE"
ENABLE_ENV = "IRIS_PLUGINS_ENABLE"

# Built-in fallback when the shipped profile file is missing or unreadable —
# config-with-defaults, like DEFAULT_INTERCEPTS: a runtime built against a bare
# config dir (tests, a stripped deployment) still mounts the reference plugin.
# Mounted when the shipped profile is missing or invalid. This is not "the smallest
# useful set" — it is "what the core used to do on its own", so a runtime built
# against a bare config dir behaves as it did before these capabilities became
# plugins. A capability extracted from the core is added here as it leaves.
#: The fallback when the shipped profile file is missing or invalid. Core-only since
#: M6.1b, matching `config/profiles/default.yaml`: the domain plugins live in a second
#: root the harness cannot assume is installed (OSS plan M6, decisions 2 and 10), so
#: naming them here would make a bare config dir try to mount plugins that are absent.
DEFAULT_PLUGINS: tuple[str, ...] = (
    "system",
    "telegram_channel",
    "web_channel",
    "web_push_channel",
    "research",
    "code_exec",
)


class PluginRef(BaseModel):
    """One row of a profile's ``plugins`` list."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, pattern=r"^[a-z][a-z0-9_-]*$")
    enabled: bool = True
    trust: Literal["in-process", "mcp"] | None = None  # None → manifest default
    # Mount only if installed and loadable here; otherwise left out, not failed.
    optional: bool = False


class ProfileDocument(BaseModel):
    """The on-disk shape of one profile layer."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    description: str = ""
    plugins: list[PluginRef] = Field(default_factory=list)
    intercept_order: list[str] | None = None
    # Read from the `default` profile only, and only when no profile is named: the
    # profiles to run instead, first whose plugins are all installed wins.
    prefer_when_installed: list[str] = Field(default_factory=list)
    # For that preference: each group needs at least one available plugin.
    requires_one_of: list[list[str]] = Field(default_factory=list)


@dataclass
class EffectiveProfile:
    """The merged result of all layers, plus provenance for ``--dump-config``."""

    name: str
    description: str
    plugins: list[PluginRef]
    intercept_order: list[str]
    layers: list[str] = field(default_factory=list)  # human-readable, in apply order
    provenance: dict[str, str] = field(default_factory=dict)  # plugin name → layer that last set it
    files: list[Path] = field(default_factory=list)  # the YAML layers that applied, in order

    def enabled_plugins(self) -> list[PluginRef]:
        return [p for p in self.plugins if p.enabled]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "layers": list(self.layers),
            "plugins": [
                {**p.model_dump(), "set_by": self.provenance.get(p.name, "?")} for p in self.plugins
            ],
            "intercept_order": list(self.intercept_order),
        }


def iris_home() -> Path:
    """``$IRIS_HOME`` (tests relocate it) else ``~/.iris`` — same rule as governance paths."""
    from iris_harness.foundation.plugin_dirs import iris_home as _home

    return _home()


def profiles_dir(config_dir: Path) -> Path:
    return config_dir / "profiles"


def list_profiles(config_dir: Path) -> list[str]:
    root = profiles_dir(config_dir)
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.yaml"))


def _read_layer(path: Path) -> ProfileDocument | None:
    if not path.is_file():
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("profile layer unreadable (%s): %s", path, exc)
        return None
    if not isinstance(raw, dict):
        logger.warning("profile layer is not a mapping: %s", path)
        return None
    try:
        return ProfileDocument.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 — an invalid layer is skipped; logged
        logger.warning("profile layer invalid (%s): %s", path, exc)
        return None


def _csv_env(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _installed_plugins() -> tuple[str, ...]:
    """What this installation actually ships: the built-ins plus anything installed.

    The fallback for a missing or invalid profile file. It is ``DEFAULT_PLUGINS`` (the
    reference plugins that live in this package) plus every plugin registered in the
    ``iris_harness.plugins`` entry-point group -- which, on a machine with the domains
    installed, is the six from ``iris_personal`` (OSS plan M6, decisions 2 and 10).

    Discovering them rather than naming them is the point: the core knows no domain
    plugin's name, and a public install with nothing extra installed falls back to the
    five it has. Before M6.1b the six were hard-coded here, which a core-only install
    could not have honoured.
    """
    import importlib.metadata

    # Imported here, not at module scope: `loader` imports this module, so the constant
    # can only travel the other way at call time.
    from .loader import ENTRY_POINT_GROUP

    names = list(DEFAULT_PLUGINS)
    try:
        for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
            if ep.name not in names:
                names.append(ep.name)
    except Exception:  # metadata scan is best-effort, never fatal
        logger.debug("entry-point scan for the default profile failed", exc_info=True)
    return tuple(names)


def _plugin_available(name: str, home: Path) -> bool:
    """``name`` is installed here and would load: found, and its ``requires`` met.

    The same two checks the loader makes before it calls ``setup`` -- a plugin missing
    here, or one whose required packages are absent, would not mount, so a profile
    that names it cannot be the one an unnamed run picks.
    """
    # Imported here, not at module scope: `loader` imports this module.
    from .loader import _requirements_problem, discover_plugin

    try:
        source = discover_plugin(name, home_dir=home)
    except Exception:  # a broken manifest is "not available"; logged
        logger.debug("profile preference: plugin %r not discoverable", name, exc_info=True)
        return False
    return source is not None and _requirements_problem(source.manifest) is None


def _preferred_profile(config_dir: Path, home: Path) -> tuple[str, str | None]:
    """The profile an unnamed run uses, and why when it is not ``default``.

    ``default``'s ``prefer_when_installed`` names the candidates, in order; the first
    whose every enabled, non-optional plugin is available, and each of whose
    ``requires_one_of`` groups has an available plugin, wins. A candidate file that is
    missing or invalid is skipped, as is one that names no plugins.
    """
    base = _read_layer(profiles_dir(config_dir) / f"{DEFAULT_PROFILE_NAME}.yaml")
    if base is None:
        return DEFAULT_PROFILE_NAME, None
    for candidate in base.prefer_when_installed:
        if candidate == DEFAULT_PROFILE_NAME:
            continue
        doc = _read_layer(profiles_dir(config_dir) / f"{candidate}.yaml")
        if doc is None:
            logger.debug("profile preference %r: file missing or invalid; skipped", candidate)
            continue
        wanted = [ref.name for ref in doc.plugins if ref.enabled and not ref.optional]
        if not wanted:
            continue
        if not all(_plugin_available(plugin, home) for plugin in wanted):
            continue
        if all(
            any(_plugin_available(plugin, home) for plugin in group)
            for group in doc.requires_one_of
        ):
            return candidate, (
                f"selected: {candidate} (no profile named; {DEFAULT_PROFILE_NAME} prefers "
                "it and all its plugins are installed)"
            )
    return DEFAULT_PROFILE_NAME, None


def load_profile(
    config_dir: Path,
    name: str | None = None,
    *,
    home_dir: Path | None = None,
    env: dict[str, str] | None = None,
) -> EffectiveProfile:
    """Resolve the effective profile from the shipped file, the home overlay, and env.

    A missing or invalid shipped profile is not fatal: the base layer falls back
    to ``DEFAULT_PLUGINS`` (the capabilities the core used to provide itself), so a
    runtime built against a bare config dir still boots with the same deterministic
    behaviour — config-with-defaults, like the rest of ``config/``.

    With neither ``name`` nor ``IRIS_PROFILE``, ``default``'s ``prefer_when_installed``
    may pick another profile (module docstring).
    """
    environ = os.environ if env is None else env
    home = home_dir or iris_home()
    layers: list[str] = []
    profile_name = name or environ.get(PROFILE_ENV) or ""
    if not profile_name:
        profile_name, selected = _preferred_profile(config_dir, home)
        if selected is not None:
            layers.append(selected)

    plugins: dict[str, PluginRef] = {}
    provenance: dict[str, str] = {}
    files: list[Path] = []
    description = ""
    intercept_order: list[str] = []

    base_path = profiles_dir(config_dir) / f"{profile_name}.yaml"
    base = _read_layer(base_path)
    if base is None:
        fallback = _installed_plugins()
        layers.append(
            f"shipped: {base_path} (missing/invalid → built-in default: {', '.join(fallback)})"
        )
        for plugin_name in fallback:
            plugins[plugin_name] = PluginRef(name=plugin_name)
            provenance[plugin_name] = "built-in default"
    else:
        layers.append(f"shipped: {base_path}")
        files.append(base_path)
        description = base.description
        for ref in base.plugins:
            plugins[ref.name] = ref
            provenance[ref.name] = "shipped"
        if base.intercept_order is not None:
            intercept_order = list(base.intercept_order)

    overlay_path = home / "profile.yaml"
    overlay = _read_layer(overlay_path)
    if overlay is not None:
        layers.append(f"home: {overlay_path}")
        files.append(overlay_path)
        if overlay.description:
            description = overlay.description
        for ref in overlay.plugins:
            current = plugins.get(ref.name)
            if current is None:
                plugins[ref.name] = ref
            else:
                merged = current.model_copy(
                    update={
                        "enabled": ref.enabled,
                        "trust": ref.trust if ref.trust is not None else current.trust,
                        "optional": ref.optional or current.optional,
                    }
                )
                plugins[ref.name] = merged
            provenance[ref.name] = "home"
        if overlay.intercept_order is not None:
            intercept_order = list(overlay.intercept_order)

    disabled = (
        _csv_env(DISABLE_ENV)
        if env is None
        else [s.strip() for s in environ.get(DISABLE_ENV, "").split(",") if s.strip()]
    )
    enabled = (
        _csv_env(ENABLE_ENV)
        if env is None
        else [s.strip() for s in environ.get(ENABLE_ENV, "").split(",") if s.strip()]
    )
    if disabled or enabled:
        layers.append(f"env: {DISABLE_ENV}={','.join(disabled)} {ENABLE_ENV}={','.join(enabled)}")
    for plugin_name in enabled:
        current = plugins.get(plugin_name)
        # Asked for by name: required from here on, so a missing one shows as failed.
        plugins[plugin_name] = (
            PluginRef(name=plugin_name)
            if current is None
            else current.model_copy(update={"enabled": True, "optional": False})
        )
        provenance[plugin_name] = "env"
    for plugin_name in disabled:
        current = plugins.get(plugin_name)
        if current is not None:
            plugins[plugin_name] = current.model_copy(update={"enabled": False})
            provenance[plugin_name] = "env"

    for plugin_name, ref in list(plugins.items()):
        if ref.optional and ref.enabled and not _plugin_available(plugin_name, home):
            del plugins[plugin_name]
            provenance.pop(plugin_name, None)
            layers.append(f"optional: {plugin_name} not installed or not loadable here; left out")

    return EffectiveProfile(
        name=profile_name,
        description=description,
        plugins=list(plugins.values()),
        intercept_order=intercept_order,
        layers=layers,
        provenance=provenance,
        files=files,
    )


__all__ = [
    "DEFAULT_PROFILE_NAME",
    "EffectiveProfile",
    "PluginRef",
    "ProfileDocument",
    "iris_home",
    "list_profiles",
    "load_profile",
    "profiles_dir",
]
