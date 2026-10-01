"""Config-vs-runtime drift detection — observability for the framework itself.

IRIS is meant to be YAML-first: behavior is declared in config and wired up in
code. Over time those can drift — a skill dir with no loaded package, a tier in
llm_tiers.yaml the router never built, a channel connector registered with no
config entry. This module surfaces those gaps in one report.

It compares what is DECLARED (config YAML on disk) against what is REGISTERED
(live in the runtime), for the surfaces that have a YAML source today: skills,
LLM tiers, and channels. As more of the harness moves to YAML (Phase 4 —
intercepts, intents, built-in tools), each slots into the same report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from iris_harness.foundation.paths import config_dir as resolve_config_dir


@dataclass(frozen=True)
class SurfaceDrift:
    """Drift for one surface (skills / tiers / channels)."""

    surface: str
    declared_only: tuple[str, ...] = ()  # in config YAML, not registered at runtime
    registered_only: tuple[str, ...] = ()  # live in the runtime, no config entry
    in_sync: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.declared_only and not self.registered_only


@dataclass(frozen=True)
class DriftReport:
    surfaces: tuple[SurfaceDrift, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.surfaces)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "surfaces": [
                {
                    "surface": s.surface,
                    "ok": s.ok,
                    "declared_only": list(s.declared_only),
                    "registered_only": list(s.registered_only),
                    "in_sync": list(s.in_sync),
                }
                for s in self.surfaces
            ],
        }


def _compare(surface: str, declared: set[str], registered: set[str]) -> SurfaceDrift:
    return SurfaceDrift(
        surface=surface,
        declared_only=tuple(sorted(declared - registered)),
        registered_only=tuple(sorted(registered - declared)),
        in_sync=tuple(sorted(declared & registered)),
    )


def _config_dir() -> Path:
    return resolve_config_dir()


def _declared_skills(config_dir: Path) -> set[str]:
    """Skill names declared by a ``manifest.yaml`` under ``config/skills``
    (excluding the ``auto`` quarantine, which the registry also skips)."""
    root = config_dir / "skills"
    names: set[str] = set()
    if not root.exists():
        return names
    for manifest in root.rglob("manifest.yaml"):
        if "auto" in manifest.relative_to(root).parts:
            continue
        try:
            data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        name = data.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _declared_tiers(config_dir: Path) -> set[str]:
    path = config_dir / "llm_tiers.yaml"
    if not path.exists():
        return set()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return set()
    tiers = data.get("tiers", data)
    return {str(k) for k in tiers} if isinstance(tiers, dict) else set()


def _declared_channels(config_dir: Path) -> set[str]:
    path = config_dir / "channels.yaml"
    if not path.exists():
        return set()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return set()
    channels = data.get("channels", [])
    names: set[str] = set()
    if isinstance(channels, list):
        for entry in channels:
            if isinstance(entry, dict) and entry.get("type"):
                names.add(str(entry["type"]))
            elif isinstance(entry, str):
                names.add(entry)
    return names


def _registered_skills(runtime: Any) -> set[str]:
    registry = getattr(runtime, "skill_registry", None)
    if registry is None:
        return set()
    try:
        return {p.manifest.name for p in registry.list_packages()}
    except Exception:  # noqa: BLE001 — drift is best-effort observability
        return set()


def _registered_tiers(runtime: Any) -> set[str]:
    router = getattr(runtime, "tier_router", None)
    tiers = getattr(router, "_tiers", None)
    return set(tiers.keys()) if isinstance(tiers, dict) else set()


def _registered_channels(runtime: Any) -> set[str]:
    channels = getattr(runtime, "channels", None)
    if channels is None:
        return set()
    try:
        return set(channels.channels())
    except Exception:  # noqa: BLE001
        return set()


def _declared_plugin_tools(runtime: Any) -> set[str]:
    """Tools declared under ``tools:`` in loaded plugins' manifests (ADR-0110)."""
    registry = getattr(runtime, "plugin_registry", None)
    if registry is None:
        return set()
    declared: set[str] = set()
    for rec in getattr(registry, "_plugins", {}).values():
        manifest = getattr(rec, "manifest", None)
        declared.update(getattr(manifest, "tools", {}) or {})
    return declared


def _registered_plugin_tools(runtime: Any) -> set[str]:
    """Tool names plugins actually registered on the loop."""
    registry = getattr(runtime, "plugin_registry", None)
    if registry is None:
        return set()
    return {t.name for t in registry.tools()}


def _declared_intercepts(runtime: Any) -> set[str]:
    """Intercept names in the runtime's chain (declared in intercepts.yaml)."""
    chain = getattr(runtime, "intercept_chain", None) or ()
    return {getattr(s, "name", "") for s in chain} - {""}


def _wired_intercepts(runtime: Any) -> set[str]:
    """Declared intercepts whose handler actually exists: a runtime method, or a
    plugin registration of the same name (``handler: plugin:<name>`` rows).

    A declared intercept with neither shows up as declared-only — the YAML names
    a `_handle_*_turn` that was renamed or never written, or a plugin that did
    not load.
    """
    chain = getattr(runtime, "intercept_chain", None) or ()
    registry = getattr(runtime, "plugin_registry", None)
    wired: set[str] = set()
    for s in chain:
        name = getattr(s, "name", "")
        if not name:
            continue
        if registry is not None and registry.intercept(name) is not None:
            wired.add(name)
        elif getattr(runtime, getattr(s, "handler", ""), None) is not None:
            wired.add(name)
    return wired


def _plugin_uses(runtime: Any) -> tuple[set[str], set[str]]:
    """The permission contract's grants, and which of them name a registered tool.

    One-sided on purpose: a grant (``uses: tools``) naming no registered tool is a loose
    end — a permission for nothing (the closure rule, plugin-capabilities §5) — while a
    registered tool no plugin lists is normal (the model can still use it). So the
    "registered" side is the grants that resolve, and only dangling grants show as drift.
    """
    registry = getattr(runtime, "plugin_registry", None)
    uses_tools = getattr(registry, "uses_tools", None)
    if registry is None or uses_tools is None:
        return set(), set()
    granted = {tool for tools in uses_tools().values() for tool in tools}
    return granted, granted & {t.name for t in registry.tools()}


def _plugin_capabilities(runtime: Any) -> tuple[set[str], set[str]]:
    """Capabilities the mounted manifests declare under ``provides``, and those provided.

    Two-sided, as ``plugin_tools`` is: a declared capability no ``api.provide`` registered is
    a loose end (declared but never registered, the closure rule, plugin-capabilities §5); an
    undeclared one cannot appear, because ``api.provide`` refuses it. Keyed
    ``plugin:capability``, because several plugins may provide one capability and each
    declaration closes on its own.
    """
    registry = getattr(runtime, "plugin_registry", None)
    declared_capabilities = getattr(registry, "declared_capabilities", None)
    if registry is None or declared_capabilities is None:
        return set(), set()
    declared = {
        f"{plugin}:{cap}"
        for plugin, roles in declared_capabilities().items()
        for cap in roles["provides"]
    }
    provided = {
        f"{owner}:{cap}"
        for cap, owners in registry.provided_capabilities().items()
        for owner in owners
    }
    return declared, provided


def _capability_uses(runtime: Any) -> tuple[set[str], set[str]]:
    """Capabilities mounted plugins use or require, and which of them something provides.

    One-sided, like ``plugin_uses``: a ``uses`` nobody provides is a consumer stuck on its
    degraded path for a dependency that is not there (a ``requires`` nobody provides keeps
    its plugin unloaded, so it never counts here -- that is a red row in System Health).
    """
    registry = getattr(runtime, "plugin_registry", None)
    declared_capabilities = getattr(registry, "declared_capabilities", None)
    if registry is None or declared_capabilities is None:
        return set(), set()
    consumed = {
        cap
        for roles in declared_capabilities().values()
        for cap in (*roles["uses"], *roles["requires"])
    }
    return consumed, consumed & set(registry.provided_capabilities())


def build_drift_report(runtime: Any, *, config_dir: Path | None = None) -> DriftReport:
    """Compare declared config vs runtime-registered across every surface."""
    cfg = config_dir or _config_dir()
    return DriftReport(
        surfaces=(
            _compare("skills", _declared_skills(cfg), _registered_skills(runtime)),
            _compare("llm_tiers", _declared_tiers(cfg), _registered_tiers(runtime)),
            _compare("channels", _declared_channels(cfg), _registered_channels(runtime)),
            _compare("intercepts", _declared_intercepts(runtime), _wired_intercepts(runtime)),
            _compare(
                "plugin_tools", _declared_plugin_tools(runtime), _registered_plugin_tools(runtime)
            ),
            _compare("plugin_uses", *_plugin_uses(runtime)),
            _compare("plugin_capabilities", *_plugin_capabilities(runtime)),
            _compare("capability_uses", *_capability_uses(runtime)),
        )
    )
