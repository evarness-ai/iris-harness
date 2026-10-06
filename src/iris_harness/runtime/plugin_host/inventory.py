"""The plugin inventory: what the profile asked for, what loaded, and its YAML.

The post-boot counterpart of ``dump.py``. ``--dump-config`` reads manifests without
booting; this reads the live :class:`PluginRegistry` a runtime built, so it can say
what each plugin actually registered (its agents, tools, heartbeats…) beside what
its manifest declared, which bus topics it subscribed to (and on which bus: a
subscription on the wrong one fails silently), which core seams it filled (API
routers, public callbacks, agent panels, learned sources), and the YAML that
configures it. Read-only: nothing
here changes a profile, a manifest, or a registration. ``GET /plugins`` and
``GET /plugins/{name}`` are thin wrappers over the two public functions, and
``iris plugins`` / ``iris plugins show NAME`` read those.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .api import SEARCH_PROVIDER_SEAM
from .loader import MANIFEST_FILENAME, discover_plugin
from .manifest import PluginManifest, RegistrationKind
from .profile import EffectiveProfile, list_profiles
from .registry import PluginRecord, PluginRegistry, PluginStatus

YAML_SUFFIXES = (".yaml", ".yml")
# A plugin's config files are small; anything bigger is data, not configuration,
# and is listed without its content rather than shipped to the browser.
MAX_YAML_BYTES = 256_000
MAX_YAML_FILES = 50


def _manifest_and_directory(rec: PluginRecord) -> tuple[PluginManifest | None, Path | None]:
    """The record's manifest, or — for a plugin that never loaded — a discovery read.

    A disabled plugin has no manifest on its record because loading stops before
    discovery. Reading it here (no import, no ``setup``) lets the inventory still
    describe what switching it on would bring.
    """
    if rec.manifest is not None:
        return rec.manifest, rec.directory
    try:
        source = discover_plugin(rec.name)
    except ValueError:
        return None, None
    if source is None:
        return None, None
    return source.manifest, source.directory


def _yaml_files(directory: Path | None) -> list[dict[str, Any]]:
    """Every YAML file under a plugin's directory, the manifest first."""
    # Only a directory that holds a manifest is a plugin's own. An entry point that
    # names a bare module resolves to its parent — site-packages — which is not.
    if directory is None or not (directory / MANIFEST_FILENAME).is_file():
        return []
    root = directory.resolve()
    found: list[Path] = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part == "__pycache__" or part.startswith(".") for part in rel.parts):
            continue
        if path.suffix not in YAML_SUFFIXES or not path.is_file():
            continue
        # A symlink out of the plugin's directory is not the plugin's configuration.
        if not path.resolve().is_relative_to(root):
            continue
        found.append(path)
    found.sort(key=lambda p: (p.name != MANIFEST_FILENAME, str(p.relative_to(root))))
    return [_file_entry(p, p.relative_to(root).as_posix()) for p in found[:MAX_YAML_FILES]]


def _file_entry(path: Path, label: str) -> dict[str, Any]:
    size = path.stat().st_size
    entry: dict[str, Any] = {"path": label, "size": size, "content": None, "truncated": False}
    if size > MAX_YAML_BYTES:
        entry["truncated"] = True
        return entry
    try:
        entry["content"] = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        entry["content"] = None
    return entry


def _profile_ref(profile: EffectiveProfile | None, name: str) -> tuple[bool | None, str | None]:
    if profile is None:
        return None, None
    for ref in profile.plugins:
        if ref.name == name:
            return ref.enabled, profile.provenance.get(name)
    return None, None


def _subscriptions(registry: PluginRegistry | None, name: str) -> list[dict[str, str]]:
    if registry is None:
        return []
    return [
        {"topic": topic, "scope": scope}
        for plugin, topic, scope in registry.subscriptions()
        if plugin == name
    ]


def _seams(registry: PluginRegistry | None, name: str) -> list[dict[str, str]]:
    if registry is None:
        return []
    return [{"seam": seam, "key": key} for plugin, seam, key in registry.seams() if plugin == name]


def _capabilities(
    registry: PluginRegistry | None, name: str, manifest: PluginManifest | None
) -> dict[str, list[dict[str, Any]]]:
    """Who provides and who uses what, from this plugin's side (plugin-capabilities §2).

    For each capability it provides: whether it did (``provided``) and which mounted plugins
    declare they use or require it. For each it uses or requires: the mounted plugins that
    provide it -- an empty list is the degraded path (``uses``) or a load failure
    (``requires``).
    """
    empty: dict[str, list[dict[str, Any]]] = {"provides": [], "uses": [], "requires": []}
    if manifest is None:
        return empty
    declared = registry.declared_capabilities() if registry is not None else {}

    def providers(cap: str) -> list[str]:
        return list(registry.capability_providers(cap)) if registry is not None else []

    def consumers(cap: str) -> list[str]:
        return sorted(
            plugin
            for plugin, roles in declared.items()
            if cap in roles["uses"] or cap in roles["requires"]
        )

    caps = manifest.capabilities
    return {
        "provides": [
            {"name": cap, "provided": name in providers(cap), "used_by": consumers(cap)}
            for cap in caps.provides
        ],
        "uses": [{"name": cap, "providers": providers(cap)} for cap in caps.uses],
        "requires": [{"name": cap, "providers": providers(cap)} for cap in caps.requires],
    }


def _summary(
    rec: PluginRecord,
    profile: EffectiveProfile | None,
    manifest: PluginManifest | None,
    registry: PluginRegistry | None = None,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for reg in rec.registrations:
        counts[reg.kind.value] = counts.get(reg.kind.value, 0) + 1
    enabled, set_by = _profile_ref(profile, rec.name)
    return {
        "name": rec.name,
        "status": rec.status.value,
        "source": rec.source,
        "version": manifest.version if manifest else None,
        "description": manifest.description.strip() if manifest else "",
        "trust": rec.trust,
        "flavor": manifest.flavor if manifest else None,
        "provides": [k.value for k in manifest.provides] if manifest else [],
        "enabled": enabled,
        "set_by": set_by,
        # ADR-0120: why the app may not turn it off, when it may not.
        "locked": manifest.locked if manifest else None,
        "registration_counts": counts,
        "subscription_count": len(_subscriptions(registry, rec.name)),
        "seam_count": len(_seams(registry, rec.name)),
        "agents": [r.name for r in rec.registrations if r.kind is RegistrationKind.INTENT_HANDLER],
        "declared_tools": len(manifest.tools) if manifest else 0,
        "search_providers": list(manifest.search_providers) if manifest else [],
        "failure_count": rec.failure_count,
        "last_error": rec.last_error,
        "load_error": rec.load_error,
        # Why a MOUNTED plugin is answering in a degraded way (a failed guarded call, or an
        # optional capability nothing provides); None when healthy. The registry's one
        # source -- the same string the System Health line carries -- so surfaces agree.
        "degraded_reason": registry.degraded_reason(rec.name) if registry is not None else None,
    }


def agent_plugins(registry: PluginRegistry | None) -> dict[str, str]:
    """``agent_type -> plugin`` for every agent a plugin registered.

    An agent absent from the map was registered by the core itself.
    """
    if registry is None:
        return {}
    owners: dict[str, str] = {}
    for rec in registry.plugins():
        for reg in rec.registrations:
            if reg.kind is RegistrationKind.INTENT_HANDLER:
                owners[reg.name] = rec.name
    return owners


def plugins_inventory(
    registry: PluginRegistry | None,
    profile: EffectiveProfile | None,
    *,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """Every plugin in the profile, in load order, with the profile that chose them."""
    records = registry.plugins() if registry is not None else []
    plugins = [_summary(rec, profile, _manifest_and_directory(rec)[0], registry) for rec in records]
    totals = {status.value: 0 for status in PluginStatus}
    for rec in records:
        totals[rec.status.value] += 1
    profile_view: dict[str, Any] | None = None
    if profile is not None:
        profile_view = {
            "name": profile.name,
            "description": profile.description,
            "layers": list(profile.layers),
            "intercept_order": list(profile.intercept_order),
            "available_profiles": list_profiles(config_dir) if config_dir is not None else [],
            "files": [_file_entry(p, str(p)) for p in profile.files if p.is_file()],
        }
    return {"profile": profile_view, "count": len(plugins), "totals": totals, "plugins": plugins}


def plugin_detail(
    registry: PluginRegistry | None, profile: EffectiveProfile | None, name: str
) -> dict[str, Any] | None:
    """One plugin: summary, manifest, registrations, subscriptions, seams, drift, YAML."""
    rec = registry.get(name) if registry is not None else None
    if rec is None:
        return None
    manifest, directory = _manifest_and_directory(rec)
    registered_tools = {r.name for r in rec.registrations if r.kind is RegistrationKind.TOOL}
    registered_kinds = {r.kind for r in rec.registrations}
    declared_kinds = set(manifest.provides) if manifest else set()
    tools = [
        {
            "name": tool_name,
            "effect": decl.effect,
            "confirm": decl.confirm_mode,
            "pinned": decl.pinned,
            "answers_directly": decl.answers_directly,
            "undo": decl.undo,
            "undo_window_days": decl.undo_window_days,
            "content": decl.content,
            "sends_to": decl.sends_to,
            "executes_code": decl.executes_code,
            "verify": decl.verify,
            "guidance": decl.guidance.strip(),
            "registered": tool_name in registered_tools,
        }
        for tool_name, decl in (manifest.tools.items() if manifest else [])
    ]
    # Drift only means something once setup ran; a disabled or failed plugin
    # registered nothing by definition.
    loaded = rec.status in (PluginStatus.LOADED, PluginStatus.DEGRADED)
    declared_tools = set(manifest.tools) if manifest else set()
    registered_providers = {
        s["key"] for s in _seams(registry, name) if s["seam"] == SEARCH_PROVIDER_SEAM
    }
    declared_providers = set(manifest.search_providers) if manifest else set()
    search_providers = [
        {"name": provider, "registered": provider in registered_providers}
        for provider in (manifest.search_providers if manifest else ())
    ]
    capabilities = _capabilities(registry, name, manifest)
    drift = {
        "capabilities_declared_not_provided": (
            [c["name"] for c in capabilities["provides"] if not c["provided"]] if loaded else []
        ),
        "tools_declared_not_registered": (
            sorted(declared_tools - registered_tools) if loaded else []
        ),
        "tools_registered_not_declared": (
            sorted(registered_tools - declared_tools) if loaded and manifest else []
        ),
        # The reverse cannot happen: an undeclared provider is refused at registration.
        "search_providers_declared_not_registered": (
            sorted(declared_providers - registered_providers) if loaded else []
        ),
        "provides_not_registered": (
            sorted(k.value for k in declared_kinds - registered_kinds) if loaded else []
        ),
        "registered_not_provided": (
            sorted(k.value for k in registered_kinds - declared_kinds)
            if loaded and manifest
            else []
        ),
    }
    return {
        **_summary(rec, profile, manifest, registry),
        "directory": str(directory) if directory else None,
        "manifest": manifest.model_dump(mode="json") if manifest else None,
        "registrations": [
            {"kind": r.kind.value, "name": r.name, "detail": r.detail} for r in rec.registrations
        ],
        "subscriptions": _subscriptions(registry, name),
        "seams": _seams(registry, name),
        "capabilities": capabilities,
        "tools": tools,
        "search_providers": search_providers,
        "drift": drift,
        "files": _yaml_files(directory),
    }


__all__ = [
    "MAX_YAML_BYTES",
    "MAX_YAML_FILES",
    "agent_plugins",
    "plugin_detail",
    "plugins_inventory",
]
