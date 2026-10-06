"""Discover and load the plugins a profile names.

Discovery order for a plugin ``name`` (first hit wins):

1. **builtin** — the package ``iris_harness.plugins_builtin.<name>`` shipped with
   the harness (reference plugins: ``system``, later ``code_exec``, ``research``).
2. **entry point** — an installed distribution exposing ``name`` in the
   ``iris_harness.plugins`` entry-point group (community plugins via pip).
3. **home** — ``$IRIS_HOME/plugins/<name>/`` with ``manifest.yaml`` + ``plugin.py``
   (a personal plugin, no packaging needed).

A fourth source is not discovered at all: an **in-process** plugin
(:class:`InProcessPlugin`) is handed to ``build_runtime`` as its ``setup`` callable and
manifest, by code that builds the runtime itself -- a plugin's own tests and the
examples, through ``iris_harness.testing.harness`` (OSS plan R16/R18). It is added to
the effective profile under its manifest name (:func:`add_in_process`) and then mounts
exactly as a discovered plugin does: requirements, capabilities, ``setup(api)`` under
the fault boundary, a registry record.

Each plugin directory carries a ``manifest.yaml`` (see :mod:`.manifest`); an
entry point may point at the ``setup`` callable directly, in which case the
manifest is synthesised from the distribution metadata.

Loading = requirements check → import → ``setup(api)`` under the fault boundary. A
``flavor: declarative`` plugin ships no ``setup``: the loader builds one from its manifest
(``declarative.py``), binding each declared tool to its ``impl`` function, so it mounts
from any source -- an entry point naming just its package, a home directory, in-process.
A plugin that fails to load is recorded ``FAILED`` with the reason (it shows red
in System Health) and the harness boots without it.

Declared capabilities (docs/architecture/plugin-capabilities.md §2) shape the order and
the check: plugins mount in profile order except that a capability's providers mount
before its consumers, and a ``capabilities: requires`` entry no mounted plugin provides is
a requirement unmet, exactly like a missing package. A missing ``uses`` is not: the plugin
loads and ``api.capability`` returns None.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

from .api import HarnessServices, PluginAPI
from .declarative import declarative_setup
from .manifest import PluginManifest, load_manifest
from .profile import EffectiveProfile, PluginRef, iris_home
from .registry import PluginRecord, PluginRegistry, PluginStatus

logger = logging.getLogger(__name__)

# Where plugins live is shared with memory, which reads installed plugins' vocabulary
# without a runtime (foundation/plugin_dirs.py).
from iris_harness.foundation.plugin_dirs import (  # noqa: E402
    BUILTIN_PACKAGE as BUILTIN_PACKAGE,
)
from iris_harness.foundation.plugin_dirs import (  # noqa: E402
    ENTRY_POINT_GROUP as ENTRY_POINT_GROUP,
)
from iris_harness.foundation.plugin_dirs import (  # noqa: E402
    MANIFEST_FILENAME as MANIFEST_FILENAME,
)


@dataclass(frozen=True)
class PluginSource:
    """Where a plugin was found and how to reach its ``setup``."""

    name: str
    kind: str  # "builtin" | "entry_point" | "home" | "in_process"
    location: str  # package name, entry-point value, directory path, or setup's qualname
    manifest: PluginManifest
    directory: Path | None = None
    entry_point: importlib.metadata.EntryPoint | None = None
    setup: Callable[[PluginAPI], None] | None = None  # in_process only

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.location}"


# ---------------------------------------------------------------- in-process
IN_PROCESS = "in-process"


@dataclass(frozen=True)
class InProcessPlugin:
    """A plugin supplied as a ``setup`` callable plus its manifest, not discovered.

    Built with :func:`in_process_plugin`. The manifest is the same contract an installed
    plugin's ``manifest.yaml`` is (``tools:`` declarations, ``capabilities:``,
    ``requires:``): supplying a plugin in-process skips discovery, never the checks. A
    ``flavor: declarative`` plugin is its manifest alone: ``setup`` is None.
    """

    setup: Callable[[PluginAPI], None] | None
    manifest: PluginManifest

    @property
    def name(self) -> str:
        return self.manifest.name


def in_process_plugin(
    setup: Callable[[PluginAPI], None] | None = None,
    *,
    name: str | None = None,
    manifest: PluginManifest | Mapping[str, Any] | Path | str | None = None,
) -> InProcessPlugin:
    """An :class:`InProcessPlugin` from ``setup`` and a manifest.

    ``manifest`` is a :class:`PluginManifest`, a mapping of the ``manifest.yaml`` shape,
    or the path of a ``manifest.yaml``; ``None`` is the minimal manifest (``name`` only),
    as for an entry point that ships none. ``name`` fills a manifest that has none and
    must agree with one that has. ``setup`` is None for a ``flavor: declarative`` manifest
    (the loader binds its tools) and required for any other.
    """
    if setup is not None and not callable(setup):
        raise TypeError(f"in-process plugin setup must be callable, got {type(setup).__name__}")
    resolved: PluginManifest
    if isinstance(manifest, PluginManifest):
        resolved = manifest
    elif isinstance(manifest, Mapping):
        raw = dict(manifest)
        if name is not None:
            raw.setdefault("name", name)
        resolved = PluginManifest.model_validate(raw)
    elif isinstance(manifest, (Path, str)):
        resolved = load_manifest(Path(manifest))
    else:
        if name is None:
            raise ValueError("an in-process plugin needs a name or a manifest that has one")
        resolved = PluginManifest(name=name, description="in-process plugin")
    if name is not None and name != resolved.name:
        raise ValueError(f"in-process plugin name {name!r} disagrees with its manifest's")
    declarative = resolved.flavor == "declarative"
    if setup is None and not declarative:
        raise TypeError(f"in-process plugin {resolved.name!r} needs a setup callable")
    if setup is not None and declarative:
        raise ValueError(
            f"in-process plugin {resolved.name!r} is 'flavor: declarative': its manifest is the "
            "plugin, so it takes no setup"
        )
    return InProcessPlugin(setup=setup, manifest=resolved)


def add_in_process(
    profile: EffectiveProfile, plugins: Sequence[InProcessPlugin]
) -> EffectiveProfile:
    """``profile`` with each in-process plugin appended as a row, after the profile's own.

    A name the profile already lists (or a name supplied twice) is refused: an
    in-process plugin never silently stands in for a discovered one.
    """
    if not plugins:
        return profile
    listed = {ref.name for ref in profile.plugins}
    refs = list(profile.plugins)
    provenance = dict(profile.provenance)
    for plugin in plugins:
        if plugin.name in listed:
            raise ValueError(
                f"in-process plugin {plugin.name!r} is already in profile {profile.name!r}"
            )
        listed.add(plugin.name)
        refs.append(PluginRef(name=plugin.name))
        provenance[plugin.name] = IN_PROCESS
    layers = [*profile.layers, f"{IN_PROCESS}: {', '.join(p.name for p in plugins)}"]
    return replace(profile, plugins=refs, provenance=provenance, layers=layers)


def _in_process_source(plugin: InProcessPlugin) -> PluginSource:
    if plugin.setup is None:
        location = f"declarative:{plugin.name}"
    else:
        qualname = getattr(plugin.setup, "__qualname__", type(plugin.setup).__name__)
        location = f"{getattr(plugin.setup, '__module__', '?')}:{qualname}"
    return PluginSource(
        name=plugin.name,
        kind="in_process",
        location=location,
        manifest=plugin.manifest,
        setup=plugin.setup,
    )


# ----------------------------------------------------------------- discovery
def _builtin_source(name: str, package: str) -> PluginSource | None:
    module_name = f"{package}.{name.replace('-', '_')}"
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    directory = Path(spec.origin).parent
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        logger.warning("builtin plugin %r has no %s; skipped", name, MANIFEST_FILENAME)
        return None
    manifest = load_manifest(manifest_path)
    return PluginSource(
        name=name, kind="builtin", location=module_name, manifest=manifest, directory=directory
    )


def _entry_point_manifest(
    ep: importlib.metadata.EntryPoint,
) -> tuple[PluginManifest | None, Path | None]:
    """The ``manifest.yaml`` shipped beside the module the entry point names.

    An entry point is a module path, so the manifest is found exactly as it is for a
    builtin: resolve the module, look next to it. Without this an installed plugin
    silently loses everything the manifest declares — its ``cli:`` commands, its
    ``provides``, its ``requires`` — and the loss looks like nothing at all, because
    a synthesized manifest loads fine and just registers less (OSS plan M6.1b, where
    the six domain plugins start arriving this way).
    """
    module_name = ep.value.split(":", 1)[0].strip()
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return None, None
    if spec is None or spec.origin is None:
        return None, None
    directory = Path(spec.origin).parent
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None, directory
    try:
        return load_manifest(manifest_path), directory
    except Exception:  # noqa: BLE001 — a malformed manifest degrades to the synthetic one
        logger.warning("entry-point plugin %r has an unreadable %s", ep.name, MANIFEST_FILENAME)
        return None, directory


def _entry_point_source(name: str) -> PluginSource | None:
    try:
        eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 — metadata scan is best-effort
        return None
    for ep in eps:
        if ep.name != name:
            continue
        manifest, directory = _entry_point_manifest(ep)
        if manifest is None:
            # A plugin that ships no manifest still loads: name + entry point is the
            # minimum contract, and this is what third-party plugins may rely on.
            dist_version = "0.0.0"
            dist = getattr(ep, "dist", None)
            if dist is not None:
                dist_version = str(getattr(dist, "version", dist_version))
            manifest = PluginManifest(
                name=name,
                version=dist_version,
                description=f"entry point {ep.value}",
                entrypoint=ep.value if ":" in ep.value else f"{ep.value}:setup",
            )
        return PluginSource(
            name=name,
            kind="entry_point",
            location=ep.value,
            manifest=manifest,
            entry_point=ep,
            directory=directory,
        )
    return None


def _home_source(name: str, home: Path) -> PluginSource | None:
    directory = home / "plugins" / name
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    manifest = load_manifest(manifest_path)
    return PluginSource(
        name=name, kind="home", location=str(directory), manifest=manifest, directory=directory
    )


def discover_plugin(
    name: str,
    *,
    home_dir: Path | None = None,
    builtin_package: str = BUILTIN_PACKAGE,
    skip_entry_points: bool = False,
) -> PluginSource | None:
    """Locate ``name`` by the documented order; ``None`` when nowhere."""
    found = _builtin_source(name, builtin_package)
    if found is not None:
        return found
    if not skip_entry_points:
        found = _entry_point_source(name)
        if found is not None:
            return found
    return _home_source(name, home_dir or iris_home())


# ------------------------------------------------------------------- loading
def _requirements_problem(manifest: PluginManifest) -> str | None:
    for package in manifest.requires.packages:
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return f"required package not installed: {package}"
    for var in manifest.requires.env_vars:
        if not os.environ.get(var):
            return f"required environment variable unset: {var}"
    return None


def _import_setup(source: PluginSource) -> Callable[[PluginAPI], None]:
    manifest = source.manifest
    module: ModuleType
    if manifest.flavor == "declarative":
        # No setup() of the plugin's: the loader binds the manifest's tools (decision 5).
        return declarative_setup(manifest)
    if source.setup is not None:
        return source.setup
    if source.entry_point is not None:
        obj = source.entry_point.load()
        if callable(obj) and not isinstance(obj, type):
            return obj  # type: ignore[no-any-return]
        module = obj
    elif source.kind == "builtin":
        module = importlib.import_module(f"{source.location}.{manifest.entry_module}")
    else:
        assert source.directory is not None
        file_path = source.directory / f"{manifest.entry_module.replace('.', '/')}.py"
        module_name = f"iris_plugin_{source.name.replace('-', '_')}_{manifest.entry_module}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load plugin module: {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    setup = getattr(module, manifest.entry_function, None)
    if not callable(setup):
        raise ImportError(
            f"plugin {source.name!r}: {manifest.entrypoint} is not a callable setup()"
        )
    return setup  # type: ignore[no-any-return]


def _capability_problem(manifest: PluginManifest, registry: PluginRegistry) -> str | None:
    """A ``capabilities: requires`` entry no mounted plugin provides, as a load error.

    Providers mount before consumers (:func:`_mount_order`), so by now every provider
    that could satisfy it has run its ``setup``.
    """
    missing = [
        name for name in manifest.capabilities.requires if not registry.capability_providers(name)
    ]
    if not missing:
        return None
    return f"required capability not provided: {', '.join(missing)} (no mounted plugin provides it)"


def _find(
    ref: PluginRef,
    *,
    home_dir: Path | None,
    builtin_package: str,
    skip_entry_points: bool,
    in_process: Mapping[str, InProcessPlugin] | None = None,
) -> PluginSource | PluginRecord:
    """Where ``ref`` is, or the record saying why it will not load (disabled, not found)."""
    if not ref.enabled:
        return PluginRecord(name=ref.name, source="profile", status=PluginStatus.DISABLED)
    supplied = (in_process or {}).get(ref.name)
    if supplied is not None:
        return _in_process_source(supplied)
    try:
        source = discover_plugin(
            ref.name,
            home_dir=home_dir,
            builtin_package=builtin_package,
            skip_entry_points=skip_entry_points,
        )
    except ValueError as exc:  # bad manifest
        return PluginRecord(
            name=ref.name, source="profile", status=PluginStatus.FAILED, load_error=str(exc)
        )
    if source is None:
        return PluginRecord(
            name=ref.name,
            source="profile",
            status=PluginStatus.FAILED,
            load_error=(
                "not found as builtin, entry point, or $IRIS_HOME/plugins/<name>/manifest.yaml"
            ),
        )
    return source


def _mount_order(
    found: list[tuple[PluginRef, PluginSource | PluginRecord]],
) -> list[tuple[PluginRef, PluginSource | PluginRecord]]:
    """Profile order, except that a capability's providers mount before its consumers.

    Each step takes the first plugin (in profile order) whose providers -- the other
    plugins in the profile declaring a capability it ``uses`` or ``requires`` -- have all
    been placed. A cycle has no such plugin; the first remaining one is taken, and the
    capability it consumes is simply not there yet when it mounts (``uses``: None;
    ``requires``: not loaded, with the reason).
    """
    providers: dict[str, set[str]] = {}
    for ref, item in found:
        if isinstance(item, PluginSource):
            for name in item.manifest.capabilities.provides:
                providers.setdefault(name, set()).add(ref.name)

    def needs(ref: PluginRef, item: PluginSource | PluginRecord) -> set[str]:
        if not isinstance(item, PluginSource):
            return set()
        return {
            owner
            for name in item.manifest.capabilities.consumes
            for owner in providers.get(name, ())
            if owner != ref.name
        }

    remaining = list(found)
    placed: set[str] = set()
    order: list[tuple[PluginRef, PluginSource | PluginRecord]] = []
    while remaining:
        ready = next((i for i, (r, it) in enumerate(remaining) if needs(r, it) <= placed), None)
        if ready is None:
            ready = 0
            logger.warning(
                "plugins: capability cycle among %s; mounting %r first",
                [r.name for r, _it in remaining],
                remaining[0][0].name,
            )
        ref, item = remaining.pop(ready)
        order.append((ref, item))
        placed.add(ref.name)
    return order


def load_plugin(
    ref: PluginRef,
    *,
    services: HarnessServices,
    registry: PluginRegistry,
    home_dir: Path | None = None,
    builtin_package: str = BUILTIN_PACKAGE,
    skip_entry_points: bool = False,
) -> PluginRecord:
    """Discover, check, import and ``setup`` one plugin; always returns a record."""
    found = _find(
        ref,
        home_dir=home_dir,
        builtin_package=builtin_package,
        skip_entry_points=skip_entry_points,
    )
    return _mount(ref, found, services=services, registry=registry)


def _mount(
    ref: PluginRef,
    found: PluginSource | PluginRecord,
    *,
    services: HarnessServices,
    registry: PluginRegistry,
) -> PluginRecord:
    """Check, import and ``setup`` a discovered plugin; always returns a record."""
    if isinstance(found, PluginRecord):
        return registry.add_plugin(found)
    source = found
    trust = ref.trust or source.manifest.trust
    # Issue #97: a manifest that says nothing about `party` reads untrusted; say so, once per
    # mount. A manifest the loader synthesised for a manifest-less entry point is covered
    # (it never sets the field); a plugin supplied from code is the caller's own.
    if "party" not in source.manifest.model_fields_set and source.kind != "in_process":
        logger.warning(
            "plugin %r does not declare `party` in its manifest; treating it as untrusted "
            "(declare `party: first-party | trusted-third-party | untrusted` in manifest.yaml)",
            ref.name,
        )
    record = PluginRecord(
        name=ref.name,
        source=source.label,
        status=PluginStatus.LOADED,
        manifest=source.manifest,
        trust=trust,
        directory=source.directory,
    )
    if trust == "mcp":
        record.status = PluginStatus.UNSUPPORTED
        record.load_error = "trust: mcp (out-of-process host) is not available yet; not loaded"
        return registry.add_plugin(record)
    problem = _requirements_problem(source.manifest) or _capability_problem(
        source.manifest, registry
    )
    if problem is not None:
        record.status = PluginStatus.FAILED
        record.load_error = problem
        return registry.add_plugin(record)
    registry.add_plugin(record)
    api = PluginAPI(plugin=ref.name, services=services, registry=registry)
    try:
        setup = _import_setup(source)
        setup(api)
    except Exception as exc:  # a bad plugin must not stop the boot
        record.status = PluginStatus.FAILED
        record.load_error = f"setup failed: {type(exc).__name__}: {exc}"
        logger.exception("plugin %r failed to load", ref.name)
    return record


def load_plugins(
    profile: EffectiveProfile,
    *,
    services: HarnessServices,
    registry: PluginRegistry,
    home_dir: Path | None = None,
    builtin_package: str = BUILTIN_PACKAGE,
    skip_entry_points: bool = False,
    in_process: Sequence[InProcessPlugin] = (),
) -> list[PluginRecord]:
    """Load every plugin the profile lists: profile order, providers before consumers.

    ``in_process`` supplies plugins by ``setup`` instead of discovery; each must already
    be a row of ``profile`` (:func:`add_in_process`), or it is not mounted.
    """
    supplied = {plugin.name: plugin for plugin in in_process}
    found = [
        (
            ref,
            _find(
                ref,
                home_dir=home_dir,
                builtin_package=builtin_package,
                skip_entry_points=skip_entry_points,
                in_process=supplied,
            ),
        )
        for ref in profile.plugins
    ]
    records = [
        _mount(ref, item, services=services, registry=registry) for ref, item in _mount_order(found)
    ]
    loaded = [r.name for r in records if r.status is PluginStatus.LOADED]
    failed = [r.name for r in records if r.status is PluginStatus.FAILED]
    logger.info("plugins: profile=%s loaded=%s failed=%s", profile.name, loaded, failed or "none")
    return records


def describe_sources(
    profile: EffectiveProfile,
    *,
    home_dir: Path | None = None,
    builtin_package: str = BUILTIN_PACKAGE,
) -> list[dict[str, Any]]:
    """Discovery-only view (no setup) for ``iris --dump-config``."""
    rows: list[dict[str, Any]] = []
    for ref in profile.plugins:
        row: dict[str, Any] = {
            "name": ref.name,
            "enabled": ref.enabled,
            "set_by": profile.provenance.get(ref.name, "?"),
        }
        try:
            source = discover_plugin(ref.name, home_dir=home_dir, builtin_package=builtin_package)
        except ValueError as exc:
            row.update({"source": None, "error": str(exc)})
            rows.append(row)
            continue
        if source is None:
            row.update({"source": None, "error": "not found"})
        else:
            row.update(
                {
                    "source": source.label,
                    "version": source.manifest.version,
                    "trust": ref.trust or source.manifest.trust,
                    "party": source.manifest.party,
                    "provides": [k.value for k in source.manifest.provides],
                    "capabilities": {
                        "provides": list(source.manifest.capabilities.provides),
                        "uses": list(source.manifest.capabilities.uses),
                        "requires": list(source.manifest.capabilities.requires),
                    },
                    # ADR-0125: the owner-identity kinds this plugin may supply.
                    "identity": list(source.manifest.identity.provides),
                    # The search providers it may add to the research chain.
                    "search_providers": list(source.manifest.search_providers),
                    # Issue #103: where its code may connect (empty: nowhere).
                    "egress": source.manifest.egress.summary(),
                    "description": source.manifest.description,
                }
            )
        rows.append(row)
    return rows


__all__ = [
    "BUILTIN_PACKAGE",
    "ENTRY_POINT_GROUP",
    "IN_PROCESS",
    "InProcessPlugin",
    "PluginSource",
    "add_in_process",
    "describe_sources",
    "discover_plugin",
    "in_process_plugin",
    "load_plugin",
    "load_plugins",
]
