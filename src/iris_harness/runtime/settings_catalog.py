"""The settings catalog of THIS install, and an agent's own settings in it (ADR-0120).

The server builds its catalog from the plugins it loaded (``plugin_registry``). The
CLI and the agent console have no running harness, so they build it from the
effective profile and each listed plugin's manifest instead — the same declarations,
read from disk. Either way, an agent's settings are the catalog entries that name it
(``agent:``), which is how the core stopped keeping a list of any plugin agent's
switches.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from iris_harness.foundation.settings.catalog import (
    Catalog,
    CatalogEntry,
    build_catalog,
    load_core_catalog,
)

logger = logging.getLogger(__name__)

# The words an on/off setting reads as on (``env_overrides.normalize`` writes "1"/"0").
_TRUE = frozenset({"1", "true", "yes", "on"})


def installed_catalog(config_dir: Path) -> Catalog:
    """Core + every plugin the effective profile lists, read from their manifests."""
    from iris_harness.runtime.plugin_host.loader import discover_plugin
    from iris_harness.runtime.plugin_host.profile import load_profile

    declared = []
    for ref in load_profile(config_dir).plugins:
        try:
            source = discover_plugin(ref.name)
        except Exception:  # one unreadable plugin must not hide the rest
            logger.warning("settings catalog: could not read plugin %s", ref.name, exc_info=True)
            continue
        if source is not None:
            declared.append((f"plugin:{source.manifest.name}", dict(source.manifest.settings)))
    return build_catalog(load_core_catalog(), declared)


def registry_catalog(registry: Any) -> Catalog:
    """Core + every plugin a running harness loaded."""
    records = registry.plugins() if registry is not None else []
    return build_catalog(
        load_core_catalog(),
        [
            (f"plugin:{r.manifest.name}", dict(r.manifest.settings))
            for r in records
            if r.manifest is not None
        ],
    )


def agent_settings(catalog: Catalog, agent: str) -> list[CatalogEntry]:
    """The settings that switch ``agent``'s behaviour, editable ones only."""
    return [
        entry
        for entry in catalog.entries.values()
        if entry.declaration.agent == agent and entry.declaration.editable
    ]


def bool_setting(name: str, catalog: Catalog) -> bool:
    """An on/off setting as the code that owns it reads it: the environment's value
    when set (after the owner's saved overrides, ADR-0120), else the declared
    ``default`` — so an unset setting that defaults on counts as on. A name no loaded
    owner declares as a ``bool`` and the environment does not set reads as off."""
    raw = os.environ.get(name)
    if raw is not None and raw.strip():
        return raw.strip().lower() in _TRUE
    entry = catalog.get(name)
    if entry is None or entry.declaration.kind != "bool":
        logger.info("setting %s is not a declared on/off setting here; reading it as off", name)
        return False
    return bool(entry.declaration.default)
