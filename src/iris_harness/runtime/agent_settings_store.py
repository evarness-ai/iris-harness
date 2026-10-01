"""Curated per-agent settings WRITES (ADR-0074 §4).

An agent's toggles are the on/off settings the catalog says switch that agent
(``agent:`` on a declaration, in the core catalog or the owning plugin's manifest;
ADR-0120) — the core keeps no list of them. Edits persist in the
settings store on the data volume (ADR-0120) — they used to go to
``config/agent_settings.local.yaml``, which on the cloud VM sat inside the container and
was lost on every deploy; a file left from before is imported once — and are
hot-applied to the live process. Flags re-read at use-time
(``applies="next run"``) take effect immediately; start-time-wired ones
(``applies="restart"``) take effect on the next restart — the writer says which.

Writes are user-initiated only (API behind the write gate, CLI) — never a chat
tool: a model must not flip the user's configuration.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from iris_harness.foundation.settings.catalog import Catalog, CatalogEntry
from iris_harness.foundation.settings.env_overrides import (
    ENV_SECTION,
    apply_env_overrides,
    set_env_override,
)
from iris_harness.foundation.settings.store import SettingsStore

logger = logging.getLogger(__name__)

OVERRIDE_FILENAME = "agent_settings.local.yaml"


def override_path(config_dir: Path) -> Path:
    return config_dir / OVERRIDE_FILENAME


APPLIES_TEXT = {"now": "now", "next_run": "next run", "restart": "restart"}


def toggles_for_agent(name: str, catalog: Catalog) -> dict[str, CatalogEntry]:
    """The on/off settings that switch agent ``name``, keyed by env var."""
    return {
        entry.name: entry
        for entry in catalog.entries.values()
        if entry.declaration.agent == name
        and entry.declaration.editable
        and entry.declaration.kind == "bool"
    }


def load_overrides(config_dir: Path) -> dict[str, bool]:
    """The persisted toggle overrides ({env_key: bool}), or {} if none."""
    path = override_path(config_dir)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        logger.warning("ignoring malformed %s", path)
        return {}
    toggles = data.get("toggles") or {}
    return {str(k): bool(v) for k, v in toggles.items()}


def apply_overrides_to_env(config_dir: Path, *, store: SettingsStore | None = None) -> None:
    """Import a pre-ADR-0120 override file into the settings store (once), then apply
    every saved override — call at startup BEFORE flags are read."""
    store = store or SettingsStore()
    legacy = override_path(config_dir)
    for key, enabled in load_overrides(config_dir).items():
        if store.get(ENV_SECTION, key) is None:
            store.set(
                ENV_SECTION,
                key,
                "1" if enabled else "0",
                old=os.environ.get(key),
                actor="migration",
            )
    if legacy.exists():
        try:
            legacy.rename(legacy.with_suffix(".yaml.migrated"))
        except OSError:
            logger.warning("could not rename %s after importing it", legacy)
    apply_env_overrides(store)


def set_toggle(
    env_key: str,
    enabled: bool,
    *,
    catalog: Catalog,
    store: SettingsStore | None = None,
    actor: str = "service",
) -> dict[str, object]:
    """Persist + hot-apply one agent toggle. Returns the applied state and whether a
    restart is needed. Raises ValueError for a key that is not an agent's on/off
    setting, and for a guarded one: that needs the owner's confirm, which only the
    Settings screen (``PATCH /settings``) asks for."""
    entry = catalog.get(env_key)
    decl = entry.declaration if entry is not None else None
    if decl is None or decl.agent is None or not decl.editable or decl.kind != "bool":
        raise ValueError(f"unknown or non-editable toggle: {env_key}")
    if decl.guarded:
        raise ValueError(
            f"{env_key} is guarded ({decl.guard_reason}); change it in Settings, which asks "
            "you to confirm"
        )

    # Hot-apply: a use-time-read flag picks this up immediately; a start-time flag
    # applies on the next restart (still set here so the process is consistent).
    set_env_override(env_key, "1" if enabled else "0", actor=actor, store=store)

    applies = APPLIES_TEXT[decl.applies]
    restart_required = decl.applies == "restart"
    logger.info(
        "agent toggle set: %s=%s (applies %s)%s",
        env_key,
        enabled,
        applies,
        " — restart to take effect" if restart_required else "",
    )
    return {
        "key": env_key,
        "enabled": bool(enabled),
        "applies": applies,
        "restart_required": restart_required,
    }
