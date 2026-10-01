"""The owner's edits to ``health_watch.yaml``, saved on the data volume (ADR-0120).

On the cloud VM ``health_watch.yaml`` is a read-only mount and a deploy replaces it, so a
change made from the app is saved in the settings store (section ``health_watch``, key
``config``: the fields that differ from the file) and laid over the file in
``load_watch_config``. The running watcher reads its config on every tick, so an edit
also swaps the live watcher's config: it applies at the next tick.

On/off is not here: ``IRIS_HEALTH_WATCH_ENABLED`` (a catalog setting on the same tab)
already does it, and two switches for one thing would disagree.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iris_harness.foundation.settings.store import SettingsStore
    from iris_harness.services.health.watch import WatchConfig

logger = logging.getLogger(__name__)

SETTINGS_SECTION = "health_watch"
KEY = "config"
# field -> (kind, low, high); bools have no range.
EDITABLE: dict[str, tuple[str, float, float]] = {
    "confirm_ticks": ("int", 1, 60),
    "renotify_hours": ("float", 0.5, 168),
    "max_attempts": ("int", 0, 10),
    "recurring_days": ("int", 1, 90),
    "notify_recovered": ("bool", 0, 0),
    "notify_self_healed": ("bool", 0, 0),
    "heartbeat_retry": ("bool", 0, 0),
    "diagnose_credentials": ("bool", 0, 0),
}
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class WatchEditError(ValueError):
    """An edit that does not fit a health-watch field."""


def validate(changes: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, raw in changes.items():
        spec = EDITABLE.get(key)
        if spec is None:
            raise WatchEditError(f"{key!r} is not an editable health-watch field")
        kind, low, high = spec
        if kind == "bool":
            text = str(raw).strip().lower()
            if isinstance(raw, bool):
                out[key] = raw
            elif text in _TRUE or text in _FALSE:
                out[key] = text in _TRUE
            else:
                raise WatchEditError(f"{key} must be on or off")
            continue
        number: float
        try:
            if isinstance(raw, bool):
                raise TypeError
            number = float(raw) if kind == "float" else int(str(raw).strip())
        except (TypeError, ValueError):
            raise WatchEditError(f"{key} must be a number") from None
        if not low <= number <= high:
            raise WatchEditError(f"{key} must be between {low:g} and {high:g}")
        out[key] = number
    return out


def fields(config: WatchConfig) -> dict[str, Any]:
    return {f: getattr(config, f) for f in EDITABLE}


def apply_saved(config: WatchConfig, store: SettingsStore) -> WatchConfig:
    """``config`` with the saved edits laid over it; a saved edit that no longer fits is
    ignored with a warning (the file's value is a safe place to land)."""
    try:
        saved = store.get(SETTINGS_SECTION, KEY)
    except Exception:  # a broken store must not stop the watch
        logger.exception("health watch: could not read saved edits; using the file")
        return config
    if not isinstance(saved, dict):
        return config
    good: dict[str, Any] = {}
    for key, value in saved.items():
        try:
            good.update(validate({key: value}))
        except WatchEditError as exc:
            logger.warning("health watch: ignoring saved %s (%s)", key, exc)
    return replace(config, **good)


def update(
    file_config: WatchConfig,
    current: WatchConfig,
    store: SettingsStore,
    changes: dict[str, Any],
    *,
    actor: str,
) -> WatchConfig:
    """Save ``changes`` (the fields that differ from the file) and return the new config."""
    target = replace(current, **validate(changes))
    if fields(target) == fields(current):
        return current
    diff = {f: v for f, v in fields(target).items() if v != getattr(file_config, f)}
    if diff:
        store.set(SETTINGS_SECTION, KEY, diff, old=fields(current), new=fields(target), actor=actor)
    else:
        store.clear(SETTINGS_SECTION, KEY, old=fields(current), new=fields(target), actor=actor)
    return target


def reset(
    file_config: WatchConfig, current: WatchConfig, store: SettingsStore, *, actor: str
) -> WatchConfig:
    return update(file_config, current, store, fields(file_config), actor=actor)
