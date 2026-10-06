"""Heartbeat configuration loading from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import HeartbeatDefinition


class HeartbeatConfigError(ValueError):
    """Raised when a heartbeat definition is malformed."""


def load_heartbeats(path: Path) -> list[HeartbeatDefinition]:
    """Load and validate heartbeat definitions from a YAML file.

    Returns an empty list if the file does not exist (heartbeats are optional).
    """
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise HeartbeatConfigError(f"{path}: top-level must be a mapping")
    entries = raw.get("heartbeats", [])
    if not isinstance(entries, list):
        raise HeartbeatConfigError(f"{path}: 'heartbeats' must be a list")
    return [_parse_entry(entry, path) for entry in entries]


def _parse_entry(entry: Any, path: Path) -> HeartbeatDefinition:
    if not isinstance(entry, dict):
        raise HeartbeatConfigError(
            f"{path}: heartbeat entry must be a mapping, got {type(entry).__name__}"
        )
    try:
        name = str(entry["name"])
        handler = str(entry["handler"])
        schedule = str(entry["schedule"])
    except KeyError as exc:
        raise HeartbeatConfigError(
            f"{path}: heartbeat missing required field {exc.args[0]!r}"
        ) from exc
    enabled = bool(entry.get("enabled", True))
    description = str(entry.get("description", ""))
    params_raw = entry.get("params", {})
    if not isinstance(params_raw, dict):
        raise HeartbeatConfigError(f"{path}: heartbeat {name!r} 'params' must be a mapping")
    platforms_raw = entry.get("platforms", [])
    if not isinstance(platforms_raw, list) or not all(
        isinstance(p, str) and p for p in platforms_raw
    ):
        raise HeartbeatConfigError(
            f"{path}: heartbeat {name!r} 'platforms' must be a list of names like darwin"
        )
    record_runs = entry.get("record_runs")
    if record_runs is not None and not isinstance(record_runs, bool):
        raise HeartbeatConfigError(
            f"{path}: heartbeat {name!r} 'record_runs' must be true or false"
        )
    retry_after = entry.get("retry_skipped_after_minutes")
    if retry_after is not None and (
        isinstance(retry_after, bool) or not isinstance(retry_after, int) or retry_after < 1
    ):
        raise HeartbeatConfigError(
            f"{path}: heartbeat {name!r} 'retry_skipped_after_minutes' must be a whole "
            "number of minutes (1 or more)"
        )
    plugin = entry.get("plugin", "")
    if not isinstance(plugin, str):
        raise HeartbeatConfigError(f"{path}: heartbeat {name!r} 'plugin' must be a plugin name")
    return HeartbeatDefinition(
        name=name,
        handler=handler,
        schedule=schedule,
        enabled=enabled,
        description=description,
        params=dict(params_raw),
        platforms=tuple(platforms_raw),
        record_runs=record_runs,
        retry_skipped_after_minutes=retry_after,
        plugin=plugin.strip(),
    )
