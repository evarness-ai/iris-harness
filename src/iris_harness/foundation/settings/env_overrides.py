"""The owner's ``IRIS_*`` setting changes, laid over the deploy's environment (ADR-0120).

A setting the app changes is saved in the settings store (section ``env``), never in
``server.env``, which the deploy owns. Every process that reads settings applies the
saved overrides to ``os.environ`` when it starts, before its modules load — the three
servers do it in ``iris_harness/server/__init__.py`` and the CLI in ``main`` — so a
setting read at import, at build or per call all see the same value after a restart,
and a per-call setting sees a change at once (the writer also sets ``os.environ``).

The deploy's own value is remembered the first time an override replaces it, so a
reset returns to exactly what ``server.env`` (or nothing) said.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from iris_harness.foundation.process_state import track_globals
from iris_harness.foundation.settings.catalog import SettingDeclaration
from iris_harness.foundation.settings.store import SettingChange, SettingsStore

logger = logging.getLogger(__name__)

ENV_SECTION = "env"

_lock = threading.Lock()
# name -> the value the environment had before any override (None: unset).
_deploy_values: dict[str, str | None] = {}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_MAX_TEXT = 500


class SettingValueError(ValueError):
    """A value that does not fit the setting's declared kind."""


def _remember(name: str) -> None:
    with _lock:
        _deploy_values.setdefault(name, os.environ.get(name))


def deploy_value(name: str) -> str | None:
    """What the deploy's environment says for ``name``, overrides aside."""
    with _lock:
        if name in _deploy_values:
            return _deploy_values[name]
    return os.environ.get(name)


def apply_env_overrides(store: SettingsStore | None = None) -> dict[str, Any]:
    """Put every saved override into ``os.environ``. Never raises: a broken store must
    not stop a process from starting on its deploy's settings."""
    try:
        overrides = (store or SettingsStore()).section(ENV_SECTION)
    except Exception:
        logger.exception("settings: could not read saved overrides; using the deploy's")
        return {}
    for name, value in overrides.items():
        if not name.startswith("IRIS_"):
            continue
        _remember(name)
        os.environ[name] = str(value)
    if overrides:
        logger.info("settings: applied %d saved override(s)", len(overrides))
    return overrides


def normalize(declaration: SettingDeclaration, raw: Any) -> str:
    """The env string for ``raw`` under ``declaration``'s kind, or SettingValueError."""
    kind = declaration.kind
    if kind == "bool":
        text = str(raw).strip().lower() if not isinstance(raw, bool) else str(raw).lower()
        if text in _TRUE:
            return "1"
        if text in _FALSE:
            return "0"
        raise SettingValueError(f"expected on/off, got {raw!r}")
    if kind == "int":
        try:
            if isinstance(raw, bool):
                raise TypeError
            return str(int(str(raw).strip()))
        except (TypeError, ValueError):
            raise SettingValueError(f"expected a whole number, got {raw!r}") from None
    if kind == "float":
        try:
            if isinstance(raw, bool):
                raise TypeError
            return repr(float(str(raw).strip()))
        except (TypeError, ValueError):
            raise SettingValueError(f"expected a number, got {raw!r}") from None
    if kind in {"str", "enum", "list"}:
        text = str(raw).strip()
        if "\n" in text or "\r" in text or len(text) > _MAX_TEXT:
            raise SettingValueError("expected one line of at most 500 characters")
        if kind == "list":
            # An empty list is a value ("none"), e.g. no plugin turned off.
            return ",".join(part.strip() for part in text.split(",") if part.strip())
        if not text:
            raise SettingValueError("expected a value; use reset to go back to the default")
        return text
    raise SettingValueError(f"a {kind} setting is never set from the app")


def set_env_override(
    name: str, value: str, *, actor: str, store: SettingsStore | None = None
) -> SettingChange:
    """Save ``value`` for ``name`` and apply it to this process now."""
    _remember(name)
    change = (store or SettingsStore()).set(
        ENV_SECTION, name, value, old=os.environ.get(name), actor=actor
    )
    os.environ[name] = value
    return change


def clear_env_override(
    name: str, *, actor: str, store: SettingsStore | None = None
) -> SettingChange | None:
    """Drop the override and return ``name`` to the deploy's value; None if unchanged."""
    store = store or SettingsStore()
    if store.get(ENV_SECTION, name) is None:
        return None
    original = deploy_value(name)
    change = store.clear(ENV_SECTION, name, old=os.environ.get(name), new=original, actor=actor)
    if original is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = original
    return change


def _reset_for_tests() -> None:
    with _lock:
        _deploy_values.clear()


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_deploy_values")
