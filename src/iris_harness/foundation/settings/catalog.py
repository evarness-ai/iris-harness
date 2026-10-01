"""The settings catalog: every ``IRIS_*`` setting the code reads, described (ADR-0120).

Each setting is declared once, by whoever owns it:

* the harness core, in ``catalog.yaml`` next to this module;
* a plugin, in the ``settings:`` section of its own ``manifest.yaml`` — the core
  carries no plugin's vocabulary, so it never lists a plugin's switches.
* a sidecar process the deployment runs beside the harness (an LLM proxy, say), in a
  catalog file that names its ``owner``, listed in ``IRIS_SETTINGS_SIDECAR_CATALOGS``.
  The core knows no deployment: it reads whatever catalogs that setting names. The app
  lists a sidecar's settings but cannot change them (the sidecar does not read the
  app's store).

A declaration says what the setting is (``kind``, ``default``), when a change takes
effect (``applies``), where the app shows it (``tab``), whether changing it needs the
owner's confirm (``guarded``) and whether the app may change it at all (``editable``;
secrets, paths and URLs never are). ``not_settings`` lists ``IRIS_*`` literals that are
not settings at all (a prefix matched against, a name the harness sets for a child
process), so a completeness test can account for every literal in the code.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CORE_CATALOG = Path(__file__).with_name("catalog.yaml")
SIDECAR_CATALOGS_ENV = "IRIS_SETTINGS_SIDECAR_CATALOGS"

logger = logging.getLogger(__name__)

SettingKind = Literal["bool", "int", "float", "str", "enum", "list", "path", "url", "secret"]
Applies = Literal["now", "next_run", "restart"]
Tab = Literal["features", "agents", "models", "health", "guards", "advanced", "none"]

# Kinds the app never edits: a secret is never shown, and a path or URL names a place on
# this machine or a service, which the deploy owns.
NEVER_EDITABLE: frozenset[str] = frozenset({"secret", "path", "url"})


class CatalogError(ValueError):
    """A catalog file or a plugin's ``settings:`` section is malformed."""


class SettingDeclaration(BaseModel):
    """One setting, as its owner declares it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: SettingKind
    default: Any = None
    applies: Applies
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1)
    tab: Tab
    guarded: bool = False
    guard_reason: str = ""
    editable: bool = True
    not_editable_reason: str = ""
    # The agent (its ``agent_type``) whose behaviour this switches, when it switches one:
    # the agent dashboard and ``iris agents --set`` show an agent's settings from here,
    # so the core keeps no list of any plugin agent's switches.
    agent: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")

    @model_validator(mode="after")
    def _coherent(self) -> SettingDeclaration:
        if self.guarded and not self.guard_reason.strip():
            raise ValueError("a guarded setting says why (guard_reason)")
        if self.editable and self.kind in NEVER_EDITABLE:
            raise ValueError(f"a {self.kind} setting is never editable from the app")
        if not self.editable and not self.not_editable_reason.strip():
            raise ValueError("a setting the app may not change says why (not_editable_reason)")
        if not self.editable and self.tab != "none":
            raise ValueError("a setting the app may not change lives on no tab (tab: none)")
        if self.editable and self.tab == "none":
            raise ValueError("an editable setting names the tab it appears on")
        return self


@dataclass(frozen=True)
class CatalogEntry:
    """A declared setting and who declared it (``core`` or ``plugin:<name>``)."""

    name: str
    owner: str
    declaration: SettingDeclaration


@dataclass(frozen=True)
class Catalog:
    entries: Mapping[str, CatalogEntry]
    not_settings: Mapping[str, str]

    def get(self, name: str) -> CatalogEntry | None:
        return self.entries.get(name)


def parse_settings(raw: Any, *, source: str) -> dict[str, SettingDeclaration]:
    """Validate a ``{IRIS_NAME: declaration}`` mapping from ``source``."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise CatalogError(f"{source}: settings must be a mapping of IRIS_* names")
    out: dict[str, SettingDeclaration] = {}
    for name, body in raw.items():
        if not isinstance(name, str) or not name.startswith("IRIS_"):
            raise CatalogError(f"{source}: {name!r} is not an IRIS_* name")
        try:
            out[name] = SettingDeclaration.model_validate(body)
        except ValueError as exc:
            raise CatalogError(f"{source}: {name}: {exc}") from exc
    return out


def load_catalog_file(path: Path) -> tuple[dict[str, SettingDeclaration], dict[str, str]]:
    """The ``settings`` and ``not_settings`` sections of one catalog file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise CatalogError(f"{path}: top level must be a mapping")
    not_settings = raw.get("not_settings") or {}
    if not isinstance(not_settings, dict) or not all(
        isinstance(v, str) and v.strip() for v in not_settings.values()
    ):
        raise CatalogError(f"{path}: not_settings maps each name to the reason it is not one")
    return parse_settings(raw.get("settings"), source=str(path)), dict(not_settings)


def build_catalog(
    core: tuple[dict[str, SettingDeclaration], dict[str, str]],
    plugins: Iterable[tuple[str, dict[str, SettingDeclaration]]] = (),
) -> Catalog:
    """Merge the core file and each other owner's declarations; a name is declared once.

    ``plugins`` pairs an owner label (``plugin:<name>``, or a sidecar's own name, see
    :func:`load_sidecar_catalogs`) with what it declares.
    """
    core_settings, not_settings = core
    entries: dict[str, CatalogEntry] = {
        name: CatalogEntry(name, "core", decl) for name, decl in core_settings.items()
    }
    for owner, declared in plugins:
        for name, decl in declared.items():
            if name in entries:
                raise CatalogError(f"{name} is declared by both {entries[name].owner} and {owner}")
            entries[name] = CatalogEntry(name, owner, decl)
    both = set(entries) & set(not_settings)
    if both:
        raise CatalogError(f"declared as a setting and as not one: {sorted(both)}")
    return Catalog(entries=dict(sorted(entries.items())), not_settings=dict(not_settings))


def load_core_catalog() -> tuple[dict[str, SettingDeclaration], dict[str, str]]:
    return load_catalog_file(CORE_CATALOG)


def load_sidecar_catalogs(
    raw: str | None = None,
) -> list[tuple[str, dict[str, SettingDeclaration]]]:
    """``(owner, declarations)`` for each catalog ``IRIS_SETTINGS_SIDECAR_CATALOGS`` names.

    The value is a list of YAML files separated by ``os.pathsep``; each carries the
    ``settings`` a sidecar process reads and an ``owner`` naming it (not ``core``, not
    ``plugin:<x>``). A listed file that does not exist is logged and skipped: the app
    still lists every other setting.
    """
    raw = os.environ.get(SIDECAR_CATALOGS_ENV, "") if raw is None else raw
    out: list[tuple[str, dict[str, SettingDeclaration]]] = []
    for entry in (part.strip() for part in raw.split(os.pathsep)):
        if not entry:
            continue
        path = Path(entry)
        if not path.is_file():
            logger.warning("%s names %s, which is not a file; skipped", SIDECAR_CATALOGS_ENV, path)
            continue
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        owner = loaded.get("owner") if isinstance(loaded, dict) else None
        if not isinstance(owner, str) or not owner.strip():
            raise CatalogError(f"{path}: a sidecar catalog names its owner (owner: <name>)")
        owner = owner.strip()
        if owner == "core" or owner.startswith("plugin:"):
            raise CatalogError(f"{path}: owner {owner!r} is reserved for the core and plugins")
        out.append((owner, load_catalog_file(path)[0]))
    return out
