"""The settings catalog accounts for every IRIS_* name the code reads (ADR-0120).

These tests read the real source tree and the real catalog files, so adding a setting
without declaring it — or deleting the last read of one without removing its entry —
fails here, not in the app.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

import pytest

from iris_harness.foundation.settings.catalog import (
    Catalog,
    CatalogError,
    SettingDeclaration,
    build_catalog,
    load_core_catalog,
    load_sidecar_catalogs,
)
from iris_harness.runtime.plugin_host.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[5]
SRC = ROOT / "src"
# Directories holding a manifest: a bare glob of */ also catches __pycache__.
PLUGIN_DIRS = sorted(
    m.parent
    for m in [
        *(SRC / "iris_personal" / "plugins").glob("*/manifest.yaml"),
        *(SRC / "iris_harness" / "plugins_builtin").glob("*/manifest.yaml"),
    ]
)
# The personal domain packages each plugin is built on: a plugin declares the settings
# its package reads, because the package has no manifest of its own.
PLUGIN_PACKAGES = {
    "email_workflows": ("email",),
    "file_organizer": ("filemanager",),
    "finance_workflows": ("finance", "market"),
}
# Core files that still name a plugin's setting. Each is a leak of plugin vocabulary
# into the core, burned down by PRs 2b and 2c-1 (agent toggles moved to their plugins). The list
# only shrinks: a test below fails when an entry no longer holds.
#
# TODO(plan PR 7): the core stops reading IRIS_CALENDAR_APPLE_WRITE (it moves behind the
# calendar plugin); then these entries, and the exemptions below that read them, go.
KNOWN_LEAKS: dict[str, set[str]] = {
    "iris_harness/playground/service.py": {"IRIS_CALENDAR_APPLE_WRITE"},
    "iris_harness/services/learning/preflight.py": {"IRIS_CALENDAR_APPLE_WRITE"},
}

_LITERAL = re.compile(r"[\"'](IRIS_[A-Z0-9_]+)[\"']")


def _literals(root: Path) -> dict[str, set[str]]:
    """Every quoted IRIS_* literal under ``root``, with the files it appears in."""
    found: dict[str, set[str]] = defaultdict(set)
    for path in root.rglob("*.py"):
        for match in _LITERAL.finditer(path.read_text(encoding="utf-8")):
            found[match.group(1)].add(path.relative_to(SRC).as_posix())
    return found


def _leaks_in_this_tree(catalog: Catalog) -> dict[str, set[str]]:
    """``KNOWN_LEAKS`` as far as this tree goes: the names some plugin HERE declares.

    A leak is a core read of a plugin's setting. Where the plugin that declares the name
    is not in the tree (the public export ships no src/iris_personal), the core's read
    is still there but no declaration is, so the entry describes nothing to check.
    """
    plugin_owned = {n for n, e in catalog.entries.items() if e.owner.startswith("plugin:")}
    return {
        file: names & plugin_owned for file, names in KNOWN_LEAKS.items() if names & plugin_owned
    }


def _undeclared_known_leaks(catalog: Catalog) -> set[str]:
    """Known-leak names that nothing in this tree declares (their plugin is absent)."""
    return {name for names in KNOWN_LEAKS.values() for name in names if catalog.get(name) is None}


def _plugin_declarations() -> list[tuple[str, dict[str, SettingDeclaration]]]:
    out = []
    for directory in PLUGIN_DIRS:
        manifest = load_manifest(directory / "manifest.yaml")
        out.append((f"plugin:{manifest.name}", dict(manifest.settings)))
    return out


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return build_catalog(load_core_catalog(), _plugin_declarations())


@pytest.fixture(scope="module")
def literals() -> dict[str, set[str]]:
    found: dict[str, set[str]] = defaultdict(set)
    for root in (SRC / "iris_harness", SRC / "iris_personal"):
        for name, files in _literals(root).items():
            found[name] |= files
    return found


def test_every_iris_name_in_the_code_is_declared(
    catalog: Catalog, literals: dict[str, set[str]]
) -> None:
    # A known leak's name is declared by its plugin; in a tree without that plugin the
    # core's (known, shrink-only) read is all that is left of it. TODO(plan PR 7).
    exempt = _undeclared_known_leaks(catalog)
    missing = {
        name: sorted(files)[:3]
        for name, files in literals.items()
        if catalog.get(name) is None and name not in catalog.not_settings and name not in exempt
    }
    assert missing == {}, (
        "declare these in foundation/settings/catalog.yaml (core) or the owning plugin's "
        f"manifest.yaml settings (ADR-0120): {missing}"
    )


# The reverse leak: names the CORE catalog lists that only a domain package reads. In a
# tree without src/iris_personal (the public export) nothing reads them, so they are not
# stale there; where the domains are present they must still be read. Shrink-only.
# TODO(plan PR 7): declare IRIS_GOOGLE_OAUTH_TEST_BASE with the code that reads it
# (src/iris_personal/connections/google.py, #660) instead of in the core's not_settings.
KNOWN_CORE_DECLARED_DOMAIN_NAMES = frozenset({"IRIS_GOOGLE_OAUTH_TEST_BASE"})


def test_every_declared_name_is_still_in_the_code(
    catalog: Catalog, literals: dict[str, set[str]]
) -> None:
    domains_here = (SRC / "iris_personal").is_dir()
    stale = sorted(
        name
        for name in [*catalog.entries, *catalog.not_settings]
        if name not in literals and (domains_here or name not in KNOWN_CORE_DECLARED_DOMAIN_NAMES)
    )
    assert stale == [], f"nothing reads these any more; remove them from the catalog: {stale}"
    if domains_here:
        outside = {
            name
            for name in KNOWN_CORE_DECLARED_DOMAIN_NAMES
            if not all(f.startswith("iris_personal/") for f in literals.get(name, {"?"}))
        }
        assert (
            not outside
        ), f"no longer read only by a domain package; drop from the list: {outside}"


def test_a_plugin_declares_only_what_it_reads(catalog: Catalog) -> None:
    wrong: dict[str, list[str]] = {}
    for directory in PLUGIN_DIRS:
        manifest = load_manifest(directory / "manifest.yaml")
        roots = [directory] + [
            SRC / "iris_personal" / pkg for pkg in PLUGIN_PACKAGES.get(manifest.name, ())
        ]
        read = set().union(*(_literals(root).keys() for root in roots))
        extra = sorted(set(manifest.settings) - read)
        if extra:
            wrong[manifest.name] = extra
    assert wrong == {}, f"declared by a plugin that never reads them: {wrong}"


def test_the_core_names_no_plugin_setting_outside_the_known_leaks(catalog: Catalog) -> None:
    plugin_owned = {n for n, e in catalog.entries.items() if e.owner.startswith("plugin:")}
    leaks: dict[str, set[str]] = defaultdict(set)
    for name, files in _literals(SRC / "iris_harness").items():
        if name in plugin_owned:
            for file in files:
                if "/plugins_builtin/" not in file:
                    leaks[file].add(name)
    known = _leaks_in_this_tree(catalog)
    new = {f: sorted(n - known.get(f, set())) for f, n in leaks.items()}
    assert {
        f: n for f, n in new.items() if n
    } == {}, "the core must not carry a plugin's settings; read them through the plugin"
    gone = {f: sorted(n - leaks.get(f, set())) for f, n in known.items()}
    assert {
        f: n for f, n in gone.items() if n
    } == {}, "these leaks are fixed; remove them from KNOWN_LEAKS so the list only shrinks"


# -- sidecar catalogs: what the deployment runs beside the harness ------------------------

_SIDECAR = """
owner: {owner}
settings:
  IRIS_DEMO_SIDECAR_PORT:
    kind: int
    default: 4000
    applies: restart
    label: Demo sidecar port
    description: Where the demo sidecar listens.
    tab: advanced
"""


def test_sidecar_catalogs_load_from_the_setting(tmp_path: Path) -> None:
    first, second = tmp_path / "a.yaml", tmp_path / "b.yaml"
    first.write_text(_SIDECAR.format(owner="proxy"), encoding="utf-8")
    second.write_text(
        _SIDECAR.format(owner="mailer").replace("DEMO_SIDECAR", "DEMO_MAILER"), encoding="utf-8"
    )
    loaded = load_sidecar_catalogs(
        f"{first}{os.pathsep}{tmp_path / 'absent.yaml'}{os.pathsep}{second}"
    )
    assert [(owner, sorted(decls)) for owner, decls in loaded] == [
        ("proxy", ["IRIS_DEMO_SIDECAR_PORT"]),
        ("mailer", ["IRIS_DEMO_MAILER_PORT"]),
    ]
    catalog = build_catalog(load_core_catalog(), loaded)
    assert catalog.entries["IRIS_DEMO_SIDECAR_PORT"].owner == "proxy"


def test_no_sidecar_catalogs_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_SETTINGS_SIDECAR_CATALOGS", raising=False)
    assert load_sidecar_catalogs() == []


@pytest.mark.parametrize("owner", ["core", "plugin:demo", "''"])
def test_a_sidecar_catalog_names_an_owner_of_its_own(tmp_path: Path, owner: str) -> None:
    path = tmp_path / "sidecar.yaml"
    path.write_text(_SIDECAR.format(owner=owner), encoding="utf-8")
    with pytest.raises(CatalogError):
        load_sidecar_catalogs(str(path))


# -- the declaration rules ---------------------------------------------------------

_BASE = {
    "kind": "bool",
    "default": False,
    "applies": "now",
    "label": "A switch",
    "description": "Turns a thing on.",
    "tab": "features",
}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"guarded": True}, "guard_reason"),
        ({"kind": "secret"}, "never editable"),
        ({"kind": "path"}, "never editable"),
        ({"editable": False, "tab": "none"}, "not_editable_reason"),
        ({"editable": False, "not_editable_reason": "x"}, "tab: none"),
        ({"tab": "none"}, "names the tab"),
        ({"applies": "whenever"}, "applies"),
        ({"surprise": 1}, "surprise"),
    ],
)
def test_incoherent_declarations_are_refused(change: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SettingDeclaration.model_validate({**_BASE, **change})


def test_a_name_is_declared_once() -> None:
    decl = SettingDeclaration.model_validate(_BASE)
    with pytest.raises(CatalogError, match="declared by both core and plugin:x"):
        build_catalog(({"IRIS_A": decl}, {}), [("plugin:x", {"IRIS_A": decl})])
    with pytest.raises(CatalogError, match="as not one"):
        build_catalog(({"IRIS_A": decl}, {"IRIS_A": "why"}))


def test_the_shipped_catalog_obeys_the_owners_rules(catalog: Catalog) -> None:
    """The guarded set the owner named (ADR-0120 decision 5) is guarded."""
    must_be_guarded = {
        "IRIS_GOVERNANCE_ENABLED",
        "IRIS_GOVERNANCE_COST_ENFORCE",
        "IRIS_GOVERNANCE_DAILY_COST_CAP_USD",
        "IRIS_WEBUI_ALLOW_WRITES",
        "IRIS_CHANNEL_GATEWAY_TELEGRAM_ENABLED",
        "IRIS_CURATOR_LEAK_JUDGE",
    }
    unguarded = sorted(n for n in must_be_guarded if not catalog.entries[n].declaration.guarded)
    assert unguarded == []
    secrets_editable = sorted(
        n
        for n, e in catalog.entries.items()
        if e.declaration.kind == "secret" and e.declaration.editable
    )
    assert secrets_editable == []


@pytest.mark.parametrize(
    ("name", "default", "guarded"),
    [
        ("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER", True, True),
        ("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL", False, False),
    ],
)
def test_the_side_effect_ledger_is_two_plain_booleans(
    catalog: Catalog, name: str, default: bool, guarded: bool
) -> None:
    """On/off stays type safe: the ledger and its scope are separate bools, so an explicit
    ``true`` of the ledger cannot widen what is recorded."""
    from iris_harness.foundation.settings.env_overrides import SettingValueError, normalize

    decl = catalog.entries[name].declaration
    assert (decl.kind, decl.default, decl.guarded, decl.applies) == (
        "bool",
        default,
        guarded,
        "restart",
    )
    for raw, out in (("true", "1"), ("on", "1"), ("1", "1"), ("No", "0"), (False, "0")):
        assert normalize(decl, raw) == out
    for raw in ("all", "high-risk", "maybe", ""):
        with pytest.raises(SettingValueError):
            normalize(decl, raw)
