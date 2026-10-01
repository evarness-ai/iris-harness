"""CI guard: the "SDK only" import contracts forbid every core package but the SDK.

import-linter's ``forbidden`` contract blocks only the modules it names, and it cannot
say "all of ``iris_harness`` except ``iris_harness.sdk``" (forbidding the root would
forbid the SDK, its child). So each SDK-only contract names the other core packages
one by one, and a package added to ``src/iris_harness`` later would be silently
allowed. This test closes that gap: a new top-level package or module fails here
until it is added to every contract below (or, if plugins may use it, published
through the SDK instead).

Contracts that are not in this tree (the private ``domains-use-the-sdk`` is stripped
from the public export) are skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

_ROOT = Path(__file__).resolve()
while not (_ROOT / "src" / "iris_harness").is_dir():
    if _ROOT == _ROOT.parent:
        raise RuntimeError("could not locate repo root (src/iris_harness)")
    _ROOT = _ROOT.parent

# Contracts whose forbidden list must be "every core package except the SDK".
SDK_ONLY_CONTRACTS = ("plugins-use-the-public-api", "domains-use-the-sdk")
# The packages a plugin may import: the SDK itself.
ALLOWED = {"iris_harness.sdk"}
# The contract's own source package is not forbidden to itself.
SOURCES_INSIDE_CORE = {"plugins-use-the-public-api": {"iris_harness.plugins_builtin"}}


def _core_top_level() -> set[str]:
    """Every importable top-level name under ``iris_harness``: packages and modules."""
    root = _ROOT / "src" / "iris_harness"
    names = {p.name for p in root.iterdir() if p.is_dir() and (p / "__init__.py").is_file()}
    names |= {p.stem for p in root.glob("*.py") if p.stem != "__init__"}
    return {f"iris_harness.{name}" for name in names}


def _contracts() -> dict[str, dict[str, object]]:
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    contracts = data["tool"]["importlinter"]["contracts"]
    return {str(c["id"]): c for c in contracts if "id" in c}


def test_the_core_has_the_packages_this_guard_expects() -> None:
    # A sanity floor, so an empty scan (a moved src/ tree) cannot pass vacuously.
    assert {"iris_harness.sdk", "iris_harness.runtime", "iris_harness.main"} <= _core_top_level()


@pytest.mark.parametrize("contract_id", SDK_ONLY_CONTRACTS)
def test_sdk_only_contract_forbids_every_other_core_package(contract_id: str) -> None:
    contract = _contracts().get(contract_id)
    if contract is None:
        pytest.skip(f"{contract_id} is not in this tree")
    listed = {
        str(m)
        for m in contract["forbidden_modules"]  # type: ignore[attr-defined]
        if str(m).count(".") == 1 and str(m).startswith("iris_harness.")
    }
    expected = _core_top_level() - ALLOWED - SOURCES_INSIDE_CORE.get(contract_id, set())
    missing = sorted(expected - listed)
    assert not missing, (
        f"{contract_id} does not forbid {missing}: a core package it does not name is "
        "silently allowed. Add it to forbidden_modules in pyproject.toml, or publish "
        "what plugins need from it through iris_harness.sdk."
    )
    assert not (listed & ALLOWED), f"{contract_id} forbids the SDK itself"


# --- Plugin-capabilities step 3c: every domain plugin and library sits in a domain ------
#
# The <domain>-imports-no-other-domain contracts name their members and the libraries
# they may not import one by one, so a new plugin or library would be silently unchecked.
# These tests fail until it is placed: in a domain contract's source_modules, or (a
# library no plugin owns) in SHARED_LIBRARIES.
#
# The public export carries ONE domain, email (OSS plan R2): the other domains' packages
# and contracts are not in that tree. A domain whose anchor package is absent is skipped,
# per domain; one whose package is present must have its contract, in either tree.

# contract -> the package whose presence says the domain is in this tree.
DOMAIN_ANCHORS = {
    "email-imports-no-other-domain": "iris_personal.email",
    "finance-imports-no-other-domain": "iris_personal.finance",
    "files-imports-no-other-domain": "iris_personal.filemanager",
    "calendar-imports-no-other-domain": "iris_personal.calendar",
    "planner-imports-no-other-domain": "iris_personal.plugins.planner",
}
DOMAIN_CONTRACTS = tuple(DOMAIN_ANCHORS)
# Libraries no plugin owns: importing one reaches no other plugin.
SHARED_LIBRARIES = {"iris_personal.market"}


def _packages(parent: Path, prefix: str) -> set[str]:
    return {
        f"{prefix}.{p.name}"
        for p in parent.iterdir()
        if p.is_dir() and (p / "__init__.py").is_file() and p.name != "plugins"
    }


def _domain_tree() -> tuple[set[str], set[str]]:
    """The domain libraries and the domain plugins, or skip where the tree has none."""
    root = _ROOT / "src" / "iris_personal"
    if not root.is_dir():
        pytest.skip("iris_personal is not in this tree")
    return _packages(root, "iris_personal"), _packages(root / "plugins", "iris_personal.plugins")


def _present(libraries: set[str], plugins: set[str]) -> list[str]:
    """The domain contracts whose domain is in this tree."""
    return [cid for cid, anchor in DOMAIN_ANCHORS.items() if anchor in libraries | plugins]


def _domain_contracts(present: list[str]) -> dict[str, dict[str, object]]:
    contracts = _contracts()
    missing = sorted(set(present) - set(contracts))
    assert not missing, f"domain contracts missing from pyproject.toml: {missing}"
    return {cid: contracts[cid] for cid in present}


def _modules(contract: dict[str, object], key: str) -> set[str]:
    return {str(m) for m in contract[key]}  # type: ignore[attr-defined]


def test_every_domain_plugin_and_library_is_in_exactly_one_domain() -> None:
    libraries, plugins = _domain_tree()
    # A sanity floor, so an empty scan cannot pass vacuously: email is in every tree that
    # has iris_personal (the public one ships only it), and its plugin with it.
    assert "iris_personal.email" in libraries
    assert "iris_personal.plugins.email_workflows" in plugins
    seen: dict[str, str] = {}
    for cid, contract in _domain_contracts(_present(libraries, plugins)).items():
        for module in _modules(contract, "source_modules"):
            assert module not in seen, f"{module} is in both {seen[module]} and {cid}"
            seen[module] = cid
    unplaced = sorted(((libraries - SHARED_LIBRARIES) | plugins) - set(seen))
    assert not unplaced, (
        f"{unplaced} belong to no domain contract, so their imports of other domains are "
        "unchecked. Add each to a <domain>-imports-no-other-domain contract in "
        "pyproject.toml (or, for a library no plugin owns, to SHARED_LIBRARIES here)."
    )


@pytest.mark.parametrize("contract_id", DOMAIN_CONTRACTS)
def test_domain_contract_forbids_every_other_domain_library(contract_id: str) -> None:
    libraries, plugins = _domain_tree()
    if contract_id not in _present(libraries, plugins):
        pytest.skip(f"{DOMAIN_ANCHORS[contract_id]} is not in this tree")
    contract = _domain_contracts([contract_id])[contract_id]
    own = _modules(contract, "source_modules")
    forbidden = _modules(contract, "forbidden_modules")
    missing = sorted(libraries - SHARED_LIBRARIES - own - forbidden)
    assert not missing, f"{contract_id} does not forbid the other domains' libraries {missing}"
    assert not (own & forbidden), f"{contract_id} forbids its own members"


def test_libraries_import_no_plugin_covers_every_library() -> None:
    libraries, _ = _domain_tree()
    contract = _contracts().get("libraries-import-no-plugin")
    assert contract is not None, "libraries-import-no-plugin is missing from pyproject.toml"
    missing = sorted(libraries - _modules(contract, "source_modules"))
    assert not missing, f"libraries-import-no-plugin does not check {missing}"
