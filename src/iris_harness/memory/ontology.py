"""The memory ontology IRIS runs with: ``config/memory/{ontology,shapes,mappings}.yaml``
plus the vocabulary fragments installed plugins ship.

memris compiles it; this module only finds the directory (honouring
``IRIS_CONFIG_DIR`` the way every other memory config does), finds the fragments, and
caches the result. A broken ontology raises here, loudly, rather than letting a caller
guess.

**Plugin vocabulary** (memris PR 10): a plugin whose ``manifest.yaml`` names an
``ontology:`` directory contributes that fragment — the finance plugin's ``fin:`` terms.
Every INSTALLED plugin's fragment is loaded, mounted or not: the ontology is grammar,
and what was stored with a plugin's terms must stay readable when the plugin is off.
The core knows no plugin by name.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from iris_harness.foundation.paths import config_path, default_config_dir
from iris_harness.foundation.plugin_dirs import MANIFEST_FILENAME, installed_plugin_dirs
from iris_harness.foundation.process_state import track_globals
from memris.ontology import MappingRule, Ontology, load_or_raise

logger = logging.getLogger(__name__)

_CACHE: dict[tuple[Path, tuple[Path, ...]], Ontology] = {}


def memory_config_dir() -> Path:
    """``<config_dir()>/memory``, or the shipped one when the override has no ontology."""
    base = config_path("memory")
    if (base / "ontology.yaml").exists():
        return base
    return default_config_dir() / "memory"


def vocabulary_fragments() -> list[Path]:
    """The ``ontology:`` directory of every installed plugin that declares one."""
    found: list[Path] = []
    for name, directory in installed_plugin_dirs():
        manifest = directory / MANIFEST_FILENAME
        if not manifest.is_file():
            continue
        try:
            declared = (yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}).get("ontology")
        except (OSError, yaml.YAMLError, AttributeError) as exc:
            logger.warning("plugin %r: unreadable %s (%s)", name, MANIFEST_FILENAME, exc)
            continue
        if not declared:
            continue
        fragment = (directory / str(declared)).resolve()
        if not fragment.is_relative_to(directory.resolve()):
            logger.warning("plugin %r: ontology %s is outside the plugin; skipped", name, declared)
            continue
        if (fragment / "ontology.yaml").is_file():
            found.append(fragment)
        else:
            logger.warning(
                "plugin %r declares ontology %s, which has no ontology.yaml", name, fragment
            )
    return found


def memory_ontology(directory: Path | None = None, *, fragments: bool = True) -> Ontology:
    """The compiled ontology: ``directory`` (default the config's) plus plugin fragments."""
    root = (directory or memory_config_dir()).resolve()
    parts = tuple(vocabulary_fragments()) if fragments else ()
    key = (root, parts)
    if key not in _CACHE:
        _CACHE[key] = load_or_raise(root, parts)
    return _CACHE[key]


def reset_cache() -> None:
    _CACHE.clear()


def normalise_fact_key(key: str) -> str:
    """The same folding fact_keys.canonical_key applies: case, spaces and dashes."""
    return re.sub(r"[\s\-]+", "_", (key or "").strip().lower())


def fact_mappings(ontology: Ontology) -> dict[str, MappingRule]:
    """Normalised fact key → the mapping that turns it into a statement."""
    return {
        normalise_fact_key(key): rule
        for rule in ontology.mappings
        if rule.source_type == "fact"
        for key in rule.keys
    }


__all__ = [
    "fact_mappings",
    "memory_config_dir",
    "memory_ontology",
    "normalise_fact_key",
    "reset_cache",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_CACHE")
