"""Load scenario suites from YAML and locate the default suite directory."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from iris_harness.foundation.paths import config_path

from .models import ScenarioSuite


def default_scenario_dir() -> Path:
    """Where suites live by default: ``config/playground`` under the repo/config.

    Honors ``IRIS_PLAYGROUND_DIR``, then the resolved config directory
    (``foundation.paths.config_dir()``: ``IRIS_CONFIG_DIR``, the checkout, the packaged
    defaults), so tests and adopters can point it elsewhere without editing code.
    """
    override = os.environ.get("IRIS_PLAYGROUND_DIR")
    if override:
        return Path(override).expanduser()
    return config_path("playground")


def discover_suites(scenario_dir: Path | None = None) -> list[Path]:
    """Return every ``*.yaml`` suite file under *scenario_dir*, sorted."""
    root = scenario_dir or default_scenario_dir()
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.yaml") if p.is_file())


def load_suite(path: Path) -> ScenarioSuite:
    """Parse one YAML file into a validated ``ScenarioSuite``.

    Raises ``ValueError`` with the file path on malformed YAML or schema
    violations, so a bad suite fails loudly instead of silently skipping cases.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    raw.setdefault("name", path.stem)
    try:
        return ScenarioSuite.model_validate(raw)
    except Exception as exc:  # re-raise with the offending file
        raise ValueError(f"{path}: {exc}") from exc


def load_all_suites(scenario_dir: Path | None = None) -> list[ScenarioSuite]:
    """Load every discoverable suite."""
    return [load_suite(p) for p in discover_suites(scenario_dir)]
