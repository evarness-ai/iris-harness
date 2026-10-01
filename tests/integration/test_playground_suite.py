"""Run the committed deterministic suite through the service — the CI net.

Proves config/playground/core-deterministic.yaml passes against a real runtime
built by the playground service (no LLM: every scenario is a deterministic
intercept). If the bootstrap decomposition changes intercept behavior, this
turns red.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.playground.loader import discover_suites, load_suite
from iris_harness.playground.service import run_suite

_SUITE_DIR = Path(__file__).resolve().parents[2] / "config" / "playground"


@pytest.mark.smoke
def test_core_deterministic_suite_passes() -> None:
    suite = load_suite(_SUITE_DIR / "core-deterministic.yaml")
    result = run_suite(suite)
    assert result.ok, [
        (r.scenario_name, [a.detail for a in r.failed_assertions])
        for r in result.results
        if not r.passed
    ]
    assert result.total == 4


def test_discovery_ignores_example_templates() -> None:
    # The .example template must not be picked up as a runnable suite.
    names = [p.name for p in discover_suites(_SUITE_DIR)]
    assert "core-deterministic.yaml" in names
    assert "intercepts-live.yaml.example" not in names
