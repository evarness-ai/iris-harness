"""End-to-end: the playground engine against a real in-process runtime.

Uses the deterministic time/date intercept, which never calls the LLM, so the
whole path — chat -> intercept -> timeline event -> handler detection ->
assertions — is exercised without Ollama. This is the proof that the scenario
engine is a real regression net for the bootstrap decomposition.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.playground.models import Scenario, ScenarioExpectation, ScenarioSuite
from iris_harness.playground.runner import PlaygroundRunner
from iris_harness.runtime import build_runtime

pytestmark = pytest.mark.smoke


@pytest.fixture()
def runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    config_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    rt = build_runtime(
        config_dir=config_dir,
        data_dir=data_dir,
        use_background_scheduler=False,
    )
    rt.startup()
    try:
        yield rt
    finally:
        rt.shutdown()


def test_time_query_fires_deterministic_intercept(runtime) -> None:
    scenario = Scenario(
        name="what-time",
        message="what time is it?",
        expect=ScenarioExpectation(
            handler="time_date",
            metadata={"deterministic_time_date": True},
            response_contains=("Current local time",),
            no_pii_leak=True,
        ),
    )
    result = PlaygroundRunner(runtime.chat).run_scenario(scenario)
    assert result.passed, [a.detail for a in result.failed_assertions]
    assert result.handler == "time_date"


def test_suite_runs_multiple_scenarios(runtime) -> None:
    suite = ScenarioSuite(
        name="clock",
        scenarios=(
            Scenario(
                name="time",
                message="what time is it?",
                expect=ScenarioExpectation(handler="time_date"),
            ),
            Scenario(
                name="date",
                message="what's today's date?",
                expect=ScenarioExpectation(
                    handler="time_date", response_contains=("Today's date",)
                ),
            ),
        ),
    )
    suite_result = PlaygroundRunner(runtime.chat).run_suite(suite)
    assert suite_result.ok, [
        (r.scenario_name, [a.detail for a in r.failed_assertions])
        for r in suite_result.results
        if not r.passed
    ]
    assert suite_result.total == 2
    assert suite_result.passed == 2
