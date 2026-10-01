"""IRIS playground — a YAML scenario engine to test, tweak and observe the harness.

The playground runs a message through the real runtime under a declared set of
flags and asserts on the outcome (resolved intent, which intercept/handler
answered, tools used, guardrails, response). Scenarios are YAML, so they double
as a deterministic regression net (the safety belt for the bootstrap
decomposition) and as living documentation of how the harness behaves.

Surfaced three ways per the thin-renderer rule: ``iris playground`` (CLI), the
``/playground`` API, and the web Playground screen. The logic lives here.
"""

from .models import (
    AssertionResult,
    Scenario,
    ScenarioExpectation,
    ScenarioResult,
    ScenarioSuite,
    SuiteResult,
)

__all__ = [
    "AssertionResult",
    "Scenario",
    "ScenarioExpectation",
    "ScenarioResult",
    "ScenarioSuite",
    "SuiteResult",
]
