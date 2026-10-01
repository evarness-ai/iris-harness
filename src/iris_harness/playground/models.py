"""Scenario + result models for the playground engine.

A *scenario* declares an input and what the harness should do with it; the
engine runs it and produces a *result*. Everything an adopter needs to write a
regression case lives in YAML — no Python.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ScenarioExpectation(BaseModel):
    """What the harness should do with the scenario's message.

    Every field is optional; only the ones you set are asserted. This keeps a
    scenario focused ("this must NOT web-search") rather than pinning the whole
    response. Deterministic-first: prefer intent/handler/sources/guardrail
    assertions (stable) over exact response text (stochastic under a live LLM).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Resolved routing intent (e.g. "finance", "profile_query").
    intent: str | None = None
    # Agent that handled the turn (e.g. "finance", "general").
    agent_type: str | None = None
    # Which intercept answered, by its pipeline phase name minus ".end"
    # (e.g. "dues_request", "time_date"). Use "" to assert NO intercept fired
    # (the turn went to the agent loop).
    handler: str | None = None
    # Tool/source names that must (not) appear in ChatResult.sources.
    sources_include: tuple[str, ...] = Field(default_factory=tuple)
    sources_exclude: tuple[str, ...] = Field(default_factory=tuple)
    # Case-insensitive substrings the response must (not) contain.
    response_contains: tuple[str, ...] = Field(default_factory=tuple)
    response_not_contains: tuple[str, ...] = Field(default_factory=tuple)
    # Regex the response must match (re.search, multiline).
    response_regex: str | None = None
    # Whether the turn is expected to carry an error.
    has_errors: bool | None = None
    # Subset match against ChatResult.metadata (each key must equal).
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Guardrail: response must contain no email address and no raw prompt marker.
    no_pii_leak: bool = False


class Scenario(BaseModel):
    """One playground case: an input, the flags it runs under, and expectations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1)
    description: str = ""
    message: str = Field(..., min_length=1)
    channel: str = "console"
    # Prior turns run (in order) to establish conversational state before the
    # asserted message. Their outcomes are not asserted.
    setup_messages: tuple[str, ...] = Field(default_factory=tuple)
    # Per-scenario env overrides applied for the turn (e.g. feature flags).
    # Best-effort for flags read per-turn; flags read at build time need a
    # suite-level override (the runner rebuilds per suite, not per scenario).
    env: dict[str, str] = Field(default_factory=dict)
    tags: tuple[str, ...] = Field(default_factory=tuple)
    expect: ScenarioExpectation = Field(default_factory=ScenarioExpectation)


class ScenarioSuite(BaseModel):
    """A named collection of scenarios, loaded from one YAML file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1)
    description: str = ""
    # Suite-level env applied before the runtime is built, so build-time flags
    # (most IRIS_* intercept toggles) take effect for every scenario in the file.
    env: dict[str, str] = Field(default_factory=dict)
    # Run on a throwaway memory: a fresh data directory (memory.db, the index) and a
    # fresh identity workspace (USER.md, where confirmed facts are projected), removed
    # afterwards. A suite that TELLS the assistant things ("my bank is Barclays") needs
    # this, or what it says is remembered in the owner's real memory. Scenarios in the
    # suite share it, in order, so a later scenario can ask what an earlier one said.
    isolated: bool = False
    scenarios: tuple[Scenario, ...] = Field(default_factory=tuple)


class AssertionResult(BaseModel):
    """Outcome of one expectation field."""

    model_config = ConfigDict(frozen=True)

    field: str
    ok: bool
    expected: Any = None
    actual: Any = None
    detail: str = ""


class ScenarioResult(BaseModel):
    """What actually happened when a scenario ran, plus per-assertion verdicts."""

    model_config = ConfigDict(frozen=True)

    scenario_name: str
    passed: bool
    assertions: tuple[AssertionResult, ...] = Field(default_factory=tuple)
    # Observed turn outcome (for the report + before/after diff).
    response: str = ""
    intent: str | None = None
    agent_type: str | None = None
    handler: str | None = None
    sources: tuple[str, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None

    @property
    def failed_assertions(self) -> tuple[AssertionResult, ...]:
        return tuple(a for a in self.assertions if not a.ok)


class SuiteResult(BaseModel):
    """Aggregate result for a whole suite."""

    model_config = ConfigDict(frozen=True)

    suite_name: str
    results: tuple[ScenarioResult, ...] = Field(default_factory=tuple)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def ok(self) -> bool:
        return self.failed == 0
