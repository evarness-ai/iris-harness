"""Declarative config for threat detection (sub-phase 6a.1).

Loaded from ``config/governance/threat-detection.yaml``. Every default here
matches the resolved decisions D1-D5 in
``docs/architecture/governance-phase6-threat-detection.md`` §9, so the system
behaves correctly even with an empty/absent file (``disabled()`` fallback).

``extra="forbid"`` on every model means a stray or misspelled key fails loudly
at load time rather than silently dropping policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

#: Maps onto the kernel's ``HookOutcome`` set (plus ``transform`` for G2).
#: Consumed by the enforcement hooks in later sub-phases; inert in 6a.1.
OnDetect = Literal["allow", "warn", "require_approval", "deny", "transform"]

#: Global enforcement mode. ``shadow`` downgrades every guard's ``on_detect`` to
#: log-only (allow + audit) so thresholds can be tuned against real traffic
#: before anything is blocked; ``enforce`` honors the per-surface ``on_detect``.
#: Shadow precedes enforcement for each guard (plan §8 rollout).
ThreatMode = Literal["shadow", "enforce"]


class BackendConfig(BaseModel):
    """One classifier backend (provider + model + optional endpoint)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    endpoint: str | None = None


class BackendBundle(BaseModel):
    """The two model backends: Prompt Guard (G1/G2) and the output guard (G3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_guard: BackendConfig = Field(
        default_factory=lambda: BackendConfig(
            provider="transformers", model="meta-llama/Llama-Prompt-Guard-2-86M"
        )
    )
    output_guard: BackendConfig = Field(
        default_factory=lambda: BackendConfig(
            provider="ollama",
            model="llama-guard3:1b",
            endpoint="http://localhost:11434",
        )
    )


class LatencyBudget(BaseModel):
    """Per-surface p95 budgets in milliseconds (D5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inbound: int = 150
    retrieved: int = 150
    output: int = 400


class InboundConfig(BaseModel):
    """G1 — inbound user-turn screening."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    on_detect: OnDetect = "require_approval"  # D2


class RetrievedConfig(BaseModel):
    """G2 — retrieved/tool content screening (indirect injection).

    Which results are scanned is not configured here: a tool declares its output
    ``content: external`` (``kernel/governance/hooks/tool_payload.py``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    on_detect: OnDetect = "transform"  # D3


class OutputConfig(BaseModel):
    """G3 — outbound response screening. Categories split per D4."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    #: Hazard categories that block the response (``deny``).
    enforce: tuple[str, ...] = (
        "violent_crimes",
        "sex_crimes",
        "child_exploitation",
        "self_harm",
        "privacy",
        "indiscriminate_weapons",
    )
    #: Categories that are audited but shipped (observe before enforcing).
    log_only: tuple[str, ...] = (
        "hate",
        "defamation",
        "specialized_advice",
        "elections",
        "intellectual_property",
    )


class InputSafetyConfig(BaseModel):
    """The user turn screened for hazards (self-harm, violence, ...) at ``PRE_TURN``.

    Opt-in twice over: the ``IRIS_GOVERNANCE_INPUT_SAFETY`` flag installs it, and this
    section must be enabled. It reuses the output guard's model, so a user message is
    judged the way a response is. ``enforce`` / ``log_only`` left unset mean the output
    guard's lists, so ``self_harm`` is covered with no new configuration
    (docs/architecture/deterministic-path-parity.md, step b2).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    enforce: tuple[str, ...] | None = None
    log_only: tuple[str, ...] | None = None


class ThreatDetectionConfig(BaseModel):
    """Top-level threat-detection policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = 1
    enabled: bool = True
    mode: ThreatMode = "shadow"
    backend: BackendBundle = Field(default_factory=BackendBundle)
    latency_budget: LatencyBudget = Field(default_factory=LatencyBudget)
    inbound: InboundConfig = Field(default_factory=InboundConfig)
    retrieved: RetrievedConfig = Field(default_factory=RetrievedConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    input_safety: InputSafetyConfig = Field(default_factory=InputSafetyConfig)

    def input_safety_categories(self) -> tuple[frozenset[str], frozenset[str]]:
        """(enforce, log_only) for the input screen, defaulting to the output guard's."""
        section = self.input_safety
        enforce = section.enforce if section.enforce is not None else self.output.enforce
        log_only = section.log_only if section.log_only is not None else self.output.log_only
        return frozenset(enforce), frozenset(log_only)

    def budget_for(self, surface: Literal["inbound", "retrieved", "output"]) -> int:
        return int(getattr(self.latency_budget, surface))

    @classmethod
    def disabled(cls) -> ThreatDetectionConfig:
        """A fully-inert config. The safe default when no file is present."""
        return cls(
            enabled=False,
            inbound=InboundConfig(enabled=False),
            retrieved=RetrievedConfig(enabled=False),
            output=OutputConfig(enabled=False),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> ThreatDetectionConfig:
        """Load + validate policy from disk.

        A *present* but malformed file raises ``ValueError`` (loud failure on
        real misconfiguration). An *absent* file degrades to ``disabled()`` so
        installs without the config simply run no guards.
        """
        if not path.exists():
            return cls.disabled()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"threat-detection config must decode to a mapping: {path}")
        return cls.model_validate(raw)
