"""Governance Phase 6 — threat detection (sub-phase 6a.1).

Model-driven, swappable content guards for inbound / retrieved / outbound
surfaces. 6a.1 ships the provider abstraction, backends, config loader, and a
shadow-mode detector facade. No enforcement yet — see
``docs/architecture/governance-phase6-threat-detection.md``.
"""

from __future__ import annotations

from iris_harness.kernel.governance.threat.backends import (
    LLAMA_GUARD_CATEGORIES,
    LlamaGuardClassifier,
    NullClassifier,
    PromptGuardClassifier,
    default_ollama_invoke,
)
from iris_harness.kernel.governance.threat.battery import (
    BatteryReport,
    CorpusCase,
    load_corpus,
    run_battery,
)
from iris_harness.kernel.governance.threat.config import (
    BackendBundle,
    BackendConfig,
    InboundConfig,
    LatencyBudget,
    OutputConfig,
    RetrievedConfig,
    ThreatDetectionConfig,
)
from iris_harness.kernel.governance.threat.detector import ThreatDetector, build_threat_detector
from iris_harness.kernel.governance.threat.types import (
    ThreatClassifier,
    ThreatLabel,
    ThreatSurface,
    ThreatVerdict,
)

__all__ = [
    "LLAMA_GUARD_CATEGORIES",
    "BackendBundle",
    "BackendConfig",
    "BatteryReport",
    "CorpusCase",
    "InboundConfig",
    "LatencyBudget",
    "LlamaGuardClassifier",
    "NullClassifier",
    "OutputConfig",
    "PromptGuardClassifier",
    "RetrievedConfig",
    "ThreatClassifier",
    "ThreatDetectionConfig",
    "ThreatDetector",
    "ThreatLabel",
    "ThreatSurface",
    "ThreatVerdict",
    "build_threat_detector",
    "default_ollama_invoke",
    "load_corpus",
    "run_battery",
]
