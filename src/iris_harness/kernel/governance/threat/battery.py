"""Red-team detection battery for the Phase 6 threat guards (sub-phase 6a.5).

Runs a labeled corpus of injection / jailbreak / unsafe-output attacks (plus a
benign control set for false-positive measurement) through a scorer and grades
detection rate vs. false-positive rate. Used two ways:

- **Unit** — against a deterministic reference scorer, to validate the harness
  math + corpus integrity in CI (no model weights needed).
- **Live** — against a real ``ThreatDetector`` (opt-in), to decide whether a
  guard is ready to flip from ``mode: shadow`` to ``mode: enforce``.

This is the gate on enforcement: do not flip a guard to enforce until the live
battery clears the thresholds (plan §8 rollout).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

from iris_harness.kernel.governance.threat.types import ThreatSurface, ThreatVerdict


@runtime_checkable
class Scorer(Protocol):
    """Anything that scores text for a surface — a ``ThreatClassifier`` or a
    ``ThreatDetector`` (both expose this method)."""

    async def score(self, *, text: str, surface: ThreatSurface) -> ThreatVerdict: ...


@dataclass(frozen=True)
class CorpusCase:
    """One labeled red-team example. ``malicious`` is the ground truth."""

    id: str
    surface: ThreatSurface
    text: str
    malicious: bool
    category: str = ""


@dataclass(frozen=True)
class SurfaceStats:
    malicious: int = 0
    benign: int = 0
    detected: int = 0
    false_positives: int = 0


@dataclass(frozen=True)
class BatteryReport:
    """Aggregate detection metrics over a corpus run."""

    total: int
    malicious_total: int
    benign_total: int
    detected: int
    false_positives: int
    errors: int
    by_surface: dict[str, SurfaceStats] = field(default_factory=dict)

    @property
    def detection_rate(self) -> float:
        return self.detected / self.malicious_total if self.malicious_total else 1.0

    @property
    def false_positive_rate(self) -> float:
        return self.false_positives / self.benign_total if self.benign_total else 0.0

    def passes(self, *, min_detection: float, max_false_positive: float) -> bool:
        return (
            self.detection_rate >= min_detection and self.false_positive_rate <= max_false_positive
        )

    def summary(self) -> str:
        return (
            f"detection={self.detection_rate:.1%} ({self.detected}/{self.malicious_total}) "
            f"fpr={self.false_positive_rate:.1%} ({self.false_positives}/{self.benign_total}) "
            f"errors={self.errors}"
        )


def load_corpus(path: Path) -> list[CorpusCase]:
    """Load + validate the labeled corpus YAML. Raises on malformed entries."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError(f"corpus must have a non-empty 'cases' list: {path}")
    cases: list[CorpusCase] = []
    seen: set[str] = set()
    for entry in cases_raw:
        if not isinstance(entry, dict):
            raise ValueError(f"corpus case must be a mapping: {entry!r}")
        case = CorpusCase(
            id=str(entry["id"]),
            surface=entry["surface"],
            text=str(entry["text"]),
            malicious=bool(entry["malicious"]),
            category=str(entry.get("category", "")),
        )
        if case.id in seen:
            raise ValueError(f"duplicate corpus case id: {case.id}")
        if case.surface not in ("inbound", "retrieved", "output"):
            raise ValueError(f"invalid surface for {case.id}: {case.surface}")
        seen.add(case.id)
        cases.append(case)
    return cases


async def run_battery(scorer: Scorer, cases: list[CorpusCase]) -> BatteryReport:
    """Score every case and aggregate detection / false-positive metrics.

    A classifier ``error`` verdict counts as *not detected* (conservative) and is
    tracked separately so a degraded backend is visible rather than silently
    inflating or deflating the detection rate.
    """
    detected = 0
    false_positives = 0
    errors = 0
    surfaces: dict[str, dict[str, int]] = {}

    for case in cases:
        bucket = surfaces.setdefault(
            case.surface, {"malicious": 0, "benign": 0, "detected": 0, "false_positives": 0}
        )
        verdict = await scorer.score(text=case.text, surface=case.surface)
        is_error = verdict.label == "error"
        if is_error:
            errors += 1
        flagged = verdict.is_threat  # error -> False

        if case.malicious:
            bucket["malicious"] += 1
            if flagged:
                detected += 1
                bucket["detected"] += 1
        else:
            bucket["benign"] += 1
            if flagged:
                false_positives += 1
                bucket["false_positives"] += 1

    malicious_total = sum(1 for c in cases if c.malicious)
    benign_total = len(cases) - malicious_total
    return BatteryReport(
        total=len(cases),
        malicious_total=malicious_total,
        benign_total=benign_total,
        detected=detected,
        false_positives=false_positives,
        errors=errors,
        by_surface={s: SurfaceStats(**vals) for s, vals in surfaces.items()},
    )
