"""Threat-detection provider contract (Governance Phase 6, sub-phase 6a.1).

Model-driven content guards screen three surfaces for prompt injection,
jailbreak attempts, and unsafe output:

- ``inbound``   — the raw user turn (guard G1, future hook at ``PRE_CLASSIFY``)
- ``retrieved`` — RAG / tool results (guard G2, future hook at ``POST_TOOL_USE``)
- ``output``    — the final response (guard G3, future curator ``JudgeSignal``)

This module defines the swappable provider contract. Backends live in
``backends.py`` and are selected from ``config/governance/threat-detection.yaml``
(see ``config.py``). The orchestrating facade is ``detector.ThreatDetector``.

Sub-phase 6a.1 ships the abstraction + backends + config loader only.
**Nothing here enforces yet** — the kernel hooks / curator signal that consume
these verdicts land in later sub-phases (6a.2-6a.4). See
``docs/architecture/governance-phase6-threat-detection.md``.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

#: The content surface a verdict applies to.
ThreatSurface = Literal["inbound", "retrieved", "output"]

#: Verdict label. ``error`` means the classifier could not produce a verdict
#: (model down, parse failure); in shadow mode it is logged, never raised.
ThreatLabel = Literal["benign", "injection", "jailbreak", "unsafe", "error"]


class ThreatVerdict(BaseModel):
    """One classifier's verdict for one piece of content.

    ``score`` is the model's malicious/unsafe probability in ``[0, 1]``
    (``1.0`` for binary detectors like Llama Guard). ``categories`` carries
    backend-specific labels (e.g. Llama Guard 3 hazard names). The verdict is
    advisory data — callers decide what to do with it.
    """

    model_config = ConfigDict(frozen=True)

    label: ThreatLabel
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    surface: ThreatSurface
    categories: tuple[str, ...] = Field(default_factory=tuple)
    backend: str = "unknown"
    latency_ms: float | None = None
    detail: str | None = None

    @property
    def is_threat(self) -> bool:
        """True when the content was flagged (excludes ``benign`` and ``error``)."""
        return self.label not in ("benign", "error")

    @classmethod
    def benign(
        cls,
        *,
        surface: ThreatSurface,
        backend: str,
        score: float = 0.0,
        latency_ms: float | None = None,
    ) -> ThreatVerdict:
        return cls(
            label="benign",
            score=score,
            surface=surface,
            backend=backend,
            latency_ms=latency_ms,
        )

    @classmethod
    def failure(
        cls,
        *,
        surface: ThreatSurface,
        backend: str,
        detail: str,
        latency_ms: float | None = None,
    ) -> ThreatVerdict:
        """A verdict the classifier could not compute. Fail-safe in shadow mode."""
        return cls(
            label="error",
            score=0.0,
            surface=surface,
            backend=backend,
            detail=detail,
            latency_ms=latency_ms,
        )


@runtime_checkable
class ThreatClassifier(Protocol):
    """The swappable provider contract every backend satisfies.

    Implementations must never raise from ``score`` — they return an ``error``
    verdict instead, so a degraded guard can fail-safe rather than break the
    request path. Enforcement (deny / approval / transform) is the caller's
    job in later sub-phases, driven by ``config.fail_mode``.
    """

    name: str

    async def score(self, *, text: str, surface: ThreatSurface) -> ThreatVerdict: ...
