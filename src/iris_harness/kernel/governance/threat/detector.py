"""ThreatDetector facade + backend factory (sub-phase 6a.1).

``ThreatDetector`` routes each surface to its classifier (Prompt Guard for
``inbound``/``retrieved``, the output guard for ``output``), honors the
per-surface + global ``enabled`` switches, and emits **shadow-mode** logs.

Shadow mode is the whole point of 6a.1: ``score_*`` always returns a verdict
and never enforces or raises. A flagged verdict is logged at WARNING so we can
tune thresholds against real traffic before any guard denies a request in
6a.2+. Over-budget latencies are logged too, never acted on.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from iris_harness.kernel.governance.threat.backends import (
    LlamaGuardClassifier,
    NullClassifier,
    PromptGuardClassifier,
    default_ollama_invoke,
)
from iris_harness.kernel.governance.threat.config import ThreatDetectionConfig
from iris_harness.kernel.governance.threat.types import (
    ThreatClassifier,
    ThreatSurface,
    ThreatVerdict,
)

logger = logging.getLogger(__name__)


class ThreatDetector:
    """Routes content to the right classifier and shadow-logs the verdict."""

    def __init__(
        self,
        *,
        config: ThreatDetectionConfig,
        prompt_guard: ThreatClassifier,
        output_guard: ThreatClassifier,
    ) -> None:
        self._config = config
        self._prompt_guard = prompt_guard
        self._output_guard = output_guard

    @property
    def config(self) -> ThreatDetectionConfig:
        return self._config

    @property
    def prompt_guard(self) -> ThreatClassifier:
        return self._prompt_guard

    @property
    def output_guard(self) -> ThreatClassifier:
        return self._output_guard

    def _surface_enabled(self, surface: ThreatSurface) -> bool:
        if not self._config.enabled:
            return False
        return bool(getattr(self._config, surface).enabled)

    def _classifier_for(self, surface: ThreatSurface) -> ThreatClassifier:
        return self._output_guard if surface == "output" else self._prompt_guard

    async def score(self, *, text: str, surface: ThreatSurface) -> ThreatVerdict:
        """Score ``text`` for ``surface``. Always returns a verdict; never enforces."""
        if not self._surface_enabled(surface):
            return ThreatVerdict.benign(surface=surface, backend="disabled")

        verdict = await self._classifier_for(surface).score(text=text, surface=surface)
        self._shadow_log(verdict)
        return verdict

    async def score_inbound(self, text: str) -> ThreatVerdict:
        return await self.score(text=text, surface="inbound")

    async def score_retrieved(self, text: str) -> ThreatVerdict:
        return await self.score(text=text, surface="retrieved")

    async def score_output(self, text: str) -> ThreatVerdict:
        return await self.score(text=text, surface="output")

    def _shadow_log(self, verdict: ThreatVerdict) -> None:
        budget = self._config.budget_for(verdict.surface)
        over_budget = verdict.latency_ms is not None and verdict.latency_ms > budget
        if verdict.is_threat:
            logger.warning(
                "threat-detection[shadow] flagged surface=%s label=%s score=%.3f "
                "categories=%s backend=%s latency_ms=%s",
                verdict.surface,
                verdict.label,
                verdict.score,
                ",".join(verdict.categories) or "-",
                verdict.backend,
                f"{verdict.latency_ms:.0f}" if verdict.latency_ms is not None else "-",
            )
        elif verdict.label == "error":
            logger.warning(
                "threat-detection[shadow] backend error surface=%s backend=%s detail=%s",
                verdict.surface,
                verdict.backend,
                verdict.detail,
            )
        if over_budget:
            logger.warning(
                "threat-detection[shadow] over budget surface=%s latency_ms=%.0f budget_ms=%d",
                verdict.surface,
                verdict.latency_ms,
                budget,
            )


def build_threat_detector(
    config: ThreatDetectionConfig,
    *,
    prompt_guard: ThreatClassifier | None = None,
    output_guard: ThreatClassifier | None = None,
    output_invoke: Callable[[str, str], str] | None = None,
) -> ThreatDetector:
    """Construct a ``ThreatDetector`` from config, injecting/falling back per backend.

    Tests and the future wiring layer can inject ready classifiers (or just an
    ``output_invoke`` transport). When nothing is injected, build the defaults
    from config; any backend that can't be constructed degrades to
    ``NullClassifier`` (logged), which keeps shadow mode safe.
    """
    if prompt_guard is None:
        prompt_guard = _build_prompt_guard(config)
    if output_guard is None:
        output_guard = _build_output_guard(config, invoke=output_invoke)
    return ThreatDetector(config=config, prompt_guard=prompt_guard, output_guard=output_guard)


def _build_prompt_guard(config: ThreatDetectionConfig) -> ThreatClassifier:
    if not config.enabled or not (config.inbound.enabled or config.retrieved.enabled):
        return NullClassifier(name="prompt_guard", reason="disabled")
    # Use the stricter of the two surface thresholds for the shared classifier.
    threshold = min(config.inbound.threshold, config.retrieved.threshold)
    backend = config.backend.prompt_guard
    if backend.provider in ("transformers", "onnx"):
        return PromptGuardClassifier(model_id=backend.model, threshold=threshold)
    logger.warning("unknown prompt_guard provider %r; running NullClassifier", backend.provider)
    return NullClassifier(name="prompt_guard", reason=f"provider {backend.provider}")


def _build_output_guard(
    config: ThreatDetectionConfig, *, invoke: Callable[[str, str], str] | None
) -> ThreatClassifier:
    if not config.enabled or not config.output.enabled:
        return NullClassifier(name="llama_guard", reason="disabled")
    backend = config.backend.output_guard
    if invoke is None and backend.provider == "ollama" and backend.endpoint:
        timeout_s = max(0.1, config.budget_for("output") / 1000.0 * 3)
        invoke = default_ollama_invoke(
            endpoint=backend.endpoint, model=backend.model, timeout_s=timeout_s
        )
    if invoke is None:
        logger.warning(
            "no output-guard transport for provider %r; running NullClassifier",
            backend.provider,
        )
        return NullClassifier(name="llama_guard", reason=f"provider {backend.provider}")
    return LlamaGuardClassifier(invoke=invoke)
