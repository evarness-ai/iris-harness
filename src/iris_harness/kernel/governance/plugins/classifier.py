"""Stage 1 data classifier — regex packs.

Scans inbound text for credential and PII patterns and returns the
highest-severity classification found.

Severity order: ``secret`` > ``personal`` > ``internal`` > ``public``.

Stage 2 (configurable LLM classifier on ambiguous inputs) and Stage 1B
(Presidio NER for unstructured PII) are separate plugins; both are off
by default and add their own dependencies. This plugin alone is enough
to enforce the credential half of the privacy promise (no API keys,
JWTs, or vault handles leaving local tiers).

Design references: §6.1 taxonomy, §6.2 classifier (two-stage), §6.5
output classification (the ``PostToolUse`` companion plugin reuses
``DataClassifier``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from iris_harness.kernel.governance.hooks.types import (
    DataClassification,
    HookContext,
    HookDecision,
    HookPoint,
)
from iris_harness.kernel.governance.plugins.regex_packs import ALL_PACKS, RegexEntry

logger = logging.getLogger(__name__)

# Severity order — index 0 is the most restrictive.
_SEVERITY: tuple[DataClassification, ...] = ("secret", "personal", "internal", "public")


@dataclass(frozen=True)
class ClassificationResult:
    """Outcome of a classifier scan."""

    classification: DataClassification
    matched_patterns: tuple[str, ...] = field(default_factory=tuple)


class DataClassifier:
    """Stage 1 classifier: regex packs over inbound text.

    Pure function of input text — no I/O, no model loading, no side
    effects. Safe to instantiate as a singleton at process start.
    """

    def __init__(self, *, packs: dict[str, list[RegexEntry]] | None = None) -> None:
        self._packs = packs if packs is not None else ALL_PACKS

    def classify(self, text: str) -> ClassificationResult:
        """Classify ``text`` against all configured packs."""
        if not text:
            return ClassificationResult(classification="public")

        matches: list[tuple[str, DataClassification]] = []
        for pack_name, patterns in self._packs.items():
            for name, pattern, classification in patterns:
                if pattern.search(text):
                    matches.append((f"{pack_name}/{name}", classification))

        if not matches:
            return ClassificationResult(classification="public")

        best = min(matches, key=lambda m: _SEVERITY.index(m[1]))
        return ClassificationResult(
            classification=best[1],
            matched_patterns=tuple(name for name, _ in matches),
        )


class DataClassifierHook:
    """``PreClassify`` hook wrapping ``DataClassifier``.

    Reads ``ctx.payload['text' | 'prompt' | 'input' | 'message']``,
    classifies it, and annotates the running context via
    ``set_classification``. Downstream hooks at the same point (e.g. a
    future PromptInjectionDetector) see the classification, and the
    caller threads the final context into the next hook point.
    """

    name: str = "data_classifier"
    hook_point: HookPoint = HookPoint.PRE_CLASSIFY
    priority: int = 10

    _TEXT_KEYS: tuple[str, ...] = ("text", "prompt", "input", "message")

    def __init__(self, classifier: DataClassifier | None = None) -> None:
        self._classifier = classifier or DataClassifier()

    async def __call__(self, ctx: HookContext) -> HookDecision:
        text = self._extract_text(ctx.payload)
        result = self._classifier.classify(text)
        logger.debug(
            "data_classifier run_id=%s class=%s matched=%s",
            ctx.run_id,
            result.classification,
            result.matched_patterns,
        )
        return HookDecision(
            outcome="allow",
            reason=f"classified as {result.classification}",
            set_classification=result.classification,
            audit_metadata={"matched_patterns": list(result.matched_patterns)},
        )

    @classmethod
    def _extract_text(cls, payload: dict[str, Any]) -> str:
        for key in cls._TEXT_KEYS:
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return ""
