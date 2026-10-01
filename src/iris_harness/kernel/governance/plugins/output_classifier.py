"""OutputClassifierHook — a tool's result raises the run's data classification.

Design §6.5. The gap this closes: the run's classification was computed once, from the
user's question, and cached for the rest of the run
(``agent/agentic_core.py``, ``_governance_pre_llm`` only classifies when it is
``None``). So "wondering how my day looks like ?" was classified ``public``, the
``daily_plan`` tool returned the user's inbox, and every later ``egress_gate`` decision
in that run was still made against ``public`` — with personal data in the prompt. The
gate answered correctly for the inputs it was given; the inputs were stale.

The question, not the answer, is what the user typed. A tool is where the personal data
actually enters a turn, so the tool's result is where the label has to be re-derived.

Classification only ever **rises**. A tool that returns nothing sensitive leaves the run
where it was; it can never talk the label back down to ``public`` after something
personal has already entered the prompt, because the prompt keeps that content for the
rest of the run.
"""

from __future__ import annotations

import logging
from typing import Any

from iris_harness.kernel.governance.hooks.types import (
    DataClassification,
    HookContext,
    HookDecision,
    HookPoint,
)
from iris_harness.kernel.governance.plugins.classifier import DataClassifier

logger = logging.getLogger(__name__)

# Most restrictive first, matching `plugins/classifier.py`.
_SEVERITY: tuple[DataClassification, ...] = ("secret", "personal", "internal", "public")


def more_restrictive(
    left: DataClassification | None, right: DataClassification | None
) -> DataClassification | None:
    """The stricter of two labels; ``None`` means "nothing said", not "public"."""
    if left is None:
        return right
    if right is None:
        return left
    return left if _SEVERITY.index(left) <= _SEVERITY.index(right) else right


class OutputClassifierHook:
    """``PostToolUse`` hook that raises the run's classification from a tool result."""

    name: str = "output_classifier"
    hook_point: HookPoint = HookPoint.POST_TOOL_USE
    # Before the side-effect ledger (40) and the injection guard (45): both are better
    # off seeing the label this derives than the stale one the turn came in with.
    priority: int = 10

    _RESULT_KEYS: tuple[str, ...] = ("result", "observation", "output", "text")

    def __init__(self, classifier: DataClassifier | None = None) -> None:
        self._classifier = classifier or DataClassifier()

    async def __call__(self, ctx: HookContext) -> HookDecision:
        text = self._extract_text(ctx.payload)
        if not text:
            return HookDecision(outcome="allow", reason="output_classifier: tool returned no text")

        observed = self._classifier.classify(text)
        raised = more_restrictive(ctx.classification, observed.classification)

        if raised == ctx.classification:
            return HookDecision(
                outcome="allow",
                reason=(
                    f"output_classifier: {self._tool_name(ctx)} output is "
                    f"{observed.classification}; run stays {ctx.classification}"
                ),
            )

        logger.info(
            "output_classifier run_id=%s tool=%s raised %s -> %s",
            ctx.run_id,
            self._tool_name(ctx),
            ctx.classification,
            raised,
        )
        return HookDecision(
            outcome="allow",
            reason=(
                f"output_classifier: {self._tool_name(ctx)} output raised the run "
                f"from {ctx.classification} to {raised}"
            ),
            severity="warn",
            set_classification=raised,
            audit_metadata={
                "from": ctx.classification,
                "to": raised,
                "tool_name": self._tool_name(ctx),
                "matched_patterns": list(observed.matched_patterns),
            },
        )

    @staticmethod
    def _tool_name(ctx: HookContext) -> str:
        """The tool this result came from; ``PreToolUse`` puts it on the payload."""
        name = ctx.payload.get("tool_name")
        return name if isinstance(name, str) and name else "tool"

    @classmethod
    def _extract_text(cls, payload: dict[str, Any]) -> str:
        for key in cls._RESULT_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""


__all__ = ["OutputClassifierHook", "more_restrictive"]
