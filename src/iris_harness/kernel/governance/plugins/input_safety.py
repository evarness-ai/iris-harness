"""InputSafetyHook — the user turn screened for hazards at ``PreTurn`` (opt-in).

Every turn passes the ``PRE_TURN`` screen before anything answers it, deterministic
handlers included (docs/architecture/deterministic-path-parity.md). The default
screens are model-free (credential / PII patterns) and cannot recognise a hazardous
request such as self-harm; the 2026-07-06 red-team found one captured by a keyword
handler and answered with no guard at all. This hook runs the output guard's model
(Llama Guard) on the user's message, with the output guard's category lists.

It is opt-in (``IRIS_GOVERNANCE_INPUT_SAFETY``) and follows the threat config's
``mode``: in ``shadow`` a hazard is logged and audited but allowed; in ``enforce`` an
enforced category refuses the turn and a ``log_only`` one is audited and allowed. A
slow or missing model lets the turn through — the inbound Prompt Guard's rule — so an
input check never breaks chat.
"""

from __future__ import annotations

import logging
from typing import Any

from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.threat.types import ThreatClassifier

logger = logging.getLogger(__name__)


class InputSafetyHook:
    """Llama Guard on the raw user message at ``PRE_TURN``."""

    name: str = "input_safety"
    hook_point: HookPoint = HookPoint.PRE_TURN
    priority: int = 7  # after the inbound Prompt Guard (5), before the data classifier (10)

    _TEXT_KEYS: tuple[str, ...] = ("message", "text", "prompt", "input")

    def __init__(
        self,
        *,
        classifier: ThreatClassifier,
        enforce: frozenset[str],
        log_only: frozenset[str] = frozenset(),
        shadow: bool = True,
    ) -> None:
        self._classifier = classifier
        self._enforce = enforce
        self._log_only = log_only
        self._shadow = shadow

    async def __call__(self, ctx: HookContext) -> HookDecision:
        text = self._extract_text(ctx.payload)
        if not text.strip():
            return HookDecision(outcome="allow", reason="input_safety: empty message")

        verdict = await self._classifier.score(text=text, surface="inbound")

        if verdict.label == "error":
            return HookDecision(
                outcome="allow",
                reason="input_safety: guard unavailable",
                severity="warn",
                audit_metadata={"backend": verdict.backend, "detail": verdict.detail},
            )
        if not verdict.is_threat:
            return HookDecision(
                outcome="allow",
                reason="input_safety: safe",
                audit_metadata={"backend": verdict.backend},
            )

        categories = tuple(verdict.categories)
        enforced = tuple(c for c in categories if c in self._enforce)
        audit: dict[str, Any] = {
            "categories": list(categories),
            "enforced": list(enforced),
            "backend": verdict.backend,
            "shadow": self._shadow,
        }
        # An unsafe verdict naming no category we know fails safe, as the output guard does.
        blocks = bool(enforced) or not categories
        if self._shadow:
            logger.warning(
                "input_safety[shadow] flagged categories=%s run_id=%s", categories, ctx.run_id
            )
            return HookDecision(
                outcome="allow",
                reason=f"input_safety[shadow] flagged {', '.join(categories) or 'uncategorized'}",
                severity="critical",
                audit_metadata=audit,
            )
        if blocks:
            return HookDecision(
                outcome="deny",
                reason=f"unsafe request: {', '.join(enforced) or 'uncategorized'}",
                severity="critical",
                audit_metadata=audit,
            )
        return HookDecision(
            outcome="allow",
            reason=f"input_safety (log-only): {', '.join(categories)}",
            severity="warn",
            audit_metadata=audit,
        )

    @classmethod
    def _extract_text(cls, payload: dict[str, Any]) -> str:
        for key in cls._TEXT_KEYS:
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return ""


__all__ = ["InputSafetyHook"]
