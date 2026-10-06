"""PromptGuardInboundHook — Phase 6 G1 inbound prompt-injection guard (6a.3).

A ``PreClassify`` hook that screens the raw user turn for prompt injection /
jailbreak with the model-driven Prompt Guard 2 classifier (the existing regex
classifier remains the cheap fast-path at priority 10). Runs at **priority 5**,
before ``DataClassifierHook`` (10), so a jailbreak is caught before it can
influence classification or tier routing.

Shadow-first (plan §8): when ``shadow`` is set, a detection is logged + audited
but **allowed** (outcome ``allow``, severity ``critical``) so thresholds can be
tuned against real traffic before anything is blocked. With ``shadow=False`` the
configured ``on_detect`` (``require_approval`` / ``deny``) is honored. A degraded
guard (``error`` verdict) always allows — inbound screening must never break the
request path on a missing/slow model.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from iris_harness.kernel.governance.hooks.tool_payload import (
    RESULT,
    is_external,
    result_of,
    tool_name_of,
)
from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.threat.config import OnDetect
from iris_harness.kernel.governance.threat.types import ThreatClassifier

logger = logging.getLogger(__name__)

#: Marker substituted for a retrieved segment flagged as injection (G2 transform).
REDACTION_MARKER = "[redacted: possible prompt injection]"


class PromptGuardInboundHook:
    """Model-driven inbound injection/jailbreak guard at ``PreClassify``."""

    name: str = "prompt_guard_inbound"
    hook_point: HookPoint = HookPoint.PRE_CLASSIFY
    priority: int = 5  # before DataClassifierHook (10)

    _TEXT_KEYS: tuple[str, ...] = ("text", "prompt", "input", "message")

    def __init__(
        self,
        *,
        classifier: ThreatClassifier,
        on_detect: OnDetect = "require_approval",
        shadow: bool = True,
    ) -> None:
        self._classifier = classifier
        self._on_detect = on_detect
        self._shadow = shadow

    async def __call__(self, ctx: HookContext) -> HookDecision:
        text = self._extract_text(ctx.payload)
        if not text.strip():
            return HookDecision(outcome="allow", reason="prompt_guard_inbound: empty text")

        verdict = await self._classifier.score(text=text, surface="inbound")

        if verdict.label == "error":
            # Degraded guard: never block inbound on a missing/slow model.
            return HookDecision(
                outcome="allow",
                reason="prompt_guard_inbound: guard unavailable",
                severity="warn",
                audit_metadata={"backend": verdict.backend, "detail": verdict.detail},
            )

        if not verdict.is_threat:
            return HookDecision(
                outcome="allow",
                reason="prompt_guard_inbound: benign",
                audit_metadata={"score": verdict.score, "backend": verdict.backend},
            )

        audit: dict[str, Any] = {
            "label": verdict.label,
            "score": verdict.score,
            "categories": list(verdict.categories),
            "backend": verdict.backend,
            "shadow": self._shadow,
            "on_detect": self._on_detect,
        }

        # Shadow mode (or an explicit warn policy): log + audit, but allow.
        if self._shadow or self._on_detect in ("allow", "warn"):
            logger.warning(
                "prompt_guard_inbound[shadow] flagged label=%s score=%.3f run_id=%s",
                verdict.label,
                verdict.score,
                ctx.run_id,
            )
            return HookDecision(
                outcome="allow",
                reason=f"prompt_guard_inbound[shadow] flagged {verdict.label}",
                severity="critical",
                audit_metadata=audit,
            )

        if self._on_detect == "deny":
            return HookDecision(
                outcome="deny",
                reason=f"prompt injection detected ({verdict.label})",
                severity="critical",
                audit_metadata=audit,
            )

        # require_approval (D2 default) — and transform is not meaningful for a
        # whole inbound turn, so it falls through to the HITL path too.
        return HookDecision(
            outcome="require_approval",
            reason=f"prompt injection suspected ({verdict.label})",
            severity="critical",
            audit_metadata=audit,
        )

    @classmethod
    def _extract_text(cls, payload: dict[str, Any]) -> str:
        for key in cls._TEXT_KEYS:
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return ""


class PromptGuardRetrievedHook:
    """Phase 6 G2: indirect-injection guard over tool/RAG results at ``PostToolUse``.

    The highest-value guard for a connected agent — an attacker can plant
    injection inside content IRIS *retrieves* (RAG docs, emails, files), not just
    the prompt. Runs at priority 45 (after ``PostToolUseLedgerHook`` @40) and only
    for a tool that declares its output ``content: external`` -- text a third party
    wrote. The runner stamps the declaration on the context (``tool_content``, see
    ``kernel/governance/hooks/tool_payload.py``); there is no list of tool names.

    The result text is split into segments and each is scored; on ``transform``
    only the flagged segments are replaced with ``REDACTION_MARKER`` so the rest
    of the document stays usable (plan D3). Shadow mode audits but allows; a
    degraded guard allows; the kernel re-binds ``transformed_payload`` for
    downstream hooks.

    Opt-in and fail-open, so it is not what a default install relies on: the
    deterministic floor under it (``plugins/external_content_floor.py``, always on)
    marks and tripwire-scans the same results with no model.
    """

    name: str = "prompt_guard_retrieved"
    hook_point: HookPoint = HookPoint.POST_TOOL_USE
    priority: int = 45  # after PostToolUseLedgerHook (40)

    #: Keys searched (in order) for the scannable text inside a dict result.
    _RESULT_TEXT_KEYS: tuple[str, ...] = ("output", "content", "text", "result", "body")
    #: Cap on segments scored per result; beyond this the whole text is one scan.
    _MAX_SEGMENTS: int = 32

    def __init__(
        self,
        *,
        classifier: ThreatClassifier,
        on_detect: OnDetect = "transform",
        shadow: bool = True,
    ) -> None:
        self._classifier = classifier
        self._on_detect = on_detect
        self._shadow = shadow

    async def __call__(self, ctx: HookContext) -> HookDecision:
        tool = tool_name_of(ctx.payload)
        if not tool or not is_external(ctx.metadata):
            return HookDecision(
                outcome="allow", reason="prompt_guard_retrieved: not external content"
            )

        # A capability result arrives as a field map (``payload["fields"]``) beside its joined
        # text; the map is what reaches the consumer, so it is what gets scanned and redacted,
        # field by field -- rewriting only ``result`` would be dropped (plugin-capabilities §4).
        fields = ctx.payload.get("fields")
        by_field = isinstance(fields, dict) and bool(fields)
        text_key: str | None = None
        if by_field:
            assert isinstance(fields, dict)
            sources = {str(path): str(text) for path, text in fields.items()}
        else:
            text, text_key = self._extract_result_text(result_of(ctx.payload))
            sources = {"": text}
        if not any(text.strip() for text in sources.values()):
            return HookDecision(outcome="allow", reason="prompt_guard_retrieved: no scannable text")

        segmented = {key: self._segment(text) for key, text in sources.items() if text.strip()}
        segments = [(key, i, seg) for key, segs in segmented.items() for i, seg in enumerate(segs)]
        verdicts = [
            await self._classifier.score(text=seg, surface="retrieved") for _k, _i, seg in segments
        ]
        errored = any(v.label == "error" for v in verdicts)
        flagged_idx = [i for i, v in enumerate(verdicts) if v.is_threat]

        if not flagged_idx:
            reason = (
                "prompt_guard_retrieved: guard unavailable"
                if errored
                else "prompt_guard_retrieved: benign"
            )
            severity = "warn" if errored else "info"
            if not errored:
                return HookDecision(outcome="allow", reason=reason, severity=severity)
            # An unscanned result must be traceable from the ledger alone: which tool's
            # text went through unscanned, how much of it, and why the classifier said no.
            failed = [v for v in verdicts if v.label == "error"]
            return HookDecision(
                outcome="allow",
                reason=reason,
                severity=severity,
                audit_metadata={
                    "tool": tool,
                    "scanned": False,
                    "segments_total": len(segments),
                    "segments_unscanned": len(failed),
                    "backend": failed[0].backend,
                    "detail": failed[0].detail,
                },
            )

        max_score = max(verdicts[i].score for i in flagged_idx)
        audit: dict[str, Any] = {
            "tool": tool,
            "segments_total": len(segments),
            "segments_flagged": len(flagged_idx),
            "max_score": max_score,
            "shadow": self._shadow,
            "on_detect": self._on_detect,
        }

        if self._shadow or self._on_detect in ("allow", "warn"):
            logger.warning(
                "prompt_guard_retrieved[shadow] flagged tool=%s segments=%d/%d run_id=%s",
                tool,
                len(flagged_idx),
                len(segments),
                ctx.run_id,
            )
            return HookDecision(
                outcome="allow",
                reason=f"prompt_guard_retrieved[shadow] flagged {len(flagged_idx)} segment(s)",
                severity="critical",
                audit_metadata=audit,
            )

        if self._on_detect == "deny":
            return HookDecision(
                outcome="deny",
                reason=f"injection in retrieved content from {tool}",
                severity="critical",
                audit_metadata=audit,
            )

        # transform (D3 default): redact only the flagged segments, keep the rest.
        flagged = {(segments[i][0], segments[i][1]) for i in flagged_idx}
        redacted = {
            key: "\n\n".join(
                REDACTION_MARKER if (key, i) in flagged else seg for i, seg in enumerate(segs)
            )
            for key, segs in segmented.items()
        }
        if by_field:
            new_fields = {key: redacted.get(key, text) for key, text in sources.items()}
            payload = {
                **ctx.payload,
                "fields": new_fields,
                RESULT: "\n".join(new_fields.values()),
            }
        else:
            new_result = self._replace_result_text(result_of(ctx.payload), redacted[""], text_key)
            payload = {**ctx.payload, RESULT: new_result}
        return HookDecision(
            outcome="transform",
            reason=f"redacted {len(flagged_idx)} injected segment(s) from {tool}",
            transformed_payload=payload,
            severity="critical",
            audit_metadata=audit,
        )

    @classmethod
    def _extract_result_text(cls, result: object) -> tuple[str, str | None]:
        """Return ``(text, key)`` — ``key`` is the dict field the text came from
        (``None`` when the result is a bare string)."""
        if isinstance(result, str):
            return result, None
        if isinstance(result, dict):
            for key in cls._RESULT_TEXT_KEYS:
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value, key
        return "", None

    @staticmethod
    def _replace_result_text(result: object, redacted: str, text_key: str | None) -> object:
        if text_key is None:
            return redacted
        if isinstance(result, dict):
            return {**result, text_key: redacted}
        return redacted

    @classmethod
    def _segment(cls, text: str) -> list[str]:
        parts = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not parts:
            parts = [text]
        if len(parts) > cls._MAX_SEGMENTS:
            # Too granular to scan economically — fall back to one whole-text scan.
            return [text]
        return parts
