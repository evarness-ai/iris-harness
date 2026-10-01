"""LLMJudge — opt-in post-run safety review (story 12.gov-3.8).

Design §9.2: an async, **opt-in** judge that scores a finished trace
on hallucination / goal-alignment / tool-misuse / danger and returns
a recommendation. Off by default; runs *after* the user-facing
response has shipped so it never blocks the in-flight loop.

The judge is intentionally decoupled from the kernel: it isn't a hook
(no kernel firing point), it's an orchestration helper that callers
(``AgenticCore`` or ``iris-code``) invoke once a run completes. The
verdict is written to the same audit log every kernel hook writes to
so ``iris run inspect`` can surface it next to evaluator signals.

Failure modes (timeout, parse error, judge LLM down) **must not**
halt the run — they write a ``severity=warn`` row to audit and
return ``None``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.evaluator.judge_template import (
    SYSTEM_PROMPT,
    render_trace_text,
    render_user_prompt,
)
from iris_harness.kernel.governance.hooks.types import LLMTier

logger = logging.getLogger(__name__)


JUDGE_PLUGIN_NAME: Final[str] = "llm_judge"
"""Identifier written into ``audit_log.plugin`` so the inspect CLI can find judge rows."""

JUDGE_CLOUD_OPT_IN_ENV: Final[str] = "IRIS_GOVERNANCE_JUDGE_CLOUD_OPT_IN"
"""Operator must set this env var to allow a tier_3 (cloud) judge."""

JudgeRecommendation = Literal["allow", "warn", "halt_next"]


class JudgeVerdict(BaseModel):
    """Structured judge output. Persisted as ``audit_log`` payload."""

    model_config = ConfigDict(frozen=True)

    hallucination: float = Field(..., ge=0.0, le=1.0)
    goal_alignment: float = Field(..., ge=0.0, le=1.0)
    tool_misuse: float = Field(..., ge=0.0, le=1.0)
    danger: float = Field(..., ge=0.0, le=1.0)
    recommend: JudgeRecommendation
    rationale: str = Field(..., min_length=1, max_length=2000)


@runtime_checkable
class JudgeClient(Protocol):
    """Minimal LLM interface the judge talks to. Tests pass a mock.

    The client is asked to complete a chat-style ``(system, user)``
    pair and return raw text. The judge parses it as JSON. Streaming,
    retries, and provider selection are concerns of the client impl
    — the judge stays a pure transformation.
    """

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_tier: LLMTier,
        timeout_s: float,
    ) -> str: ...


class LLMJudge:
    """Opt-in async safety judge.

    Construct one per process. ``judge_trace`` is the only entry
    point; calls may be skipped entirely by passing ``enabled=False``
    at construction. Cloud (tier_3) judging requires both
    ``model_tier='tier_3'`` AND the ``IRIS_GOVERNANCE_JUDGE_CLOUD_OPT_IN``
    env var being truthy — otherwise the judge falls back to tier_2.
    """

    def __init__(
        self,
        *,
        client: JudgeClient,
        audit_log: AuditLog | None = None,
        enabled: bool = False,
        model_tier: LLMTier = "tier_2",
        timeout_s: float = 30.0,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        self._client = client
        self._audit_log = audit_log
        self._enabled = enabled
        self._model_tier = _resolve_model_tier(model_tier)
        self._timeout_s = timeout_s

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def model_tier(self) -> LLMTier:
        return self._model_tier

    async def judge_trace(
        self,
        *,
        run_id: str,
        agent_type: str,
        original_task: str,
        steps: list[dict[str, str | None]],
    ) -> JudgeVerdict | None:
        """Run the judge against a completed trace.

        Returns ``None`` when:
        - the judge is disabled,
        - the client raised any exception,
        - the client returned text that didn't parse as a valid
          ``JudgeVerdict``.

        In every "failure" branch a ``severity=warn`` row is written
        to the audit log (when one is configured) so the operator
        has visibility without the judge ever blocking the run.
        """
        if not self._enabled:
            return None

        trace_text = render_trace_text(steps)
        user_prompt = render_user_prompt(original_task=original_task, trace_text=trace_text)

        try:
            raw = await self._client.complete(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model_tier=self._model_tier,
                timeout_s=self._timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - judge must never halt the run
            logger.warning("llm_judge: client raised (%s); writing warn audit row", exc)
            self._audit_warn(
                run_id=run_id,
                agent_type=agent_type,
                reason=f"llm_judge: client error ({exc.__class__.__name__})",
                payload={"error": str(exc)},
            )
            return None

        try:
            verdict = _parse_verdict(raw)
        except (ValueError, ValidationError) as exc:
            logger.warning(
                "llm_judge: parse failure on judge output (%s); writing warn audit row", exc
            )
            self._audit_warn(
                run_id=run_id,
                agent_type=agent_type,
                reason=f"llm_judge: parse failure ({exc.__class__.__name__})",
                payload={"error": str(exc), "raw_excerpt": raw[:500]},
            )
            return None

        self._audit_verdict(run_id=run_id, agent_type=agent_type, verdict=verdict)
        return verdict

    def _audit_verdict(self, *, run_id: str, agent_type: str, verdict: JudgeVerdict) -> None:
        if self._audit_log is None:
            return
        severity = "warn" if verdict.recommend != "allow" else "info"
        try:
            self._audit_log.record(
                run_id=run_id,
                step_id=None,
                agent_type=agent_type,
                hook_point="post_run",
                plugin=JUDGE_PLUGIN_NAME,
                decision="allow",  # judge is informational, not a gate
                severity=severity,
                reason=f"llm_judge: recommend={verdict.recommend}",
                tier=self._model_tier,
                payload=verdict.model_dump(mode="json"),
            )
        except Exception as exc:  # noqa: BLE001 - audit must not crash
            logger.warning("llm_judge: audit write failed (%s)", exc)

    def _audit_warn(
        self,
        *,
        run_id: str,
        agent_type: str,
        reason: str,
        payload: dict[str, Any],
    ) -> None:
        if self._audit_log is None:
            return
        try:
            self._audit_log.record(
                run_id=run_id,
                step_id=None,
                agent_type=agent_type,
                hook_point="post_run",
                plugin=JUDGE_PLUGIN_NAME,
                decision="allow",
                severity="warn",
                reason=reason,
                tier=self._model_tier,
                payload=payload,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm_judge: audit write failed (%s)", exc)


def _resolve_model_tier(requested: LLMTier) -> LLMTier:
    """Enforce the cloud-opt-in rule (AC-3).

    A caller asking for ``tier_3`` without the env opt-in is silently
    downgraded to ``tier_2`` with a warning logged. The cap is
    intentional: a researcher should have to *think* before sending
    full traces to a cloud LLM, because trace content commonly
    contains personal/internal data the rest of the governance layer
    holds back from the cloud.
    """
    if requested != "tier_3":
        return requested
    raw = os.getenv(JUDGE_CLOUD_OPT_IN_ENV, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return "tier_3"
    logger.warning(
        "llm_judge: tier_3 requested but %s is not set; downgrading to tier_2",
        JUDGE_CLOUD_OPT_IN_ENV,
    )
    return "tier_2"


def _parse_verdict(raw: str) -> JudgeVerdict:
    """Parse the judge's raw output as a JudgeVerdict.

    The judge is instructed to return a single JSON object. We accept
    leading/trailing whitespace and also tolerate a fenced ```json
    code block since some models like to wrap output even when told
    not to. Anything else raises ``ValueError``.
    """
    candidate = raw.strip()
    if candidate.startswith("```"):
        # Strip surrounding fence: ```json ... ``` or ``` ... ```.
        candidate = candidate.lstrip("`")
        # After lstrip the leading 'json' (if present) and the
        # newline need to go.
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.lstrip("\n").rstrip("`").strip()

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge output is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("judge output must be a JSON object")
    return JudgeVerdict.model_validate(data)


def judge_from_env(*, client: JudgeClient, audit_log: AuditLog | None) -> LLMJudge:
    """Build an LLMJudge using env-var opt-in (used by wiring + CLI flags).

    Reads:
    - ``IRIS_GOVERNANCE_JUDGE_ENABLED``     truthy → enable
    - ``IRIS_GOVERNANCE_JUDGE_MODEL_TIER``  one of tier_1|tier_2|tier_3
    - ``IRIS_GOVERNANCE_JUDGE_TIMEOUT_S``   float seconds

    Cloud tier still requires ``IRIS_GOVERNANCE_JUDGE_CLOUD_OPT_IN``.
    """
    enabled_raw = os.getenv("IRIS_GOVERNANCE_JUDGE_ENABLED", "0").strip().lower()
    enabled = enabled_raw in {"1", "true", "yes", "on"}

    tier_raw = os.getenv("IRIS_GOVERNANCE_JUDGE_MODEL_TIER", "tier_2").strip().lower()
    tier: LLMTier
    if tier_raw in {"tier_1", "tier_2", "tier_3"}:
        tier = tier_raw  # type: ignore[assignment]
    else:
        logger.warning(
            "llm_judge: invalid IRIS_GOVERNANCE_JUDGE_MODEL_TIER=%r; using tier_2", tier_raw
        )
        tier = "tier_2"

    timeout_raw = os.getenv("IRIS_GOVERNANCE_JUDGE_TIMEOUT_S", "30.0").strip()
    try:
        timeout_s = float(timeout_raw)
    except ValueError:
        timeout_s = 30.0
    if timeout_s <= 0:
        timeout_s = 30.0

    return LLMJudge(
        client=client,
        audit_log=audit_log,
        enabled=enabled,
        model_tier=tier,
        timeout_s=timeout_s,
    )
