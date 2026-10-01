"""Merge, validate, and format multi-agent results for delivery."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

from iris_harness.agent.agent_executor import AgentResult
from iris_harness.agent.escalation import (
    EscalationConfig,
    EscalationJudgeClient,
    EscalationVerdict,
    parse_escalation_verdict,
)
from iris_harness.foundation.observability.session_log import bind_context
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.hooks.response_payload import (
    current_audience,
    pre_response_payload,
)
from iris_harness.kernel.governance.hooks.types import HookContext, HookPoint
from iris_harness.kernel.governance.plugins.response_safety import check_response

if TYPE_CHECKING:
    from iris_harness.kernel.governance.kernel import GovernanceKernel

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# The model-free response check is regex and literal matching; a fire that takes this
# long has failed, and the curator falls back to running the check directly.
_KERNEL_FIRE_TIMEOUT_S = 5.0

# A token shaped like a URL / domain / email — never a credential or canary, so it's
# excluded from the identity-egress secret-literal set (issue 0022).
# A local model sometimes emits its raw ReAct scaffolding (a ``{"thought": ...}``
# JSON blob, or a ``Thought:/Action:`` plaintext step) instead of a final answer;
# without a clean answer that scratchpad can leak to the user. These guards
# recover the embedded answer when present, else suppress the leak.
_REACT_JSON_RE = re.compile(r'^\s*\{\s*"(?:thought|action|action_input)"\s*:', re.IGNORECASE)
_REACT_TEXT_RE = re.compile(r"^\s*thought\s*:", re.IGNORECASE)
_REACT_STEP_RE = re.compile(r"\b(?:action|action[ _]input|observation)\s*:", re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(r"\bfinal[ _]answer\s*:", re.IGNORECASE)
_REACT_LEAK_FALLBACK = (
    "Sorry — I couldn't put that together cleanly. Could you rephrase or ask again?"
)
# What is left when a reply was nothing but "run <tool>" instructions.
_NO_ANSWER_AFTER_STRIP = (
    "I couldn't find that in what I have access to. Ask me to search your inbox or "
    "your stored data and I'll look again."
)


# ── Internal tool names must not reach the user ───────────────────────────────
#
# Cloud trial, 2026-09-19: "you can run the `finance_lookup` tool with the query
# 'electricity bill'". `finance_lookup` is an internal tool the MODEL calls; the user
# has no such command, so the instruction is both useless and confusing. The ReAct-leak
# guard above only catches a whole scratchpad, not a sentence like this one.
#
# Narrow by design: only a sentence that tells the reader to RUN something, naming a
# snake_case or backticked identifier. Prose that merely mentions a word is untouched.
_TOOL_INSTRUCTION_RE = re.compile(
    r"[^.!?\n]*\b(?:run|use|call|execute|invoke|try)\b[^.!?\n]*"
    r"(?:`[a-z][a-z0-9_]*`|\b[a-z][a-z0-9]*_[a-z0-9_]+\b)"
    r"[^.!?\n]*(?:[.!?]|$)",
    re.IGNORECASE,
)
_TOOL_WORD_RE = re.compile(r"\btool\b|\bfunction\b|`", re.IGNORECASE)


def _strip_tool_instructions(text: str) -> tuple[str, list[str]]:
    """Remove sentences telling the user to run an internal tool.

    Returns the cleaned text and what was removed (for the judge signal). Only
    sentences that both instruct an action and name a tool-shaped identifier go.
    """
    removed: list[str] = []

    def _drop(match: re.Match[str]) -> str:
        sentence = match.group(0)
        if not _TOOL_WORD_RE.search(sentence):
            return sentence
        removed.append(sentence.strip())
        return ""

    cleaned = _TOOL_INSTRUCTION_RE.sub(_drop, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, removed


def _recover_react_answer(text: str) -> str | None:
    """If ``text`` is a raw ReAct JSON blob whose action is a final answer, return
    that embedded answer; otherwise None."""
    s = text.strip()
    if not _REACT_JSON_RE.match(s):
        return None
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    action = str(obj.get("action", "")).strip().lower().replace("_", " ")
    if action in {"final answer", "finalanswer", "final", "respond", "answer"}:
        for key in ("action_input", "input", "answer", "final_answer"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _looks_like_react_scaffolding(text: str) -> bool:
    """True when ``text`` is recognizably a raw ReAct step (no real answer)."""
    s = text.strip()
    if not s:
        return False
    if _REACT_JSON_RE.match(s):
        return True
    # plaintext "Thought: …" followed by an Action/Observation step, with no
    # surfaced "Final Answer:" — i.e. the scratchpad, not a reply.
    if _REACT_TEXT_RE.match(s) and _REACT_STEP_RE.search(s) and not _FINAL_ANSWER_RE.search(s):
        return True
    return False


@dataclass
class CuratedResponse:
    """Final formatted response ready for delivery to the user."""

    text: str
    sources: list[str] = field(default_factory=list)
    has_errors: bool = False
    error_summary: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def governance_warning_banner(warnings: tuple[str, ...]) -> str:
    """The banner a shipped answer carries when a check was not fully satisfied.

    One wording for a generated answer (``curate``) and a deterministic one (the
    ``guard`` stage), e.g. when the output guard timed out and failed open.
    """
    return "[Governance warning] Some response checks were not fully satisfied: " + "; ".join(
        warnings
    )


# The one refusal voice (deterministic-path parity, decision C): a response the guards
# halt and a user turn the PRE_TURN screen denies both answer with this text.
GOVERNANCE_BLOCKED_TEXT = (
    "I can’t provide that response because governance policy flagged it as unsafe. "
    "Please rephrase your request."
)


@runtime_checkable
class FaithfulnessJudgeClient(Protocol):
    """Optional async LLM judge for response faithfulness."""

    async def judge(self, *, query: str, response: str) -> str:
        """Return a JSON-ish verdict payload describing faithfulness."""


@runtime_checkable
class LeakJudgeClient(Protocol):
    """Optional async LLM judge for system-prompt / identity-leak intent.

    Consulted ONLY when the deterministic dump-phrase guard flags a response, to
    decide whether it is a genuine self-description (allow) or an actual dump of the
    system prompt / identity docs (block). Lets the curator distinguish *intent*,
    which content matching cannot — fixing benign over-blocks without weakening the
    hard credential/secret-literal guards, which never consult the judge.
    """

    async def judge(self, *, response: str) -> str:
        """Return a JSON-ish verdict: {is_leak: bool, confidence: float, reason: str}."""


@dataclass(frozen=True)
class OutputSafetyVerdict:
    """Result of the output-safety guard (Phase 6 G3, Llama Guard 3).

    ``categories`` are hazard names (e.g. ``violent_crimes``, ``privacy``). The
    curator maps them against its ``enforce`` / ``log_only`` sets to decide halt
    vs. warn — see ``ResponseCurator._judge_output_safety``.
    """

    unsafe: bool
    categories: tuple[str, ...] = ()


@runtime_checkable
class OutputSafetyJudgeClient(Protocol):
    """Optional async guard for unsafe model output (Phase 6 G3).

    The deterministic ``_SAFETY_PATTERNS`` regex remains the cheap fast-path;
    this model-driven guard is the primary signal. ``judge`` must raise on
    backend failure so the curator can fail-open-with-banner (warn, not halt)
    rather than blocking every response when the guard is unavailable.
    """

    async def judge(self, *, response: str) -> OutputSafetyVerdict:
        """Return whether the response is unsafe and which hazards matched."""


@runtime_checkable
class GroundingJudgeClient(Protocol):
    """Optional async judge for response grounding (Phase 5).

    Checks whether the response's factual claims are supported by the retrieval
    context the agent actually used. Only consulted when retrieved context is
    present (a non-RAG answer is never penalized). Returns a JSON-ish verdict:
    ``{grounded: bool, confidence: float, unsupported: str}``.
    """

    async def judge(self, *, query: str, response: str, retrieved_context: str) -> str:
        """Return a JSON verdict describing whether the response is grounded."""


SignalVerdict = Literal["pass", "warn", "retry", "halt", "skipped"]


@dataclass(frozen=True)
class JudgeSignal:
    """One pre-response judge signal result."""

    name: str
    verdict: SignalVerdict
    reason: str
    retryable: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeBundle:
    """Aggregated pre-response signal state."""

    signals: tuple[JudgeSignal, ...]
    retry_budget: int
    retries_used: int
    strict: bool

    @property
    def halted(self) -> bool:
        return any(signal.verdict == "halt" for signal in self.signals)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            f"{signal.name}: {signal.reason}"
            for signal in self.signals
            if signal.verdict in {"warn", "retry"}
        )


class ResponseCurator:
    """Merge and format results from one or more agent executions."""

    _NAME_FACT_RE = re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z\-']{1,40})\b", re.IGNORECASE)
    _NAME_REPLY_RE = re.compile(r"\byour name is\s+([A-Za-z][A-Za-z\-']{1,40})\b", re.IGNORECASE)
    _TZ_FACT_RE = re.compile(r"\bmy timezone is\s+([A-Za-z0-9_./+-]{2,64})\b", re.IGNORECASE)
    _TZ_REPLY_RE = re.compile(r"\byour timezone is\s+([A-Za-z0-9_./+-]{2,64})\b", re.IGNORECASE)

    def __init__(
        self,
        *,
        strict_default: bool = False,
        retry_budget: int = 2,
        audit_log: AuditLog | None = None,
        faithfulness_judge: FaithfulnessJudgeClient | None = None,
        faithfulness_timeout_s: float = 8.0,
        leak_judge: LeakJudgeClient | None = None,
        leak_judge_timeout_s: float = 8.0,
        output_safety_judge: OutputSafetyJudgeClient | None = None,
        output_safety_timeout_s: float = 8.0,
        output_safety_enforce: frozenset[str] = frozenset(),
        output_safety_log_only: frozenset[str] = frozenset(),
        grounding_judge: GroundingJudgeClient | None = None,
        grounding_timeout_s: float = 8.0,
        escalation_judge: EscalationJudgeClient | None = None,
        escalation_config: EscalationConfig | None = None,
        escalation_timeout_s: float = 8.0,
        kernel: GovernanceKernel | None = None,
    ) -> None:
        self._strict_default = strict_default
        # The kernel the model-free response check fires through (PRE_RESPONSE).
        self._kernel = kernel
        self._retry_budget = max(0, retry_budget)
        self._audit_log = audit_log
        self._faithfulness_judge = faithfulness_judge
        self._faithfulness_timeout_s = max(0.1, faithfulness_timeout_s)
        self._leak_judge = leak_judge
        self._leak_judge_timeout_s = max(0.1, leak_judge_timeout_s)
        self._output_safety_judge = output_safety_judge
        self._output_safety_timeout_s = max(0.1, output_safety_timeout_s)
        self._output_safety_enforce = output_safety_enforce
        self._output_safety_log_only = output_safety_log_only
        self._grounding_judge = grounding_judge
        self._grounding_timeout_s = max(0.1, grounding_timeout_s)
        self._escalation_judge = escalation_judge
        self._escalation_config = escalation_config or EscalationConfig()
        self._escalation_timeout_s = max(0.1, escalation_timeout_s)

    @property
    def escalation_config(self) -> EscalationConfig:
        """The escalation policy (read by the runtime to decide whether to act)."""
        return self._escalation_config

    def curate(
        self,
        results: list[AgentResult],
        *,
        query: str = "",
        strict: bool | None = None,
        session_id: str = "default",
        conversation_history: tuple[str, ...] = (),
        intent: str = "",
    ) -> CuratedResponse:
        if not results:
            return CuratedResponse(
                text="No results were produced.",
                has_errors=True,
                error_summary="empty result set",
            )

        failed = [r for r in results if not r.success]
        succeeded = [r for r in results if r.success]

        if not succeeded:
            errors = "; ".join(r.error or "unknown error" for r in failed)
            return CuratedResponse(
                text="I encountered an error while processing your request.",
                has_errors=True,
                error_summary=errors,
            )

        if len(succeeded) == 1:
            result = succeeded[0]
            text = result.output.strip()
        else:
            parts = []
            for r in succeeded:
                parts.append(f"[{r.agent_type}]\n{r.output.strip()}")
            text = "\n\n".join(parts)

        error_summary: str | None = None
        if failed:
            error_summary = f"{len(failed)} sub-task(s) failed: " + "; ".join(
                r.error or "unknown" for r in failed
            )

        prompt_tokens = sum(int(str(r.metadata.get("prompt_tokens") or 0)) for r in results)
        completion_tokens = sum(int(str(r.metadata.get("completion_tokens") or 0)) for r in results)

        metadata: dict[str, object] = {
            "total_tasks": len(results),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "total_latency_ms": sum(r.latency_ms for r in results),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        token_keys = {"prompt_tokens", "completion_tokens", "total_tokens"}
        for result in succeeded:
            for key, value in result.metadata.items():
                if key in token_keys or key in metadata:
                    continue
                metadata[key] = value

        # Guard: a local model can surface its raw ReAct scaffolding instead of a
        # reply. Recover the embedded final answer when present, else suppress the
        # leak — so the scratchpad never reaches the user (and judges run clean).
        recovered = _recover_react_answer(text)
        if recovered is not None:
            metadata["react_answer_recovered"] = True
            text = recovered
        elif _looks_like_react_scaffolding(text):
            metadata["react_leak_suppressed"] = True
            text = _REACT_LEAK_FALLBACK

        # A reply may still tell the user to run one of the model's own tools.
        cleaned, removed_instructions = _strip_tool_instructions(text)
        if removed_instructions:
            metadata["tool_instruction_stripped"] = removed_instructions
            text = cleaned or _NO_ANSWER_AFTER_STRIP

        strict_mode = self._strict_default if strict is None else strict
        judged_text, judge_bundle = self._run_pre_response_judges(
            text=text,
            query=query,
            strict=strict_mode,
            metadata=metadata,
            conversation_history=conversation_history,
            session_id=session_id,
            intent=intent,
        )
        metadata["judge_bundle"] = {
            "strict": judge_bundle.strict,
            "retry_budget": judge_bundle.retry_budget,
            "retries_used": judge_bundle.retries_used,
            "halted": judge_bundle.halted,
            "signals": [
                {
                    "name": signal.name,
                    "verdict": signal.verdict,
                    "reason": signal.reason,
                    "retryable": signal.retryable,
                    "metadata": signal.metadata,
                }
                for signal in judge_bundle.signals
            ],
        }
        # First-class flag when the output-safety guard failed open (timeout /
        # backend error) rather than actually clearing the response — so health
        # + audit can COUNT degraded turns instead of parsing the prose banner
        # (red-team 2a follow-up). A degraded guard is a fail-open: the response
        # shipped without a real safety verdict.
        degraded = next(
            (
                s
                for s in judge_bundle.signals
                if s.name == "output_safety" and s.metadata.get("degraded")
            ),
            None,
        )
        if degraded is not None:
            metadata["output_safety_degraded"] = True
            metadata["output_safety_degraded_reason"] = degraded.metadata.get(
                "degraded_reason", "unknown"
            )

        if judge_bundle.halted:
            halt_reason = next(
                (signal.reason for signal in judge_bundle.signals if signal.verdict == "halt"),
                "response blocked by safety policy",
            )
            return CuratedResponse(
                text=GOVERNANCE_BLOCKED_TEXT,
                sources=[r.agent_type for r in succeeded],
                has_errors=True,
                error_summary=halt_reason,
                metadata=metadata,
            )

        if judge_bundle.warnings:
            banner = governance_warning_banner(judge_bundle.warnings)
            judged_text = f"{banner}\n\n{judged_text}" if judged_text else banner
            metadata["warning_banner"] = banner

        return CuratedResponse(
            text=judged_text,
            sources=[r.agent_type for r in succeeded],
            has_errors=bool(failed),
            error_summary=error_summary,
            metadata=metadata,
        )

    def _run_pre_response_judges(
        self,
        *,
        text: str,
        query: str,
        strict: bool,
        metadata: dict[str, object],
        conversation_history: tuple[str, ...],
        session_id: str,
        intent: str = "",
    ) -> tuple[str, JudgeBundle]:
        signals: list[JudgeSignal] = []
        retries_used = 0
        remaining = self._retry_budget
        current_text = text

        safety = self._judge_safety(current_text, session_id=session_id)
        signals.append(safety)
        self._audit_signal(session_id=session_id, signal=safety)
        if safety.verdict == "halt":
            return current_text, JudgeBundle(
                signals=tuple(signals),
                retry_budget=self._retry_budget,
                retries_used=retries_used,
                strict=strict,
            )

        schema_signal, repaired = self._judge_schema(current_text, metadata)
        if schema_signal.verdict == "retry" and remaining > 0:
            retries_used += 1
            remaining -= 1
            current_text = repaired
            schema_signal, _ = self._judge_schema(current_text, metadata)
            if schema_signal.verdict == "retry":
                schema_signal = JudgeSignal(
                    name="schema",
                    verdict="warn",
                    reason="schema validation still failing after retry",
                    retryable=True,
                    metadata=schema_signal.metadata,
                )
        elif schema_signal.verdict == "retry" and remaining <= 0:
            schema_signal = JudgeSignal(
                name="schema",
                verdict="warn",
                reason="schema validation failed and retry budget is exhausted",
                retryable=True,
                metadata=schema_signal.metadata,
            )
        signals.append(schema_signal)
        self._audit_signal(session_id=session_id, signal=schema_signal)

        consistency = self._judge_consistency(current_text, conversation_history)
        signals.append(consistency)
        self._audit_signal(session_id=session_id, signal=consistency)

        faithfulness = self._judge_faithfulness(current_text, query=query, strict=strict)
        if faithfulness.verdict == "retry" and remaining > 0:
            retries_used += 1
            remaining -= 1
            current_text = f'In response to your question "{query}":\n{current_text}'.strip()
            faithfulness = self._judge_faithfulness(current_text, query=query, strict=strict)
            if faithfulness.verdict == "retry":
                faithfulness = JudgeSignal(
                    name="faithfulness",
                    verdict="warn",
                    reason="faithfulness check still failing after retry",
                    retryable=True,
                    metadata=faithfulness.metadata,
                )
        elif faithfulness.verdict == "retry" and remaining <= 0:
            faithfulness = JudgeSignal(
                name="faithfulness",
                verdict="warn",
                reason="faithfulness check failed and retry budget is exhausted",
                retryable=True,
                metadata=faithfulness.metadata,
            )
        signals.append(faithfulness)
        self._audit_signal(session_id=session_id, signal=faithfulness)

        grounding = self._judge_grounding(
            current_text, query=query, metadata=metadata, strict=strict
        )
        signals.append(grounding)
        self._audit_signal(session_id=session_id, signal=grounding)

        output_safety = self._judge_output_safety(current_text)
        signals.append(output_safety)
        self._audit_signal(session_id=session_id, signal=output_safety)

        # Tier-escalation judge (ADR-0068). SHADOW-ONLY in L2: it diagnoses and
        # records, but its signal is always non-acting ("skipped") so it can
        # never halt/retry/banner the response. Runs last — after the response
        # is otherwise cleared — and only when the response did not halt.
        escalation = self._judge_escalation(
            current_text, query=query, intent=intent, metadata=metadata
        )
        signals.append(escalation)
        self._audit_signal(session_id=session_id, signal=escalation)

        # The judges may have rewritten the text (a schema repair, the faithfulness
        # prefix). The guard must pass the text that actually ships, so it runs again
        # on the final text whenever that differs from what it first checked.
        if current_text != text:
            final_safety = self._judge_safety(current_text, session_id=session_id)
            signals.append(final_safety)
            self._audit_signal(session_id=session_id, signal=final_safety)

        return current_text, JudgeBundle(
            signals=tuple(signals),
            retry_budget=self._retry_budget,
            retries_used=retries_used,
            strict=strict,
        )

    def guard(
        self, text: str, *, session_id: str = "default", handler: str | None = None
    ) -> JudgeSignal:
        """The model-free response check for an answer no model wrote.

        A deterministic handler's answer never reaches ``curate``, so the turn pipeline's
        ``guard`` stage calls this instead: the same kernel-fired check generated answers
        pass inside ``curate``, marked ``deterministic`` with the handler's name in the
        audit row (deterministic-path parity, step c). Dump phrasing has no reason to
        appear in a templated answer, and with no leak judge configured it fails closed.
        """
        signal = self._judge_safety(
            text,
            session_id=session_id,
            audit_payload={"deterministic": True, "handler": handler or "unknown"},
        )
        self._audit_signal(session_id=session_id, signal=signal)
        return signal

    def guard_output(
        self, text: str, *, session_id: str = "default", handler: str | None = None
    ) -> JudgeSignal:
        """The model-based output guard (Llama Guard) on a deterministic answer.

        Only for a handler that declared ``guard_output`` — its answer repeats text
        someone else wrote (deterministic-path parity, decision B). Same judge, same
        verdicts and the same fail-open-with-warning as a generated answer gets in
        ``curate``; ``skipped`` when the output guard is not enabled
        (``IRIS_CURATOR_OUTPUT_SAFETY``). Audited with the deterministic marker.
        """
        signal = self._judge_output_safety(text)
        signal = JudgeSignal(
            name=signal.name,
            verdict=signal.verdict,
            reason=signal.reason,
            retryable=signal.retryable,
            metadata={**signal.metadata, "deterministic": True, "handler": handler or "unknown"},
        )
        self._audit_signal(session_id=session_id, signal=signal)
        return signal

    def _judge_safety(
        self,
        text: str,
        *,
        session_id: str = "default",
        audit_payload: dict[str, object] | None = None,
    ) -> JudgeSignal:
        """The model-free response check, then the leak judge for dump phrasing.

        The check itself is ``kernel.governance.plugins.response_safety`` — one
        implementation every answer passes (deterministic-path parity, step b). It fires
        through the kernel's ``PRE_RESPONSE`` hook when a kernel is wired, so the kernel
        writes the audit row; with governance disabled it runs directly, so response
        safety never weakens with it. Dump phrasing is only flagged by the check: telling
        a dump from a self-description is the leak judge's call, failing closed.
        """
        verdict, pattern, reason, audited = self._check_response(
            text, session_id=session_id, audit_payload=audit_payload
        )
        if verdict == "halt":
            return JudgeSignal(
                name="safety",
                verdict="halt",
                reason=reason,
                metadata={"pattern": pattern, "kernel_audited": audited, **(audit_payload or {})},
            )
        if verdict == "flag_dump":
            return self._arbitrate_dump(text)
        return JudgeSignal(
            name="safety",
            verdict="pass",
            reason=reason,
            metadata={"kernel_audited": audited, **(audit_payload or {})},
        )

    def _check_response(
        self,
        text: str,
        *,
        session_id: str,
        audit_payload: dict[str, object] | None = None,
    ) -> tuple[str, str | None, str, bool]:
        """(verdict, pattern, reason, kernel_audited) for ``text``."""
        if self._kernel is not None:
            kernel = self._kernel
            ctx = HookContext(
                hook_point=HookPoint.PRE_RESPONSE,
                run_id=session_id or "default",
                agent_type="chat",
                # Who reads it is the turn's to say (a group chat's members, not the owner).
                payload=pre_response_payload(
                    text, audience=current_audience(), audit=audit_payload
                ),
            )
            try:
                decision, _ = _await_sync(
                    lambda: kernel.fire(HookPoint.PRE_RESPONSE, ctx),
                    timeout_s=_KERNEL_FIRE_TIMEOUT_S,
                )
            except Exception:
                logger.warning(
                    "PRE_RESPONSE fire failed; running the response check directly",
                    exc_info=True,
                )
            else:
                pattern = decision.audit_metadata.get("pattern")
                pattern = pattern if isinstance(pattern, str) else None
                if decision.outcome in ("deny", "require_approval"):
                    return "halt", pattern, decision.reason, True
                if decision.audit_metadata.get("dump_flag"):
                    return "flag_dump", pattern, decision.reason, True
                return "pass", None, decision.reason, True
        check = check_response(text)
        return check.verdict, check.pattern, check.reason, False

    def _arbitrate_dump(self, text: str) -> JudgeSignal:
        """Decide a dump-flagged response via the semantic leak-judge. FAIL-CLOSED.

        The deterministic guards above already cleared this text of credential/PII/
        secret-literal content; what's left is the "mention vs dump" intent question.
        With no judge (or any judge failure / unparseable verdict / "leak" verdict)
        we keep the safe deterministic halt — the judge can only *clear* a response it
        is confident is a legitimate self-description, never make things less safe.
        """
        halt = JudgeSignal(
            name="safety",
            verdict="halt",
            reason="detected unsafe content pattern: prompt_regurgitation",
            metadata={"pattern": "prompt_regurgitation"},
        )
        if self._leak_judge is None:
            return halt
        judge = self._leak_judge
        try:
            raw = _await_sync(
                lambda: judge.judge(response=text),
                timeout_s=self._leak_judge_timeout_s,
            )
        except Exception:  # noqa: BLE001 - any judge failure must fail closed (halt)
            return halt
        parsed = _parse_leak_verdict(raw)
        if parsed is None or parsed[0]:  # unparseable, or judge says is_leak=True
            return halt
        _is_leak, confidence, reason = parsed
        return JudgeSignal(
            name="safety",
            verdict="pass",
            reason=f"leak judge cleared flagged response: {reason}",
            metadata={"pattern": "prompt_regurgitation", "judge": "leak", "confidence": confidence},
        )

    def _judge_schema(
        self,
        text: str,
        metadata: dict[str, object],
    ) -> tuple[JudgeSignal, str]:
        expected = metadata.get("expected_schema")
        if expected is None:
            return (
                JudgeSignal(name="schema", verdict="pass", reason="no expected schema provided"),
                text,
            )

        candidate_text = _extract_json_candidate(text)
        try:
            parsed = json.loads(candidate_text)
        except json.JSONDecodeError:
            repaired = _try_json_repair(text)
            if repaired is not None:
                return (
                    JudgeSignal(
                        name="schema",
                        verdict="retry",
                        reason="response is not valid JSON for expected schema",
                        retryable=True,
                        metadata={"repair": "extract_json_block"},
                    ),
                    repaired,
                )
            return (
                JudgeSignal(
                    name="schema",
                    verdict="retry",
                    reason="response is not valid JSON for expected schema",
                    retryable=True,
                ),
                text,
            )

        model_cls = (
            expected if isinstance(expected, type) and issubclass(expected, BaseModel) else None
        )
        if model_cls is not None:
            try:
                model_cls.model_validate(parsed)
            except ValidationError as exc:
                return (
                    JudgeSignal(
                        name="schema",
                        verdict="retry",
                        reason="schema validation failed",
                        retryable=True,
                        metadata={"error": str(exc)},
                    ),
                    text,
                )
            return (
                JudgeSignal(name="schema", verdict="pass", reason="schema validation passed"),
                text,
            )

        if isinstance(expected, dict):
            required_raw = expected.get("required")
            if isinstance(required_raw, list) and required_raw:
                missing = [
                    item for item in required_raw if isinstance(item, str) and item not in parsed
                ]
                if missing:
                    return (
                        JudgeSignal(
                            name="schema",
                            verdict="retry",
                            reason="schema validation failed",
                            retryable=True,
                            metadata={"missing_keys": missing},
                        ),
                        text,
                    )
        return JudgeSignal(name="schema", verdict="pass", reason="schema validation passed"), text

    def _judge_consistency(self, text: str, history: tuple[str, ...]) -> JudgeSignal:
        known_name: str | None = None
        known_tz: str | None = None
        for entry in history:
            name_match = self._NAME_FACT_RE.search(entry)
            if name_match:
                known_name = name_match.group(1)
            tz_match = self._TZ_FACT_RE.search(entry)
            if tz_match:
                known_tz = tz_match.group(1)

        reply_name = self._NAME_REPLY_RE.search(text)
        if known_name and reply_name and reply_name.group(1).lower() != known_name.lower():
            return JudgeSignal(
                name="consistency",
                verdict="warn",
                reason="response may contradict known user name",
                metadata={"expected": known_name, "found": reply_name.group(1)},
            )

        reply_tz = self._TZ_REPLY_RE.search(text)
        if known_tz and reply_tz and reply_tz.group(1).lower() != known_tz.lower():
            return JudgeSignal(
                name="consistency",
                verdict="warn",
                reason="response may contradict known user timezone",
                metadata={"expected": known_tz, "found": reply_tz.group(1)},
            )

        return JudgeSignal(
            name="consistency", verdict="pass", reason="no contradiction pattern found"
        )

    def _judge_faithfulness(self, text: str, *, query: str, strict: bool) -> JudgeSignal:
        if not query.strip():
            return JudgeSignal(name="faithfulness", verdict="pass", reason="empty query")
        if not strict:
            return JudgeSignal(
                name="faithfulness",
                verdict="skipped",
                reason="non-strict mode defers heavy faithfulness checks",
            )
        if self._faithfulness_judge is not None:
            llm_signal = self._judge_faithfulness_llm(text=text, query=query)
            if llm_signal is not None:
                return llm_signal

        query_tokens = _keyword_tokens(query)
        if not query_tokens:
            return JudgeSignal(name="faithfulness", verdict="pass", reason="query too short")

        response_tokens = _keyword_tokens(text)
        overlap = len(query_tokens & response_tokens)
        ratio = overlap / max(1, len(query_tokens))
        if ratio < 0.2:
            return JudgeSignal(
                name="faithfulness",
                verdict="retry",
                reason="response may not address the user question",
                retryable=True,
                metadata={"overlap_ratio": round(ratio, 3)},
            )
        return JudgeSignal(
            name="faithfulness",
            verdict="pass",
            reason="response appears to address the user question",
            metadata={"overlap_ratio": round(ratio, 3), "judge_mode": "heuristic"},
        )

    def _judge_faithfulness_llm(self, *, text: str, query: str) -> JudgeSignal | None:
        if self._faithfulness_judge is None:
            return None
        judge = self._faithfulness_judge
        try:
            raw = _await_sync(
                lambda: judge.judge(query=query, response=text),
                timeout_s=self._faithfulness_timeout_s,
            )
        except TimeoutError:
            return JudgeSignal(
                name="faithfulness",
                verdict="retry",
                reason="faithfulness judge timed out",
                retryable=True,
                metadata={"judge_mode": "llm"},
            )
        except Exception as exc:  # noqa: BLE001 - fall back to heuristic on failures
            return JudgeSignal(
                name="faithfulness",
                verdict="retry",
                reason="faithfulness judge failed",
                retryable=True,
                metadata={"judge_mode": "llm", "error": str(exc)},
            )

        parsed = _parse_faithfulness_verdict(raw)
        if parsed is None:
            return JudgeSignal(
                name="faithfulness",
                verdict="retry",
                reason="faithfulness judge returned invalid verdict",
                retryable=True,
                metadata={"judge_mode": "llm"},
            )

        passes, confidence, rationale = parsed
        verdict: SignalVerdict = "pass" if passes else "retry"
        return JudgeSignal(
            name="faithfulness",
            verdict=verdict,
            reason=rationale,
            retryable=not passes,
            metadata={
                "judge_mode": "llm",
                "confidence": confidence,
            },
        )

    def _judge_grounding(
        self,
        text: str,
        *,
        query: str,
        metadata: dict[str, object],
        strict: bool,
    ) -> JudgeSignal:
        """Phase 5: check the response is supported by the retrieved context.

        No retrieved context → ``skipped`` (a non-RAG answer is never penalized
        for not citing sources it never had — the key false-positive guard). With
        a judge configured (opt-in), the LLM judge runs; otherwise the cheap
        token-overlap heuristic runs only in ``strict`` mode. Unsupported claims
        → ``retry``; never ``halt`` (grounding is a quality signal, not safety).
        """
        context = str(metadata.get("retrieved_context") or "")
        if not context.strip():
            return JudgeSignal(name="grounding", verdict="skipped", reason="no retrieved context")

        if self._grounding_judge is not None:
            llm_signal = self._judge_grounding_llm(text=text, query=query, context=context)
            if llm_signal is not None:
                return llm_signal

        if not strict:
            return JudgeSignal(
                name="grounding",
                verdict="skipped",
                reason="non-strict mode defers heavy grounding checks",
            )

        context_tokens = _keyword_tokens(context)
        response_tokens = _keyword_tokens(text)
        if not response_tokens:
            return JudgeSignal(name="grounding", verdict="pass", reason="empty response")
        overlap = len(response_tokens & context_tokens)
        ratio = overlap / max(1, len(response_tokens))
        if ratio < 0.2:
            return JudgeSignal(
                name="grounding",
                verdict="retry",
                reason="response may not be grounded in the retrieved context",
                retryable=True,
                metadata={"overlap_ratio": round(ratio, 3), "judge_mode": "heuristic"},
            )
        return JudgeSignal(
            name="grounding",
            verdict="pass",
            reason="response appears grounded in the retrieved context",
            metadata={"overlap_ratio": round(ratio, 3), "judge_mode": "heuristic"},
        )

    def _judge_grounding_llm(self, *, text: str, query: str, context: str) -> JudgeSignal | None:
        if self._grounding_judge is None:
            return None
        judge = self._grounding_judge
        try:
            raw = _await_sync(
                lambda: judge.judge(query=query, response=text, retrieved_context=context),
                timeout_s=self._grounding_timeout_s,
            )
        except TimeoutError:
            return JudgeSignal(
                name="grounding",
                verdict="warn",
                reason="grounding judge timed out",
                metadata={"judge_mode": "llm"},
            )
        except Exception as exc:  # noqa: BLE001 - degrade to a warn, never block
            return JudgeSignal(
                name="grounding",
                verdict="warn",
                reason="grounding judge failed",
                metadata={"judge_mode": "llm", "error": str(exc)},
            )

        parsed = _parse_grounding_verdict(raw)
        if parsed is None:
            return JudgeSignal(
                name="grounding",
                verdict="warn",
                reason="grounding judge returned invalid verdict",
                metadata={"judge_mode": "llm"},
            )

        grounded, confidence, unsupported = parsed
        if grounded:
            return JudgeSignal(
                name="grounding",
                verdict="pass",
                reason="response grounded in retrieved context",
                metadata={"judge_mode": "llm", "confidence": confidence},
            )
        return JudgeSignal(
            name="grounding",
            verdict="retry",
            reason=unsupported or "response contains claims unsupported by retrieved context",
            retryable=True,
            metadata={"judge_mode": "llm", "confidence": confidence},
        )

    def _judge_escalation(
        self,
        text: str,
        *,
        query: str,
        intent: str,
        metadata: dict[str, object],
    ) -> JudgeSignal:
        """Tier-escalation diagnosis (ADR-0068), SHADOW-ONLY in L2.

        The returned signal is ALWAYS ``skipped`` so it is purely informational —
        it never halts, retries, or banners the response. The would-be decision
        rides in ``metadata['escalation']`` for logging + learning-signal capture.
        Acting on the verdict is L3 work, deliberately not wired here.
        """
        cfg = self._escalation_config
        if self._escalation_judge is None or not cfg.enabled:
            return JudgeSignal(
                name="escalation", verdict="skipped", reason="escalation judge not configured"
            )
        if not text.strip() or not query.strip():
            return JudgeSignal(
                name="escalation", verdict="skipped", reason="no query/response to judge"
            )
        sampled_out = cfg.sample_rate < 1.0 and random.random() >= cfg.sample_rate  # noqa: S311
        if sampled_out:
            return JudgeSignal(name="escalation", verdict="skipped", reason="not sampled this turn")

        context = str(metadata.get("retrieved_context") or "")
        verdict = self._judge_escalation_llm(text=text, query=query, intent=intent, context=context)
        if verdict is None:
            return JudgeSignal(
                name="escalation",
                verdict="skipped",
                reason="escalation judge inconclusive",
                metadata={"judge_mode": "llm", "shadow": True},
            )
        # Shadow: record the diagnosis + would-be route, but do not act.
        return JudgeSignal(
            name="escalation",
            verdict="skipped",
            reason=(
                f"shadow: would {verdict.action} ({verdict.diagnosis}, "
                f"conf={verdict.confidence:.2f})"
            ),
            metadata={
                "judge_mode": "llm",
                "shadow": True,
                "mode": cfg.mode,
                "escalation": verdict.to_metadata(),
            },
        )

    def _judge_escalation_llm(
        self, *, text: str, query: str, intent: str, context: str
    ) -> EscalationVerdict | None:
        judge = self._escalation_judge
        if judge is None:
            return None
        try:
            raw = _await_sync(
                lambda: judge.judge(query=query, response=text, intent=intent, context=context),
                timeout_s=self._escalation_timeout_s,
            )
        except TimeoutError:
            logger.debug("escalation judge timed out")
            return None
        except Exception:  # shadow telemetry must never break curation
            logger.debug("escalation judge failed", exc_info=True)
            return None
        return parse_escalation_verdict(raw)

    def warm_output_safety(self) -> bool:
        """Pre-load the output-safety guard model so its first real call doesn't
        pay Ollama's cold-load cost and blow the per-judge timeout.

        The guard fails OPEN on timeout by design (a slow guard must not block
        every response), which made the fail-open reachable while llama-guard was
        cold — the first calls exceeded the budget and shipped unchecked
        (2026-07-06 red-team, finding 2a). Warming closes that window: warm calls
        return in ~150ms. Best-effort; returns True if the guard responded.
        Given a generous timeout because this IS the cold call.
        """
        if self._output_safety_judge is None:
            return False
        judge = self._output_safety_judge
        try:
            _await_sync(
                lambda: judge.judge(response="ok"),
                timeout_s=max(self._output_safety_timeout_s, 30.0),
            )
            return True
        except Exception:  # warm-up is a pure optimization
            logger.debug("output-safety guard warm-up skipped", exc_info=True)
            return False

    def _judge_output_safety(self, text: str) -> JudgeSignal:
        """Phase 6 G3: model-driven output-safety guard (Llama Guard 3).

        ``enforce`` categories halt the response; ``log_only`` categories warn
        (ship + audit). An unsafe verdict with no mappable category fails safe
        (halt). A backend failure/timeout fails open with a warning banner — a
        slow or missing guard must not block every response (plan §5).
        """
        if self._output_safety_judge is None:
            return JudgeSignal(
                name="output_safety",
                verdict="skipped",
                reason="no output-safety guard configured",
            )
        if not text.strip():
            return JudgeSignal(name="output_safety", verdict="pass", reason="empty response")

        judge = self._output_safety_judge
        try:
            verdict = _await_sync(
                lambda: judge.judge(response=text),
                timeout_s=self._output_safety_timeout_s,
            )
        except TimeoutError:
            # Fail-open: the response ships without a real safety verdict. Log at
            # WARNING (not just the user banner) + tag degraded so it's countable.
            logger.warning(
                "output-safety guard FAILED OPEN (timeout after %.1fs) — response unchecked",
                self._output_safety_timeout_s,
            )
            return JudgeSignal(
                name="output_safety",
                verdict="warn",
                reason="output-safety guard timed out",
                metadata={"judge_mode": "llm", "degraded": True, "degraded_reason": "timeout"},
            )
        except Exception as exc:  # noqa: BLE001 - fail-open-with-banner on guard failure
            logger.warning("output-safety guard FAILED OPEN (error: %s) — response unchecked", exc)
            return JudgeSignal(
                name="output_safety",
                verdict="warn",
                reason="output-safety guard failed",
                metadata={
                    "judge_mode": "llm",
                    "error": str(exc),
                    "degraded": True,
                    "degraded_reason": "error",
                },
            )

        if not verdict.unsafe:
            return JudgeSignal(
                name="output_safety",
                verdict="pass",
                reason="response passed output-safety guard",
                metadata={"judge_mode": "llm"},
            )

        categories = tuple(verdict.categories)
        enforced = tuple(c for c in categories if c in self._output_safety_enforce)
        if enforced or not categories:
            blocked = ", ".join(enforced) if enforced else "uncategorized"
            return JudgeSignal(
                name="output_safety",
                verdict="halt",
                reason=f"unsafe output blocked: {blocked}",
                metadata={
                    "judge_mode": "llm",
                    "categories": list(categories),
                    "enforced": list(enforced),
                },
            )
        return JudgeSignal(
            name="output_safety",
            verdict="warn",
            reason=f"unsafe output (log-only): {', '.join(categories)}",
            metadata={"judge_mode": "llm", "categories": list(categories)},
        )

    def _audit_signal(self, *, session_id: str, signal: JudgeSignal) -> None:
        if self._audit_log is None:
            return
        if signal.metadata.get("kernel_audited"):
            return  # the kernel wrote this decision's row when PRE_RESPONSE fired
        decision = "deny" if signal.verdict == "halt" else "allow"
        severity = (
            "critical"
            if signal.verdict == "halt"
            else "warn" if signal.verdict in {"warn", "retry"} else "info"
        )
        try:
            self._audit_log.record(
                run_id=session_id,
                step_id=None,
                agent_type="chat",
                hook_point="pre_response",
                plugin=f"curator_{signal.name}",
                decision=decision,
                severity=severity,
                reason=signal.reason,
                payload={
                    "verdict": signal.verdict,
                    "retryable": signal.retryable,
                    **signal.metadata,
                },
            )
        except Exception:
            # Audit writes never block the user response — but a dropped governance
            # decision must still leave a trace (the log is the audit of last resort).
            logger.warning(
                "governance audit-signal write failed; decision '%s' for session %s "
                "was NOT persisted to the audit log",
                signal.name,
                session_id,
                exc_info=True,
            )
            return


def _extract_json_candidate(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        body = stripped.strip("`")
        if body.startswith("json"):
            body = body[4:]
        return body.strip()
    return stripped


def _try_json_repair(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return json.dumps(parsed, sort_keys=True)


def _keyword_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", text.lower())
        if token not in {"this", "that", "with", "from", "have", "what", "when", "where"}
    }


def _parse_faithfulness_verdict(raw: str) -> tuple[bool, float, str] | None:
    candidate = _extract_json_candidate(raw)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    addresses = parsed.get("addresses_question")
    if not isinstance(addresses, bool):
        return None

    confidence_raw = parsed.get("confidence", 0.0)
    if not isinstance(confidence_raw, int | float):
        return None
    confidence = max(0.0, min(1.0, float(confidence_raw)))

    rationale_raw = parsed.get("rationale", "")
    if not isinstance(rationale_raw, str) or not rationale_raw.strip():
        rationale = "faithfulness judge did not provide rationale"
    else:
        rationale = rationale_raw.strip()

    return addresses, confidence, rationale


def _parse_grounding_verdict(raw: str) -> tuple[bool, float, str] | None:
    """Parse a grounding judge reply → ``(grounded, confidence, unsupported)``."""
    candidate = _extract_json_candidate(raw)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    grounded = parsed.get("grounded")
    if not isinstance(grounded, bool):
        return None

    confidence_raw = parsed.get("confidence", 0.0)
    if not isinstance(confidence_raw, int | float):
        return None
    confidence = max(0.0, min(1.0, float(confidence_raw)))

    unsupported_raw = parsed.get("unsupported", "")
    unsupported = unsupported_raw.strip() if isinstance(unsupported_raw, str) else ""

    return grounded, confidence, unsupported


def _parse_leak_verdict(raw: str) -> tuple[bool, float, str] | None:
    """Parse the leak-judge verdict: {is_leak: bool, confidence: float, reason: str}."""
    candidate = _extract_json_candidate(raw)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    is_leak = parsed.get("is_leak")
    if not isinstance(is_leak, bool):
        return None

    confidence_raw = parsed.get("confidence", 0.0)
    if not isinstance(confidence_raw, int | float):
        return None
    confidence = max(0.0, min(1.0, float(confidence_raw)))

    reason_raw = parsed.get("reason", "")
    reason = (
        reason_raw.strip() if isinstance(reason_raw, str) and reason_raw.strip() else "no reason"
    )
    return is_leak, confidence, reason


def _await_sync(factory: Callable[[], Awaitable[_T]], *, timeout_s: float) -> _T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(asyncio.wait_for(factory(), timeout=timeout_s))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(bind_context(_run_async_callable), factory, timeout_s)
        try:
            return future.result(timeout=timeout_s + 0.5)
        except FuturesTimeoutError as exc:
            raise TimeoutError("async faithfulness judge timed out") from exc


def _run_async_callable(factory: Callable[[], Awaitable[_T]], timeout_s: float) -> _T:
    return asyncio.run(asyncio.wait_for(factory(), timeout=timeout_s))
