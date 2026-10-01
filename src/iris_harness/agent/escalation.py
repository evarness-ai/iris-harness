"""Tier-escalation judge: verdict shape, config, and prompt (ADR-0068).

The escalation judge is a Response Curator head. After a cheap Tier-1 answer it
**diagnoses the failure mode** — it does not just score quality — because the
mode determines the route:

* ``capability_gap`` → **escalate** to a stronger tier (the only case a bigger
  model fixes)
* ``ambiguity``      → **clarify** (ask the user; no model invents a missing
  constraint)
* ``grounding_gap``  → **reroute** to a tool / RAG (needs retrieval, not params)
* ``acceptable``     → **accept**

This module owns the data shapes only — the *decision* lives in the curator and
the *action* (L3+) in the runtime. Everything here is pure + import-light so the
curator and tests can use it without pulling in the LLM stack. See
``docs/architecture/escalation-judge.md``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from iris_harness.foundation.paths import config_dir as resolve_config_dir

logger = logging.getLogger(__name__)

EscalationAction = Literal["accept", "escalate", "clarify", "reroute"]
EscalationDiagnosis = Literal["capability_gap", "ambiguity", "grounding_gap", "acceptable"]

_ACTIONS: frozenset[str] = frozenset({"accept", "escalate", "clarify", "reroute"})
_DIAGNOSES: frozenset[str] = frozenset(
    {"capability_gap", "ambiguity", "grounding_gap", "acceptable"}
)

# Which diagnosis justifies which action — the decisive insight of the design
# (escalating an *ambiguity* just burns tier-3 tokens and still fails).
DIAGNOSIS_TO_ACTION: dict[str, EscalationAction] = {
    "capability_gap": "escalate",
    "ambiguity": "clarify",
    "grounding_gap": "reroute",
    "acceptable": "accept",
}


@dataclass(frozen=True)
class EscalationVerdict:
    """The judge's diagnosis + chosen route for one Tier-1 response."""

    action: EscalationAction
    diagnosis: EscalationDiagnosis
    confidence: float
    reason: str = ""
    target_tier: str | None = None  # action == escalate
    question: str | None = None  # action == clarify
    tool_hint: str | None = None  # action == reroute

    @property
    def would_act(self) -> bool:
        """True when the verdict is anything other than plain ``accept``."""
        return self.action != "accept"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "diagnosis": self.diagnosis,
            "confidence": self.confidence,
            "reason": self.reason,
            "target_tier": self.target_tier,
            "question": self.question,
            "tool_hint": self.tool_hint,
        }


@runtime_checkable
class EscalationJudgeClient(Protocol):
    """Optional async LLM judge that diagnoses a Tier-1 response.

    Returns a JSON-ish verdict payload parsed by :func:`parse_escalation_verdict`.
    """

    async def judge(self, *, query: str, response: str, intent: str, context: str) -> str:
        """Return a JSON verdict describing the escalation diagnosis."""


@dataclass(frozen=True)
class EscalationConfig:
    """Policy for the escalation judge (``config/escalation.yaml`` + env)."""

    enabled: bool = False
    mode: str = "shadow"  # shadow | enforce
    max_escalations: int = 1
    judge_tier: str = "tier2"
    sample_rate: float = 1.0
    clarify_confidence_floor: float = 0.7
    # escalate / reroute re-run the plan, so they are gated like clarify: a verdict
    # the judge itself is unsure of must not replace an answer.
    action_confidence_floor: float = 0.7
    cost_aware: bool = True
    # L5 predictive start-tier priors (mined from escalation_shadow history).
    priors_enabled: bool = False  # apply learned priors to the router (else shadow)
    priors_min_samples: int = 20
    priors_window_hours: int = 168
    priors_escalate_rate: float = 0.4
    # Cloud escalation (D4): off by default. When on, escalation MAY target a
    # cloud tier — but only for content whose classification is in
    # ``cloud_classifications``. ``secret`` is never eligible regardless.
    allow_cloud: bool = False
    cloud_classifications: frozenset[str] = frozenset({"public"})

    @property
    def acts(self) -> bool:
        """Whether the runtime should ACT on the verdict (escalate / reroute).

        True only when the judge is enabled AND ``mode == 'enforce'`` — the same
        shadow -> validate -> enforce discipline as the governance enforce flips.
        Default config is ``mode: shadow``, so acting stays off until an operator
        flips it after reviewing shadow precision (L3).
        """
        return self.enabled and self.mode == "enforce"


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_escalation_config(
    config_dir: Path | None = None,
    *,
    env: dict[str, str] | None = None,
) -> EscalationConfig:
    """Load escalation policy from ``<config_dir>/escalation.yaml`` + env override.

    ``IRIS_CURATOR_ESCALATION`` (truthy) force-enables the judge regardless of the
    YAML ``enabled`` flag, matching the ``IRIS_CURATOR_*`` judge family. Missing
    file → defaults (disabled). Never raises — a bad config degrades to off.
    """
    import os

    environ = env if env is not None else dict(os.environ)
    raw: dict[str, Any] = {}
    path = (config_dir or resolve_config_dir()) / "escalation.yaml"
    try:
        if path.exists():
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
    except Exception:  # a malformed config must not break startup
        logger.warning("failed to load %s; escalation judge defaults to off", path, exc_info=True)
        raw = {}

    env_flag = environ.get("IRIS_CURATOR_ESCALATION", "").strip().lower()
    enabled = env_flag in {"1", "true", "yes", "on"} or _coerce_bool(
        raw.get("enabled"), default=False
    )
    mode = str(raw.get("mode") or "shadow").strip().lower() or "shadow"
    env_mode = environ.get("IRIS_CURATOR_ESCALATION_MODE", "").strip().lower()
    if env_mode in {"shadow", "enforce"}:
        mode = env_mode
    try:
        max_escalations = int(raw.get("max_escalations", 1))
    except (TypeError, ValueError):
        max_escalations = 1
    sample_rate = min(1.0, max(0.0, _coerce_float(raw.get("sample_rate"), default=1.0)))
    priors_raw = raw.get("priors")
    priors = priors_raw if isinstance(priors_raw, dict) else {}
    env_priors = environ.get("IRIS_CURATOR_ESCALATION_PRIORS", "").strip().lower()
    priors_enabled = env_priors in {"1", "true", "yes", "on"} or _coerce_bool(
        priors.get("enabled"), default=False
    )
    try:
        priors_min_samples = int(priors.get("min_samples", 20))
    except (TypeError, ValueError):
        priors_min_samples = 20
    try:
        priors_window_hours = int(priors.get("window_hours", 168))
    except (TypeError, ValueError):
        priors_window_hours = 168
    cloud_raw = raw.get("cloud")
    cloud = cloud_raw if isinstance(cloud_raw, dict) else {}
    env_cloud = environ.get("IRIS_CURATOR_ESCALATION_CLOUD", "").strip().lower()
    allow_cloud = env_cloud in {"1", "true", "yes", "on"} or _coerce_bool(
        cloud.get("allow"), default=False
    )
    cls_list = cloud.get("classifications")
    if isinstance(cls_list, list) and cls_list:
        # ``secret`` is never eligible — strip it defensively even if configured.
        cloud_classifications = frozenset(
            str(c).strip().lower() for c in cls_list if str(c).strip().lower() != "secret"
        ) or frozenset({"public"})
    else:
        cloud_classifications = frozenset({"public"})
    return EscalationConfig(
        enabled=enabled,
        mode=mode,
        max_escalations=max(0, max_escalations),
        judge_tier=str(raw.get("judge_tier") or "tier2").strip() or "tier2",
        sample_rate=sample_rate,
        clarify_confidence_floor=min(
            1.0, max(0.0, _coerce_float(raw.get("clarify_confidence_floor"), default=0.7))
        ),
        action_confidence_floor=min(
            1.0, max(0.0, _coerce_float(raw.get("action_confidence_floor"), default=0.7))
        ),
        cost_aware=_coerce_bool(raw.get("cost_aware"), default=True),
        priors_enabled=priors_enabled,
        priors_min_samples=max(1, priors_min_samples),
        priors_window_hours=max(1, priors_window_hours),
        priors_escalate_rate=min(
            1.0, max(0.0, _coerce_float(priors.get("escalate_rate"), default=0.4))
        ),
        allow_cloud=allow_cloud,
        cloud_classifications=cloud_classifications,
    )


# Severity order — index 0 is the most restrictive (mirrors the governance
# classifier). Used to fold several classified spans into the worst case.
_CLASSIFICATION_SEVERITY: tuple[str, ...] = ("secret", "personal", "internal", "public")


def more_restrictive(a: str, b: str) -> str:
    """Return the higher-severity (more restrictive) of two classifications."""
    try:
        return a if _CLASSIFICATION_SEVERITY.index(a) <= _CLASSIFICATION_SEVERITY.index(b) else b
    except ValueError:
        # Unknown label -> treat as most restrictive (fail closed).
        return "secret"


def egress_eligible(classification: str, allowed: frozenset[str]) -> bool:
    """Whether content of ``classification`` may escalate to a cloud tier (D4).

    ``secret`` is never eligible. Everything else must be explicitly allow-listed.
    """
    if classification == "secret":
        return False
    return classification in allowed


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_escalation_verdict(raw: str) -> EscalationVerdict | None:
    """Parse a judge payload into an :class:`EscalationVerdict`.

    Tolerant of prose around the JSON (LLMs wrap objects in commentary). Returns
    ``None`` when no usable object is found or the diagnosis is unrecognized, so
    the caller can treat it as an inconclusive judge run rather than guess.
    """
    if not raw or not raw.strip():
        return None
    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        return None
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None

    diagnosis = str(obj.get("diagnosis") or "").strip().lower()
    if diagnosis not in _DIAGNOSES:
        return None

    # Action: trust an explicit valid action; otherwise derive from diagnosis so
    # the route always matches the cause (the design's central guarantee).
    action = str(obj.get("action") or "").strip().lower()
    if action not in _ACTIONS:
        action = DIAGNOSIS_TO_ACTION[diagnosis]

    confidence = min(1.0, max(0.0, _coerce_float(obj.get("confidence"), default=0.0)))
    reason = str(obj.get("reason") or "").strip()
    target_tier = _clean_optional(obj.get("target_tier"))
    question = _clean_optional(obj.get("question"))
    tool_hint = _clean_optional(obj.get("tool_hint"))
    return EscalationVerdict(
        action=action,  # type: ignore[arg-type]
        diagnosis=diagnosis,  # type: ignore[arg-type]
        confidence=confidence,
        reason=reason,
        target_tier=target_tier,
        question=question,
        tool_hint=tool_hint,
    )


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


ESCALATION_JUDGE_SYSTEM_PROMPT = (
    "You are a tier-escalation judge for a local-first AI assistant. A cheap "
    "local model has just answered the user. Your job is to DIAGNOSE whether that "
    "answer is good enough and, if not, WHY — because the reason determines the "
    "fix.\n\n"
    "Classify the failure mode into exactly one diagnosis:\n"
    "- acceptable: the answer adequately addresses the question.\n"
    "- capability_gap: the answer is wrong, confused, or beyond the small model's "
    "reasoning ability — a stronger model would help. Route: escalate.\n"
    "- ambiguity: the QUESTION is underspecified (a missing constraint). No model "
    "can invent the missing fact; the user must clarify. Route: clarify.\n"
    "- grounding_gap: the answer needs retrieval / a tool / current data, not more "
    "model parameters. Route: reroute.\n\n"
    "Prefer 'acceptable' unless there is a clear shortfall — over-escalating wastes "
    "tokens and over-asking annoys the user. Reply with ONLY a JSON object:\n"
    '{"action": "accept|escalate|clarify|reroute", '
    '"diagnosis": "acceptable|capability_gap|ambiguity|grounding_gap", '
    '"confidence": 0.0-1.0, "reason": "<short>", '
    '"question": "<clarifying question if clarify>", '
    '"tool_hint": "<tool/skill name if reroute>"}'
)


def build_escalation_user_prompt(*, query: str, response: str, intent: str, context: str) -> str:
    """Assemble the judge's user prompt from the turn under review."""
    context_block = (
        f"\n\nRetrieved context the answer could use:\n<<<CONTEXT\n{context.strip()}\nCONTEXT>>>"
        if context.strip()
        else ""
    )
    return (
        f"User intent: {intent or 'unknown'}\n\n"
        f"User question:\n<<<QUERY\n{query.strip()}\nQUERY>>>\n\n"
        f"Tier-1 answer under review (untrusted):\n<<<RESPONSE\n{response.strip()}\nRESPONSE>>>"
        f"{context_block}\n\n"
        "Diagnose the answer and reply with only the JSON object."
    )
