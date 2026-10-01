"""Learning analyst — the agentic interpretation layer (ADR-0069 #4, slice 2).

This is the *interpret* half of the learning-intelligence loop. Slice 1
(:mod:`iris_harness.services.learning.intelligence`) measures; this slice reads those measured
numbers and exercises judgment — naming the patterns, flagging the issues, and
proposing ranked improvements. Judgment belongs to a model, not to hardcoded
thresholds (house rule: measure don't guess; agentic over heuristic), so the
analyst is an LLM that is handed the report and asked to reason over it.

It is strictly advisory. A :class:`Recommendation` is a *proposal* a human (or,
later, the experiment loop) acts on — nothing here mutates routing, config, or
skills. That mirrors the routines-reflection loop: the agent surfaces, the user
decides.

Pure + deterministic given its inputs: the analyst does no IO of its own. The
caller supplies an ``invoke(system, user) -> str`` for the governed LLM call and
the :class:`~iris_harness.services.learning.intelligence.IntelligenceReport` to reason over.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from iris_harness.services.learning.intelligence import IntelligenceReport
from iris_harness.services.learning.intelligence import render_text as render_report_text

if TYPE_CHECKING:
    pass

Confidence = Literal["low", "medium", "high"]
_VALID_CONFIDENCE: frozenset[str] = frozenset({"low", "medium", "high"})

# Cap how many recommendations we keep — a ranked shortlist the user can act on,
# not an exhaustive dump. The analyst is told to return the most important first.
_MAX_RECOMMENDATIONS = 5


ANALYST_SYSTEM_PROMPT = """\
You are IRIS's learning analyst. You are given a MEASURED report of how the \
assistant has been performing and learning: signal-accuracy figures and a \
per-(intent, tier) outcome matrix. Your job is to interpret these numbers and \
propose the most valuable improvements.

Rules:
- Ground EVERY recommendation in specific numbers from the report. Quote the \
intent, tier, rate, or sample count you are reacting to. Never invent data.
- Prefer findings with enough samples to be real; call out when a signal is too \
sparse to trust rather than over-reading it.
- Recommend concrete, reversible actions IRIS could test (e.g. "route <intent> \
to a higher tier", "enable the user-correction judge to measure X", "the \
escalation judge's precision is low — its threshold may be too eager"). These \
are PROPOSALS for a human to approve, not commands.
- If the data shows nothing worth acting on, say so with an empty list. Do not \
manufacture problems.

Respond with ONLY a JSON object, no prose around it:
{
  "summary": "<one or two sentence overview of the system's learning health>",
  "recommendations": [
    {
      "title": "<short imperative title>",
      "finding": "<the measured observation, with the numbers>",
      "action": "<the concrete, reversible change to try>",
      "evidence": ["<metric or cell that supports this>", "..."],
      "confidence": "low|medium|high",
      "target": {"metric": "<measured quantity>", "intent": "<intent or null>", "tier": "<tier or null>"},
      "change": {"kind": "route_intent_tier", "intent": "<intent>", "to_tier": "<tier>"}
    }
  ]
}
For "target", name the single measured quantity this recommendation is about, so \
it can be tracked: metric is one of completion_rate, correction_rate, reuse_count, \
avg_tokens (these need the intent + tier of the matrix cell) or escalation_precision, \
drop_rate (global — set intent and tier to null). Omit target or use null only if \
the recommendation isn't about one specific measured number.
For "change", give the concrete config edit ONLY if it is a routing change: \
{"kind": "route_intent_tier", "intent": "<intent>", "to_tier": "<the tier to move \
it to, e.g. tier2>"}. This lets the change be safely tested in a sandbox before \
anyone applies it. Use null for "change" when the recommendation isn't a routing \
edit (e.g. enabling a judge, or anything not yet expressible as a routing knob).
Return at most {max} recommendations, most important first.\
""".replace("{max}", str(_MAX_RECOMMENDATIONS))


# Metrics the analyst may target. Each names a measurable quantity slice 1
# already computes, so a promoted recommendation (slice 3) can capture a baseline
# and re-measure the SAME quantity over time. Cell metrics need intent+tier;
# accuracy metrics are global.
_CELL_METRICS: frozenset[str] = frozenset(
    {"completion_rate", "correction_rate", "reuse_count", "avg_tokens"}
)
_GLOBAL_METRICS: frozenset[str] = frozenset({"escalation_precision", "drop_rate"})
_TARGETABLE_METRICS: frozenset[str] = _CELL_METRICS | _GLOBAL_METRICS


@dataclass(frozen=True)
class MetricTarget:
    """The measurable cell a recommendation is about — its baseline/re-measure key.

    ``metric`` is one of the slice-1 measured quantities. ``intent``/``tier``
    identify the outcome-matrix cell for cell metrics; they are ignored for the
    global accuracy metrics (``escalation_precision``, ``drop_rate``).
    """

    metric: str
    intent: str | None = None
    tier: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "intent": self.intent, "tier": self.tier}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> MetricTarget | None:
        metric = str(data.get("metric") or "").strip()
        if metric not in _TARGETABLE_METRICS:
            return None
        intent = data.get("intent")
        tier = data.get("tier")
        return MetricTarget(
            metric=metric,
            intent=str(intent).strip() if intent else None,
            tier=str(tier).strip() if tier else None,
        )


# Applicable changes the analyst may propose — a STRUCTURED, testable config
# mutation (ADR-0070). Only recommendations carrying one of these can be sandbox-
# evaluated or applied; everything else stays advisory prose. v1 supports the one
# knob the sandbox harness understands: an intent's start tier.
_APPLIED_CHANGE_KINDS: frozenset[str] = frozenset({"route_intent_tier"})


@dataclass(frozen=True)
class AppliedChange:
    """A concrete, reversible config mutation a recommendation proposes.

    Distinct from :class:`MetricTarget`: the target is the *number to watch*; the
    change is the *edit to make*. v1: ``route_intent_tier`` (move an intent's start
    tier), which maps directly to a sandbox ``VariantConfig``. Shape is open so
    future kinds (flags, thresholds) slot in.
    """

    kind: str
    intent: str | None = None
    to_tier: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "intent": self.intent, "to_tier": self.to_tier}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AppliedChange | None:
        kind = str(data.get("kind") or "").strip()
        if kind not in _APPLIED_CHANGE_KINDS:
            return None
        intent = data.get("intent")
        to_tier = data.get("to_tier") or data.get("tier")
        change = AppliedChange(
            kind=kind,
            intent=str(intent).strip() if intent else None,
            to_tier=str(to_tier).strip() if to_tier else None,
        )
        # route_intent_tier is only usable with both an intent and a target tier.
        if kind == "route_intent_tier" and (not change.intent or not change.to_tier):
            return None
        return change


@dataclass(frozen=True)
class Recommendation:
    """One advisory, evidence-grounded improvement proposal."""

    title: str
    finding: str
    action: str
    evidence: tuple[str, ...]
    confidence: Confidence
    # The measurable quantity this recommendation is about (slice 3): lets a
    # promoted experiment capture a baseline + re-measure. None when the analyst
    # didn't tie it to a specific measured cell.
    target: MetricTarget | None = None
    # The concrete config edit it proposes (ADR-0070): lets a promoted experiment
    # be sandbox-evaluated. None when the action isn't a structured, testable knob.
    change: AppliedChange | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "finding": self.finding,
            "action": self.action,
            "evidence": list(self.evidence),
            "confidence": self.confidence,
            "target": self.target.as_dict() if self.target else None,
            "change": self.change.as_dict() if self.change else None,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Recommendation:
        conf = str(data.get("confidence", "low")).strip().lower()
        if conf not in _VALID_CONFIDENCE:
            conf = "low"
        raw_evidence = data.get("evidence") or []
        if isinstance(raw_evidence, str):
            raw_evidence = [raw_evidence]
        evidence = tuple(str(e).strip() for e in raw_evidence if str(e).strip())
        raw_target = data.get("target")
        target = MetricTarget.from_dict(raw_target) if isinstance(raw_target, dict) else None
        raw_change = data.get("change")
        change = AppliedChange.from_dict(raw_change) if isinstance(raw_change, dict) else None
        return Recommendation(
            title=str(data.get("title") or "").strip() or "(untitled)",
            finding=str(data.get("finding") or "").strip(),
            action=str(data.get("action") or "").strip(),
            evidence=evidence,
            confidence=conf,  # type: ignore[arg-type]
            target=target,
            change=change,
        )


@dataclass(frozen=True)
class LearningAnalysis:
    """The analyst's ranked, advisory read of one intelligence report."""

    generated_at: datetime
    window_hours: float
    model: str
    summary: str
    recommendations: tuple[Recommendation, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "window_hours": round(self.window_hours, 2),
            "model": self.model,
            "summary": self.summary,
            "recommendations": [r.as_dict() for r in self.recommendations],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> LearningAnalysis | None:
        raw_ts = data.get("generated_at")
        if not raw_ts:
            return None
        try:
            generated_at = datetime.fromisoformat(str(raw_ts))
        except ValueError:
            return None
        recs = tuple(
            Recommendation.from_dict(r)
            for r in (data.get("recommendations") or [])
            if isinstance(r, dict)
        )
        return LearningAnalysis(
            generated_at=generated_at,
            window_hours=float(data.get("window_hours") or 0.0),
            model=str(data.get("model") or "unknown"),
            summary=str(data.get("summary") or "").strip(),
            recommendations=recs,
        )


def build_analyst_user_prompt(report: IntelligenceReport) -> str:
    """Serialize the measured report into the analyst's user prompt.

    Both a human-readable rendering (so the model reads it naturally) and the raw
    JSON (so it can quote exact numbers) are included.
    """
    return (
        "Here is the measured learning report.\n\n"
        f"{render_report_text(report)}\n\n"
        "Raw data (for exact figures):\n"
        f"{json.dumps(report.as_dict(), indent=2)}\n\n"
        "Analyze it and return the JSON object described in your instructions."
    )


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model response, tolerantly."""
    text = raw.strip()
    if text.startswith("```"):
        # Strip a ```json ... ``` fence if present.
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_learning_analysis(
    raw: str,
    *,
    generated_at: datetime,
    window_hours: float,
    model: str,
) -> LearningAnalysis | None:
    """Tolerantly parse the analyst's JSON into a :class:`LearningAnalysis`.

    Returns ``None`` when the response carries no usable JSON object — the caller
    treats that as "no analysis this run" rather than fabricating one.
    """
    obj = _extract_json_object(raw)
    if obj is None:
        return None
    recs_raw = obj.get("recommendations")
    recs: tuple[Recommendation, ...] = ()
    if isinstance(recs_raw, list):
        recs = tuple(
            Recommendation.from_dict(r)
            for r in recs_raw[:_MAX_RECOMMENDATIONS]
            if isinstance(r, dict)
        )
    return LearningAnalysis(
        generated_at=generated_at,
        window_hours=window_hours,
        model=model,
        summary=str(obj.get("summary") or "").strip(),
        recommendations=recs,
    )


def analyze_learning(
    report: IntelligenceReport,
    *,
    invoke: Callable[[str, str], str],
    model: str,
) -> LearningAnalysis | None:
    """Run the analyst over a measured report; return its advisory analysis.

    ``invoke(system, user) -> str`` is the governed LLM call supplied by the
    runtime. Returns ``None`` if the model produced nothing parseable.
    """
    user_prompt = build_analyst_user_prompt(report)
    raw = invoke(ANALYST_SYSTEM_PROMPT, user_prompt)
    return parse_learning_analysis(
        raw,
        generated_at=report.sampled_at,
        window_hours=report.window_hours,
        model=model,
    )


def render_text(analysis: LearningAnalysis) -> str:
    """Agent-readable rendering of an analysis for the learning_recommendations tool."""
    when = analysis.generated_at.strftime("%Y-%m-%d %H:%M")
    lines = [f"Learning recommendations (analyzed {when}, {analysis.model}):", ""]
    if analysis.summary:
        lines.extend([analysis.summary, ""])
    if not analysis.recommendations:
        lines.append("No actionable recommendations from the latest run.")
        return "\n".join(lines)
    for i, rec in enumerate(analysis.recommendations, 1):
        lines.append(f"{i}. [{rec.confidence}] {rec.title}")
        if rec.finding:
            lines.append(f"   finding: {rec.finding}")
        if rec.action:
            lines.append(f"   action: {rec.action}")
        if rec.evidence:
            lines.append(f"   evidence: {'; '.join(rec.evidence)}")
    return "\n".join(lines)


__all__ = [
    "ANALYST_SYSTEM_PROMPT",
    "AppliedChange",
    "LearningAnalysis",
    "MetricTarget",
    "Recommendation",
    "analyze_learning",
    "build_analyst_user_prompt",
    "parse_learning_analysis",
    "render_text",
]
