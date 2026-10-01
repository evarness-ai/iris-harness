"""Deterministic memory triage for the IRIS operating framework.

This module classifies a learned item before any caller decides where to write
it. It is intentionally rule-based: local models can propose observations, but
these rails keep destinations explicit, auditable, and easy to test.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class MemoryTriageKind(StrEnum):
    """Classification lanes for learned memory items."""

    ACTIVE_CONTEXT = "active_context"
    USER_FACT = "user_fact"
    PATTERN = "pattern"
    ROUTINE = "routine"
    BEHAVIOR = "behavior"
    SOURCE_PREFERENCE = "source_preference"
    WIKI_KNOWLEDGE = "wiki_knowledge"
    LEARNING_SIGNAL = "learning_signal"
    AUTOMATION_CANDIDATE = "automation_candidate"


class MemoryDestination(StrEnum):
    """Durable stores or queues that a triaged item can target."""

    ACTIVE_MD = "active_md"
    USER_MD = "user_md"
    EPISODIC_MD = "episodic_md"
    WIKI = "wiki"
    BEHAVIOR = "behavior"
    ROUTINE_STORE = "routine_store"
    SKILL_QUEUE = "skill_queue"
    LEARNING_DB = "learning_db"


class MemoryDurability(StrEnum):
    """Expected lifetime of a triaged item."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    PERMANENT = "permanent"


class MemoryActionability(StrEnum):
    """Whether a triaged item implies follow-up action."""

    NONE = "none"
    REMINDER = "reminder"
    WORKFLOW = "workflow"
    AUTOMATION_CANDIDATE = "automation_candidate"


class MemoryTriageResult(BaseModel):
    """Decision about where a learned item belongs."""

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    schema_version: str = Field(
        default="memory-triage/v1",
        description="Stable contract version for tools and future MCP adapters.",
    )
    kind: MemoryTriageKind = Field(..., description="Classification lane for the item.")
    destination: MemoryDestination = Field(..., description="Store or queue to update.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Rule confidence.")
    durability: MemoryDurability = Field(..., description="Expected lifetime.")
    actionability: MemoryActionability = Field(
        default=MemoryActionability.NONE,
        description="Follow-up action implied by the item.",
    )
    requires_review: bool = Field(
        default=False,
        description="Whether a human should approve before durable write or execution.",
    )
    reason: str = Field(..., min_length=1, description="Human-readable routing rationale.")


_ACTIVE_PHRASES = (
    "we need to",
    "need to",
    "todo",
    "to do",
    "next step",
    "pending",
    "in progress",
    "working on",
    "start fresh",
    "current branch",
    "follow up",
)
_USER_FACT_PHRASES = (
    "remember that",
    "my name is",
    "call me",
    "i prefer",
    "i like",
    "i don't like",
    "i do not like",
    "my timezone",
    "my location",
    "i live in",
    "i work at",
)
_PATTERN_PHRASES = ("always", "usually", "often", "repeatedly", "every time", "tend to")
_SOURCE_PREF_PHRASES = (
    "prefer official docs",
    "preferred source",
    "source preference",
    "trust docs",
    "use official documentation",
    "use the docs",
)
_ROUTINE_TERMS = ("brief", "briefing", "digest", "check", "scan", "summarize", "report")
_SCHEDULE_TERMS = (
    "daily",
    "every morning",
    "every evening",
    "weekly",
    "each day",
    "every day",
    "every week",
    "at 7",
    "at 8",
)
_AUTOMATION_PHRASES = (
    "automate this",
    "make this a skill",
    "reusable skill",
    "promote this",
    "skill proposal",
    "do this automatically",
)
_LEARNING_PHRASES = (
    "failed",
    "failure",
    "error",
    "slow",
    "latency",
    "correction",
    "worked",
    "succeeded",
    "success",
)
_WIKI_PHRASES = (
    "architecture",
    "design decision",
    "decision record",
    "concept",
    "entity",
    "project history",
    "source notes",
    "wiki",
)
_WHEN_DO_RE = re.compile(
    r"\bwhen\b.+\b(?:do|use|ask|write|create|avoid|prefer|explain|respond)\b",
    re.DOTALL,
)
_REMINDER_RE = re.compile(r"\bremind\b|\breminder\b", re.IGNORECASE)


def triage_memory_item(
    text: str,
    *,
    signal_type: str = "conversation",
    repeated_count: int = 1,
    tool_outcome: str = "",
) -> MemoryTriageResult:
    """Classify a learned item into the operating-framework memory lanes.

    ``signal_type`` lets trusted callers preserve intent for already-typed
    events, such as extracted facts or tool telemetry. Free-form conversation
    still goes through deterministic phrase rules.
    """
    cleaned = _compact(text)
    if not cleaned:
        return MemoryTriageResult(
            kind=MemoryTriageKind.ACTIVE_CONTEXT,
            destination=MemoryDestination.ACTIVE_MD,
            confidence=0.0,
            durability=MemoryDurability.SHORT_TERM,
            actionability=MemoryActionability.NONE,
            requires_review=False,
            reason="empty item; nothing durable to store",
        )

    signal = signal_type.strip().lower()
    lower = cleaned.lower()
    outcome = tool_outcome.strip().lower()

    if signal in {"extracted_fact", "user_fact"}:
        return _result(
            kind=MemoryTriageKind.USER_FACT,
            destination=MemoryDestination.USER_MD,
            confidence=0.92,
            durability=MemoryDurability.LONG_TERM,
            reason="caller supplied a typed user fact",
        )
    if signal in {"tool_result", "learning_signal"} or _contains_any(
        f"{lower} {outcome}", _LEARNING_PHRASES
    ):
        return _result(
            kind=MemoryTriageKind.LEARNING_SIGNAL,
            destination=MemoryDestination.LEARNING_DB,
            confidence=0.86,
            durability=MemoryDurability.LONG_TERM,
            reason="outcome evidence belongs in the learning signal store",
        )
    if signal == "wiki_ingest":
        return _result(
            kind=MemoryTriageKind.WIKI_KNOWLEDGE,
            destination=MemoryDestination.WIKI,
            confidence=0.84,
            durability=MemoryDurability.LONG_TERM,
            reason="caller supplied curated knowledge for wiki ingestion",
        )
    if _contains_any(lower, _AUTOMATION_PHRASES):
        return _result(
            kind=MemoryTriageKind.AUTOMATION_CANDIDATE,
            destination=MemoryDestination.SKILL_QUEUE,
            confidence=0.9,
            durability=MemoryDurability.LONG_TERM,
            actionability=MemoryActionability.AUTOMATION_CANDIDATE,
            requires_review=True,
            reason="explicit request to automate or promote a reusable capability",
        )
    if _WHEN_DO_RE.search(lower):
        return _result(
            kind=MemoryTriageKind.BEHAVIOR,
            destination=MemoryDestination.BEHAVIOR,
            confidence=0.88,
            durability=MemoryDurability.PERMANENT,
            actionability=MemoryActionability.WORKFLOW,
            requires_review=True,
            reason="conditional conduct recipe should become a reviewed behavior",
        )
    if _contains_any(lower, _SCHEDULE_TERMS) and _contains_any(lower, _ROUTINE_TERMS):
        return _result(
            kind=MemoryTriageKind.ROUTINE,
            destination=MemoryDestination.ROUTINE_STORE,
            confidence=0.86,
            durability=MemoryDurability.LONG_TERM,
            actionability=MemoryActionability.WORKFLOW,
            requires_review=True,
            reason="scheduled recurring work should be drafted as a routine",
        )
    if _contains_any(lower, _SOURCE_PREF_PHRASES):
        return _result(
            kind=MemoryTriageKind.SOURCE_PREFERENCE,
            destination=MemoryDestination.EPISODIC_MD,
            confidence=0.84,
            durability=MemoryDurability.LONG_TERM,
            reason="source preference is compact retrieval guidance for episodic memory",
        )
    if _contains_any(lower, _USER_FACT_PHRASES):
        return _result(
            kind=MemoryTriageKind.USER_FACT,
            destination=MemoryDestination.USER_MD,
            confidence=0.82,
            durability=MemoryDurability.LONG_TERM,
            reason="stable first-person fact or preference belongs in user.md",
        )
    if _REMINDER_RE.search(lower):
        return _result(
            kind=MemoryTriageKind.ACTIVE_CONTEXT,
            destination=MemoryDestination.ACTIVE_MD,
            confidence=0.78,
            durability=MemoryDurability.SHORT_TERM,
            actionability=MemoryActionability.REMINDER,
            reason="reminder request is actionable open context until scheduled",
        )
    if _contains_any(lower, _ACTIVE_PHRASES):
        return _result(
            kind=MemoryTriageKind.ACTIVE_CONTEXT,
            destination=MemoryDestination.ACTIVE_MD,
            confidence=0.8,
            durability=MemoryDurability.SHORT_TERM,
            reason="open-loop wording belongs in active context",
        )
    if repeated_count > 1 or _contains_any(lower, _PATTERN_PHRASES):
        return _result(
            kind=MemoryTriageKind.PATTERN,
            destination=MemoryDestination.EPISODIC_MD,
            confidence=0.76 if repeated_count <= 1 else 0.88,
            durability=MemoryDurability.LONG_TERM,
            reason="recurring behavior should be indexed as an episodic pattern",
        )
    if _contains_any(lower, _WIKI_PHRASES) or len(cleaned) >= 180:
        return _result(
            kind=MemoryTriageKind.WIKI_KNOWLEDGE,
            destination=MemoryDestination.WIKI,
            confidence=0.7,
            durability=MemoryDurability.LONG_TERM,
            reason="durable concept or detailed note belongs in the wiki",
        )
    return _result(
        kind=MemoryTriageKind.ACTIVE_CONTEXT,
        destination=MemoryDestination.ACTIVE_MD,
        confidence=0.35,
        durability=MemoryDurability.SHORT_TERM,
        requires_review=True,
        reason="low-confidence default; keep reviewable before durable writes",
    )


def _result(
    *,
    kind: MemoryTriageKind,
    destination: MemoryDestination,
    confidence: float,
    durability: MemoryDurability,
    reason: str,
    actionability: MemoryActionability = MemoryActionability.NONE,
    requires_review: bool = False,
) -> MemoryTriageResult:
    return MemoryTriageResult(
        kind=kind,
        destination=destination,
        confidence=max(0.0, min(1.0, confidence)),
        durability=durability,
        actionability=actionability,
        requires_review=requires_review,
        reason=reason,
    )


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)
