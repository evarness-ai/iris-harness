"""Routine authoring driven by the brief skill manifest registry.

The previous implementation hardcoded two routine templates
(``morning_briefing`` / ``daily_repo_brief``), the sections each could
include, the labels for those sections, and the tool-callback strings —
all in Python dicts and regexes. That metadata already lives in the
brief skill manifests (``config/skills/builtin/*/manifest.yaml``); this
module reads it instead of re-encoding it.

Template selection uses the same SemanticSkillRouter the runtime uses
for skill selection (cosine over manifest text). Section selection
ranks slot keys + their tool descriptions against the user's message.
Tool callbacks and capabilities are derived from each slot's
``skill``/``tool`` fields. Adding a new brief skill therefore needs no
Python edits — drop a new manifest into ``config/skills/`` and it
becomes an addressable routine target.

Schedule, time, delivery-channel, and content-style extractors remain
deterministic value parsers — those are precise grammatical
conversions ("9am" → 09:00, "telegram" → telegram), not domain
classification decisions.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from iris_harness.tools.skills.models import BriefToolSlot, SkillPackage, SkillToolManifest, ToolArg

from .models import RoutineApprovalStatus, RoutineSpec, create_routine_spec

logger = logging.getLogger(__name__)

LLMCaller = Callable[[str], str]
"""Sync Tier-2 LLM call shape — takes a fully-rendered prompt, returns text."""

RoutineAuthoringAction = Literal["none", "draft", "clarify", "approve", "cancel"]

_ROUTINE_SIGNAL_RE = re.compile(
    # Bare "each" is NOT a recurrence signal — "a brief summary of each repo" is a
    # one-shot ask, not a routine (issue 0019). "each" only counts when temporal
    # ("each morning", "each Monday").
    r"\b(every|everyday|daily|weekly|weekday|weekdays|routine|recurring|hourly|nightly|monthly)\b"
    r"|\beach\s+(?:day|morning|afternoon|evening|night|week|weekday|month|hour|"
    r"mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b",
    re.IGNORECASE,
)
_ROUTINE_VERB_RE = re.compile(
    # "set up" added by the Phase 3 multiturn scenario: "set up a routine
    # every morning at 7" carried a routine signal but no recognized verb
    # and silently fell through to the general pipeline.
    r"\b(send|brief|summary|summarize|notify|create|set\s+up|give|show|run|fetch|schedule)\b",
    re.IGNORECASE,
)
_HOW_TO_SETUP_RE = re.compile(
    r"^\s*how\s+(?:to|do\s+(?:i|we)|can\s+(?:i|we)|would\s+(?:i|we)|should\s+(?:i|we))\b",
    re.IGNORECASE,
)
# A one-off PREVIEW of a brief — "show me a sample of my daily brief", "what would
# my briefing look like" — is a render-once request, NOT routine authoring. Defer
# it to the ReAct path (which renders the brief via its brief-render tool) unless
# the message ALSO carries explicit creation/recurrence intent. The recurrence set
# deliberately omits bare "daily": in a preview that word is the brief's NAME
# ("daily brief"), not a schedule directive.
_BRIEF_PREVIEW_RE = re.compile(r"\b(sample|preview|example|look\s+like)\b", re.IGNORECASE)
_BRIEF_NOUN_RE = re.compile(r"\b(brief|briefing|summary|digest)\b", re.IGNORECASE)
_ROUTINE_RECURRENCE_RE = re.compile(
    r"\b(every|each|recurring|recurrence|automate|automated|set\s*up|"
    r"schedule|create|nightly|weekly|monthly|hourly)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_EMAIL_WORKFLOW_DOMAIN_RE = re.compile(
    r"\b(gmail|e-?mails?|inbox|mailbox)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_EMAIL_WORKFLOW_ACTION_RE = re.compile(
    r"\b(connect|authorize|authenticate|extract|read|scan|pull|import|classify|label)\b",
    re.IGNORECASE,
)
_APPROVAL_RE = re.compile(
    # ``schedule`` is an approval verb when bare ("schedule", "schedule it",
    # "schedule that") but a creation verb when followed by an object
    # ("schedule my healthcheck every morning"). The negative lookahead
    # excludes the latter so user-named routines aren't misread as
    # approvals.
    r"^\s*(yes|yep|yeah|approve|approved|go ahead|do it|looks good)\b"
    r"|^\s*scheduled?\b(?!\s+(?:my|the|a|an|your|our))",
    re.IGNORECASE,
)
_CANCEL_RE = re.compile(r"^\s*(no|cancel|discard|never mind|stop)\b", re.IGNORECASE)
_TIME_WITH_PREFIX_RE = re.compile(
    r"\b(?:at|by|around)\s+(?P<hour>1[0-2]|0?[1-9]|2[0-3])"
    r"(?::(?P<minute>[0-5]\d))?\s*(?P<meridiem>am|pm)?\b",
    re.IGNORECASE,
)
_TIME_WITH_MERIDIEM_RE = re.compile(
    r"\b(?P<hour>1[0-2]|0?[1-9])(?::(?P<minute>[0-5]\d))?\s*(?P<meridiem>am|pm)\b",
    re.IGNORECASE,
)
_WEEKDAYS: dict[str, int] = {
    "sunday": 0,
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
}
_ALL_SECTIONS_TOKENS_RE = re.compile(
    r"\b(all|everything|default|defaults|standard|full)\b",
    re.IGNORECASE,
)


class RoutineAuthoringResult(BaseModel):
    """Parsed user intent for conversational routine authoring."""

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    action: RoutineAuthoringAction
    title: str = ""
    goal: str = ""
    schedule: str = ""
    template: str = ""
    delivery_channel: str = "console"
    source_preferences: tuple[str, ...] = Field(default_factory=tuple)
    required_capabilities: tuple[str, ...] = Field(default_factory=tuple)
    missing_slots: tuple[str, ...] = Field(default_factory=tuple)
    reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_draft_spec(self) -> RoutineSpec:
        """Build a draft routine spec from a complete draft parse."""

        if self.action not in {"draft", "clarify"}:
            raise ValueError("only draft or complete clarification results can become RoutineSpec")
        return create_routine_spec(
            title=self.title,
            goal=self.goal or self.title,
            schedule=self.schedule,
            template=self.template,
            delivery_channel=self.delivery_channel,
            source_preferences=self.source_preferences,
            required_capabilities=self.required_capabilities,
            approval_status=RoutineApprovalStatus.DRAFT,
            metadata=self.metadata,
        )


# ---------------------------------------------------------------------------
# Manifest-driven helpers (replace hardcoded section / template dicts).
# ---------------------------------------------------------------------------


def brief_tool_slot_keys(package: SkillPackage) -> tuple[str, ...]:
    """Return the user-selectable tool slot keys of a brief skill.

    Literal slots (e.g. ``date``) are computed at render time and are
    not user-selectable; only tool slots are returned here.
    """

    brief = package.manifest.brief
    if brief is None:
        return ()
    # Synthesis/headline slots (e.g. "today's plan") render but are not
    # user-selectable sections — excluding them keeps a bare "morning briefing"
    # request asking which sections, rather than auto-binding the headline whose
    # text collides with the brief's own name (ADR-0059).
    return tuple(
        key
        for key, slot in brief.slots.items()
        if isinstance(slot, BriefToolSlot) and not slot.synthesis
    )


def brief_slot_label(slot_key: str) -> str:
    """Humanize a slot key for user-facing display."""

    return slot_key.replace("_", " ").replace("-", " ").strip()


def brief_slot_callbacks(
    package: SkillPackage,
    slot_keys: Sequence[str],
) -> tuple[str, ...]:
    """Return the de-duplicated tool names invoked by the given slots."""

    brief = package.manifest.brief
    if brief is None:
        return ()
    seen: list[str] = []
    for key in slot_keys:
        slot = brief.slots.get(key)
        if isinstance(slot, BriefToolSlot) and slot.tool not in seen:
            seen.append(slot.tool)
    return tuple(seen)


def brief_slot_capabilities(
    package: SkillPackage,
    slot_keys: Sequence[str],
) -> tuple[str, ...]:
    """Return capability identifiers needed by the given slots.

    Capabilities derive from the slot key plus an always-present
    ``channel_delivery`` capability so downstream consumers can filter
    on what the routine needs at run time.
    """

    capabilities: list[str] = ["channel_delivery"]
    for key in slot_keys:
        cap = key.replace("-", "_")
        if cap and cap not in capabilities:
            capabilities.append(cap)
    return tuple(capabilities)


def format_brief_sections(slot_keys: Sequence[str]) -> str:
    """Return a human-readable section list for the chosen brief slots."""

    labels = [brief_slot_label(key) for key in slot_keys if key]
    return ", ".join(labels) if labels else "none"


def match_routine_capability(
    message: str,
    packages: Sequence[SkillPackage],
    router: Any,
) -> SkillPackage | None:
    """Pick the best skill (any kind) to bind a routine to via semantic ranking.

    Routines are generic — they can bind to a brief skill (rendered as
    a multi-section brief), a regular tool skill (called on schedule),
    or any future capability. Filtering the candidate set here would
    re-introduce the brief-only assumption Phase A is removing. The
    caller is responsible for handling kind-specific behavior after
    the match (slot selection for briefs, arg checking for tool
    skills).
    """

    if router is None or not message.strip():
        return None
    candidates = tuple(p for p in packages if getattr(p, "is_loadable", True))
    if not candidates:
        return None
    return cast("SkillPackage | None", router.best_match(message, candidates))


# Backwards-compatible alias for any caller that still imports the
# brief-leaning name. Remove once Phase A propagates everywhere.
match_brief_template = match_routine_capability


_CLAUSE_SPLIT_RE = re.compile(
    r"[,;.]|\s+\band\b\s+|\s+\bwith\b\s+|\s+\bfinally\b\s+", re.IGNORECASE
)


def _user_clauses(message: str) -> tuple[str, ...]:
    """Split a user message into semantically distinct clauses.

    Whole-message cosine against a short slot snippet is noisy — the
    user's enumeration ("reminders, activities, top 10 git repos, AI
    news…") gets diluted by all the unrelated tokens. Splitting on
    natural boundaries (commas, periods, ``and``, ``with``) lets each
    clause stand alone for the per-slot comparison.
    """

    parts = (clause.strip() for clause in _CLAUSE_SPLIT_RE.split(message))
    return tuple(clause for clause in parts if len(clause) >= 2)


def match_brief_slots(
    message: str,
    package: SkillPackage,
    router: Any,
    *,
    threshold: float = 0.30,
    margin: float = 0.12,
) -> tuple[str, ...]:
    """Pick the slots a user's message references in the chosen brief.

    Returns every tool slot when the user uses ``all|everything|default``
    shorthand. Otherwise splits the message into clauses (comma / period /
    ``and`` / ``with`` boundaries) and scores each clause against each slot's
    ``"<key> — <item template>"`` representation.

    Selection is per-clause **winner-band**, not a flat threshold: for each
    clause we take the single best-scoring slot and keep only slots within
    *margin* of that clause's best (and above *threshold*). This stops a strong
    clause from dragging in weakly-related neighbours that merely share a token —
    e.g. "due reminders" selects ``reminders`` (its clear winner) without also
    pulling in ``due_today`` / ``bills_due`` — while still selecting a genuine
    cluster when several slots score close together (e.g. a bare "news" clause
    keeping ai/usa/global news). A flat threshold can't do this: raising it to
    drop the neighbours would also drop legitimately low-but-best matches like
    ``stocks``. The 0.30 floor reflects that a short clause vs. a short slot
    snippet is a noisier signal than a full-document match.
    """

    brief = package.manifest.brief
    if brief is None:
        return ()
    tool_keys = brief_tool_slot_keys(package)
    if not tool_keys:
        return ()
    if _ALL_SECTIONS_TOKENS_RE.search(message):
        return tool_keys
    if router is None:
        return ()

    slot_docs: dict[str, _SlotDoc] = {}
    for key in tool_keys:
        slot = brief.slots[key]
        if not isinstance(slot, BriefToolSlot):
            continue
        # An explicit ``summary`` is a clean, author-controlled semantic
        # descriptor — use it alone so it isn't diluted by the functional
        # item_template/empty noise (lets confusable slots like stocks vs
        # portfolio separate). Otherwise fall back to the auto-derived snippet.
        if slot.summary:
            snippet = f"{brief_slot_label(key)} {slot.summary}".strip()
        else:
            snippet_parts = [brief_slot_label(key), slot.item_template or "", slot.empty or ""]
            snippet = " ".join(p for p in snippet_parts if p).strip() or key
        slot_docs[key] = _SlotDoc(snippet)
    if not slot_docs:
        return ()

    clauses = _user_clauses(message) or (message,)
    selected: set[str] = set()
    for clause in clauses:
        scores = {key: router.score(clause, doc) for key, doc in slot_docs.items()}
        clause_best = max(scores.values(), default=0.0)
        if clause_best < threshold:
            # This clause doesn't strongly reference any slot — skip it rather
            # than admit a borderline neighbour.
            continue
        cutoff = max(threshold, clause_best - margin)
        for key, score in scores.items():
            if score >= cutoff:
                selected.add(key)
    # Preserve manifest slot order for stable, readable output.
    return tuple(key for key in tool_keys if key in selected)


class _SlotDoc:
    """Thin shim so SemanticSkillRouter.score can rank a slot snippet.

    The router's ``score`` expects a package-like object whose
    ``_skill_doc(pkg)`` join produces a string. We give it a tiny
    object with the right shape rather than building a full package.
    """

    def __init__(self, text: str) -> None:
        self._text = text

        class _Manifest:
            name = ""
            description = text
            tools: tuple[Any, ...] = ()

        self.manifest = _Manifest()
        self.agent_context = None
        self.is_loadable = True


def _tool_required_args(
    manifest_tool: SkillToolManifest,
    tool_class: Any,
) -> tuple[ToolArg, ...]:
    """Return the args the user must supply before drafting the routine.

    Manifest-declared ``args`` are the source of truth — that block is
    the only place that distinguishes "the user must choose this" from
    "this has a default the agent can silently accept". When the
    manifest omits an ``args`` block (legacy skill yet to migrate),
    fall back to introspecting ``tool.args_schema`` and synthesize a
    minimal ``ToolArg`` per required-without-default field so the
    downstream clarification UX has *something* to render.
    """

    if manifest_tool.args:
        return tuple(arg for arg in manifest_tool.args if arg.required)

    schema = getattr(tool_class, "args_schema", None)
    if schema is None:
        return ()
    fields = getattr(schema, "model_fields", None)
    if not isinstance(fields, dict):
        return ()
    synthesized: list[ToolArg] = []
    for name, field in fields.items():
        try:
            if not field.is_required():
                continue
        except Exception:  # noqa: BLE001, S112 - optionality probe; skip exotic fields
            continue
        description = getattr(field, "description", None) or name.replace("_", " ")
        synthesized.append(ToolArg(name=name, description=description, type="string"))
    return tuple(synthesized)


def _match_tool_in_skill(
    message: str,
    package: SkillPackage,
    router: Any,
) -> tuple[SkillToolManifest, Any] | None:
    """Pick which tool in a multi-tool skill the user intends.

    Returns ``(manifest_tool, tool_class)`` or ``None``. The caller
    needs the full manifest entry (not just the name) so it can read
    the declared ``args`` block for clarification UX. For single-tool
    skills, returns that tool unconditionally. For multi-tool skills,
    scores each tool's description against the user message and returns
    the best one; falls back to the first tool when the router is
    missing.
    """

    pairs = list(zip(package.manifest.tools, package.tool_classes, strict=False))
    if not pairs:
        return None
    if len(pairs) == 1 or router is None:
        return pairs[0]

    best: tuple[SkillToolManifest, Any] | None = None
    best_score = -1.0
    for manifest_tool, tool_class in pairs:
        snippet = f"{manifest_tool.name.replace('_', ' ')} {manifest_tool.description}"
        score = router.score(message, _SlotDoc(snippet))
        if score > best_score:
            best_score = score
            best = (manifest_tool, tool_class)
    return best


_USER_NAMED_TITLE_PROMPT = """\
You are parsing a user instruction that schedules a recurring assistant routine.

Decide whether the user explicitly named this routine. Examples:
- "schedule my healthcheck every morning at 8" -> healthcheck
- "call it flashcards and run it daily at 7am" -> flashcards
- "set up my appointments routine" -> appointments
- "every morning at 8 send me my reminders" -> (no name; reminders is content, not a name)
- "send me a morning briefing daily" -> (no name; describes the brief, not a name)

Respond with ONLY the name as a single lowercase token (no quotes, no
punctuation, no explanation), or the empty string if the user did not
name the routine.

Message: {message}
Name:"""

_HEDGED_TITLE_RESPONSES = frozenset(
    {"none", "no", "n/a", "na", "empty", "null", "unknown", "unspecified"}
)


def extract_user_named_title(
    message: str,
    llm_caller: LLMCaller | None,
) -> str | None:
    """Return the routine name the user explicitly chose, or ``None``.

    Q-A3: title extraction is LLM-driven (no heuristic keyword matching
    — see the agentic-over-heuristic feedback memory). When the LLM
    caller is unavailable, the helper degrades cleanly: returns
    ``None`` and lets the caller fall back to the matched skill's
    title. Hedged responses (multi-token, multi-line, or known
    "no-name" tokens like ``none`` / ``unknown``) are treated as no
    name so the agent can ask the user to confirm rather than guess.
    """

    if llm_caller is None:
        logger.debug("routine title extraction skipped: no LLM caller (degraded mode)")
        return None

    try:
        raw = llm_caller(_USER_NAMED_TITLE_PROMPT.format(message=message))
    except Exception:
        logger.warning(
            "routine title extraction LLM call failed; falling back to skill title",
            exc_info=True,
        )
        return None

    if raw is None:
        return None
    candidate = raw.strip().strip("\"'").rstrip(".,;:")
    if not candidate:
        return None
    # The prompt asks for a single token; treat anything else as a
    # hedged / explanatory reply rather than a confident name.
    if "\n" in candidate:
        candidate = candidate.split("\n", 1)[0].strip().strip("\"'").rstrip(".,;:")
    if not candidate or " " in candidate or len(candidate) > 50:
        logger.debug("routine title extraction returned non-token response: %r", raw)
        return None
    if candidate.lower() in _HEDGED_TITLE_RESPONSES:
        return None
    return candidate


# ---------------------------------------------------------------------------
# Phase A-2 / Q-A1: parse the user's reply to a tool-args clarification.
# ---------------------------------------------------------------------------

_TOOL_ARG_REPLY_PROMPT = """\
You are parsing a user's reply to a routine setup question. The user was
asked to choose values for these arguments:

{args_block}
{prior_block}\
Their reply:
{reply}

Return ONLY a JSON object with one key per argument name that you can
resolve confidently, plus an optional "missing" array listing the
argument names you cannot resolve from the reply. Examples of the
exact shape:

{{"type": "git", "category": "git-repositories", "limit": 10}}
{{"type": "git", "missing": ["category", "limit"]}}

Rules:
- Use the exact option strings shown under "allowed" for enum args.
- Integers and numbers must be numeric, not strings.
- Do not invent values the user did not imply.
- Do not include any prose, markdown fences, or trailing text — JSON only.
"""

_BOOL_TRUE = frozenset({"true", "yes", "y", "1", "on"})
_BOOL_FALSE = frozenset({"false", "no", "n", "0", "off"})
_KEY_VALUE_RE = re.compile(r"(\w[\w\-]*)\s*[=:]\s*([^,;\n]+)")


def parse_tool_arg_reply(
    reply: str,
    pending_args: Sequence[ToolArg],
    llm_caller: LLMCaller | None,
    *,
    prior_answers: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve a user's natural-language reply into structured arg values.

    Q-A1: when a Tier-2 LLM caller is wired in, the helper ships the
    args schema plus the user's reply and asks for JSON. That lets
    users answer naturally ("git, top repos, 10 of them", "the
    trending repositories one"). When the caller is unavailable —
    cloud disabled, runtime degraded — the helper falls back to a
    deterministic ``key=value`` / comma-positional grammar so the flow
    keeps working.

    Returns ``(values, missing)``:
    - ``values`` — dict of fully-validated ``arg_name -> coerced_value``
      pairs. Only entries that pass the ``ToolArg`` type / options /
      pattern / min / max checks appear here.
    - ``missing`` — names of pending args that could not be resolved
      (absent, typed wrong, out of range, or flagged by the LLM in its
      own ``missing`` array). The caller re-asks only these.

    *prior_answers* are values resolved by an earlier clarification
    turn; passed to the LLM as context so it does not ask the user to
    repeat them. They are NOT returned in ``values`` — the caller
    merges across turns.
    """

    if not pending_args:
        return ({}, [])

    if llm_caller is None:
        return _deterministic_tool_arg_parse(reply, pending_args)

    try:
        prompt = _format_tool_arg_reply_prompt(reply, pending_args, prior_answers or {})
        raw = llm_caller(prompt)
    except Exception:
        logger.warning(
            "tool arg reply LLM call failed; falling back to deterministic parser",
            exc_info=True,
        )
        return _deterministic_tool_arg_parse(reply, pending_args)

    parsed = _coerce_json_response(raw)
    if parsed is None:
        logger.debug(
            "tool arg reply LLM response not parseable as JSON; falling back: %r",
            raw,
        )
        return _deterministic_tool_arg_parse(reply, pending_args)

    return _validate_resolved_args(parsed, pending_args)


def _format_tool_arg_reply_prompt(
    reply: str,
    pending_args: Sequence[ToolArg],
    prior_answers: Mapping[str, Any],
) -> str:
    args_lines: list[str] = []
    for arg in pending_args:
        line = f"- {arg.name} (type: {arg.type})"
        if arg.options:
            line += f"; allowed: {', '.join(arg.options)}"
        if arg.default is not None:
            line += f"; default: {arg.default}"
        if arg.examples:
            line += f"; examples: {', '.join(str(e) for e in arg.examples)}"
        if arg.description:
            line += f" — {arg.description}"
        args_lines.append(line)
    args_block = "\n".join(args_lines)

    prior_block = ""
    if prior_answers:
        prior_lines = "\n".join(f"- {k}: {v}" for k, v in prior_answers.items())
        prior_block = "Already answered (do not re-ask the user):\n" + prior_lines + "\n\n"

    return _TOOL_ARG_REPLY_PROMPT.format(
        args_block=args_block,
        prior_block=prior_block,
        reply=reply,
    )


def _coerce_json_response(raw: Any) -> dict[str, Any] | None:
    """Extract the JSON object from an LLM response, tolerating fences."""

    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1)
        text = re.sub(r"\s*```\s*$", "", text, count=1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        result = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


def _validate_resolved_args(
    parsed: Mapping[str, Any],
    pending_args: Sequence[ToolArg],
) -> tuple[dict[str, Any], list[str]]:
    """Coerce each pending arg through its ``ToolArg`` validation rules."""

    llm_missing: set[str] = set()
    raw_missing = parsed.get("missing")
    if isinstance(raw_missing, list):
        llm_missing = {str(name) for name in raw_missing}

    values: dict[str, Any] = {}
    missing: list[str] = []
    for arg in pending_args:
        if arg.name in llm_missing:
            missing.append(arg.name)
            continue
        raw_value = parsed.get(arg.name)
        coerced = _coerce_value_to_tool_arg(raw_value, arg)
        if coerced is None:
            missing.append(arg.name)
        else:
            values[arg.name] = coerced
    return values, missing


def _coerce_value_to_tool_arg(value: Any, arg: ToolArg) -> Any:
    """Validate and coerce a single value against the ``ToolArg`` schema.

    Returns the coerced value on success, ``None`` on failure. The
    caller treats ``None`` as "unresolved" and adds the arg name to
    ``missing``.
    """

    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None

    if arg.type == "enum":
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        return next(
            (opt for opt in arg.options if opt.lower() == normalized),
            None,
        )
    if arg.type == "int":
        if isinstance(value, bool):
            return None
        try:
            coerced_int = int(value) if not isinstance(value, str) else int(value.strip())
        except (TypeError, ValueError):
            return None
        if arg.min is not None and coerced_int < arg.min:
            return None
        if arg.max is not None and coerced_int > arg.max:
            return None
        return coerced_int
    if arg.type == "number":
        if isinstance(value, bool):
            return None
        try:
            coerced_num = float(value) if not isinstance(value, str) else float(value.strip())
        except (TypeError, ValueError):
            return None
        if arg.min is not None and coerced_num < arg.min:
            return None
        if arg.max is not None and coerced_num > arg.max:
            return None
        return coerced_num
    if arg.type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in _BOOL_TRUE:
                return True
            if normalized in _BOOL_FALSE:
                return False
        return None
    if arg.type == "string":
        text = value if isinstance(value, str) else str(value)
        stripped = text.strip()
        if not stripped:
            return None
        if arg.pattern is not None and not re.fullmatch(arg.pattern, stripped):
            return None
        return stripped
    return None


def _deterministic_tool_arg_parse(
    reply: str,
    pending_args: Sequence[ToolArg],
) -> tuple[dict[str, Any], list[str]]:
    """LLM-free fallback parser.

    Prefers ``key=value`` / ``key: value`` pairs (comma- or newline-
    separated). When the reply contains no key markers, treats it as
    positional — each comma-separated token feeds the next pending arg
    in order. Logged as degraded so operators can see when the Tier-2
    caller is missing.
    """

    logger.debug("tool arg reply: deterministic parser (no LLM caller)")

    raw_values: dict[str, Any] = {}
    pairs = _KEY_VALUE_RE.findall(reply)
    if pairs:
        for key, raw_value in pairs:
            raw_values[key.lower()] = raw_value.strip()
    else:
        parts = [p.strip() for p in re.split(r"[,;\n]", reply) if p.strip()]
        for arg, part in zip(pending_args, parts, strict=False):
            raw_values[arg.name] = part

    return _validate_resolved_args(raw_values, pending_args)


def parse_free_form_tool_arg_reply(
    reply: str,
    arg: ToolArg,
    llm_caller: LLMCaller,
) -> Any:
    """Q-A2-c (deferred): extract a value for a free-form string arg.

    The surface exists so the runtime can already branch on free-form
    args; the implementation is intentionally deferred until a real
    skill needs it. When a skill author declares a ``type="string"``
    arg with no ``pattern`` and no ``options`` (genuinely open-ended
    input — e.g. a stock ticker, a free-text query), the agent should
    ask the user with no enumeration and route the response through an
    LLM that extracts a structured value plus optional validation.

    Tracked under "Open long-term questions" in
    ``docs/architecture/routine-engine-roadmap.md``. Promote the
    implementation when a skill genuinely needs it; until then,
    ``parse_tool_arg_reply`` handles enumerated and bounded args.
    """

    raise NotImplementedError(
        "Free-form tool-arg asking (Q-A2-c) is deferred — see "
        "docs/architecture/routine-engine-roadmap.md section 4."
    )


def parse_routine_authoring(
    message: str,
    *,
    candidate_packages: Sequence[SkillPackage] = (),
    router: Any = None,
    origin_channel: str | None = None,
    llm_caller: LLMCaller | None = None,
    brief_packages: Sequence[SkillPackage] | None = None,
) -> RoutineAuthoringResult:
    """Parse a chat turn into a routine-authoring action when possible.

    *candidate_packages* is the registry-derived tuple of all loadable
    skill packages — brief and non-brief alike. The semantic router
    ranks the user's message against this entire set so a routine can
    bind to any addressable capability (per the "routines are generic"
    principle). When empty, the parser still handles
    approval/cancel/schedule signals but cannot resolve a capability.

    *brief_packages* is a deprecated alias for *candidate_packages*
    retained until Phase B propagates the new name everywhere.

    *origin_channel* is the channel the user is messaging from
    (``"telegram"``, ``"cli"``, etc.); used as the default delivery
    channel when the message itself doesn't name one.

    *llm_caller* is the Tier-2 LLM client used for agentic decisions
    (Q-A3 title extraction now; Q-A1 reply parsing next). When absent,
    the parser still works — anything that needs the caller degrades
    to skill-derived defaults rather than crashing.
    """

    if brief_packages is not None and not candidate_packages:
        candidate_packages = brief_packages

    text = " ".join(message.strip().split())
    lowered = text.lower()
    if not text:
        return RoutineAuthoringResult(action="none")
    if _APPROVAL_RE.search(lowered):
        return RoutineAuthoringResult(action="approve", confidence=0.9)
    if _CANCEL_RE.search(lowered):
        return RoutineAuthoringResult(action="cancel", confidence=0.9)
    if _should_defer_to_intent_router(lowered):
        return RoutineAuthoringResult(action="none")
    if not _looks_like_routine_request(lowered):
        return RoutineAuthoringResult(action="none")

    schedule = _extract_schedule(lowered)
    capability = match_routine_capability(text, candidate_packages, router)
    template = capability.manifest.name if capability is not None else ""
    is_brief = (
        capability is not None
        and capability.manifest.kind == "brief"
        and capability.manifest.brief is not None
    )
    title = (
        capability.manifest.brief.subject
        if is_brief and capability is not None and capability.manifest.brief is not None
        else (capability.manifest.name if capability is not None else "")
    )
    user_named_title = extract_user_named_title(text, llm_caller)
    if user_named_title:
        title = user_named_title
    metadata: dict[str, Any] = {"original_query": text}
    if user_named_title:
        metadata["user_named_title"] = user_named_title
    metadata.update(extract_content_style(text))
    metadata.update(extract_formatting(text))  # header / footer / tone capture

    source_preferences: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    required_args: tuple[ToolArg, ...] = ()
    missing: list[str] = []
    if not schedule:
        missing.append("schedule")

    if capability is None:
        missing.append("template")
    elif is_brief:
        selected_slots = match_brief_slots(_strip_schedule_phrasing(text), capability, router)
        if selected_slots:
            source_preferences = selected_slots
            required_capabilities = brief_slot_capabilities(capability, selected_slots)
            metadata.update(
                {
                    "briefing_sections": list(selected_slots),
                    # The mention order IS the render order (roadmap "ordering" capture):
                    # the tick handler forwards section_order to render_brief_package.
                    "section_order": list(selected_slots),
                    "tool_callbacks": list(brief_slot_callbacks(capability, selected_slots)),
                    "template_confirmed": True,
                }
            )
        else:
            missing.append("briefing_sections")
    else:
        # Non-brief capability: routine binds to a tool on a regular skill.
        # Pick the tool semantically (or take the only one), then decide
        # whether args clarification is needed.
        match = _match_tool_in_skill(text, capability, router)
        if match is None:
            missing.append("tool_unavailable")
        else:
            manifest_tool, tool_class = match
            tool_name = manifest_tool.name
            required_args = _tool_required_args(manifest_tool, tool_class)
            metadata["bound_skill"] = capability.manifest.name
            metadata["bound_tool"] = tool_name
            metadata["tool_callbacks"] = [tool_name]
            metadata["template_confirmed"] = True
            required_capabilities = (capability.manifest.name, "channel_delivery")
            if required_args:
                # Surface the full ToolArg shape (options / default /
                # examples / type / prompt) so the clarification UX can
                # render enumerated choices without re-deriving them.
                metadata["pending_tool_args"] = [
                    arg.model_dump(mode="json") for arg in required_args
                ]
                missing.append("tool_args")
            else:
                source_preferences = (tool_name,)

    delivery_channel = _resolve_delivery_channel(lowered, origin_channel)

    if missing:
        return RoutineAuthoringResult(
            action="clarify",
            title=title,
            goal=_extract_goal(text, fallback=title or "Routine"),
            schedule=schedule,
            template=template,
            delivery_channel=delivery_channel,
            source_preferences=source_preferences,
            required_capabilities=required_capabilities,
            missing_slots=tuple(missing),
            reason=_clarify_reason(missing, capability, pending_tool_args=required_args),
            confidence=0.72,
            metadata=metadata,
        )

    return RoutineAuthoringResult(
        action="draft",
        title=title,
        goal=_extract_goal(text, fallback=title),
        schedule=schedule,
        template=template,
        delivery_channel=delivery_channel,
        source_preferences=source_preferences,
        required_capabilities=required_capabilities,
        confidence=0.86,
        metadata=metadata,
    )


def _looks_like_routine_request(lowered: str) -> bool:
    if not _ROUTINE_SIGNAL_RE.search(lowered):
        return False
    return bool(_ROUTINE_VERB_RE.search(lowered))


def _should_defer_to_intent_router(lowered: str) -> bool:
    if _HOW_TO_SETUP_RE.search(lowered):
        return True
    # One-off brief preview ("show me a sample of my daily brief") → render once
    # via the ReAct brief tool, NOT author a routine — unless explicit
    # creation/recurrence intent is also present ("...every morning").
    if (
        _BRIEF_PREVIEW_RE.search(lowered)
        and _BRIEF_NOUN_RE.search(lowered)
        and not _ROUTINE_RECURRENCE_RE.search(lowered)
    ):
        return True
    return bool(
        _UNSUPPORTED_EMAIL_WORKFLOW_DOMAIN_RE.search(lowered)
        and _UNSUPPORTED_EMAIL_WORKFLOW_ACTION_RE.search(lowered)
    )


_SCHEDULE_PHRASE_PATTERNS: tuple[re.Pattern[str], ...] = (
    _TIME_WITH_PREFIX_RE,
    _TIME_WITH_MERIDIEM_RE,
    re.compile(r"\bevery\s+\d+\s+(?:minutes?|hours?)\b", re.IGNORECASE),
    re.compile(
        r"\bevery\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:hourly|weekdays?|weekly|each\s+week|everyday|every\s+day|daily|each\s+day"
        r"|every\s+morning|each\s+morning|every\s+evening|every\s+night)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:in\s+the\s+)?(?:morning|afternoon|evening|night|noon|midnight)\b", re.IGNORECASE
    ),
)


def _strip_schedule_phrasing(message: str) -> str:
    """Remove scheduling / temporal phrasing so slot matching scores only content.

    The schedule clause ("every day at 8am") is boilerplate, not a content
    request, yet time tokens weakly resemble time-bearing slots (e.g. reminders).
    Stripping it before :func:`match_brief_slots` keeps section selection tight —
    e.g. "every day at 8am send me a brief with the top 10 stocks" selects only
    ``stocks``, not ``reminders``.
    """
    cleaned = message
    for pattern in _SCHEDULE_PHRASE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip() or message


def _extract_schedule(lowered: str) -> str:
    hour, minute = _extract_time(lowered)
    interval_match = re.search(r"\bevery\s+(?P<count>\d+)\s+minutes?\b", lowered)
    if interval_match:
        return f"interval:{int(interval_match.group('count')) * 60}"
    interval_match = re.search(r"\bevery\s+(?P<count>\d+)\s+hours?\b", lowered)
    if interval_match:
        return f"interval:{int(interval_match.group('count')) * 3600}"
    if re.search(r"\bhourly\b", lowered):
        return "interval:3600"
    if re.search(r"\bweekdays?\b", lowered):
        return f"cron:{minute} {hour} * * 1-5"
    weekday = _extract_weekday(lowered)
    if weekday is not None:
        return f"cron:{minute} {hour} * * {weekday}"
    if re.search(r"\b(weekly|each week)\b", lowered):
        return f"cron:{minute} {hour} * * 1"
    if re.search(r"\b(everyday|every day|daily|each day|every morning)\b", lowered):
        return f"daily:{hour:02d}:{minute:02d}"
    return ""


def _extract_time(lowered: str) -> tuple[int, int]:
    match = _TIME_WITH_PREFIX_RE.search(lowered) or _TIME_WITH_MERIDIEM_RE.search(lowered)
    if match is not None:
        hour = int(match.group("hour"))
        minute = int(match.group("minute") or 0)
        meridiem = (match.group("meridiem") or "").lower()
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        return hour, minute
    if "morning" in lowered:
        return 8, 0
    if "afternoon" in lowered:
        return 13, 0
    if "evening" in lowered:
        return 18, 0
    if "night" in lowered:
        return 20, 0
    return 9, 0


def _extract_weekday(lowered: str) -> int | None:
    for name, value in _WEEKDAYS.items():
        if re.search(rf"\b{name}s?\b", lowered):
            return value
    return None


# ---------------------------------------------------------------------------
# Section-aware helpers used by the clarification reply flow in bootstrap.
# ---------------------------------------------------------------------------


def extract_briefing_sections(
    text: str,
    package: SkillPackage,
    router: Any = None,
) -> tuple[str, ...]:
    """Pick which slots a user's clarification reply asks for."""

    return match_brief_slots(_strip_schedule_phrasing(text), package, router)


def normalize_briefing_sections(
    value: object,
    package: SkillPackage | None = None,
) -> tuple[str, ...]:
    """Coerce persisted section metadata into known slot keys.

    When *package* is provided, only slot keys present in the brief's
    manifest are returned. When *package* is ``None``, every string-like
    value is kept verbatim (used by the display path that just renders
    whatever was persisted earlier).
    """

    if isinstance(value, str):
        candidates = (item.strip() for item in value.split(","))
    elif isinstance(value, (list, tuple, set)):
        candidates = (str(item) for item in value)
    else:
        return ()

    valid_keys: set[str] | None = None
    if package is not None:
        valid_keys = set(brief_tool_slot_keys(package))

    seen: list[str] = []
    for candidate in candidates:
        key = candidate.strip()
        if not key:
            continue
        if valid_keys is not None and key not in valid_keys:
            continue
        if key not in seen:
            seen.append(key)
    return tuple(seen)


# ---------------------------------------------------------------------------
# Deterministic value extractors (precise grammar — not domain classifiers).
# ---------------------------------------------------------------------------


def extract_repo_brief_limit(text: str) -> int:
    """Extract a requested top-N count for repository briefs."""

    match = re.search(r"\btop\s+(?P<count>\d{1,3})\b", text.lower())
    if match is None:
        return 10
    return max(1, min(25, int(match.group("count"))))


_DELIVERY_VERB = (
    # Phrases that signal "use channel X for delivery". Matching a bare
    # channel word ("web", "voice") without one of these triggered false
    # positives — e.g. "don't search the web" got read as a routine
    # delivery refinement. Require a delivery verb / preposition before
    # or after the channel name.
    r"\b(?:deliver(?:ed|y|ies)?(?:\s+(?:to|via|on|over|through|by))?"
    r"|send(?:s|ing)?(?:\s+(?:to|via|on|over|through|by))?"
    r"|push(?:es|ed|ing)?(?:\s+(?:to|via|on|over|through|by))?"
    r"|notify(?:\s+(?:via|on|over|through|by))?"
    r"|over|via|on|through"
    r"|channel|gateway)\b"
)


def extract_routine_delivery_channel(text: str) -> str:
    """Extract an explicitly requested delivery channel from a user utterance.

    Requires an explicit delivery-context phrase (``deliver to``, ``send via``,
    ``over telegram``, ``channel = web``, etc.) before treating a channel
    word as a refinement. Bare mentions like ``the web`` / ``a voice
    note`` no longer trip a false positive.
    """

    lowered = text.lower()
    candidates = (
        ("telegram", r"\btelegram\b"),
        ("console", r"\b(?:console|terminal|cli)\b"),
        ("web", r"\b(?:web|browser)\b"),
        ("voice", r"\b(?:voice|speaker)\b"),
    )
    for channel, word_re in candidates:
        for match in re.finditer(word_re, lowered):
            window_start = max(0, match.start() - 25)
            window_end = min(len(lowered), match.end() + 25)
            window = lowered[window_start:window_end]
            if re.search(_DELIVERY_VERB, window):
                return channel
    return ""


def extract_content_style(text: str) -> dict[str, object]:
    """Extract deterministic content-shape preferences from a user utterance.

    Requires the explicit ``<N> lines per <unit>`` form. The previous
    bare ``<N> lines`` fallback over-matched on natural phrases like
    "in 5 lines or fewer", which short-circuited the chat flow into
    the routine-authoring path. If the user wants the bare form
    inferred, they can say "3 lines per item" (the default unit).
    """

    lowered = text.lower()
    match = re.search(
        r"\b(?P<count>\d{1,2})\s+lines?\s+per\s+"
        r"(?P<unit>news|item|items|repo|repos|repository|repositories|story|stories)\b",
        lowered,
    )
    if match is None:
        return {}
    count = max(1, min(10, int(match.group("count"))))
    unit = _normalize_content_unit(match.group("unit"))
    return {
        "content_lines_per_item": count,
        "content_unit": unit,
        "content_style": f"{count} lines per {unit}",
    }


# Per-routine presentation capture (roadmap "next slice"): a quoted header/footer line and
# a tone keyword. Section ORDER is captured separately from the matched-slot mention order.
_HEADER_RE = re.compile(
    r"\b(?:header|greeting|salutation|start(?:ing)?|begin(?:ning)?|open(?:ing)?)\b"
    r"[^'\"\n]{0,24}['\"]([^'\"\n]{1,80})['\"]",
    re.IGNORECASE,
)
_FOOTER_RE = re.compile(
    r"\b(?:footer|sign[\s-]?off|signoff|closing|close|end(?:ing)?|finish|outro)\b"
    r"[^'\"\n]{0,24}['\"]([^'\"\n]{1,80})['\"]",
    re.IGNORECASE,
)
# NB: brevity is phrase-anchored — a bare "brief" matches the NOUN ("morning brief"), which
# would wrongly read as a tone (and, via extract_routine_refinements, even mis-route a turn).
_TONE_MAP: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(concise|succinct|terse|tl;?dr)\b"
            r"|\b(?:keep|make|leave) it (?:brief|short|concise)\b"
            r"|\bbe (?:brief|short|concise)\b"
            r"|\b(?:brief|short|concise|terse) tone\b",
            re.IGNORECASE,
        ),
        "brief",
    ),
    (re.compile(r"\b(formal|professional|business[- ]?like)\b", re.IGNORECASE), "formal"),
    (re.compile(r"\b(casual|friendly|warm|informal|chatty)\b", re.IGNORECASE), "casual"),
    (re.compile(r"\b(detailed|verbose|thorough|in[\s-]?depth)\b", re.IGNORECASE), "detailed"),
)


def extract_formatting(text: str) -> dict[str, object]:
    """Deterministic presentation capture: a quoted ``header``/``footer`` line and a ``tone``
    keyword. Returns only the keys it found (the names ``_routine_render_params`` reads)."""
    out: dict[str, object] = {}
    if (m := _HEADER_RE.search(text)) is not None:
        out["header"] = m.group(1).strip()
    if (m := _FOOTER_RE.search(text)) is not None:
        out["footer"] = m.group(1).strip()
    for pattern, tone in _TONE_MAP:
        if pattern.search(text):
            out["tone"] = tone
            break
    return out


def extract_routine_refinements(text: str) -> dict[str, object]:
    """Extract deterministic routine refinements that do not require an LLM."""

    refinements: dict[str, object] = {}
    delivery_channel = extract_routine_delivery_channel(text)
    if delivery_channel:
        refinements["delivery_channel"] = delivery_channel
    refinements.update(extract_content_style(text))
    refinements.update(extract_formatting(text))
    return refinements


_POSITION_WORDS: dict[str, int] = {
    "first": 0,
    "1st": 0,
    "1": 0,
    "#1": 0,
    "one": 0,
    "second": 1,
    "2nd": 1,
    "2": 1,
    "#2": 1,
    "two": 1,
    "third": 2,
    "3rd": 2,
    "3": 2,
    "#3": 2,
    "three": 2,
    "fourth": 3,
    "4th": 3,
    "4": 3,
    "#4": 3,
    "four": 3,
    "fifth": 4,
    "5th": 4,
    "5": 4,
    "#5": 4,
    "five": 4,
}


_CAPABILITY_SWAP_VERB_RE = re.compile(
    r"\b(switch|swap|change|move|use|bind|replace|rebind)\b",
    re.IGNORECASE,
)


def detect_capability_swap_intent(
    message: str,
    candidate_packages: Sequence[SkillPackage],
    router: Any,
) -> SkillPackage | None:
    """Return the proposed skill when the user wants to rebind a routine.

    Two-stage check so this stays cheap on every incoming chat turn:
    1. Cheap regex precheck for swap verbs — "switch", "change", "use",
       "move", "replace", "rebind". Returns ``None`` quickly otherwise.
    2. Semantic-route the message against the candidate packages and
       only propose a swap when the best match clears the router's own
       threshold (``router.best_match`` returns ``None`` below it).

    Capability changes are large — Phase B requires explicit user
    confirmation before applying, so the caller treats this return as
    a *proposal*, not an immediate action.
    """

    if router is None or not candidate_packages:
        return None
    if not message.strip():
        return None
    if not _CAPABILITY_SWAP_VERB_RE.search(message):
        return None
    return cast("SkillPackage | None", router.best_match(message, candidate_packages))


def resolve_routine_pick(
    reply: str,
    candidates: Sequence[RoutineSpec],
) -> RoutineSpec | None:
    """Match a user reply to one of a small list of routines.

    Phase B disambiguation: when several scheduled routines could be
    the refinement target, the agent renders a numbered list and asks
    the user to pick. This helper resolves the reply against the same
    list using simple deterministic strategies — the user reply is
    typically very short ("first", "r1-12345", "morning briefing")
    and an LLM call would be overkill. Falls through to ``None`` when
    the reply is empty, ambiguous, or matches nothing — the caller
    re-asks.

    Strategies, in order:
    1. Exact routine id (case-insensitive).
    2. Routine id prefix when the reply is >= 8 characters and uniquely
       matches one candidate.
    3. Position keyword ("first", "1", "#2", "third", "one", ...).
    4. Unique title substring (case-insensitive).
    """

    if not candidates:
        return None
    text = " ".join(reply.strip().lower().split())
    if not text:
        return None

    # 1. exact id match
    for spec in candidates:
        if spec.id.lower() == text:
            return spec

    # 2. id prefix (require >= 8 chars + uniqueness)
    if len(text) >= 8:
        prefix_matches = [s for s in candidates if s.id.lower().startswith(text)]
        if len(prefix_matches) == 1:
            return prefix_matches[0]

    # 3. position keyword
    # Strip leading articles ("the first", "the 2nd one") to expose the token.
    stripped = re.sub(r"^(the\s+|pick\s+|number\s+)", "", text)
    stripped = re.sub(r"\s+one$", "", stripped).strip()
    if stripped in _POSITION_WORDS:
        idx = _POSITION_WORDS[stripped]
        if 0 <= idx < len(candidates):
            return candidates[idx]

    # 4. unique title substring
    title_matches = [s for s in candidates if s.title and text in s.title.lower()]
    if len(title_matches) == 1:
        return title_matches[0]

    return None


def _normalize_content_unit(unit: str) -> str:
    lowered = unit.lower()
    if lowered in {"repo", "repos", "repository", "repositories"}:
        return "repository"
    if lowered in {"story", "stories"}:
        return "story"
    if lowered in {"items"}:
        return "item"
    return lowered


def _extract_goal(text: str, *, fallback: str) -> str:
    cleaned = re.sub(r"\b(please|hey|iris)\b", "", text, flags=re.IGNORECASE).strip(" ,.")
    return cleaned[:240] or fallback


def _resolve_delivery_channel(lowered: str, origin: str | None) -> str:
    explicit = extract_routine_delivery_channel(lowered)
    if explicit:
        return explicit
    if origin:
        return origin
    return "console"


def format_tool_arg_prompt_line(arg: ToolArg) -> str:
    """Render one ``ToolArg`` as a bullet line for the clarification message.

    Honors the optional ``prompt`` override verbatim. Otherwise composes
    a per-type rendering: enums enumerate their options, numeric args
    show their range and default, bools show ``yes / no``, and strings
    fall back to the description (plus examples when present).
    """

    if arg.prompt:
        return f"- **{arg.name}**: {arg.prompt}"

    if arg.type == "enum":
        opts = " / ".join(f"`{opt}`" for opt in arg.options)
        suffix = f" (default `{arg.default}`)" if arg.default is not None else ""
        return f"- **{arg.name}**: {opts}{suffix}"

    if arg.type in {"int", "number"}:
        kind = "an integer" if arg.type == "int" else "a number"
        bounds: list[str] = []
        if arg.min is not None and arg.max is not None:
            bounds.append(f"{arg.min}–{arg.max}")
        elif arg.min is not None:
            bounds.append(f">= {arg.min}")
        elif arg.max is not None:
            bounds.append(f"<= {arg.max}")
        if arg.default is not None:
            bounds.append(f"default {arg.default}")
        suffix = f" ({', '.join(bounds)})" if bounds else ""
        return f"- **{arg.name}**: {kind}{suffix}"

    if arg.type == "bool":
        suffix = f" (default `{arg.default}`)" if arg.default is not None else ""
        return f"- **{arg.name}**: `yes` / `no`{suffix}"

    # type == "string"
    body = arg.description or arg.name.replace("_", " ")
    if arg.examples:
        body += f" — e.g. {', '.join(arg.examples)}"
    return f"- **{arg.name}**: {body}"


def _clarify_reason(
    missing: list[str],
    capability: SkillPackage | None,
    *,
    pending_tool_args: Sequence[ToolArg] = (),
) -> str:
    if missing == ["schedule"]:
        return "I need when this routine should run. Try: every morning at 8 AM."
    if missing == ["template"]:
        return (
            "I need what capability the routine should run. Try naming a brief "
            "('morning briefing'), a tool ('list my reminders'), or any skill in "
            "this workspace."
        )
    if (
        "briefing_sections" in missing
        and capability is not None
        and capability.manifest.kind == "brief"
    ):
        slot_keys = brief_tool_slot_keys(capability)
        labels = ", ".join(brief_slot_label(k) for k in slot_keys) or "all"
        return (
            f"I need what the `{capability.manifest.name}` brief should include. "
            f"Reply with one or more: {labels}; or `all`."
        )
    if "briefing_sections" in missing:
        return "I need what the briefing should include. Reply with `all`, or list the sections you want."
    if "tool_args" in missing and pending_tool_args:
        skill_label = f"`{capability.manifest.name}`" if capability is not None else "this tool"
        lines = [f"To draft this {skill_label} routine I need a few choices:"]
        lines.extend(format_tool_arg_prompt_line(arg) for arg in pending_tool_args)
        lines.append("Reply with one of each — naturally is fine.")
        return "\n".join(lines)
    if "tool_args" in missing:
        return (
            "I need argument values for the bound tool before I can draft this routine. "
            "Tell me one at a time, or all at once."
        )
    if "tool_unavailable" in missing and capability is not None:
        return (
            f"The `{capability.manifest.name}` skill is matched but exposes no callable tools. "
            "Try a different skill."
        )
    return "I need a schedule and a supported capability. Try: every morning at 8 AM list my reminders."
