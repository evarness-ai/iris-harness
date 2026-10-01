"""Pluggable intent classification and routing for the IRIS agentic core."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from iris_harness.foundation.paths import config_path, default_config_dir
from iris_harness.kernel.governance import HookContext, HookPoint, kernel_from_env
from iris_harness.kernel.governance.evaluator.embeddings import cosine_similarity
from iris_harness.kernel.governance.turn_label import apply_turn_floor
from iris_harness.llm.client import GovernedPromptCall

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntentResult:
    """Output of intent classification."""

    intent: str
    agent_type: str
    confidence: float
    is_multi_step: bool = False
    raw_query: str = ""
    # Which path produced this result: "keyword" (regex matched), "llm"
    # (router LLM call), or "fallback" (low-confidence default).
    # Defaults to "" to preserve backward compat with classifiers that
    # don't set the field. Used by the router_audit to distinguish
    # cheap keyword hits from expensive LLM router calls.
    source: str = ""


@runtime_checkable
class IIntentClassifier(Protocol):
    """Protocol for swappable intent classifiers.

    ``context`` (issue 0002) carries recent conversation turns so a follow-up
    ("what's the summary?", "yes", "tell me more") can be routed by the
    conversation topic, not its isolated keywords. Optional + keyword-only so
    existing single-arg callers keep working.
    """

    def classify(self, query: str, *, context: str | None = None) -> IntentResult: ...


# The keyword classifier's regex rules live in ``config/intent_keywords.yaml``
# (``rules:``, loaded by ``load_keyword_rules``). They were the list
# ``_KEYWORD_RULES`` here until 2026-09-28 and moved verbatim.

# Multi-step cues live in ``config/multi_step.yaml`` (ADR-0111; the owner's rule:
# vocabulary in YAML, never in code). One loader, one compiled pattern, one
# ``is_multi_step_query`` shared by the keyword, LLM and semantic classifiers. A
# missing or empty file means no turn is compound — honest degradation, no list here.
MULTI_STEP_CUES_FILENAME = "multi_step.yaml"


def load_multi_step_cues(path: str | Path) -> tuple[str, ...]:
    """Cue phrases from ``multi_step.yaml`` (``cues: [...]``); ``()`` when absent/malformed."""
    try:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — missing/malformed file → no cues
        return ()
    cues = raw.get("cues") if isinstance(raw, dict) else None
    if not isinstance(cues, list):
        return ()
    return tuple(str(c).strip() for c in cues if str(c).strip())


def compile_multi_step_pattern(cues: Iterable[str]) -> re.Pattern[str] | None:
    """Word-bounded, case-insensitive alternation; ``" ... "`` inside a cue means
    "anything in between" (``first ... then``). None when there are no cues."""
    parts: list[str] = []
    for cue in cues:
        pieces = [re.escape(piece.strip()) for piece in cue.split("...") if piece.strip()]
        if pieces:
            parts.append(r".*".join(pieces))
    if not parts:
        return None
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


def _multi_step_pattern() -> re.Pattern[str] | None:
    # Resolved per call: config_dir() follows IRIS_CONFIG_DIR, which is deliberately not
    # cached (foundation/paths.py). The cache is keyed on the resolved file, so one
    # process that reads two config dirs -- the suite, where a test points
    # IRIS_CONFIG_DIR at a temp dir -- cannot keep the first dir's answer for the second.
    return _compiled_multi_step_cues(config_path(MULTI_STEP_CUES_FILENAME))


@lru_cache(maxsize=8)
def _compiled_multi_step_cues(path: Path) -> re.Pattern[str] | None:
    return compile_multi_step_pattern(load_multi_step_cues(path))


def is_multi_step_query(query: str) -> bool:
    """True when the turn contains a compound-request cue (see ``config/multi_step.yaml``)."""
    pattern = _multi_step_pattern()
    return bool(pattern and pattern.search((query or "").lower()))


# The keyword classifier's vocabulary lives in ``config/intent_keywords.yaml`` (owner
# rule: vocabulary in YAML, never in code): ``pre_rules`` (phrase lists) then ``rules``
# (the regex rules, in order). A config dir without the file (an operator's partial
# IRIS_CONFIG_DIR, a test's tmp dir) reads the shipped copy, so routing never silently
# loses its rules.
INTENT_KEYWORDS_FILENAME = "intent_keywords.yaml"


def _keywords_file() -> Path:
    """The configured ``intent_keywords.yaml``, else the shipped one."""
    path = config_path(INTENT_KEYWORDS_FILENAME)
    return path if path.is_file() else default_config_dir() / INTENT_KEYWORDS_FILENAME


@dataclass(frozen=True)
class KeywordRule:
    """One ``rules:`` entry: ``pattern`` found in the lower-cased message -> ``intent``."""

    pattern: re.Pattern[str]
    intent: str
    agent: str


def load_keyword_rules(path: str | Path) -> tuple[KeywordRule, ...]:
    """The ``rules:`` list of ``intent_keywords.yaml``, compiled, in file order.

    Strict: an unreadable file, a missing list, or any rule with a bad regex or an
    intent / agent the router does not know raises ``ValueError`` naming the rule —
    a silently dropped rule would reroute every turn it used to catch.
    """
    import yaml

    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: cannot read keyword rules: {exc}") from exc
    rules = raw.get("rules") if isinstance(raw, dict) else None
    if not isinstance(rules, list) or not rules:
        raise ValueError(f"{path}: 'rules' must be a non-empty list")
    out: list[KeywordRule] = []
    for n, entry in enumerate(rules, 1):
        where = f"{path}: rule {n}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: must be a mapping")
        intent = str(entry.get("intent") or "").strip()
        agent = str(entry.get("agent") or "").strip()
        fragments = entry.get("pattern")
        if isinstance(fragments, str):
            fragments = [fragments]
        if intent not in _VALID_INTENTS:
            raise ValueError(f"{where}: unknown intent {intent!r}")
        if agent not in _VALID_AGENT_TYPES:
            raise ValueError(f"{where} ({intent}): unknown agent {agent!r}")
        if not isinstance(fragments, list) or not fragments:
            raise ValueError(f"{where} ({intent}): 'pattern' must be a list of fragments")
        try:
            pattern = re.compile("".join(str(f) for f in fragments))
        except re.error as exc:
            raise ValueError(f"{where} ({intent}): bad regex: {exc}") from exc
        out.append(KeywordRule(pattern, intent, agent))
    return tuple(out)


@lru_cache(maxsize=1)
def _keyword_rules() -> tuple[KeywordRule, ...]:
    path = _keywords_file()
    shipped = default_config_dir() / INTENT_KEYWORDS_FILENAME
    try:
        return load_keyword_rules(path)
    except ValueError:
        if path == shipped:
            raise  # a broken shipped file is a bug; tests must see it
        logger.exception("keyword rules: %s refused; using the shipped %s", path, shipped)
        return load_keyword_rules(shipped)


@dataclass(frozen=True)
class KeywordPreRule:
    """One YAML pre-rule: ``any`` phrase present and no ``unless`` phrase -> ``intent``."""

    intent: str
    any: re.Pattern[str]
    unless: re.Pattern[str] | None = None


def _phrase_pattern(phrases: Iterable[object]) -> re.Pattern[str] | None:
    parts = [r"\s+".join(re.escape(w) for w in str(p).split()) for p in phrases if str(p).strip()]
    if not parts:
        return None
    parts.sort(key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


def load_keyword_pre_rules(path: str | Path) -> tuple[KeywordPreRule, ...]:
    """``pre_rules`` from ``intent_keywords.yaml``; ``()`` when absent or malformed.

    A rule naming an intent the router does not know, or with no ``any`` phrase, is
    skipped (logged) rather than failing the router.
    """
    try:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — missing/malformed file → no pre-rules
        return ()
    rules = raw.get("pre_rules") if isinstance(raw, dict) else None
    if not isinstance(rules, list):
        return ()
    out: list[KeywordPreRule] = []
    for entry in rules:
        if not isinstance(entry, dict):
            continue
        intent = str(entry.get("intent") or "").strip().lower()
        any_pattern = _phrase_pattern(entry.get("any") or [])
        if intent not in _VALID_INTENTS or any_pattern is None:
            logger.warning("intent_keywords.yaml: skipping pre-rule %r", entry)
            continue
        out.append(KeywordPreRule(intent, any_pattern, _phrase_pattern(entry.get("unless") or [])))
    return tuple(out)


@lru_cache(maxsize=1)
def _keyword_pre_rules() -> tuple[KeywordPreRule, ...]:
    return load_keyword_pre_rules(_keywords_file())


def _pre_rule_intent(lowered: str, rules: Iterable[KeywordPreRule]) -> str | None:
    for rule in rules:
        if rule.any.search(lowered) and not (rule.unless and rule.unless.search(lowered)):
            return rule.intent
    return None


# Conversational catch-all intents. A keyword match on these is weak signal, so
# it's scored below the KeywordFirst threshold (0.8) to defer to the LLM router;
# precise domain intents (email, coding, calendar, …) stay authoritative at 0.85.
_AMBIGUOUS_INTENTS = frozenset({"general", "help"})


class KeywordClassifier:
    """Fast keyword-based intent classifier — no LLM required."""

    def classify(self, query: str, *, context: str | None = None) -> IntentResult:
        # Keyword rules are regex over the query only; context is for the LLM path.
        del context
        lowered = query.lower()
        is_multi = is_multi_step_query(lowered)
        pre = _pre_rule_intent(lowered, _keyword_pre_rules())
        if pre is not None:
            return IntentResult(
                intent=pre,
                agent_type=agent_for_intent(pre),
                confidence=0.85,
                is_multi_step=is_multi,
                raw_query=query,
                source="keyword",
            )
        for rule in _keyword_rules():
            if rule.pattern.search(lowered):
                return IntentResult(
                    intent=rule.intent,
                    agent_type=rule.agent,
                    confidence=0.5 if rule.intent in _AMBIGUOUS_INTENTS else 0.85,
                    is_multi_step=is_multi,
                    raw_query=query,
                    source="keyword",
                )
        return IntentResult(
            intent="general",
            agent_type="system",
            confidence=0.5,
            is_multi_step=is_multi,
            raw_query=query,
            source="fallback",
        )


class LLMClassifier:
    """LLM-backed classifier using a provided callable."""

    def __init__(self, llm_call: Callable[[str], str]) -> None:
        self._llm = llm_call
        self._keyword_fallback = KeywordClassifier()
        self._kernel = kernel_from_env()

    def classify(self, query: str, *, context: str | None = None) -> IntentResult:
        convo = f"Conversation so far:\n{context}\n\n" if context else ""
        prompt = (
            "Classify this user query. Reply with exactly:\n"
            "INTENT: <intent_label>\nAGENT: <agent_type>\nMULTI_STEP: <yes|no>\n\n"
            f"{convo}Query: {query}\n\n"
            "Use these agent types: email, calendar, filemanager, coding_agent, rag, planner, system\n"
            "Use these intents: communication, calendar, files, coding, search, planner, help, general\n"
            "Use 'planner' for whole-day synthesis ('plan my day', 'what should I focus on').\n"
            "If a follow-up depends on the conversation, classify by its topic."
        )
        try:
            raw = self._invoke_with_governance(prompt).strip()
            intent = _extract_field(raw, "INTENT") or "general"
            agent = _extract_field(raw, "AGENT") or "system"
            multi = (_extract_field(raw, "MULTI_STEP") or "no").lower() == "yes"
            return IntentResult(
                intent=intent,
                agent_type=agent,
                confidence=0.9,
                is_multi_step=multi,
                raw_query=query,
                source="llm",
            )
        except Exception as exc:
            # Routing still works, but on the keyword rules: say so, or a dead model or
            # a governance block looks like ordinary keyword routing. Exception type
            # only at WARNING: the prompt carries the user's query, and a message can
            # echo its input; the traceback goes to DEBUG.
            logger.warning(
                "intent router: LLM classify failed (%s); using the keyword router",
                type(exc).__name__,
            )
            logger.debug("intent router: LLM classify failure", exc_info=True)
            return self._keyword_fallback.classify(query)

    def _invoke_with_governance(self, prompt: str) -> str:
        # A GovernedPromptCall governs itself, at the tier it goes to (llm/client.py).
        if self._kernel is None or isinstance(self._llm, GovernedPromptCall):
            return self._llm(prompt)

        run_id = str(uuid.uuid4())
        classify_ctx = HookContext(
            hook_point=HookPoint.PRE_CLASSIFY,
            run_id=run_id,
            agent_type="intent_router",
            payload={"prompt": prompt},
        )
        classify_decision, classified_ctx = self._kernel.fire_sync(
            HookPoint.PRE_CLASSIFY, classify_ctx
        )
        if classify_decision.outcome in ("deny", "require_approval"):
            raise RuntimeError(f"governance blocked classifier call: {classify_decision.reason}")

        llm_ctx = HookContext(
            hook_point=HookPoint.PRE_LLM_CALL,
            run_id=run_id,
            agent_type="intent_router",
            # Floored at the turn's label: this prompt may look tamer than the data the
            # turn holds (kernel/governance/turn_label.apply_turn_floor).
            classification=apply_turn_floor(classified_ctx.classification),
            # An opaque callable: where it sends the prompt is unknown, so the call is
            # governed as leaving the machine (fail closed), not guessed to be local.
            tier="tier_3",
            payload={"prompt": prompt},
        )
        decision, _ = self._kernel.fire_sync(HookPoint.PRE_LLM_CALL, llm_ctx)
        if decision.outcome in ("deny", "require_approval"):
            raise RuntimeError(f"governance blocked classifier call: {decision.reason}")

        return self._llm(prompt)


class HybridClassifier:
    """Keyword-first, LLM fallback classifier."""

    def __init__(self, llm_call: Callable[[str], str] | None = None) -> None:
        self._keyword = KeywordClassifier()
        self._llm: LLMClassifier | None = LLMClassifier(llm_call) if llm_call else None

    def classify(self, query: str, *, context: str | None = None) -> IntentResult:
        result = self._keyword.classify(query)
        if result.confidence >= 0.8 or self._llm is None:
            return result
        # Uncertain keyword → defer to the LLM, passing conversation context so a
        # follow-up is routed by topic (issue 0002).
        return self._llm.classify(query, context=context)


class KeywordFirstClassifier:
    """Trust the keyword classifier on high-confidence matches; otherwise defer.

    The keyword rules are deterministic and high-precision for the cases they
    match. Small Tier-1 LLMs sometimes misclassify those same queries
    (e.g. routing ``"create a pdf of news"`` to ``rag`` instead of ``code_exec``).
    Running keywords first avoids that noise and skips the LLM call entirely
    when the rule already fires.
    """

    def __init__(
        self,
        secondary: IIntentClassifier,
        *,
        keyword: IIntentClassifier | None = None,
        threshold: float = 0.8,
    ) -> None:
        self._keyword = keyword or KeywordClassifier()
        self._secondary = secondary
        self._threshold = threshold

    def classify(self, query: str, *, context: str | None = None) -> IntentResult:
        # Keyword-first: a confident keyword match wins even on follow-ups (e.g.
        # "any finance emails?" → email), so we don't hand keyword-clear turns to
        # a weak router LLM. When the keyword is uncertain we defer to the
        # secondary AND pass the conversation context, so a referential follow-up
        # ("what's the summary?") is routed by topic instead of isolated keywords
        # (issue 0002).
        result = self._keyword.classify(query)
        if result.confidence >= self._threshold:
            return result
        return self._secondary.classify(query, context=context)


_VALID_INTENTS: frozenset[str] = frozenset(
    {
        "communication",
        "calendar",
        "calendar_create",
        "files",
        "coding",
        "code_exec",
        "search",
        "weather",
        "system",
        "help",
        "general",
        "profile_query",
        "planner",
        "finance",
        "clarify",
        # An imperative request to DO something ("update my briefing", "configure X")
        # that has no stronger domain home. Routed to the capable instruct tier so a
        # multi-step action reasons + tool-calls properly instead of landing on the
        # weak tier-1 executor (ADR-0103).
        "action",
    }
)
_VALID_AGENT_TYPES: frozenset[str] = frozenset(
    {
        "email",
        "calendar",
        "filemanager",
        "research",
        "coding_agent",
        "code_exec",
        "rag",
        "system",
        "planner",
        "finance",
        "clarify",
    }
)

# Canonical intent → agent mapping. Small Tier-1 routers occasionally emit a
# valid intent paired with the wrong agent (e.g. ``code_exec`` + ``rag``);
# that mismatch silently routes the request to the system handler. Enforcing
# consistency lets us catch this and fall back to the keyword classifier.
_INTENT_TO_AGENT: dict[str, str] = {
    "communication": "email",
    "calendar": "calendar",
    # calendar_create is consumed by the create short-circuit; if it reaches the
    # pipeline (e.g. no parseable time) it routes to the read-only calendar agent.
    "calendar_create": "calendar",
    "files": "filemanager",
    "coding": "coding_agent",
    "code_exec": "code_exec",
    # Web/factual lookups → system handler (has research). RAG (own-docs) isn't
    # a registered chat agent, so search must not route there.
    "search": "system",
    "weather": "system",
    "system": "system",
    "help": "system",
    "general": "system",
    "profile_query": "system",
    "planner": "planner",
    "finance": "finance",
    "clarify": "clarify",
    # Generic actions run on the system agent (full ReAct tool set); the `action`
    # intent only changes the *tier* (→ tier2), not the agent's capabilities.
    "action": "system",
}

# FMX9: opt-in remap of the `search` intent to a dedicated research agent.
_RESEARCH_AGENT_ENV = "IRIS_RESEARCH_AGENT"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def agent_for_intent(intent: str) -> str:
    """Resolve an intent to its agent — the single seam over ``_INTENT_TO_AGENT``.

    When ``IRIS_RESEARCH_AGENT`` is truthy (read at *call* time so the flag can be
    flipped live, e.g. by the experiment console or tests), ``search`` remaps to
    the dedicated ``research`` agent (FMX9). Everything else — including the
    ``"system"`` default for unknown intents — is byte-identical to the canonical
    mapping, so the flag off means today's routing exactly.
    """
    if intent == "search" and os.environ.get(_RESEARCH_AGENT_ENV, "").strip().lower() in _TRUTHY:
        return "research"
    return _INTENT_TO_AGENT.get(intent, "system")


# Imperative MUTATE verbs — the signal that a turn wants the agent to DO something,
# not just read/answer. Read verbs (show/list/tell/what/how-many) are deliberately
# excluded so "what are my dues" stays a read while "add dues to my brief" is an action.
_ACTION_VERB_RE = re.compile(
    r"\b(update|updating|add|adding|change|changing|modify|edit|editing|"
    r"set\s+up|setup|set|configure|enable|disable|turn\s+(?:on|off)|remove|delete|"
    r"rename|move|include|exclude|adjust|append|attach|register|"
    r"reschedule|rebalance|cancel|"
    # Trigger/automation + create-style actions (the user's "remind me when I get
    # this email", "create a notification when …"). Domain actions (calendar
    # 'remind', comms 'send', code_exec 'create a pdf') already route to a capable
    # tier, so listing them here only ever PROMOTES the weak-tier strays.
    r"remind|create|make|build|schedule|notify|alert|draft|send|book|save|"
    r"track|monitor|watch|subscribe|unsubscribe|mute|snooze|archive|flag|"
    r"label|tag|pin|automate)\b",
    re.IGNORECASE,
)

# Intents that resolve to the weak tier-1 executor. An action that lands here is the
# case we promote — a domain action (calendar/communication/coding/finance) already
# routes to a capable tier, so we leave it alone.
_WEAK_TIER_INTENTS: frozenset[str] = frozenset(
    {"general", "system", "help", "simple_query", "voice_command"}
)

# An action whose OBJECT is an IRIS-managed config surface (the briefing, settings,
# preferences) belongs on the system/action agent — which owns the configure tools —
# even when a domain noun ("dues") pulls the classifier toward that domain. So
# "add the dues section to my daily briefing" routes to `action`, not `finance`.
_CONFIG_OBJECT_RE = re.compile(
    r"\b(brief|briefing|digest|settings?|preferences?|config(?:uration)?)\b",
    re.IGNORECASE,
)


def is_action_request(message: str) -> bool:
    """True when the turn is an imperative request to DO/CHANGE something (ADR-0103).

    Keys on mutate verbs (update/add/configure/remove/…), NOT on polite framing
    ("can you") alone — "can you tell me my dues" is a read, "can you add dues to my
    brief" is an action. Used by (1) the deterministic read-intercepts, which must
    never swallow an action, and (2) the post-classification action promotion.
    """
    return bool(message) and bool(_ACTION_VERB_RE.search(message))


def promote_to_action_intent(result: IntentResult, message: str) -> IntentResult:
    """Relabel a weak-tier turn to the ``action`` intent when it's really an action.

    Predictive escalation (ADR-0103): an imperative ("update my briefing to add the
    outstanding dues") that the classifier dropped into general/system/help would run
    on tier-1 granite4 and reason/tool-call poorly. Promote it to ``action`` so the
    tier map starts it on tier-2 (instruct). Domain actions already on a capable tier
    (finance/calendar/communication/coding/…) are left untouched — EXCEPT a
    config-surface action ("add the dues section to my briefing"), which a domain noun
    misroutes to the domain agent that lacks the configure tools; those go to `action`
    (system agent) regardless of tier.
    """
    if not is_action_request(message):
        return result
    config_action = bool(_CONFIG_OBJECT_RE.search(message))
    if result.intent in _WEAK_TIER_INTENTS or config_action:
        return replace(
            result,
            intent="action",
            agent_type=agent_for_intent("action"),
            source=(result.source + "+action").lstrip("+"),
        )
    return result


# The router LLM only picks the intent label. The agent and multi-step flag are
# derived deterministically in Python (_INTENT_TO_AGENT + is_multi_step_query),
# which keeps the JSON a single key — far more reliable on a small 3B model.
_ROUTER_SYSTEM_PROMPT_TEMPLATE = (
    "You are {agent_name}'s request router. Pick the single best intent for the "
    "user message.\n"
    "\n"
    "Output JSON and NOTHING else (no prose, no markdown fences):\n"
    '{"intent": "<one of: communication, calendar, files, coding, code_exec, '
    "search, weather, system, help, general, profile_query, planner, finance, "
    'clarify>"}\n'
    "\n"
    "If a 'Conversation so far' block is present, the user message may be a "
    "FOLLOW-UP that depends on it (e.g. 'what's the summary?', 'yes', 'tell me "
    "more', 'what about finance?'). Route such follow-ups by the CONVERSATION "
    "TOPIC, not the isolated words — e.g. 'what's the summary?' right after the "
    "assistant listed an email is 'communication', not 'files'.\n"
    "\n"
    "Intent guide:\n"
    "The domain intents (communication, calendar, planner, finance, files, "
    "profile_query) are for the user's OWN data: their inbox, schedule, tasks, money, "
    "files or profile. A message that only MENTIONS a topic is 'general': small talk, "
    "greetings, thanks, jokes, stories, poems, advice, recommendations, or explaining "
    "a concept (e.g. how interest or credit scores work).\n"
    "- 'code_exec': produce a concrete artifact (PDF, Excel, CSV, image, chart) "
    "or actually RUN a script/calculation. Examples: 'create a pdf summary of X', "
    "'plot Y as a chart', 'convert this JSON to CSV', 'calculate the IRR'.\n"
    "- 'coding': writing, editing, debugging, refactoring code or building software "
    "(CLIs, agents, services) — when the user wants the SOURCE CODE itself.\n"
    "- 'communication': the user's EMAIL / Gmail / mailbox / inbox / messages — "
    "read, search, summarize, draft, reply, send. ANY ask about their emails — "
    "including 'any <topic> emails?', 'emails about/from X', 'finance/work/AI "
    "emails', 'summarize that email' — is communication (it reads or filters the "
    "inbox), even when <topic> names another domain like finance or work.\n"
    "- 'calendar': scheduling, appointments, reminders.\n"
    "- 'planner': whole-day synthesis — 'plan my day', 'what's on today', priorities.\n"
    "- 'finance': the user's OWN money — net worth, portfolio, holdings, investments, "
    "account/bank balances, their spending/expenses, their financial statements. "
    "Examples: 'what's my net worth', 'how's my portfolio doing', 'how much money do "
    "I have', 'how much did I spend last month'. NOT live market prices of a public "
    "ticker (that's 'search').\n"
    "- 'files': file/folder/document operations.\n"
    "- 'search': research / current info / looking things up on the web / factual "
    "lookups (e.g. 'what is AAPL stock worth', 'latest AI news'). The system "
    "handler fetches live data via research.\n"
    "- 'weather': weather/forecast/temperature questions.\n"
    "- 'profile_query': the user asks about THEMSELVES — name, role, preferences, "
    "stack, languages. Examples: 'what's my name', 'who am I', 'do I use Rust'.\n"
    "- 'system': general chat, time/date, identity, capabilities, IRIS itself.\n"
    "- 'help': the user asks for help or how to use something.\n"
    "- 'clarify': the message is genuinely too ambiguous to route — even given "
    "the conversation you cannot tell what the user wants. Use SPARINGLY; prefer "
    "a concrete intent whenever the topic is reasonably clear.\n"
    "- 'general': chat that needs none of the user's data: greetings, thanks, jokes, "
    "creative writing, advice, recommendations, explaining a concept or a word. "
    "Examples: 'tell me a joke', 'thanks!', 'explain compound interest', 'suggest a "
    "name for my dog', 'write a poem about email'. Also anything else / unsure."
)


def _build_router_system_prompt(agent_name: str | None = None) -> str:
    """Return the router prompt using the configured agent name."""
    if agent_name is None:
        try:
            from iris_harness.memory.identity import load_agent_name

            agent_name = load_agent_name()
        except Exception:  # routing must survive identity read issues
            logger.exception("could not load agent name for router prompt")
            agent_name = "IRIS"
    cleaned = re.sub(r"[\r\n]+", " ", str(agent_name)).strip() or "IRIS"
    return _ROUTER_SYSTEM_PROMPT_TEMPLATE.replace("{agent_name}", cleaned)


_ROUTER_SYSTEM_PROMPT = _build_router_system_prompt("IRIS")


class LLMRouterClassifier:
    """LLM-driven intent classifier with structured JSON output.

    Calls a small fast model (Tier-1, e.g. llama3.2:3b) to classify the query.
    Falls back to ``KeywordClassifier`` on any error so /chat never hard-fails.
    """

    def __init__(
        self,
        invoke: Callable[[str, str], str],
        *,
        keyword_fallback: IIntentClassifier | None = None,
    ) -> None:
        self._invoke = invoke
        self._fallback: IIntentClassifier = keyword_fallback or KeywordClassifier()

    def classify(self, query: str, *, context: str | None = None) -> IntentResult:
        user_message = (
            f"Conversation so far:\n{context}\n\nUser message: {query}" if context else query
        )
        try:
            raw = self._invoke(_build_router_system_prompt(), user_message)
        except Exception:
            logger.exception("LLM router invocation failed; using keyword fallback")
            return self._fallback.classify(query, context=context)

        parsed = _parse_router_json(raw)
        if parsed is None:
            logger.debug(
                "LLM router returned unparseable response %r; using keyword fallback",
                raw[:160] if isinstance(raw, str) else raw,
            )
            return self._fallback.classify(query, context=context)

        # The model only supplies the intent label; the agent and multi-step flag
        # are derived deterministically (small models guess the map/flags badly).
        intent = str(parsed.get("intent", "")).strip().lower()
        if intent not in _VALID_INTENTS:
            logger.debug("LLM router returned unknown intent %r; using keyword fallback", intent)
            return self._fallback.classify(query, context=context)

        agent = agent_for_intent(intent)
        is_multi = is_multi_step_query(query)

        return IntentResult(
            intent=intent,
            agent_type=agent,
            confidence=0.9,
            is_multi_step=is_multi,
            raw_query=query,
            source="llm",
        )


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
# Thinking-by-default models (qwen3.x family) wrap output in <think> blocks;
# under a tight num_predict the closing tag may never arrive. Strip both
# forms BEFORE JSON extraction — exp-005 measured 100% silent keyword
# fallback (at 5.4s/request) without this.
_THINK_BLOCK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


def _parse_router_json(raw: str) -> dict[str, Any] | None:
    """Best-effort extract a JSON object from a router response."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = _THINK_BLOCK_RE.sub("", raw).strip()
    # Strip common markdown fences (```json ... ``` or ``` ... ```).
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        return None
    try:
        loaded = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


SemanticEmbedder = Callable[[str], Sequence[float]]
DEFAULT_INTENT_SEMANTIC_THRESHOLD = 0.42  # absolute floor (query vs short anchor)
DEFAULT_INTENT_SEMANTIC_MARGIN = 0.05  # winner must beat runner-up by this much


def load_intent_anchors(path: str | Path) -> dict[str, list[str]]:
    """Load per-intent anchor phrases from a YAML file (``intents: {name: [...]}``).

    Returns ``{}`` on a missing/malformed file so the caller degrades to the
    keyword/LLM router. Only list-valued, non-empty intents are kept.
    """
    try:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — missing/malformed file → no anchors
        return {}
    intents = raw.get("intents") if isinstance(raw, dict) else None
    if not isinstance(intents, dict):
        return {}
    out: dict[str, list[str]] = {}
    for intent, phrases in intents.items():
        if isinstance(phrases, list):
            cleaned = [str(p) for p in phrases if str(p).strip()]
            if cleaned:
                out[str(intent)] = cleaned
    return out


def load_intent_defer_anchors(path: str | Path) -> list[str]:
    """Load the ``defer:`` phrases from the anchors YAML (``[]`` if absent/malformed).

    They describe messages that need no domain (small talk, creative writing,
    explaining a concept). The semantic classifier never routes to them; when they
    are the closest match it hands the message to the keyword + LLM router, which can
    say ``general``. Without them MiniLM, which has no "general" class, sends
    "good morning" to planner ("plan my day") and "write a limerick about email" to
    communication on a shared word.
    """
    try:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — missing/malformed file → no defer anchors
        return []
    phrases = raw.get("defer") if isinstance(raw, dict) else None
    if not isinstance(phrases, list):
        return []
    return [str(p) for p in phrases if str(p).strip()]


class SemanticIntentClassifier:
    """Embedding-based intent classifier (opt-in, ``IRIS_INTENT_ROUTER_SEMANTIC``).

    Picks the intent whose anchor phrases are closest to the query by cosine
    similarity (local ONNX MiniLM-L6 — no generative LLM, no per-token cost). It is
    the **primary** classifier for every turn, but only *decides* when one intent is
    confidently closest (top score ≥ threshold AND beats the runner-up by margin);
    otherwise — or when the embedder is unavailable — it **defers to a fallback
    classifier** (the keyword + LLM router). ``defer_anchors`` compete in the ranking
    like an intent: when they are closest, or the winner doesn't beat them by the
    margin, it defers too. agent_type and is_multi_step are derived
    exactly as in the other classifiers, so its output is interchangeable.
    """

    def __init__(
        self,
        embedder: SemanticEmbedder,
        anchors: dict[str, list[str]],
        *,
        fallback: IIntentClassifier,
        threshold: float = DEFAULT_INTENT_SEMANTIC_THRESHOLD,
        margin: float = DEFAULT_INTENT_SEMANTIC_MARGIN,
        defer_anchors: Sequence[str] = (),
    ) -> None:
        self._embedder = embedder
        self._anchors = {i: tuple(a) for i, a in anchors.items() if i in _VALID_INTENTS and a}
        self._defer = tuple(p for p in defer_anchors if p.strip())
        self._fallback = fallback
        self._threshold = threshold
        self._margin = margin
        self._cache: dict[str, list[float]] = {}

    def _embed(self, text: str) -> list[float] | None:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        try:
            vec = list(self._embedder(text))
        except Exception:  # any embedder failure → defer to fallback
            logger.debug("semantic intent embedding failed", exc_info=True)
            return None
        if not vec:
            return None
        self._cache[text] = vec
        return vec

    def _best_intent(self, query: str) -> tuple[str, float] | None:
        q_vec = self._embed(query)
        if q_vec is None or not self._anchors:
            return None
        ranked: list[tuple[str | None, float]] = []
        groups: list[tuple[str | None, tuple[str, ...]]] = list(self._anchors.items())
        if self._defer:
            groups.append((None, self._defer))  # None = hand to the fallback
        for intent, phrases in groups:
            best = 0.0
            for phrase in phrases:
                a_vec = self._embed(phrase)
                if a_vec is not None:
                    best = max(best, cosine_similarity(q_vec, a_vec))
            ranked.append((intent, best))
        ranked.sort(key=lambda t: t[1], reverse=True)
        top_intent, top = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        if top_intent is not None and top >= self._threshold and top >= runner + self._margin:
            return top_intent, top
        return None

    def classify(self, query: str, *, context: str | None = None) -> IntentResult:
        decided = self._best_intent(query) if (query or "").strip() else None
        if decided is None:
            return self._fallback.classify(query, context=context)
        intent, score = decided
        return IntentResult(
            intent=intent,
            agent_type=agent_for_intent(intent),
            confidence=round(float(score), 3),
            is_multi_step=is_multi_step_query(query or ""),
            raw_query=query,
            source="semantic",
        )


@dataclass
class IntentRouter:
    """Routes queries to agents based on classified intent."""

    classifier: IIntentClassifier = field(default_factory=KeywordClassifier)
    _agent_registry: dict[str, Callable[[str], str]] = field(default_factory=dict)

    def register_agent(self, agent_type: str, handler: Callable[[str], str]) -> None:
        self._agent_registry[agent_type] = handler

    def route(
        self, query: str, *, context: str | None = None
    ) -> tuple[IntentResult, Callable[[str], str] | None]:
        result = self.classifier.classify(query, context=context)
        handler = self._agent_registry.get(result.agent_type)
        return result, handler


def _extract_field(text: str, field_name: str) -> str | None:
    match = re.search(rf"^{field_name}:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else None
