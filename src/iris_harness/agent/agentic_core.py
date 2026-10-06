"""IRIS Agentic Core — ReAct loop engine with pluggable tools and memory."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, NamedTuple

from iris_harness.agent.agent_executor import ActivityChunk, TraceChunk
from iris_harness.agent.answer_guard import (
    SCAFFOLD_ECHO_NOTE,
    UNBACKED_CLAIM_NOTE,
    Verdict,
    check_final_answer,
)
from iris_harness.agent.context_budget import (
    admit_memory_parts,
    fill_recent_turns,
    trim_history,
)
from iris_harness.agent.dateparse import resolved_dates_line
from iris_harness.agent.run_review import CompletedRun, review_completed_run
from iris_harness.agent.tool_runner import (
    GovernedToolRunner,
    ToolCall,
    governance_block_message,
)
from iris_harness.foundation.clock import local_now
from iris_harness.kernel.governance import (
    DataClassification,
    GovernanceKernel,
    HookContext,
    HookDecision,
    HookPoint,
    LLMTier,
)
from iris_harness.kernel.governance.disclosure import (
    DISCLOSURE_STYLE_GUARDRAIL,
    is_architecture_disclosure,
)
from iris_harness.kernel.governance.hooks.tool_payload import ToolContent, ToolSendsTo
from iris_harness.kernel.governance.plugins.destructive_approval import card_title
from iris_harness.kernel.governance.plugins.output_classifier import more_restrictive
from iris_harness.kernel.governance.turn_label import apply_turn_floor
from iris_harness.llm.budget import estimate_tokens
from iris_harness.memory.retriever import MemoryContext
from iris_harness.memory.state import (
    ChatCheckpointPayload,
    Checkpoint,
    CheckpointStore,
)
from iris_harness.memory.state.chat import CheckpointStep

logger = logging.getLogger(__name__)

# Streaming bypasses the ResponseCurator, so run_stream screens its own final
# answer for internal-architecture disclosure (see iris_harness.kernel.governance.disclosure) and
# substitutes this user-facing refusal when a "who are you?" / SOUL-summary turn
# would otherwise stream IRIS internals straight to the user.
_ARCH_DISCLOSURE_REFUSAL = (
    "I can't share IRIS's internal architecture or configuration. I can help with "
    "things like email, calendar, files, planning, or questions about your "
    "documents — what would you like to do?"
)

# ---------------------------------------------------------------------------
# Component registry (used by existing tests + AgenticCore.describe_pipeline)
# ---------------------------------------------------------------------------


class AgenticComponent(StrEnum):
    INTENT_ROUTER = "intent_router"
    TASK_PLANNER = "task_planner"
    REACT_LOOP = "react_loop"
    AGENT_EXECUTOR = "agent_executor"
    RESPONSE_CURATOR = "response_curator"


@dataclass(frozen=True)
class AgenticCoreConfig:
    # 10, not 5: a fan-out turn ("a reminder for each due") is one lookup, one
    # ask_user, one resume and N writes before the Final Answer. The evaluator's
    # step_cap (20) stays the hard ceiling.
    max_iterations: int = 10
    timeout_seconds: int = 120
    max_tools_per_iteration: int = 3
    stall_limit: int = 2
    max_tokens: int | None = None
    streaming: bool = False
    # ADR-0077 context guardrail: cap the running ReAct transcript (Thought/Action/
    # Observation blocks) at this many estimated tokens before each prompt build, so a
    # multi-tool loop can't grow context unbounded. None = no cap (legacy behaviour).
    history_token_budget: int | None = None
    # ADR-0077 P3: cap the injected durable-memory block (identity/profile/episodic/
    # signals/recent-turns) so a fat memory block can't crowd out the transcript or the
    # response. None = no cap (legacy behaviour). Both budgets are derived from the
    # model's context window by the ContextBudgetController at wiring time.
    memory_token_budget: int | None = None
    # ADR-0106 Tier B: offer the loop an `ask_user` action, so an agent that needs
    # a decision can stop and ask instead of guessing. Off by default — a loop that
    # can pause needs somewhere for the answer to come back to, which is the
    # continuation registry the runtime wires. No tier gate (ADR-0106 D2): the
    # continuation is harness state, not model state, so which model is behind the
    # session does not change whether the pause is safe.
    allow_ask_user: bool = False
    # One line naming what this install actually has (plugins, models), built from the
    # live registry by `runtime/capabilities.py`. It replaces the hand-written primer
    # that sat in SOUL.md naming two models this box does not run.
    capabilities_line: str | None = None
    # The turn asks about the user's OWN data (a plugin lists its intent under
    # `read_first_intents`). A final answer is then accepted only after a read tool has
    # returned; the first one without is turned back, the second ends the run
    # unsuccessfully so the handler answers with the intent's deterministic digest.
    # 2026-09-29: "What is in my calendar this week?" got five invented events from a
    # model that called no tool.
    read_first: bool = False


@dataclass(frozen=True)
class ComponentState:
    name: AgenticComponent
    initialized: bool
    responsibility: str
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolDescription:
    """What a destructive call will change, in the owner's words (ADR-0118 step 4).

    A plugin's ``describe(args)`` returns it — a title ("Trash 3 emails") and one line
    per item ("Your weekly deals — Store X · 20 Sep"). Only the plugin can look those
    up; ids mean nothing to the person approving. It is read once, when the approval is
    created, and frozen into the row, so the card never changes after it is seen.
    """

    title: str
    lines: tuple[str, ...] = ()


class ToolSpec(NamedTuple):
    name: str
    description: str
    call: Callable[[dict[str, Any]], str]
    # ADR-0110. A plugin tool gets these from its manifest declaration; a core tool
    # states them here. The loop reads them: write tools are marked in the prompt and,
    # with ``confirm="once"``, are turned back until the run has asked the user.
    effect: str = "read"  # "read" | "write" | "destructive" (ADR-0118)
    # "once" | "never" for a write; "approval" for a destructive tool, which is never
    # confirmed but approved per call, and for a write declared ``approval: pinned``
    # (ADR-0118 amendment), which takes the same per-call approval path.
    confirm: str = "never"
    guidance: str = ""  # routing prose shown while the tool is on the loop
    pinned: bool = False  # the shortlist never drops it from the menu
    # Its output is already the user's answer: the loop ends on it (first tool of the
    # run) and returns it on a repeat, instead of another model call to restate it.
    answers_directly: bool = False
    # ADR-0118, tools approved per call: how the approval card describes a call, the tool
    # that reverses it, and how long that stays possible (from the manifest).
    describe: Callable[[dict[str, Any]], ToolDescription] | None = None
    undo: str | None = None
    undo_window_days: int | None = None
    # ADR-0118, tools approved per call: checks a call's arguments against the plugin's
    # own data BEFORE an approval is queued. It returns why the call cannot run (the
    # model sees that and can correct itself) or None. Without it, a call naming ids
    # the model invented became a card the owner approved to do nothing.
    validate: Callable[[dict[str, Any]], str | None] | None = None
    # Whether the output is text a third party wrote (a web page, an email, a retrieved
    # document): "external" has the retrieved-content injection guard scan it at
    # POST_TOOL_USE; "internal" (the owner's or IRIS's own data) is not scanned.
    content: ToolContent = "internal"
    # The side-effect probe (``kernel/governance/side_effects/probes.py``) that can tell,
    # at ``iris run resume``, whether a non-read call landed. None: resume cannot tell, so
    # the owner approves before it is retried.
    verify: str | None = None
    # Where the arguments go when that is a governed destination of its own
    # (``search_engine``: a web search provider, ADR-0125). Stamped on PRE_TOOL_USE.
    sends_to: ToolSendsTo | None = None
    # Whether a call runs code the model wrote (the manifest's ``executes_code``).
    executes_code: bool = False
    # Who owns the tool, stamped on its PRE/POST_TOOL_USE rows (``tool_plugin``): the
    # plugin that registered it, ``skill:<name>`` for a skill package's tool, or the
    # default ``system`` for a tool the core itself provides.
    plugin: str = "system"


class _ToolStep(NamedTuple):
    """One tool step as the loop continues from it: what the model sees next, and the
    run's data label after the call (raised by ``POST_TOOL_USE``, never lowered)."""

    observation: str
    classification: DataClassification | None


# ---------------------------------------------------------------------------
# ReAct loop primitives
# ---------------------------------------------------------------------------

_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(.+?)(?=\nAction Input:|$)", re.DOTALL)
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.+?)(?=\nObservation:|$)", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(
    # Stop the capture at the next ReAct-format keyword OR end-of-string.
    # The original pattern was r"Final Answer:\s*(.+?)$" — with DOTALL the
    # non-greedy quantifier degenerated to "everything until end" because
    # there was no alternative endpoint, so any trailing
    # "Thought: ... Action: ..." block the model hallucinated after the
    # Final Answer leaked into user-facing responses. The lookahead below
    # is belt-and-suspenders for the new stop-sequence plumbing in the
    # LLM client (which prevents the leak from being generated at all on
    # supported providers).
    r"Final Answer:\s*(.+?)(?=\nThought:|\nAction:|\nAction Input:|" r"\nObservation:|\nUser:|\Z)",
    re.DOTALL,
)
_FRESH_NEWS_RE = re.compile(r"\b(news|headlines?|updates?)\b", re.IGNORECASE)
_FRESHNESS_RE = re.compile(
    r"\b(today|latest|current|breaking|top\s+\d{1,2}|this\s+(?:week|month|year))\b",
    re.IGNORECASE,
)
_TOP_N_RE = re.compile(r"\btop\s+(\d{1,2})\b", re.IGNORECASE)
_RETRIEVAL_TOOL_NAMES = frozenset({"research", "code_exec", "memory_search"})
# Share of the memory budget the conversation transcript may fill (the rest goes to
# identity, profile, summary and the recalled blocks). At the default 12,288-token
# budget that is 35% * 12288 * 0.45 ≈ 1,900 tokens of live conversation, against the
# ~200 tokens the old fixed 3-line tail happened to fit.
_RECENT_TURNS_BUDGET_SHARE = float(os.getenv("IRIS_RECENT_TURNS_BUDGET_SHARE", "0.45"))
_INTERNAL_PLANNING_RE = re.compile(
    r"^\s*(?:"
    r"the\s+user\s+(?:wants|asked|is\s+asking|needs)|"
    r"i\s+need\s+to|"
    r"i\s+should|"
    r"let\s+me\b|"
    r"first,?\s+i(?:'ll|\s+will)?\b|"
    r"i(?:'ll|\s+will)\s+(?:check|look|search|find|read|fetch|summari[sz]e|use)\b"
    r")",
    re.IGNORECASE,
)
# The loop's own scratchpad lines. A final answer that contains one is the model
# echoing its working back, not answering (answer_guard.echoes_scaffold).
_SCRATCHPAD_HEADER = "Your steps so far on this request"
_SCRATCHPAD_FOOTER = "Continue from the last Observation"
_SCAFFOLD_MARKERS = (_SCRATCHPAD_HEADER, _SCRATCHPAD_FOOTER)
# How a call that never ran reads as its observation (_execute_tool's early returns).
_NOT_RUN = (
    "Refused:",
    "Request blocked by governance:",
    "Request needs approval by governance:",
    "Error: unknown tool",
    "Waiting for the owner's approval",
)
_MALFORMED_STEP_FALLBACK = (
    "Sorry — I couldn't put that together cleanly. Could you rephrase or ask again?"
)


@dataclass
class ReactStep:
    thought: str = ""
    action: str | None = None
    action_input: dict[str, Any] = field(default_factory=dict)
    observation: str | None = None
    final_answer: str | None = None
    is_terminal: bool = False


@dataclass
class ReactTrace:
    query: str
    steps: list[ReactStep] = field(default_factory=list)
    final_answer: str = ""
    iterations: int = 0
    success: bool = False
    elapsed_ms: float = 0.0
    stall_count: int = 0
    # Phase 3 fields — empty/None on legacy runs that don't go through
    # the evaluator (kernel=None).
    run_id: str = ""
    halted_by: str | None = None
    halt_reason: str | None = None
    # A read-first turn answered twice without reading (AgenticCoreConfig.read_first):
    # the handler answers from the intent's deterministic digest instead.
    ungrounded: bool = False
    checkpoint_id: str | None = None
    # The declared effect of every non-read tool this run invoked (ADR-0118 decision
    # 5): escalation must not re-run a turn that changed something.
    effects_executed: list[str] = field(default_factory=list)
    # ADR-0118: the approval this run halted on, if any. The turn is waiting on the
    # owner, so escalation must not re-run it (that would raise a second approval).
    pending_approval_id: str | None = None


def _step_to_dict(step: ReactStep) -> dict[str, Any]:
    return {
        "thought": step.thought,
        "action": step.action,
        "action_input": step.action_input,
        "observation": step.observation,
        "final_answer": step.final_answer,
        "is_terminal": step.is_terminal,
    }


def _extract_leading_json_object(raw: str) -> Any | None:
    """Parse the first balanced ``{...}`` object at the start of *raw*.

    Small local models often emit valid JSON followed by trailing chatter
    ("...}\\n\\nWaiting for response..."), which ``json.loads`` rejects.
    This helper salvages the leading object so the loop doesn't have to
    burn an extra turn waiting for the model to self-correct.
    """
    raw = raw.lstrip()
    if not raw.startswith("{"):
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[: i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def _unwrap_nested_input(parsed: dict[str, Any]) -> dict[str, Any]:
    """Unwrap ``{"input": "{...real args...}"}`` into the inner dict.

    Some models double-wrap action arguments by emitting the real JSON as
    a stringified value under an ``input`` key. Unwrap one level so the
    tool sees the keys it actually expects.
    """
    if list(parsed.keys()) == ["input"] and isinstance(parsed["input"], str):
        inner = _extract_leading_json_object(parsed["input"])
        if isinstance(inner, dict):
            return inner
    return parsed


def _parse_react_step(text: str) -> ReactStep:
    step = ReactStep()
    if m := _FINAL_ANSWER_RE.search(text):
        step.final_answer = m.group(1).strip()
        step.is_terminal = True
        if m2 := _THOUGHT_RE.search(text):
            step.thought = m2.group(1).strip()
        return step
    if m := _THOUGHT_RE.search(text):
        step.thought = m.group(1).strip()
    if m := _ACTION_RE.search(text):
        step.action = m.group(1).strip()
    if m := _ACTION_INPUT_RE.search(text):
        raw = m.group(1).strip()
        parsed: Any | None
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            parsed = _extract_leading_json_object(raw)
        if isinstance(parsed, dict):
            step.action_input = _unwrap_nested_input(parsed)
        else:
            step.action_input = {"input": raw}
    return step


# Model-facing context header some tools prepend to their observation (the
# research tool's "[Current date: …; provider: …]" line — see
# iris_harness.plugins_builtin.research.tool.format_result). It grounds the LLM, but it is internal
# markup: when a fallback surfaces an observation as the user's answer, the
# header must not ship with it (the 2026-07-05 campaign leaked it verbatim).
_OBSERVATION_CONTEXT_HEADER_RE = re.compile(r"^\[current date: [^\]\n]*\]\n", re.IGNORECASE)


# The observation that turns back an ask while a destructive tool waits for its card.
_ASK_TURNED_BACK = "Not asked yet."
# The loop's own notes to the model, stored as observations on the steps they turned
# back. Never an answer: a recovery path that answers from "the last good
# observation" showed the owner "Your answer says a change was made, but no tool…"
# (2026-09-28 eval, after a turn-back and a repeated read).
_HARNESS_NOTES = (_ASK_TURNED_BACK, SCAFFOLD_ECHO_NOTE, UNBACKED_CLAIM_NOTE)
# How many asks one run turns back before it stops asking the model and ends the turn
# with a plain account of what happened (owner decision 7, 2026-09-22).
_MAX_TURN_BACKS = 2
# What a destructive step's observation starts with once its approval exists in the
# queue: waiting on it, or settled on resume (_settle_pending_approval). Only these mean
# the owner was shown a card. A call the validator, the kernel or governance refused
# never reached the owner, so it is not one.
_APPROVAL_WAITING = "Waiting for the owner's approval"
_APPROVAL_APPROVED = "The owner approved."
_APPROVAL_REJECTED = "The owner rejected this:"
_APPROVAL_SETTLED = "The approval is "
_APPROVAL_UNREADABLE = "Approval could not be read"
_CARD_SHOWN_PREFIXES = (
    _APPROVAL_WAITING,
    _APPROVAL_APPROVED,
    _APPROVAL_REJECTED,
    _APPROVAL_SETTLED,
    _APPROVAL_UNREADABLE,
)
# The bound on what a question carries of the last read's result.
_FOUND_ITEMS_MAX_LINES = 15
_FOUND_ITEMS_MAX_CHARS = 1500


def _run_asked_user(steps: list[ReactStep], tool_name: str) -> bool:
    """True once this run proposed ``tool_name``, was held, and then asked the user.

    The confirm-once evidence. Only an ``ask_user`` that comes AFTER an attempt at the
    write counts: that question is the confirmation the held call told the model to
    ask, listing the concrete items. A clarifying question asked earlier ("what task
    are you checking for 5 PM?") is not consent — counting it let the reply "Python
    training" go straight to a reminder on an invented date (2026-09-15 session).
    A resumed run carries both steps back in through its checkpoint, so the evidence
    survives the pause and the writes after the answer pass without asking again.
    """
    proposed = False
    for step in steps:
        if step.action == tool_name:
            proposed = True
        elif (
            proposed
            and step.action == ASK_USER_ACTION
            # A turned-back ask never reached the user, so it is nobody's consent.
            and not (step.observation or "").startswith(_ASK_TURNED_BACK)
        ):
            return True
    return False


def _needs_card(tool: ToolSpec) -> bool:
    """True for a tool whose every call waits for the owner's approval card (ADR-0118).

    The one place the ask guard asks "is this a card tool?". Today that is a
    destructive tool; a write declared to go through the same pinned card joins here.
    """
    return tool.effect == "destructive"


def _card_shown(steps: Sequence[ReactStep], destructive: Sequence[str]) -> bool:
    """True once a call to a destructive tool in ``steps`` put a card in the queue."""
    return any(
        s.action in destructive and (s.observation or "").startswith(_CARD_SHOWN_PREFIXES)
        for s in steps
    )


@dataclass(frozen=True)
class AskGuard:
    """What the loop does with an ``ask_user`` instead of putting it to the owner.

    ``kind`` is ``"turn_back"`` (``text`` is the observation handed back to the model)
    or ``"end"`` (``text`` is the final answer that ends the turn).
    """

    kind: str
    text: str


def _guard_ask(steps: list[ReactStep], tools: list[ToolSpec]) -> AskGuard | None:
    """Whether this ``ask_user`` is turned back, ends the turn, or (``None``) goes through.

    A destructive call is never run until the owner approves it on a card listing every
    item, so the card is both the confirmation and the selection: the owner ticks off
    what goes there. Asking them first makes them answer twice, and asking "which of
    these?" hands them a question with nothing to pick from. qwen2.5 asks anyway,
    whatever the prompt says (8 of 8 on 2026-09-21), and asked again it repeats the
    question word for word (5 of 7 runs ended on a bare question, 2026-09-22 probe).

    The guard only applies once the model has ATTEMPTED a card tool in this run (owner,
    2026-09-22). Merely having one on the menu is not enough: a read-only turn ("read my
    email from John", several matches) has a legitimate "which one?" and must not be
    turned back — the same reason ``_needs_card`` stays destructive-only for now. Any
    step whose action is a card tool is an attempt, including a call refused by
    ``validate``, the kernel or governance: those are exactly the runs where the model
    then asked the owner to pick (4 of 7 in the 2026-09-22 probe).

    After an attempt, and while no card has been shown, every ask is turned back, up to
    ``_MAX_TURN_BACKS`` times, telling the model to call the tool with ids from what it
    read. Past the cap the turn ends on a plain account written here, not the model's
    question. Once a card exists the ask is about something else and goes through, as
    does the confirmation a held write asked for. Whatever reaches the owner still
    carries what the last read found (_with_found_items), attempt or no attempt. The
    last step in ``steps`` is the ask itself: an ``ask_user``, or a final answer that
    ends on a question (_answer_as_question), which share one budget.
    """
    destructive = [t.name for t in tools if _needs_card(t)]
    if not destructive:
        return None
    held = {t.name for t in tools if t.effect == "write" and t.confirm == "once"}
    earlier = steps[:-1]
    if not any(s.action in destructive for s in earlier):
        return None  # nothing was proposed to a card yet: this question is the model's own
    if _card_shown(earlier, destructive):
        return None  # the card has been raised; asking now is about something else
    if any(s.action in held for s in earlier):
        return None  # a held write told the model to ask: this ask is its confirmation
    # One budget for both ways of asking: an ask_user step, and a final answer that
    # ends on a question (no action; see _answer_as_question).
    turned_back = sum(
        1
        for s in earlier
        if s.action in (ASK_USER_ACTION, None)
        and (s.observation or "").startswith(_ASK_TURNED_BACK)
    )
    if turned_back >= _MAX_TURN_BACKS:
        return AskGuard("end", _no_card_answer(earlier, tools, destructive))
    names = " / ".join(destructive)
    return AskGuard(
        "turn_back",
        f"{_ASK_TURNED_BACK} Do not ask the owner to confirm or to pick items. {names} "
        "shows the owner an approval card listing every item before anything happens; "
        "they approve or reject the items there, so the card is both their confirmation "
        f"and their selection. Call {names} now, in one call, with the ids of the items "
        "from your results that match the request. If you have no results yet, read "
        "first, then call it.",
    )


def _no_card_answer(
    steps: Sequence[ReactStep], tools: Sequence[ToolSpec], destructive: Sequence[str]
) -> str:
    """The plain, factual end of a turn that never got a destructive call onto a card.

    Written by the loop, not the model: the model's own next move was the question the
    owner has nothing to answer with. Names the tool the model tried (or every
    destructive tool offered), says nothing changed, and shows what the last read found.
    """
    tried = [n for n in destructive if any(s.action == n for s in steps)]
    names = " / ".join(tried or destructive)
    answer = f"I couldn't prepare the {names} request, so nothing was changed."
    found = _found_items(steps, tools)
    if found:
        answer += f"\n\nWhat I found:\n{found}"
    return answer


def _found_items(steps: Sequence[ReactStep], tools: Sequence[ToolSpec]) -> str:
    """The most recent read tool's result, trimmed to show under a question, or "".

    Generic on purpose: the observation text itself, bounded to
    ``_FOUND_ITEMS_MAX_LINES`` lines and ``_FOUND_ITEMS_MAX_CHARS`` characters, with a
    note of how much was left out. An error or empty result is not something to pick
    from, so it attaches nothing.
    """
    reads = {t.name for t in tools if t.effect == "read" and t.name != ASK_USER_ACTION}
    for step in reversed(steps):
        if step.action not in reads:
            continue
        text = _usable_observation(step.observation)
        if not text:
            return ""
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        kept: list[str] = []
        used = 0
        for line in lines[:_FOUND_ITEMS_MAX_LINES]:
            if used + len(line) > _FOUND_ITEMS_MAX_CHARS:
                if not kept:
                    kept.append(line[:_FOUND_ITEMS_MAX_CHARS].rstrip() + "...")
                break
            kept.append(line)
            used += len(line) + 1
        left = len(lines) - len(kept)
        if left > 0:
            kept.append(f"(+{left} more line{'s' if left != 1 else ''} not shown)")
        return "\n".join(kept)
    return ""


def _with_found_items(question: str, steps: Sequence[ReactStep], tools: Sequence[ToolSpec]) -> str:
    """*question* with what the last read found underneath, unless it already shows it.

    "Which of these…?" with nothing under it is unanswerable: the owner never saw the
    tool result the model means (2026-09-22 probe).
    """
    found = _found_items(steps, tools)
    if not found:
        return question
    first = found.splitlines()[0].strip()
    if first and first in question:
        return question
    return f"{question}\n\n{found}"


# The observation on an ask or question the loop answered for the model by ending the turn.
_ASK_ENDED = f"{_ASK_TURNED_BACK} The turn ended without an approval card."


def _answer_as_question(step: ReactStep, raw: str) -> str:
    """The text of a final answer that ends on a question, or "".

    Either form the loops accept as the answer: a ``Final Answer:``, or plain prose
    with no thought and no action. The one signal is the trailing "?", no phrasing:
    a question put in a final answer is a question to the owner all the same, and
    while a card tool waits it gets the same treatment as ``ask_user`` (_guard_ask).
    qwen2.5 did exactly this after a refused call (2026-09-22 probe, run 2).
    """
    if step.is_terminal and step.final_answer:
        text = step.final_answer.strip()
    elif not step.action and not step.thought:
        text = raw.strip()
        if _looks_like_internal_planning(text):
            return ""
    else:
        return ""
    return text if text.endswith("?") else ""


def _hold_back_answer(step: ReactStep, said: str, observation: str) -> str:
    """Un-final *step* (it was not the answer) and return its prompt-history entry."""
    step.is_terminal = False
    step.final_answer = None
    step.observation = observation
    thought = f"Thought: {step.thought}\n" if step.thought else ""
    return f"{thought}Final Answer: {said}\nObservation: {observation}"


def _same_call(a: ReactStep, b: ReactStep) -> bool:
    """Same tool, same arguments — the signature both repeat guards compare."""
    return a.action == b.action and (a.action_input or {}) == (b.action_input or {})


def _usable_observation(observation: str | None) -> str:
    """*observation* as something the user can be shown, or "" when it is not one.

    Tool error/usage messages are not answers, and model-facing context headers are
    internal markup, so the first is refused and the second stripped.
    """
    obs = (observation or "").strip()
    if not obs or obs.lower() == "none" or obs.startswith(_HARNESS_NOTES):
        return ""
    low = obs.lower()
    if low.startswith("error:") or " failed:" in low or " unavailable:" in low:
        return ""
    return _OBSERVATION_CONTEXT_HEADER_RE.sub("", obs).strip()


def _last_good_observation(steps: list[ReactStep]) -> str:
    """Most recent USEFUL tool observation — the grounded data to fall back on
    when the model loops instead of finalizing."""
    for s in reversed(steps):
        usable = _usable_observation(s.observation)
        if usable:
            return usable
    return ""


def _looks_like_internal_planning(text: str) -> bool:
    """True when a plain-prose reply is really planning chatter, not an answer."""
    return bool(_INTERNAL_PLANNING_RE.search(text or ""))


def _resolve_tool_alias(name: str, tool_index: dict[str, Any]) -> str | None:
    """Resolve a hallucinated tool name to the obvious real tool, or None.

    Local models occasionally call a near-miss name — ``portfolio_summary`` for
    ``portfolio``, ``daily_plan_tool`` for ``daily_plan``. Two confident heuristics:
    (1) a real tool whose name is a token-substring of the called name (the called
    name is the real one plus a suffix/prefix), preferring the longest unambiguous
    match; (2) a high-cutoff fuzzy match. Deliberately conservative — when nothing is
    clearly the intended tool, return None so the caller surfaces an honest error
    rather than silently running the wrong tool.
    """
    norm = (name or "").strip().lower()
    if not norm or not tool_index:
        return None
    names = list(tool_index)
    if norm in {n.lower() for n in names}:
        return next(n for n in names if n.lower() == norm)
    # (1) A real tool name contained in the called name ("portfolio_summary" ⊃
    # "portfolio"). Require a clear longest winner so "search" stays ambiguous.
    contained = sorted((n for n in names if n.lower() in norm), key=len, reverse=True)
    if len(contained) == 1 or (len(contained) >= 2 and len(contained[0]) > len(contained[1])):
        return contained[0]
    # (2) Closest by string similarity, high cutoff to avoid wrong redirects.
    import difflib

    match = difflib.get_close_matches(norm, [n.lower() for n in names], n=1, cutoff=0.8)
    if match:
        return next(n for n in names if n.lower() == match[0])
    return None


# Transcript eviction now lives in iris_harness.agent.context_budget (the ADR-0077 P3 controller
# owns both admission and eviction). Kept as a module alias so existing call sites/tests
# that reference ``_trim_react_history`` continue to work unchanged.
_trim_react_history = trim_history


def _clock_line(query: str = "") -> str:
    """The user's local date and time, for the head of the ReAct prompt.

    At the head for the reason the general handler pins it there: local-tier prompts
    are trimmed tail-first, and a model without today's date falls back to whatever
    date it last saw. Here it did exactly that — "tomorrow" came back a day late and a
    follow-up reminder landed four days in the past (2026-09-15 session).
    """
    now = local_now()
    return (
        f"Current date: {now.strftime('%A, %B %d, %Y')}\n"
        f"Current local time: {now.strftime('%H:%M %Z')}\n"
        f"{resolved_dates_line(query, now=now)}\n"
    )


# How the tool list marks a tool that is not a plain read (ADR-0110, ADR-0118).
_EFFECT_LABELS: dict[str, str] = {
    "write": " (WRITES on the user's behalf)",
    "destructive": " (DELETES or OVERWRITES the user's data; each call needs the owner's approval)",
}
# A write approved per call (``approval: pinned``) says so, so the model does not ask first.
_PINNED_WRITE_LABEL = (
    " (WRITES on the user's behalf; each call shows the owner an approval card first, "
    "so do not ask them to confirm)"
)


def _effect_label(tool: ToolSpec) -> str:
    if tool.effect == "write" and tool.confirm == "approval":
        return _PINNED_WRITE_LABEL
    return _EFFECT_LABELS.get(tool.effect, "")


# ADR-0118: a destructive call runs only as a pinned item of an approved approval. With
# no governance kernel there is no queue to ask, so the loop refuses it outright; with a
# kernel, DestructiveApprovalHook queues the approval and the loop halts on it.
def _build_react_prompt(
    query: str,
    tools: list[ToolSpec],
    history: list[str],
    memory_context: MemoryContext | None,
    *,
    require_retrieval_before_final: bool = False,
    expected_list_count: int | None = None,
    memory_token_budget: int | None = None,
    capabilities_line: str | None = None,
) -> str:
    tool_block = (
        "\n".join(f"- {t.name}{_effect_label(t)}: {t.description}" for t in tools)
        or "No tools available."
    )

    # Identity preamble. SOUL.md (`memory_context.soul`) is the canonical
    # agent constitution — use it verbatim when available. The hardcoded
    # fallback only triggers on fresh installs where the loader didn't
    # populate soul (tests, missing files, etc.). Without this, the
    # AgenticCore ReAct path replaced SOUL.md with a one-liner, defeating
    # the entire identity layer for any system-intent turn.
    soul_text = (memory_context.soul or "").strip() if memory_context else ""
    if soul_text:
        identity_preamble = soul_text + "\n\n"
    else:
        identity_preamble = "You are IRIS, a personal AI assistant.\n\n"
    # L1 self-knowledge: what this install has, generated per turn so it cannot drift
    # the way the hand-written primer did. The detail is one iris_doc call away.
    if capabilities_line:
        identity_preamble += capabilities_line.strip() + "\n\n"

    mem_block = ""
    if memory_context:
        parts: list[str] = []
        # USER.md (curated profile) — same field the legacy general
        # handler injects at bootstrap.py:2136. Falls back to extracted
        # facts only if user_profile is empty. The explicit
        # "About the user you are talking to" header keeps small Tier-1
        # models from confusing USER.md with the agent's own preferences.
        user_profile = (memory_context.user_profile or "").strip()
        if user_profile:
            parts.append("## About the user you are talking to (from USER.md)\n\n" + user_profile)
        # Confirmed facts ride ALONGSIDE the curated profile, ranked by relevance to
        # this turn. They used to appear only when USER.md was empty, which it never
        # is — so everything IRIS learned and the owner approved stayed out of the
        # prompt. Unconfirmed facts never get here (the retriever drops them).
        if memory_context.user_facts:
            facts = ", ".join(
                f"{f.key}={f.value}" + (" (unconfirmed)" if f.uncertain else "")
                for f in memory_context.user_facts[:5]
            )
            parts.append(f"Confirmed facts about the user: {facts}")
        # The memory graph around the names this message mentions (memris PR 6), already
        # within its own budget (learning.yaml → linking.max_tokens).
        linked_text = (memory_context.linked or "").strip()
        if linked_text:
            parts.append(linked_text)
        # Active items (todos / in-flight things from active.md).
        active_text = (memory_context.active or "").strip() if memory_context.active else ""
        if active_text:
            parts.append(active_text)
        # Episodic patterns (recurring user behaviors).
        episodic_text = (
            (memory_context.episodic_digest or "").strip() if memory_context.episodic_digest else ""
        )
        if episodic_text:
            parts.append(episodic_text)
        elif memory_context.episodic_patterns:
            parts.append(
                "## What I've noticed about you\n"
                + "\n".join(f"- {p}" for p in memory_context.episodic_patterns)
            )
        # Behavior overrides matched for this intent.
        behavior_text = (memory_context.behavior or "").strip() if memory_context.behavior else ""
        if behavior_text:
            parts.append(behavior_text)
        # Conversation continuity (summary + recent turns + excerpts from other
        # sessions). L0 of the context layers: what is always worth its tokens.
        if memory_context.summary:
            parts.append(f"## Earlier in this conversation\n{memory_context.summary}")
        if memory_context.related_turns:
            parts.append(
                "## From earlier sessions (may or may not be relevant)\n"
                + "\n".join(f"  {t}" for t in memory_context.related_turns)
            )
        if memory_context.recent_turns:
            # Fill the conversation's share of the memory budget from the newest end.
            # The old fixed `[-3:]` tail showed 1.5 exchanges no matter how much window
            # was free, so the model answered follow-ups with almost no history.
            turns_budget = (
                int(memory_token_budget * _RECENT_TURNS_BUDGET_SHARE)
                if memory_token_budget is not None
                else 0
            )
            shown = fill_recent_turns(list(memory_context.recent_turns), turns_budget)
            parts.append("Recent turns:\n" + "\n".join(f"  {t}" for t in shown))
        # L1 of the context layers: notices about memory that exists but is not in this
        # prompt. One line each, so they survive admission even when content blocks do
        # not — a model that knows history exists can ask for it; one that doesn't, can't.
        if memory_context.pointers:
            parts.append("\n".join(f"Note: {p}" for p in memory_context.pointers))
        # ADR-0077 P3: admission — cap the memory block to its budget, evicting the
        # lowest-priority blocks (learning hints, then recent turns…) while pinning the
        # user profile. None = no cap (legacy behaviour; tests unaffected).
        if memory_token_budget is not None and parts:
            parts, _mem_evicted = admit_memory_parts(parts, memory_token_budget)
        if parts:
            # Natural prose intro instead of a bracketed `[Memory Context]`
            # marker. Small/medium models were echoing the marker verbatim
            # into their responses ("[End Example]" / "[Memory Context]"
            # appearing in user-facing output). Inline prose carries the
            # same information without giving the model a structural
            # pattern to imitate.
            mem_block = (
                "\nYou already know the following about your environment "
                "and the user (use it; do not echo these blocks back):\n\n"
                + "\n\n".join(parts)
                + "\n"
            )

    # The run's own steps go AFTER the question, as the assistant's work in progress
    # (multi-step loop plan, PR 4, 2026-09-14). They used to sit before the response
    # rules and the `User:` line, unlabelled, so the model was handed the question
    # fresh and answered it fresh: gpt-4o and qwen3.6:27b both re-issued their first
    # tool call three times after a complete, correct Observation, and the stall
    # guard ended every multi-step turn at step one. Standard ReAct: question, then
    # the scratchpad, then the next Thought.
    scratchpad = ""
    if history:
        scratchpad = (
            f"\n{_SCRATCHPAD_HEADER} (every Observation below is a real "
            "tool result you already have):\n"
            + "\n".join(history)
            + f"\n{_SCRATCHPAD_FOOTER}. Do NOT repeat a completed action: use "
            "its result to take the next step, or give the Final Answer.\n"
        )

    # Include a concrete few-shot example when tools are available so small local
    # models have a clear pattern to imitate. Embed it inline (no bracketed
    # markers) so the model doesn't treat it as a section to mirror.
    example_block = ""
    if tools:
        first_tool = tools[0]
        example_block = (
            "\nTwo examples.\n\n"
            'If you already know the answer (e.g. "What is 2 + 2?"), do NOT call a tool:\n\n'
            "Thought: This is basic arithmetic I can answer directly.\n"
            "Final Answer: 4\n\n"
            'If you genuinely need external or current information (e.g. "What is the '
            'latest news about AI?"):\n\n'
            f"Thought: I need to look this up using {first_tool.name}.\n"
            f"Action: {first_tool.name}\n"
            'Action Input: {"query": "latest AI news"}\n'
            "Observation: <the tool's response>\n"
            "Thought: I now have the information I need.\n"
            "Final Answer: Based on the search results, ...\n\n"
            "Those are the two formats. Use whichever fits the real user message below.\n"
        )

    # Cross-domain guidance (ADR-0077): only when BOTH the finance and email surfaces
    # are on the loop. Generic — names no specific bank/card — so it never bloats an
    # unrelated turn. Teaches the finance→email follow-up that the user expects when a
    # named account isn't in the local statements.
    # ADR-0110: routing guidance is the plugins' — each on-loop tool's declared
    # ``guidance`` is shown here, deduplicated, in pool order. The core authors none of
    # it and names no plugin tool. One rule the core does own: how a write tool is
    # used, which is what the tool policy's confirm-once backstop enforces.
    guidance_lines: list[str] = []
    for t in tools:
        text = (t.guidance or "").strip()
        if text and text not in guidance_lines:
            guidance_lines.append(text)
    cross_domain_block = ""
    if guidance_lines:
        cross_domain_block = "\n" + "\n".join(guidance_lines) + "\n"
    # Only a write tool that is actually held (``confirm="once"``) gets the ask-first
    # rule: shown for any write, it made the model ask before a confirm-never write
    # (restore_email) and before a destructive call, which the owner then confirmed
    # twice — in chat, and again on the approval card (2026-09-21). Keeping a first ask
    # off a destructive call is the loop's job (_guard_ask): prompt wording alone
    # did not move qwen2.5 at all.
    if any(t.effect == "write" and t.confirm == "once" for t in tools):
        cross_domain_block += (
            "\nSome tools WRITE on the user's behalf (marked above). The first call to one in a "
            "request is held until the user confirms: when it is, ask the user ONCE with "
            "ask_user, listing everything you intend to create or change (one item or "
            "several, each with its text, date and time); after they confirm, call the tool "
            "once per item without asking again. A clarifying question is not a "
            "confirmation.\n"
        )

    freshness_block = ""
    if require_retrieval_before_final:
        freshness_lines = [
            "\nFreshness requirement:",
            "- This query asks for current/news information.",
            "- You MUST call a retrieval tool (prefer research) before Final Answer.",
            "- Call research ONCE, then write the Final Answer from the returned "
            "content and cite the URLs. Do NOT repeat the same search.",
            "- Do not answer from prior knowledge only.",
        ]
        if expected_list_count is not None:
            freshness_lines.append(
                f"- Return exactly {expected_list_count} items as a numbered list."
            )
        freshness_block = "\n".join(freshness_lines) + "\n"

    return (
        f"{_clock_line(query)}"
        f"{identity_preamble}"
        f"Available tools:\n{tool_block}\n\n"
        "Answer directly when you can; only call a tool when you genuinely need "
        "external or current information, or a capability you don't have. For math, "
        "definitions, and well-known facts (capital cities, dates, history, geography), "
        "or anything already in this conversation, you already know the answer — skip "
        "tools and give the Final Answer. For current events, recent/changing data, or "
        "things you genuinely don't know, use research (it searches the web, crawls "
        "the top pages, and returns extracted CONTENT with citations) — call it once and "
        'answer from what it returns; for a quick fact pass "fetch_content": false.\n\n'
        "To use a tool respond EXACTLY:\n"
        "Thought: <your reasoning>\n"
        "Action: <tool_name>\n"
        "Action Input: <json args>\n\n"
        "When you can answer without a tool:\n"
        "Thought: <your reasoning>\n"
        "Final Answer: <response to user>\n"
        f"{example_block}"
        f"{cross_domain_block}"
        f"{freshness_block}"
        f"{mem_block}"
        f"{DISCLOSURE_STYLE_GUARDRAIL}\n"
        f"User: {query}\n"
        "Assistant:"
        f"{scratchpad}"
    )


def _requires_retrieval_before_final(query: str, tools: list[ToolSpec]) -> bool:
    """Return True for fresh-news queries when retrieval tools are available."""
    if not _FRESH_NEWS_RE.search(query):
        return False
    if not _FRESHNESS_RE.search(query):
        return False
    tool_names = {tool.name for tool in tools}
    return bool(tool_names.intersection(_RETRIEVAL_TOOL_NAMES))


def _top_n_requested(query: str) -> int | None:
    """Parse a user-requested top-N count from the query, if any."""
    if match := _TOP_N_RE.search(query):
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# AgenticCore — combines component management + ReAct loop
# ---------------------------------------------------------------------------

_RESPONSIBILITIES: dict[AgenticComponent, str] = {
    AgenticComponent.INTENT_ROUTER: "classify user intent and select the next workflow surface",
    AgenticComponent.TASK_PLANNER: "decompose multi-step requests into ordered sub-tasks",
    AgenticComponent.REACT_LOOP: "coordinate think-act-observe-decide iterations",
    AgenticComponent.AGENT_EXECUTOR: "dispatch planned work to governed agents or tools",
    AgenticComponent.RESPONSE_CURATOR: "organize and refine outputs for user delivery",
}


def _hash_action_input(args: dict[str, Any]) -> str:
    """Stable short hash of tool args for the action_repeat signal."""
    import hashlib

    encoded = json.dumps(args, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded, usedforsecurity=False).hexdigest()[:16]


_ACTION_INPUT_TEXT_CHARS = 300


def _render_action_input(args: dict[str, Any]) -> str:
    """Bounded, sorted-key JSON of tool args for the semantic signals (loop_detect).

    Same encoding as the hash, so equal args render equal; capped so a pasted
    document in one argument cannot dominate the embedding.
    """
    text = json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)
    return text if len(text) <= _ACTION_INPUT_TEXT_CHARS else text[:_ACTION_INPUT_TEXT_CHARS]


def _rebuild_history(steps: tuple[CheckpointStep, ...]) -> list[str]:
    """Rehydrate the prompt-history strings from a checkpoint's step list."""
    history: list[str] = []
    for step in steps:
        if not step.action:
            continue
        history.append(
            f"Thought: {step.thought}\nAction: {step.action}\n"
            f"Action Input: {json.dumps(step.action_input)}\n"
            f"Observation: {step.observation or ''}"
        )
    return history


# ADR-0106 Tier B (M5.C5c) — the resume seed.
#
# `resume_from_checkpoint` and `run_stream` both need to re-enter a halted run from
# the same decoded state, and the two loop bodies are still separate (ADR-0107
# option B deliberately not started). Decoding once, here, is what keeps them from
# drifting a third time: neither loop knows how a checkpoint is shaped, only how to
# start from a seed.
@dataclass
class ResumeSeed:
    """Decoded loop state for continuing a halted run at ``start_iteration``."""

    run_id: str
    query: str
    steps: list[ReactStep]
    history: list[str]
    start_iteration: int
    # ADR-0118: the approval the halted step waits on. Settled before the loop resumes.
    pending_approval_id: str | None = None


# The LangGraph `Command(resume=v)` analogue (ADR-0106's own comparison table): the
# human's answer becomes the observation of the step that asked for it, which is the
# only place the loop will read it. The escape clause is deliberate — the claim rule
# is "the next message in the session resumes", so a user who ignored the question
# and changed the subject lands here too, and the loop must be able to drop the old
# task rather than plough on with an answer to a question nobody asked. Judging that
# is a semantic call, so it goes to the model; the harness stays deterministic about
# *who* the turn belongs to, which is the fact the incident turned on.
_RESUME_OBSERVATION = (
    "{asked}\n"
    "User answered: {reply}\n"
    "If that reply does not answer the question, abandon the previous task and "
    "address the reply instead."
)


def resume_seed_from_checkpoint(
    checkpoint: Checkpoint, *, user_reply: str | None = None
) -> ResumeSeed:
    """Decode a checkpoint into loop state, injecting the human's answer if given."""
    payload = ChatCheckpointPayload.from_payload(checkpoint.payload)
    cp_steps = list(payload.steps)
    if user_reply and cp_steps:
        last = cp_steps[-1]
        cp_steps[-1] = last.model_copy(
            update={
                "observation": _RESUME_OBSERVATION.format(
                    asked=last.observation or "Asked the user something.",
                    reply=user_reply.strip(),
                )
            }
        )
    return ResumeSeed(
        run_id=checkpoint.run_id,
        query=payload.query,
        steps=[
            ReactStep(
                thought=cp_step.thought,
                action=cp_step.action,
                action_input=dict(cp_step.action_input),
                observation=cp_step.observation,
                final_answer=cp_step.final_answer,
                is_terminal=cp_step.is_terminal,
            )
            for cp_step in cp_steps
        ],
        history=_rebuild_history(tuple(cp_steps)),
        start_iteration=payload.iteration,
        pending_approval_id=payload.pending_approval_id,
    )


def _memory_context_to_payload(memory_context: MemoryContext | None) -> dict[str, Any] | None:
    """Serialize a MemoryContext into a JSON-friendly dict for checkpoints."""
    if memory_context is None:
        return None
    if hasattr(memory_context, "model_dump"):
        return memory_context.model_dump(mode="json")  # type: ignore[no-any-return]
    if isinstance(memory_context, dict):
        return memory_context
    return None


# ADR-0107, the owner's decision. On the STREAMING loop the evaluator enforces only
# the signals with no local equivalent. Loop hygiene — runaway iteration, repeated
# actions, tool-failure streaks — is already contained by run_stream's own
# `max_iterations`, `stall_limit` and malformed-step guards, so re-enforcing it
# through the evaluator would be two mechanisms halting the same run at two
# thresholds. Reuse what is there; add only what is missing.
STREAMING_ENFORCED_SIGNALS: frozenset[str] = frozenset(
    {
        "classification_violation",
        "cost_budget",
        "goal_drift",
        "loop_detect",  # the semantic one; the counting ones are the local guards' job
    }
)
_TERMINAL_VERDICTS: frozenset[str] = frozenset({"halt", "require_approval"})
_OBSERVATION_PREVIEW_CHARS = 120


def _enforced_signal_names(decision: HookDecision, enforced: frozenset[str]) -> list[str]:
    """Which enforced signals returned a terminal verdict in this decision.

    Reads the structured `audit_metadata["signals"]` the evaluator hook already
    records, rather than parsing the human-readable reason string — the reason is
    for people, and matching on it would break the first time someone reworded it.
    """
    signals = decision.audit_metadata.get("signals")
    if not isinstance(signals, list):
        return []
    hits: list[str] = []
    for raw in signals:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", ""))
        if name in enforced and str(raw.get("verdict", "")) in _TERMINAL_VERDICTS:
            hits.append(name)
    return hits


def _completed_work(steps: Sequence[ReactStep]) -> list[str]:
    """One line per tool call that actually ran, for the interruption summary."""
    lines: list[str] = []
    for step in steps:
        if not step.action:
            continue
        observation = (step.observation or "").strip().replace("\n", " ")
        if len(observation) > _OBSERVATION_PREVIEW_CHARS:
            observation = observation[:_OBSERVATION_PREVIEW_CHARS].rstrip() + "..."
        lines.append(f"{len(lines) + 1}. {step.action} - {observation or 'no result'}")
    return lines


# ADR-0106 Tier B. `ask_user` is a control-flow action, not a tool: the loop never
# "calls" it, it stops on it. It is still declared as a ToolSpec so the prompt
# builder advertises it exactly like the real tools — the model should not have to
# learn a second syntax to ask a question. Its `call` is never reached; the loops
# intercept the action before `_execute_tool`, and it raises rather than returning a
# plausible string so a future refactor that routes it there fails loudly.
ASK_USER_ACTION = "ask_user"
# Said when a read-first turn never read anything and no deterministic digest exists.
UNGROUNDED_REASON = "answered_without_reading"
UNGROUNDED_ANSWER = (
    "I couldn't look that up just now, so I won't guess. Please ask me again in a moment."
)


def _ask_user_unreachable(_args: dict[str, Any]) -> str:
    raise AssertionError(
        "ask_user is intercepted by the ReAct loop and must never be executed as a tool"
    )


ASK_USER_TOOL = ToolSpec(
    name=ASK_USER_ACTION,
    description=(
        "Stop and ask the user one question, when a decision is genuinely theirs to "
        'make and you cannot proceed sensibly without it. Input: {"question": "..."}. '
        "This ends your turn — you will be given their answer to continue from. Do not "
        "use it for anything you can reasonably decide or look up yourself."
        " THE RULE: if what you are about to send ENDS ON A QUESTION you expect them to "
        "answer — which story, which of these options, how many, which format — ask it "
        "HERE, not in a Final Answer. A Final Answer ends the exchange, so a question "
        "asked in one is a question nobody is waiting on: their reply arrives with no "
        "memory of what you asked, and lands wherever the router sends it."
        " The one exception is signing off ('anything else?', 'let me know if you need "
        "more') — that is a pleasantry, not a question. When the work is done, give the "
        "Final Answer instead."
    ),
    call=_ask_user_unreachable,
)


def _ask_user_question(step: ReactStep, fallback: str = "") -> str:
    """The question an `ask_user` step is putting to the user."""
    raw = step.action_input or {}
    for key in ("question", "prompt", "text", "q"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return (step.thought or fallback).strip() or "Could you tell me how you'd like to proceed?"


def rollout_mode() -> str:
    """Normalize the AgenticCore rollout flag into ``off`` / ``shadow`` / ``on``.

    Read by the composition root and by any plugin whose agent runs on this loop —
    the email agent, which left the core with its library at M6.1b (OSS plan M6,
    decision 2), and a plugin may not import the composition root (release gate 2).
    """
    import os

    raw = os.getenv("IRIS_AGENTIC_CORE_ENABLED", "1").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return "off"
    if raw == "shadow":
        return "shadow"
    return "on"


class AgenticCore:
    """Coordinate the modular IRIS agentic runtime components."""

    pipeline_order: tuple[AgenticComponent, ...] = (
        AgenticComponent.INTENT_ROUTER,
        AgenticComponent.TASK_PLANNER,
        AgenticComponent.REACT_LOOP,
        AgenticComponent.AGENT_EXECUTOR,
        AgenticComponent.RESPONSE_CURATOR,
    )

    def __init__(
        self,
        config: AgenticCoreConfig | None = None,
        llm_call: Callable[[str], str] | None = None,
        tools: list[ToolSpec] | None = None,
        *,
        kernel: GovernanceKernel | None = None,
        classification: DataClassification | None = None,
        target_tier: LLMTier | None = None,
        agent_type: str = "chat",
        checkpoint_store: CheckpointStore | None = None,
        session_id: str | None = None,
        origin_channel: str = "console",
        link_approval_checkpoint: Callable[[str, str], None] | None = None,
        read_approval: (
            Callable[[str], tuple[str, list[tuple[str, dict[str, Any]]]] | None] | None
        ) = None,
        budget_observer: Callable[[int, int], None] | None = None,
        reserve_tools: list[ToolSpec] | None = None,
        review_route: str | None = None,
        model_identity: Callable[[], tuple[str, str] | None] | None = None,
    ) -> None:
        self.config = config or AgenticCoreConfig()
        # ``(model, provider)`` of the model the NEXT step calls, asked just before each
        # PRE_LLM_CALL so the row names it as every other model call's row does. ``llm_call``
        # is opaque to the loop, so its owner says which model it is, and keeps that one
        # resolution for the call. None (a legacy caller, a test): the row carries the tier alone.
        self._model_identity = model_identity
        # The routing intent whose model this loop's LLM uses. A run that ends is handed
        # to the process's run reviewer (the governance judge, §9.2) with it, so the
        # review runs on the same route as the run; None opts the loop out.
        self._review_route = review_route
        # Tools not on the prompt's menu (the shortlist dropped them) but still
        # callable: admitted into the index on first call, governed like any other.
        self._reserve_index: dict[str, ToolSpec] = {t.name: t for t in (reserve_tools or [])}
        # ADR-0081 context-health: called once per iteration with the assembled prompt's
        # token count and the tokens evicted from the transcript that iteration, so the
        # runtime can surface the latest in-loop budget pressure. Best-effort; never on
        # the hot path's critical correctness. Optional — None on legacy callers/tests.
        self._budget_observer = budget_observer
        self._llm: Callable[[str], str] | None = llm_call
        self._tools: list[ToolSpec] = list(tools or [])
        self._tool_index: dict[str, ToolSpec] = {t.name: t for t in self._tools}
        self._component_states: dict[AgenticComponent, ComponentState] = {
            c: ComponentState(name=c, initialized=False, responsibility=_RESPONSIBILITIES[c])
            for c in self.pipeline_order
        }
        # Governance plumbing — all optional. kernel=None preserves the legacy
        # path so existing tests and callers are unaffected. See
        # docs/architecture/unified-governance-layer.md for design.
        self._kernel = kernel
        self._classification = classification
        # The strictest label a tool result earned before the run derived its own (an
        # approved call run on resume); applied under PRE_CLASSIFY's label.
        self._earned_floor: DataClassification | None = None
        self._target_tier = target_tier
        self._agent_type = agent_type
        self._checkpoint_store = checkpoint_store
        # ADR-0106 C5a: which conversation this loop is answering, stamped onto every
        # checkpoint so `by_session` can say what a session was doing. None for a run
        # with no conversation behind it (CLI, heartbeats) — the column is nullable.
        self._session_id = session_id
        # The gateway this run answers. Rides on the PostStep context so the evaluator
        # can stamp an approval with where to deliver it — the queue's `channel` column
        # is as old as the queue and, with nothing to stamp it from, has only ever held
        # its "cli" default.
        self._origin_channel = origin_channel
        # Called with (approval_id, checkpoint_id) once a halt's checkpoint exists. A
        # narrow callable rather than the ApprovalQueue itself: the core has no business
        # importing the approvals package, and the only thing it needs to do is close a
        # link it is uniquely placed to close. See `_link_approval_checkpoint`.
        self._link_approval_checkpoint = link_approval_checkpoint
        # ADR-0118: (status, pinned calls) of an approval, or None when it cannot be
        # read. Narrow for the same reason as the link above; a resumed run uses it to
        # settle the approval its halted step was waiting on.
        self._read_approval = read_approval
        # The request this run is answering, and the title of the approval it halted on:
        # the approval card quotes the first, the chat message names the second.
        self._current_query: str | None = None
        self._pending_title: str | None = None

    @property
    def tools(self) -> list[ToolSpec]:
        """The tools offered to this loop (read-only view for observability)."""
        return list(self._tools)

    # ------------------------------------------------------------------
    # Tool management
    # ------------------------------------------------------------------

    def register_tool(self, tool: ToolSpec) -> None:
        self._tools.append(tool)
        self._tool_index[tool.name] = tool

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        *,
        memory_context: MemoryContext | None = None,
    ) -> ReactTrace:
        """Execute the ReAct loop for the given query."""
        if self._llm is None:
            return ReactTrace(
                query=query,
                final_answer="No LLM configured.",
                success=False,
            )

        trace = ReactTrace(query=query, run_id=str(uuid.uuid4()))
        return self._loop(
            trace=trace,
            memory_context=memory_context,
            history=[],
            start_iteration=0,
        )

    def resume_from_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        memory_context: MemoryContext | None = None,
        user_reply: str | None = None,
    ) -> ReactTrace:
        """Continue a halted run from its last checkpoint.

        Rebuilds the ReAct history from the serialized steps and
        re-enters the loop at ``iteration = checkpoint.step_id + 1``.
        ``memory_context`` is re-fetched by the caller — checkpoints do
        not pickle retrieval state because vector indexes are not
        version-stable.

        ``user_reply`` is the Tier B case (ADR-0106 M5.C5c): the run paused on an
        ``ask_user`` step and the human has now answered, so the answer is injected
        as that step's observation before the loop re-enters.
        """
        if self._llm is None:
            return ReactTrace(
                query="",
                final_answer="No LLM configured.",
                success=False,
                run_id=checkpoint.run_id,
            )

        return self.run_from_seed(
            resume_seed_from_checkpoint(checkpoint, user_reply=user_reply),
            memory_context=memory_context,
        )

    def run_from_seed(
        self, seed: ResumeSeed, *, memory_context: MemoryContext | None = None
    ) -> ReactTrace:
        """Sync counterpart to ``run_stream(resume=seed)`` — re-enter ``_loop``.

        One decoder, two entry points. Callers holding a continuation ask
        :meth:`resume_seed` for the seed and hand it to whichever of the two loops is
        answering the turn, so a resume does not depend on that choice.
        """
        if self._llm is None:
            return ReactTrace(
                query=seed.query,
                final_answer="No LLM configured.",
                success=False,
                run_id=seed.run_id,
            )
        trace = ReactTrace(query=seed.query, run_id=seed.run_id)
        seed = self._settle_pending_approval(seed, effects=trace.effects_executed)
        trace.steps.extend(seed.steps)
        trace.iterations = seed.start_iteration
        return self._loop(
            trace=trace,
            memory_context=memory_context,
            history=list(seed.history),
            start_iteration=seed.start_iteration,
        )

    def _loop(
        self,
        *,
        trace: ReactTrace,
        memory_context: MemoryContext | None,
        history: list[str],
        start_iteration: int,
    ) -> ReactTrace:
        """Shared inner loop used by ``run`` and ``resume_from_checkpoint``."""
        assert self._llm is not None  # checked by callers
        query = trace.query
        self._current_query = query
        run_id = trace.run_id or str(uuid.uuid4())
        trace.run_id = run_id
        start = time.monotonic()
        # Stall detection compares the full (action, action_input) signature,
        # not just the tool name. Legitimate drill-down — e.g. research
        # with different queries across turns — is progress, not a stall.
        # The original `last_action == step.action` comparison killed those
        # runs incorrectly.
        last_signature: str = ""
        stall_count = 0
        # Action-history ledger: every (tool, args) signature we've already run.
        # A repeat means the model is looping over a result it already has, so we
        # short-circuit to a final-synthesis pass on the *second* occurrence
        # instead of burning iterations until the stall counter trips.
        executed_actions: set[str] = set()
        classification = self._classification
        require_retrieval = _requires_retrieval_before_final(query, self._tools)
        requested_top_n = _top_n_requested(query) if require_retrieval else None
        retrieval_used = False
        retrieval_enforcement_retries = 0
        malformed_steps = 0
        answer_retries = 0  # answer_guard turn-backs this run (one allowed)

        for iteration in range(start_iteration, self.config.max_iterations):
            if time.monotonic() - start > self.config.timeout_seconds:
                trace.final_answer = "Request timed out."
                break

            history, evicted_tokens = self._apply_context_budget(history)
            prompt = _build_react_prompt(
                query,
                self._tools,
                history,
                memory_context,
                require_retrieval_before_final=require_retrieval,
                expected_list_count=requested_top_n,
                memory_token_budget=self.config.memory_token_budget,
                capabilities_line=self.config.capabilities_line,
            )

            governance_decision, prompt, classification = self._governance_pre_llm(
                query=query,
                prompt=prompt,
                run_id=run_id,
                step_id=iteration,
                classification=classification,
                context_tokens=estimate_tokens(prompt),
                evicted_tokens=evicted_tokens,
            )
            if governance_decision is not None and governance_decision.outcome in (
                "deny",
                "require_approval",
            ):
                trace.final_answer = self._governance_block_message(governance_decision)
                trace.success = False
                trace.iterations = iteration
                break

            try:
                raw = self._llm(prompt)
            except Exception as exc:  # noqa: BLE001 — LLM error is the answer
                trace.final_answer = f"LLM error: {exc}"
                break

            step = _parse_react_step(raw)
            trace.steps.append(step)
            trace.iterations = iteration + 1

            # A final answer that ends on a question is an ask: while a card tool waits
            # with no card shown it is turned back, or ends the turn, like ask_user.
            said = _answer_as_question(step, raw)
            guard = _guard_ask(trace.steps, self._tools) if said else None
            if guard is not None and guard.kind == "end":
                _hold_back_answer(step, said, _ASK_ENDED)
                trace.final_answer = guard.text
                trace.success = True  # a true account of the turn, not a failure
                break
            if guard is not None:
                history.append(_hold_back_answer(step, said, guard.text))
                continue

            if step.is_terminal and step.final_answer:
                if require_retrieval and not retrieval_used:
                    retrieval_enforcement_retries += 1
                    if retrieval_enforcement_retries >= 2:
                        trace.final_answer = "Unable to provide fresh news reliably because no retrieval tool was used."
                        trace.success = False
                        break
                    step.observation = "Error: Fresh-news query requires a retrieval tool call before final answer."
                    step.is_terminal = False
                    step.final_answer = None
                    history.append(
                        "System: Fresh-news query detected. Call a retrieval tool "
                        "(prefer research) before Final Answer."
                    )
                    continue
                verdict = self._vet_final_answer(
                    step.final_answer, trace.steps, trace.effects_executed, answer_retries
                )
                if verdict.kind == "retry":
                    answer_retries += 1
                    history.append(_hold_back_answer(step, step.final_answer, verdict.note))
                    continue
                if verdict.kind == "fail":
                    trace.final_answer = UNGROUNDED_ANSWER
                    trace.success = False
                    trace.ungrounded = True
                    break
                trace.final_answer = (
                    verdict.text if verdict.kind == "replace" else step.final_answer
                )
                trace.success = True
                break

            if step.action:
                # Ahead of the repeat guards: a turned-back ask did not run, so asking
                # again is not a repeat of anything.
                guard = _guard_ask(trace.steps, self._tools) if self._is_ask_user(step) else None
                if guard is not None and guard.kind == "end":
                    step.observation = _ASK_ENDED
                    trace.final_answer = guard.text
                    trace.success = True  # a true account of the turn, not a failure
                    break
                if guard is not None:
                    step.observation = guard.text
                    history.append(
                        f"Thought: {step.thought}\nAction: {step.action}\n"
                        f"Action Input: {json.dumps(step.action_input)}\n"
                        f"Observation: {guard.text}"
                    )
                    continue

                signature = json.dumps(
                    {"action": step.action, "input": step.action_input or {}},
                    sort_keys=True,
                    default=str,
                )

                # De-dup guardrail: the model is re-running an action whose result
                # is already in `history`. Don't execute it again — force one final
                # synthesis pass from what we have (and fall back to the last
                # observation if the model still won't finalize).
                if signature in executed_actions:
                    logger.warning(
                        "AgenticCore: repeated action %s — forcing final synthesis",
                        step.action,
                    )
                    stall_count += 1  # recorded into trace.stall_count at loop exit
                    in_hand = self._answer_in_hand(step.action, trace.steps[:-1])
                    if in_hand:
                        logger.info(
                            "AgenticCore: %s answers directly — returning its result "
                            "in hand, no synthesis call",
                            step.action,
                        )
                        trace.final_answer = in_hand
                        trace.success = True
                        break
                    synthesized = self._force_final_synthesis(
                        query=query,
                        history=history,
                        memory_context=memory_context,
                        run_id=run_id,
                        step_id=iteration + 1,
                        classification=classification,
                        last_observation=_last_good_observation(trace.steps),
                    )
                    # The synthesis is a final answer like any other: vetted, with no
                    # turn-back left (the run is ending).
                    verdict = self._vet_final_answer(
                        synthesized, trace.steps, trace.effects_executed, retries_used=1
                    )
                    if verdict.kind == "fail":
                        trace.final_answer = UNGROUNDED_ANSWER
                        trace.success = False
                        trace.ungrounded = True
                        break
                    trace.final_answer = verdict.text if verdict.kind == "replace" else synthesized
                    trace.success = bool(trace.final_answer)
                    break

                executed_actions.add(signature)

                if signature == last_signature:
                    stall_count += 1
                else:
                    stall_count = 0
                    last_signature = signature

                if stall_count >= self.config.stall_limit:
                    trace.stall_count = stall_count
                    # The model looped instead of finalizing — surface the last
                    # grounded tool result (the data the user asked for) rather
                    # than a useless stall sentinel (issue 0001 follow-up D /
                    # issue 0003). stall_count still records the stall.
                    recovered = _last_good_observation(trace.steps)
                    if recovered:
                        trace.final_answer = recovered
                        trace.success = True
                    else:
                        trace.final_answer = "Stopped: repeated action without progress."
                    break

                if self._is_ask_user(step):
                    trace.final_answer = self._pause_for_user(
                        trace=trace,
                        step=step,
                        iteration=iteration,
                        memory_context=memory_context,
                    )
                    break

                halts: list[HookDecision] = []
                executed = self._execute_tool(
                    step.action,
                    step.action_input,
                    run_id=run_id,
                    classification=classification,
                    step_id=iteration,
                    asked_user=_run_asked_user(trace.steps, step.action),
                    effects=trace.effects_executed,
                    halts=halts,
                )
                # What POST_TOOL_USE handed on (a withheld or redacted result, never the
                # raw one it refused), and the label the result earned: rebound before the
                # next PreLLMCall so the egress gate decides against what is actually in
                # the prompt.
                observation = executed.observation
                classification = executed.classification
                step.observation = observation
                if halts:
                    trace.final_answer = self._pause_for_approval(
                        trace=trace,
                        decision=halts[0],
                        iteration=iteration,
                        memory_context=memory_context,
                    )
                    trace.success = True  # waiting on the owner is a complete turn
                    break
                if step.action in _RETRIEVAL_TOOL_NAMES:
                    retrieval_used = True
                history.append(
                    f"Thought: {step.thought}\nAction: {step.action}\n"
                    f"Action Input: {json.dumps(step.action_input)}\nObservation: {observation}"
                )
            else:
                candidate = raw.strip()
                if candidate and not step.thought and not _looks_like_internal_planning(candidate):
                    verdict = self._vet_final_answer(
                        candidate, trace.steps, trace.effects_executed, answer_retries
                    )
                    if verdict.kind == "retry":
                        answer_retries += 1
                        history.append(_hold_back_answer(step, candidate, verdict.note))
                        continue
                    if verdict.kind == "fail":
                        trace.final_answer = UNGROUNDED_ANSWER
                        trace.success = False
                        trace.ungrounded = True
                        break
                    trace.final_answer = verdict.text if verdict.kind == "replace" else candidate
                    trace.success = True
                    break
                malformed_steps += 1
                recovered = _last_good_observation(trace.steps)
                if malformed_steps >= 2:
                    trace.final_answer = recovered or _MALFORMED_STEP_FALLBACK
                    trace.success = bool(recovered)
                    break
                history.append(
                    "System: Your last reply was internal reasoning or incomplete. "
                    "Do NOT describe your plan. Either call exactly one tool, or reply with "
                    "'Final Answer: <user-facing answer>'."
                )
                continue

            # Evaluator PostStep — runs *after* the step is appended and
            # the tool observation is in. A halt here writes a resume
            # checkpoint and breaks the loop with a user-safe message.
            evaluator_decision = self._fire_post_step(
                run_id=run_id,
                iteration=iteration,
                step=step,
                classification=classification,
                original_task=query,
            )
            if evaluator_decision is not None and evaluator_decision.outcome in (
                "deny",
                "require_approval",
            ):
                trace.halted_by = "evaluator"
                trace.halt_reason = evaluator_decision.reason
                self._write_checkpoint(
                    trace=trace,
                    iteration=iteration,
                    memory_context=memory_context,
                    signal=("halt" if evaluator_decision.outcome == "deny" else "require_approval"),
                )
                self._link_approval_checkpoint_for(evaluator_decision, trace)
                trace.final_answer = self._evaluator_block_message(
                    evaluator_decision, trace.steps, run_id=run_id
                )
                trace.success = False
                break

            direct = self._direct_answer(step, trace.steps[:-1])
            if direct:
                logger.info(
                    "AgenticCore: %s answers directly — ending the run on its result",
                    step.action,
                )
                trace.final_answer = direct
                trace.success = True
                break

        if not trace.final_answer:
            trace.final_answer = "Max iterations reached without a final answer."

        trace.elapsed_ms = (time.monotonic() - start) * 1000
        trace.stall_count = stall_count
        if trace.pending_approval_id is None and trace.halt_reason != "awaiting_user_input":
            self._report_run_complete(
                CompletedRun(
                    run_id=run_id,
                    query=query,
                    steps=tuple(_step_to_dict(s) for s in trace.steps),
                    final_answer=trace.final_answer,
                    success=trace.success,
                )
            )
        return trace

    def _vet_final_answer(
        self, answer: str, steps: list[ReactStep], effects: Sequence[str], retries_used: int
    ) -> Verdict:
        """answer_guard's verdict on a final answer, logged when it is not ``ok``.

        The evidence is this run's executed effects plus any write among the steps a
        resume replayed (``effects_executed`` starts empty on a resume).
        """
        ran = [
            self._tool_index.get(s.action or "") or self._reserve_index.get(s.action or "")
            for s in steps
            if s.action and s.observation is not None and not s.observation.startswith(_NOT_RUN)
        ]
        written = [t.effect for t in ran if t is not None and t.effect != "read"]
        if self.config.read_first and not any(
            t is not None and t.effect == "read" and t.name != ASK_USER_ACTION for t in ran
        ):
            verdict = self._read_first_verdict(retries_used)
            self._log_answer_guard(verdict)
            return verdict
        verdict = check_final_answer(
            answer,
            effects_executed=[*effects, *written],
            retries_used=retries_used,
            any_tool_ran=bool(ran),
            request=self._current_query or "",
            markers=_SCAFFOLD_MARKERS,
            fallback=_last_good_observation(steps) or _MALFORMED_STEP_FALLBACK,
        )
        self._log_answer_guard(verdict)
        return verdict

    def _read_first_verdict(self, retries_used: int) -> Verdict:
        """A read-first turn answered before any read tool returned: turn it back once,
        naming the reads on this turn's menu; after that, fail the run."""
        if retries_used >= 1:
            return Verdict("fail", reason="answered_without_reading")
        reads = [t.name for t in self._tools if t.effect == "read" and t.name != ASK_USER_ACTION]
        return Verdict(
            "retry",
            note=(
                "This question is about the user's own data, and you have not looked "
                "anything up. Do not answer from memory or invent details. Call one of "
                f"these tools first: {', '.join(reads[:8]) or 'a read tool'}."
            ),
            reason="answered_without_reading",
        )

    def _log_answer_guard(self, verdict: Verdict) -> None:
        if verdict.kind == "ok":
            return
        from iris_harness.foundation.observability.session_log import log_timeline_event

        logger.warning("AgenticCore: final answer %s (%s)", verdict.kind, verdict.reason)
        log_timeline_event(
            "answer_guard",
            phase="answer_guard",
            payload={"verdict": verdict.kind, "reason": verdict.reason},
        )

    def _report_run_complete(self, run: CompletedRun) -> None:
        if self._review_route is None:
            return
        review_completed_run(run, route=self._review_route, agent_type=self._agent_type)

    def _force_final_synthesis(
        self,
        *,
        query: str,
        history: list[str],
        memory_context: MemoryContext | None,
        run_id: str,
        step_id: int,
        classification: DataClassification | None,
        last_observation: str,
    ) -> str:
        """One last inference pass after a repeated action.

        ``step_id`` is the step this pass takes in the ledger: the one after the step
        that repeated itself (the run ends here, so no loop step reuses it).

        Nudges the model to answer from the data it already has (in ``history``)
        and to stop calling tools. If it still won't produce a Final Answer, fall
        back to the last good observation so the loop always ends with the data
        rather than a "repeated action" dead-end.
        """
        nudge = (
            "System: You already ran that tool and its result is in the Observation "
            "above. Do NOT call any tool again. Write your Final Answer now, using "
            "that result."
        )
        prompt = _build_react_prompt(
            query,
            self._tools,
            [*history, nudge],
            memory_context,
            require_retrieval_before_final=False,
            expected_list_count=None,
        )
        decision, prompt, _ = self._governance_pre_llm(
            query=query,
            prompt=prompt,
            run_id=run_id,
            step_id=step_id,
            classification=classification,
        )
        if decision is not None and decision.outcome in ("deny", "require_approval"):
            return last_observation or self._governance_block_message(decision)
        if self._llm is None:
            return last_observation
        try:
            raw = self._llm(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("forced synthesis LLM call failed: %s", exc)
            return last_observation
        step = _parse_react_step(raw)
        if step.final_answer:
            return step.final_answer
        return last_observation or (step.thought or raw).strip()

    def _is_ask_user(self, step: ReactStep) -> bool:
        """True when this step is the loop asking the user something (ADR-0106 Tier B).

        Gated on the config flag as well as the action name, so a model that invents
        ``ask_user`` where it was never offered falls through to the usual
        unknown-tool handling instead of silently gaining a way to stop the turn.
        """
        return bool(self.config.allow_ask_user and step.action == ASK_USER_ACTION)

    def _pause_for_user(
        self,
        *,
        trace: ReactTrace,
        step: ReactStep,
        iteration: int,
        memory_context: MemoryContext | None,
    ) -> str:
        """Stop the loop on an ``ask_user`` step and return the question to put.

        The pause is a checkpoint, not a discard: the run is resumable from
        ``iteration + 1`` once the human answers, which is what separates Tier B from
        the re-prompt path. Recording it is best-effort — a checkpoint that cannot be
        written costs the mid-loop resume, not the question, so the user is still
        asked either way.
        """
        question = _ask_user_question(step, fallback=trace.query)
        step.observation = f"Asked the user: {question}"
        # What the owner sees carries the items the question is about; the model's own
        # history already has them, so the observation keeps the bare question.
        shown = _with_found_items(question, trace.steps[:-1], self._tools)
        trace.halted_by = "ask_user"
        trace.halt_reason = "awaiting_user_input"
        self._write_checkpoint(
            trace=trace,
            iteration=iteration,
            memory_context=memory_context,
            signal="awaiting_user_input",
        )
        logger.info("loop paused for user input at step %s (run %s)", iteration, trace.run_id)
        return shown

    def _settle_pending_approval(self, seed: ResumeSeed, *, effects: list[str]) -> ResumeSeed:
        """Resolve the approval a halted destructive call waited on (ADR-0118).

        Approved: execute exactly the calls the row pinned — never the model's next
        guess — each claiming the approval, which the kernel hook re-checks against the
        row. Rejected, expired, or unreadable: execute nothing. Either way the result
        becomes the halted step's observation, so the model continues from what really
        happened and tells the owner. A seed with no pending approval passes through.
        """
        approval_id = seed.pending_approval_id
        if not approval_id or not seed.steps:
            return seed
        verdict = self._read_approval(approval_id) if self._read_approval is not None else None
        if verdict is None:
            observation = f"{_APPROVAL_UNREADABLE} ({approval_id}). Nothing was changed."
        else:
            status, items = verdict
            listed = "; ".join(f"{tool} {json.dumps(args, sort_keys=True)}" for tool, args in items)
            if status == "approved":
                results = []
                for tool, args in items:
                    executed = self._execute_tool(
                        tool,
                        args,
                        run_id=seed.run_id,
                        classification=self._classification,
                        step_id=seed.start_iteration - 1,
                        effects=effects,
                        approved_by=approval_id,
                    )
                    results.append(f"{tool}: {executed.observation}")
                    # The label the result earned carries into the resumed loop as a
                    # floor. A set label is raised now (the runner never lowers it); an
                    # unset one stays unset, so PRE_CLASSIFY still derives it from the
                    # question, and the floor is applied under that
                    # (``_governance_pre_llm``) -- seeding the label from the result
                    # alone could come out lower than the question's.
                    self._earned_floor = more_restrictive(
                        self._earned_floor, executed.classification
                    )
                    if self._classification is not None:
                        self._classification = executed.classification
                observation = f"{_APPROVAL_APPROVED} Results:\n" + "\n".join(results)
            elif status == "rejected":
                observation = (
                    f"{_APPROVAL_REJECTED} {listed}. It was not run: nothing was "
                    "changed, sent or deleted. Tell them so plainly and do not try it again."
                )
            else:
                observation = f"{_APPROVAL_SETTLED}{status}, so nothing was changed: {listed}."
        steps = list(seed.steps)
        last = steps[-1]
        steps[-1] = ReactStep(
            thought=last.thought,
            action=last.action,
            action_input=dict(last.action_input),
            observation=observation,
            final_answer=last.final_answer,
            is_terminal=last.is_terminal,
        )
        history = list(seed.history)
        if last.action and history:
            history[-1] = (
                f"Thought: {last.thought}\nAction: {last.action}\n"
                f"Action Input: {json.dumps(last.action_input)}\n"
                f"Observation: {observation}"
            )
        logger.info("settled approval %s for run %s", approval_id, seed.run_id)
        return replace(seed, steps=steps, history=history, pending_approval_id=None)

    def _pause_for_approval(
        self,
        *,
        trace: ReactTrace,
        decision: HookDecision,
        iteration: int,
        memory_context: MemoryContext | None,
    ) -> str:
        """Stop the loop on a destructive call that waits for the owner (ADR-0118).

        The same shape as an evaluator halt: checkpoint the run, link the approval to
        it, tell the user where to answer. The checkpoint names the approval, so the
        resumed run knows to settle it — run the pinned calls, or not — before it
        continues. Nothing here claims the next chat message: only answering the
        approval resumes this run.
        """
        approval_id = str(decision.approval_request_id)
        trace.pending_approval_id = approval_id
        trace.halted_by = "approval"
        trace.halt_reason = decision.reason
        self._write_checkpoint(
            trace=trace,
            iteration=iteration,
            memory_context=memory_context,
            signal="require_approval",
            pending_approval_id=approval_id,
        )
        self._link_approval_checkpoint_for(decision, trace)
        logger.info(
            "loop paused for approval %s at step %s (run %s)", approval_id, iteration, trace.run_id
        )
        what = self._pending_title or "That needs your approval"
        return (
            f"{what}: this needs your approval, so nothing has changed yet. "
            f"[Review it in Activity](/actions), or answer with `iris approvals` or on "
            f"Telegram (approval {approval_id})."
        )

    def _answers_directly(self, name: str | None) -> bool:
        tool = self._tool_index.get(name or "")
        return bool(tool is not None and tool.answers_directly)

    def _direct_answer(self, step: ReactStep, earlier: list[ReactStep]) -> str:
        """The tool output that ends the run with no further model call, or "".

        Only for a tool declared ``answers_directly`` (ADR-0110), and only when it is
        the run's first tool: its output is then the whole answer to the request, and
        another model call would only restate it — a local model given that call
        re-issued the same tool instead (a digest turn paid three calls for one
        answer). A later step may be part of a larger answer, so the model keeps it.
        Asking the user is a clarification, not a tool result, and does not count.
        """
        if not self._answers_directly(step.action):
            return ""
        if any(s.action and s.action != ASK_USER_ACTION for s in earlier):
            return ""
        return _usable_observation(step.observation)

    def _answer_in_hand(self, name: str | None, earlier: list[ReactStep]) -> str:
        """For a repeated ``answers_directly`` tool: its result already in hand, or "".

        The repeat guard otherwise pays a whole synthesis call to have the model copy
        that result out; for a tool whose output is the answer, the copy is the result.
        """
        if not self._answers_directly(name):
            return ""
        for s in reversed(earlier):
            if s.action == name:
                usable = _usable_observation(s.observation)
                if usable:
                    return usable
        return ""

    def _execute_tool(
        self,
        name: str,
        args: dict[str, Any],
        *,
        run_id: str | None = None,
        classification: DataClassification | None = None,
        step_id: int | None = None,
        asked_user: bool = False,
        effects: list[str] | None = None,
        approved_by: str | None = None,
        halts: list[HookDecision] | None = None,
    ) -> _ToolStep:
        """Resolve the name the model wrote and run the tool through the governed runner.

        Returns what the model sees next (the output as ``POST_TOOL_USE`` left it, a
        governance message, or an error) and the run's label after the call, which the
        runner's ``POST_TOOL_USE`` step raised when the result was more sensitive.
        """
        unchanged = classification
        tool = self._tool_index.get(name)
        if tool is None and name in self._reserve_index:
            # Off the menu, not out of the pool: the shortlist trimmed the prompt, and
            # something on the loop (a plugin's guidance, an observation) named the
            # tool anyway. Admit it for the rest of the run.
            tool = self._reserve_index.pop(name)
            self._tool_index[name] = tool
            self._tools.append(tool)
            logger.info("react: admitted reserve tool %r on first call", name)
        if tool is None:
            # The model sometimes hallucinates a near-miss tool name
            # ("portfolio_summary" for "portfolio"). Resolve it to the obvious real
            # tool so its clear intent succeeds instead of stalling; only fall back to
            # an error (with a "did you mean") when there's no confident match.
            alias = _resolve_tool_alias(name, self._tool_index)
            if alias is not None:
                logger.info("react: resolved hallucinated tool %r -> %r", name, alias)
                name, tool = alias, self._tool_index[alias]
            else:
                import difflib

                hint = difflib.get_close_matches(
                    name.lower(), [n.lower() for n in self._tool_index], n=1, cutoff=0.5
                )
                suggestion = f" Did you mean '{hint[0]}'?" if hint else ""
                available = ", ".join(self._tool_index) or "none"
                return _ToolStep(
                    f"Error: unknown tool '{name}'.{suggestion} Available: {available}",
                    unchanged,
                )

        # Everything from here is governance, not the loop: the one runner every tool
        # call passes, whoever makes it (docs/architecture/plugin-capabilities.md).
        outcome = self._tool_runner().execute(
            tool,
            args,
            ToolCall(
                run_id=run_id,
                classification=classification,
                step_id=step_id,
                asked_user=asked_user,
                approved_by=approved_by,
                query=self._current_query,
            ),
            effects=effects,
        )
        if outcome.status != "held":
            return _ToolStep(outcome.text, outcome.classification)
        governance_decision = outcome.decision
        assert governance_decision is not None
        if (
            halts is not None
            and governance_decision.outcome == "require_approval"
            and governance_decision.approval_request_id is not None
        ):
            # An approval that exists in the queue is something to wait on, not advice:
            # the loop halts on it (ADR-0118). Confirm-once decisions carry no id.
            halts.append(governance_decision)
            self._pending_title = card_title(tool.name, outcome.approval_card)
            return _ToolStep(
                f"{_APPROVAL_WAITING} ({governance_decision.approval_request_id}).", unchanged
            )
        return _ToolStep(self._governance_block_message(governance_decision), unchanged)

    def _tool_runner(self) -> GovernedToolRunner:
        """The governed tool runner, bound to this run's kernel and conversation."""
        return GovernedToolRunner(
            kernel=self._kernel,
            agent_type=self._agent_type,
            origin_channel=self._origin_channel,
            session_id=self._session_id,
            resumable=(
                self._checkpoint_store is not None
                and self._link_approval_checkpoint is not None
                and self._read_approval is not None
            ),
        )

    # ------------------------------------------------------------------
    # Governance enforcement
    # ------------------------------------------------------------------

    def _apply_context_budget(self, history: list[str]) -> tuple[list[str], int]:
        """Trim the running transcript to the configured token budget (ADR-0077).

        Returns the (possibly trimmed) history and the estimated tokens evicted.
        Evicted blocks are dropped from the working transcript for good — the point
        is to stop stale observations from re-entering every subsequent prompt.
        """
        budget = self.config.history_token_budget
        if not budget:
            return history, 0
        trimmed, evicted = _trim_react_history(history, budget)
        if evicted:
            logger.debug(
                "react context guardrail evicted ~%d tokens of transcript (budget %d)",
                evicted,
                budget,
            )
        return trimmed, evicted

    def _governance_pre_llm(
        self,
        *,
        query: str,
        prompt: str,
        run_id: str,
        classification: DataClassification | None,
        step_id: int | None = None,
        context_tokens: int | None = None,
        evicted_tokens: int | None = None,
    ) -> tuple[HookDecision | None, str, DataClassification | None]:
        """Run PreClassify (lazily) and PreLLMCall against the kernel.

        ``step_id`` is the loop step the model call belongs to, so each call of a run
        has its own ``(run_id, step_id)`` in the audit ledger, as the step's tool rows
        do; without it every call of a run shared one key and the ledger could not tell
        how many model calls the run made (R14: every model call has its audit row).

        Returns the PreLLMCall decision, the possibly transformed prompt,
        and the classification cached for this ReAct run. The cache is only
        seeded here: PostToolUse output classification propagation (design
        §6.5, the runner's `post`) raises it when a tool result is more
        sensitive than the question was, and the raised value is what the
        caller passes back in on the next iteration.
        """
        # ADR-0081: surface in-loop budget pressure to the runtime's context-health view.
        # Fired before the kernel check so it works on the legacy (no-kernel) path too.
        if self._budget_observer is not None and context_tokens is not None:
            try:
                self._budget_observer(context_tokens, evicted_tokens or 0)
            except Exception:  # telemetry must never break a turn
                logger.debug("budget observer raised; ignoring", exc_info=True)

        if self._kernel is None:
            return None, prompt, classification

        if classification is None:
            classify_ctx = HookContext(
                hook_point=HookPoint.PRE_CLASSIFY,
                run_id=run_id,
                agent_type=self._agent_type,
                payload={"prompt": prompt, "query": query},
            )
            classify_decision, classified_ctx = self._kernel.fire_sync(
                HookPoint.PRE_CLASSIFY, classify_ctx
            )
            # A label a tool result already earned in this run (an approved call executed
            # on resume, before the loop derived one) is a floor under what the question
            # classifies as: never lower than what is already in the prompt.
            classification = more_restrictive(classified_ctx.classification, self._earned_floor)
            if classify_decision.outcome in ("deny", "require_approval"):
                return classify_decision, prompt, classification

        # The turn's label is a floor too, read before every model call: the screen's label
        # for what the user said, lifted by any governed call's result this turn -- a
        # plugin's code call inside a tool included, whose result this loop never sees as
        # an outcome (kernel/governance/turn_label.py). Never lowers the run's own. The same
        # rule every other model call in the turn goes through (apply_turn_floor).
        classification = apply_turn_floor(classification)

        # ADR-0077: surface context-budget telemetry so the kernel audits how much
        # context entered each LLM call and how much the guardrail evicted.
        llm_payload: dict[str, Any] = {"prompt": prompt, "query": query}
        if context_tokens is not None:
            llm_payload["context_tokens"] = context_tokens
        if evicted_tokens:
            llm_payload["evicted_tokens"] = evicted_tokens
        identity = self._next_model_identity()
        if identity is not None:
            llm_payload["model"], llm_payload["provider"] = identity
        llm_ctx = HookContext(
            hook_point=HookPoint.PRE_LLM_CALL,
            run_id=run_id,
            step_id=step_id,
            agent_type=self._agent_type,
            classification=classification,
            tier=self._target_tier,
            payload=llm_payload,
        )
        decision, final_ctx = self._kernel.fire_sync(HookPoint.PRE_LLM_CALL, llm_ctx)
        prompt_value = final_ctx.payload.get("prompt", prompt)
        transformed_prompt = prompt_value if isinstance(prompt_value, str) else prompt
        return decision, transformed_prompt, final_ctx.classification

    def _next_model_identity(self) -> tuple[str, str] | None:
        """``(model, provider)`` for the call about to be audited, or None when unknown."""
        if self._model_identity is None:
            return None
        try:
            return self._model_identity()
        except Exception:  # the ledger's detail must never break the turn
            logger.warning("could not resolve the model for the audit row", exc_info=True)
            return None

    @staticmethod
    def _governance_block_message(decision: HookDecision) -> str:
        return governance_block_message(decision)

    def _fire_post_step(
        self,
        *,
        run_id: str,
        iteration: int,
        step: ReactStep,
        classification: DataClassification | None,
        original_task: str | None = None,
    ) -> HookDecision | None:
        """Fire the evaluator at PostStep. Returns None when kernel is absent."""
        if self._kernel is None:
            return None
        tool_args_hash = _hash_action_input(step.action_input) if step.action else None
        tool_args_text = _render_action_input(step.action_input) if step.action else None
        tool_error: str | None = None
        if step.observation and step.observation.startswith("Error:"):
            tool_error = step.observation
        ctx = HookContext(
            hook_point=HookPoint.POST_STEP,
            run_id=run_id,
            agent_type=self._agent_type,
            step_id=iteration,
            payload={
                "thought": step.thought,
                "tool_name": step.action,
                "tool_args_hash": tool_args_hash,
                "tool_args_text": tool_args_text,
                "tool_error": tool_error,
                "original_task": original_task,
            },
            classification=classification,
            tier=self._target_tier,
            metadata={
                "origin_channel": self._origin_channel,
                "session_id": self._session_id,
            },
        )
        decision, _ = self._kernel.fire_sync(HookPoint.POST_STEP, ctx)
        return decision

    def _write_checkpoint(
        self,
        *,
        trace: ReactTrace,
        iteration: int,
        memory_context: MemoryContext | None,
        signal: str,
        pending_approval_id: str | None = None,
    ) -> None:
        """Persist a resume checkpoint. Failure does not break the run."""
        if self._checkpoint_store is None:
            return
        payload = ChatCheckpointPayload(
            query=trace.query,
            steps=tuple(
                CheckpointStep(
                    thought=s.thought,
                    action=s.action,
                    action_input=dict(s.action_input),
                    observation=s.observation,
                    final_answer=s.final_answer,
                    is_terminal=s.is_terminal,
                )
                for s in trace.steps
            ),
            iteration=iteration + 1,
            halt_reason=trace.halt_reason,
            memory_context=_memory_context_to_payload(memory_context),
            pending_approval_id=pending_approval_id,
        )
        try:
            cp = self._checkpoint_store.write(
                run_id=trace.run_id,
                step_id=iteration,
                agent_type=self._agent_type,
                payload=payload.to_payload(),
                signal=signal,
                session_id=self._session_id,
            )
        except Exception:  # checkpoint failure must not break the run
            logger.warning(
                "checkpoint write failed at step %s; run continues without a "
                "resumable checkpoint",
                iteration,
                exc_info=True,
            )
            return
        trace.checkpoint_id = f"{cp.run_id}:{cp.step_id}"

    def _link_approval_checkpoint_for(self, decision: HookDecision, trace: ReactTrace) -> None:
        """Point the approval this halt raised at the checkpoint that resumes it.

        Ordering is why this exists at all. The evaluator enqueues the approval at
        ``PostStep``, which is *before* the checkpoint is written — `hook.py` passes
        ``checkpoint_id=None`` and says so — and nothing ever came back to fill it in.
        The result was an approval that recorded a decision about a run no surface
        could find again, which is the whole reason approving has never continued
        anything. Here the checkpoint has just been written, and the approval id came
        back on the decision, so this is the one moment both halves are in hand.

        Best-effort, like every other write on a halt path: losing the link costs the
        one-click resume, not the user's answer or the record of the decision.
        """
        if self._link_approval_checkpoint is None:
            return
        approval_id = decision.approval_request_id
        if approval_id is None or not trace.checkpoint_id:
            return
        try:
            self._link_approval_checkpoint(str(approval_id), trace.checkpoint_id)
        except Exception:  # bookkeeping must not break the halt
            logger.warning(
                "could not link approval %s to checkpoint %s",
                approval_id,
                trace.checkpoint_id,
                exc_info=True,
            )

    @staticmethod
    def _evaluator_block_message(
        decision: HookDecision,
        steps: Sequence[ReactStep] = (),
        *,
        run_id: str | None = None,
    ) -> str:
        """What the user sees when the harness stops a run part-way.

        The owner's rule (ADR-0107): when the harness interrupts a task, say what
        was done, say why it stopped, and stop — then let the user decide. Not a
        bare verdict, and not an automatic retry: a half-finished task the user
        cannot see into is worse than one that explains itself.

        Two things this message used to get wrong, both found from a real halted web
        turn (session ``web-6d670ccd``, run ``b7c928e2``):

        - It closed with ``iris run resume <run_id>`` as a **literal placeholder** —
          the caller had ``run_id`` in scope and never interpolated it, so the one
          fact needed to act on the advice was the one fact withheld. It is now
          stated outright.
        - That advice was wrong on every channel, not just the ones without a
          terminal. ``iris run resume`` verifies *side effects* against the ledger
          and prints a replay hint; it does not re-enter the loop (mid-loop resume
          is ADR-0106 Tier B, and reaches an evaluator halt through no CLI command).
          Promising a resume that does not exist is worse than promising nothing, so
          the actionable line is the channel-neutral one — tell me what to do next —
          and the run id is offered for inspection, which does work.
        """
        verb = "halted" if decision.outcome == "deny" else "paused for approval"
        lines = [f"I stopped partway through — the run was {verb} by the evaluator."]
        done = _completed_work(steps)
        if done:
            lines.append("")
            lines.append("What I did before stopping:")
            lines.extend(done)
        lines.append("")
        lines.append(f"Why I stopped: {decision.reason}")
        if decision.approval_request_id is not None:
            lines.append(f"Approval ID: {decision.approval_request_id}")
        lines.append("")
        lines.append("Nothing further has run — tell me how you'd like to proceed.")
        if run_id:
            lines.append(f"Run ID: {run_id} (`iris run inspect {run_id}` has the full trace).")
        return "\n".join(lines)

    def run_stream(
        self,
        query: str,
        *,
        memory_context: MemoryContext | None = None,
        resume: ResumeSeed | None = None,
    ) -> Iterator[str | ActivityChunk | TraceChunk | dict[str, object]]:
        """Streaming ReAct loop: yields ActivityChunk per thought, TraceChunk per step, final text, metadata.

        ``resume`` continues a halted run instead of starting one (ADR-0106 Tier B):
        the seed carries the original query, the replayed steps, the rebuilt history
        and the iteration to start at, so the *same* loop body serves both. The
        alternative — a second streaming resume method beside this one — is how
        ``_loop`` and ``run_stream`` drifted apart in the first place (ADR-0107), and
        the seed exists to avoid repeating that.
        """
        if self._llm is None:
            yield "No LLM configured."
            yield {"iterations": 0, "success": False, "reason": "no_llm", "effects_executed": []}
            return
        self._current_query = resume.query if resume is not None else query

        effects_executed: list[str] = []
        pending_approval_id: str | None = None
        if resume is not None:
            resume = self._settle_pending_approval(resume, effects=effects_executed)
        history: list[str] = list(resume.history) if resume else []
        steps: list[ReactStep] = list(resume.steps) if resume else []
        start = time.monotonic()
        # See sync run() above — stall detection uses the full
        # (action, action_input) signature so drill-down tool calls
        # aren't killed.
        last_signature: str = ""
        stall_count = 0
        malformed_steps = 0
        answer_retries = 0  # answer_guard turn-backs this run (one allowed)
        final_answer = ""
        success = False
        reason = "max_iterations"
        run_id = resume.run_id if resume else str(uuid.uuid4())
        classification = self._classification
        # A resume re-enters the *original* task, not the reply that unblocked it —
        # the reply is already in `history` as the paused step's observation.
        if resume:
            query = resume.query
        start_iteration = resume.start_iteration if resume else 0
        # Bound before the loop: a resume whose start is already past the cap leaves
        # the range empty, and the metadata below reads `_iteration`.
        _iteration = start_iteration

        for _iteration in range(start_iteration, self.config.max_iterations):
            if time.monotonic() - start > self.config.timeout_seconds:
                final_answer = "Request timed out."
                reason = "timeout"
                yield final_answer  # same rule as every other terminal branch
                break

            history, evicted_tokens = self._apply_context_budget(history)
            prompt = _build_react_prompt(
                query,
                self._tools,
                history,
                memory_context,
                memory_token_budget=self.config.memory_token_budget,
                capabilities_line=self.config.capabilities_line,
            )
            governance_decision, prompt, classification = self._governance_pre_llm(
                query=query,
                prompt=prompt,
                run_id=run_id,
                step_id=_iteration,
                classification=classification,
                context_tokens=estimate_tokens(prompt),
                evicted_tokens=evicted_tokens,
            )
            if governance_decision is not None and governance_decision.outcome in (
                "deny",
                "require_approval",
            ):
                final_answer = self._governance_block_message(governance_decision)
                reason = (
                    "governance_denied"
                    if governance_decision.outcome == "deny"
                    else "governance_approval_required"
                )
                yield final_answer
                break

            try:
                raw = self._llm(prompt)
            except Exception as exc:  # noqa: BLE001 — LLM error is the answer
                final_answer = f"LLM error: {exc}"
                reason = "llm_error"
                # Surface the failure text like every other terminal branch (and
                # like sync ``run``, whose trace.final_answer carries it); without
                # this the stream ended with only metadata and the turn answered "".
                yield final_answer
                break

            step = _parse_react_step(raw)
            steps.append(step)

            if step.thought:
                yield ActivityChunk(text=f"Thought: {step.thought}")

            # The same rule as the sync loop: a question-shaped final answer is an ask.
            said = _answer_as_question(step, raw)
            guard = _guard_ask(steps, self._tools) if said else None
            if guard is not None and guard.kind == "end":
                _hold_back_answer(step, said, _ASK_ENDED)
                final_answer = guard.text
                success = True  # a true account of the turn, not a failure
                reason = "no_approval_card"
                yield TraceChunk(text=json.dumps(_step_to_dict(step)))
                yield final_answer
                break
            if guard is not None:
                history.append(_hold_back_answer(step, said, guard.text))
                yield TraceChunk(text=json.dumps(_step_to_dict(step)))
                continue

            if step.is_terminal and step.final_answer:
                verdict = self._vet_final_answer(
                    step.final_answer, steps, effects_executed, answer_retries
                )
                if verdict.kind == "retry":
                    answer_retries += 1
                    history.append(_hold_back_answer(step, step.final_answer, verdict.note))
                    yield TraceChunk(text=json.dumps(_step_to_dict(step)))
                    continue
                if verdict.kind == "fail":
                    # No text: the stream handler answers from the intent's digest, or
                    # with UNGROUNDED_ANSWER when there is none.
                    final_answer = ""
                    reason = UNGROUNDED_REASON
                    yield TraceChunk(text=json.dumps(_step_to_dict(step)))
                    break
                if verdict.kind == "replace":
                    step = replace(step, final_answer=verdict.text)
                    steps[-1] = step
                final_answer = step.final_answer or ""
                # Curation is skipped on the streaming path, so apply the shared
                # architecture-disclosure screen here — otherwise a SOUL / "who are
                # you?" summary streams IRIS internals straight to the user (and
                # into the trace metadata). Sanitize the step too so the trace
                # never carries the blocked text.
                if is_architecture_disclosure(final_answer):
                    logger.warning(
                        "run_stream: blocked internal-architecture disclosure in final answer"
                    )
                    final_answer = _ARCH_DISCLOSURE_REFUSAL
                    reason = "architecture_disclosure_blocked"
                    step = replace(step, final_answer=final_answer)
                    steps[-1] = step
                else:
                    reason = "final_answer"
                success = True
                yield TraceChunk(text=json.dumps(_step_to_dict(step)))
                yield final_answer
                break

            if step.action:
                # Ahead of the repeat guards: a turned-back ask did not run, so asking
                # again is not a repeat of anything.
                guard = _guard_ask(steps, self._tools) if self._is_ask_user(step) else None
                if guard is not None and guard.kind == "end":
                    step.observation = _ASK_ENDED
                    final_answer = guard.text
                    success = True  # a true account of the turn, not a failure
                    reason = "no_approval_card"
                    yield TraceChunk(text=json.dumps(_step_to_dict(step)))
                    yield final_answer
                    break
                if guard is not None:
                    step.observation = guard.text
                    yield TraceChunk(text=json.dumps(_step_to_dict(step)))
                    history.append(
                        f"Thought: {step.thought}\nAction: {step.action}\n"
                        f"Action Input: {json.dumps(step.action_input)}\n"
                        f"Observation: {guard.text}"
                    )
                    continue

                signature = json.dumps(
                    {"action": step.action, "input": step.action_input or {}},
                    sort_keys=True,
                    default=str,
                )
                if signature == last_signature:
                    stall_count += 1
                else:
                    stall_count = 0
                    last_signature = signature

                # This loop has no action ledger, so a repeat of the same call would run
                # the tool again. For a tool whose output is the answer, don't: return
                # the result already in hand.
                in_hand = (
                    self._answer_in_hand(step.action, steps[:-1])
                    if any(_same_call(s, step) for s in steps[:-1])
                    else ""
                )
                if in_hand:
                    logger.info(
                        "run_stream: %s answers directly — returning its result in hand",
                        step.action,
                    )
                    final_answer = in_hand
                    success = True
                    reason = "answered_by_tool"
                    yield TraceChunk(text=json.dumps(_step_to_dict(step)))
                    yield final_answer
                    break

                if stall_count >= self.config.stall_limit:
                    # Recover the last grounded observation instead of the stall
                    # sentinel (issue 0001 follow-up D / issue 0003).
                    recovered = _last_good_observation(steps)
                    if recovered:
                        final_answer = recovered
                        success = True
                        reason = "recovered_observation"
                    else:
                        final_answer = "Stopped: repeated action without progress."
                        reason = "stalled"
                    yield TraceChunk(text=json.dumps(_step_to_dict(step)))
                    yield final_answer
                    break

                if self._is_ask_user(step):
                    pause_trace = ReactTrace(query=query, run_id=run_id)
                    pause_trace.steps.extend(steps)
                    final_answer = self._pause_for_user(
                        trace=pause_trace,
                        step=step,
                        iteration=_iteration,
                        memory_context=memory_context,
                    )
                    success = True  # asking is a complete turn, not a failure
                    reason = "awaiting_user_input"
                    yield TraceChunk(text=json.dumps(_step_to_dict(step)))
                    yield final_answer
                    break

                halts: list[HookDecision] = []
                executed = self._execute_tool(
                    step.action,
                    step.action_input,
                    run_id=run_id,
                    classification=classification,
                    step_id=_iteration,
                    asked_user=_run_asked_user(steps, step.action),
                    effects=effects_executed,
                    halts=halts,
                )
                # Same as the sync loop — a behaviour the two paths must share, or the
                # streaming surface hands the model a result governance withheld, or
                # egresses under a stale label.
                observation = executed.observation
                classification = executed.classification
                step.observation = observation
                if halts:
                    pause_trace = ReactTrace(query=query, run_id=run_id)
                    pause_trace.steps.extend(steps)
                    final_answer = self._pause_for_approval(
                        trace=pause_trace,
                        decision=halts[0],
                        iteration=_iteration,
                        memory_context=memory_context,
                    )
                    success = True  # waiting on the owner is a complete turn
                    reason = "awaiting_approval"
                    pending_approval_id = pause_trace.pending_approval_id
                    yield TraceChunk(text=json.dumps(_step_to_dict(step)))
                    yield final_answer
                    break
                yield TraceChunk(text=json.dumps(_step_to_dict(step)))

                # ADR-0107: the evaluator now runs on the streaming path too, after
                # the observation is in — the same point `_loop` fires it. Only the
                # signals with no local equivalent are enforced here
                # (STREAMING_ENFORCED_SIGNALS); step-cap / action-repeat /
                # tool-failure-streak are already contained by the guards above, and
                # halting the same run twice at two thresholds helps nobody.
                evaluator_decision = self._fire_post_step(
                    run_id=run_id,
                    iteration=_iteration,
                    step=step,
                    classification=classification,
                    original_task=query,
                )
                if evaluator_decision is not None and evaluator_decision.outcome in (
                    "deny",
                    "require_approval",
                ):
                    fired = _enforced_signal_names(evaluator_decision, STREAMING_ENFORCED_SIGNALS)
                    if fired:
                        halt_trace = ReactTrace(query=query, run_id=run_id)
                        halt_trace.steps.extend(steps)
                        halt_trace.halt_reason = evaluator_decision.reason
                        self._write_checkpoint(
                            trace=halt_trace,
                            iteration=_iteration,
                            memory_context=memory_context,
                            signal=(
                                "halt"
                                if evaluator_decision.outcome == "deny"
                                else "require_approval"
                            ),
                        )
                        self._link_approval_checkpoint_for(evaluator_decision, halt_trace)
                        # The owner's rule: summarise what ran, say why it stopped,
                        # and stop. The user decides what happens next.
                        final_answer = self._evaluator_block_message(
                            evaluator_decision, steps, run_id=run_id
                        )
                        success = False
                        reason = f"evaluator_{fired[0]}"
                        logger.warning(
                            "run_stream: halted by evaluator signals %s at step %s",
                            ", ".join(fired),
                            _iteration,
                        )
                        yield final_answer
                        break
                    logger.info(
                        "run_stream: evaluator returned %s but no enforced signal "
                        "fired; local guards own this case (%s)",
                        evaluator_decision.outcome,
                        evaluator_decision.reason,
                    )

                direct = self._direct_answer(step, steps[:-1])
                if direct:
                    logger.info(
                        "run_stream: %s answers directly — ending the run on its result",
                        step.action,
                    )
                    final_answer = direct
                    success = True
                    reason = "answered_by_tool"
                    yield final_answer
                    break

                history.append(
                    f"Thought: {step.thought}\nAction: {step.action}\n"
                    f"Action Input: {json.dumps(step.action_input)}\nObservation: {observation}"
                )
            else:
                candidate = raw.strip()
                if candidate and not step.thought and not _looks_like_internal_planning(candidate):
                    verdict = self._vet_final_answer(
                        candidate, steps, effects_executed, answer_retries
                    )
                    if verdict.kind == "retry":
                        answer_retries += 1
                        history.append(_hold_back_answer(step, candidate, verdict.note))
                        yield TraceChunk(text=json.dumps(_step_to_dict(step)))
                        continue
                    if verdict.kind == "fail":
                        final_answer = ""
                        reason = UNGROUNDED_REASON
                        yield TraceChunk(text=json.dumps(_step_to_dict(step)))
                        break
                    final_answer = verdict.text if verdict.kind == "replace" else candidate
                    success = True
                    reason = "plain_answer"
                    yield TraceChunk(text=json.dumps(_step_to_dict(step)))
                    yield final_answer
                    break
                malformed_steps += 1
                if malformed_steps >= 2:
                    recovered = _last_good_observation(steps)
                    final_answer = recovered or _MALFORMED_STEP_FALLBACK
                    success = bool(recovered)
                    reason = "recovered_observation" if recovered else "malformed_react_step"
                    yield TraceChunk(text=json.dumps(_step_to_dict(step)))
                    yield final_answer
                    break
                history.append(
                    "System: Your last reply was internal reasoning or incomplete. "
                    "Do NOT describe your plan. Either call exactly one tool, or reply with "
                    "'Final Answer: <user-facing answer>'."
                )
                continue
        else:
            final_answer = final_answer or "Max iterations reached without a final answer."
            yield final_answer

        if not success and reason in {"timeout", "llm_error", "stalled", "max_iterations"}:
            # When the loop terminated without an emitted final, ensure caller sees one.
            pass

        if pending_approval_id is None and reason != "awaiting_user_input":
            self._report_run_complete(
                CompletedRun(
                    run_id=run_id,
                    query=self._current_query or query,
                    steps=tuple(_step_to_dict(s) for s in steps),
                    final_answer=final_answer,
                    success=success,
                )
            )
        yield {
            "iterations": len(steps),
            "stall_count": stall_count,
            "success": success,
            "reason": reason,
            "trace": [_step_to_dict(s) for s in steps],
            "effects_executed": list(effects_executed),
            "pending_approval_id": pending_approval_id,
            "elapsed_ms": (time.monotonic() - start) * 1000,
            # ADR-0106 Tier B: the resume point, so the runtime can record a
            # continuation that points back into this halted run rather than only
            # at the intent that produced it.
            "run_id": run_id,
            "paused_at_step": _iteration if reason == "awaiting_user_input" else None,
        }

    # ------------------------------------------------------------------
    # Component lifecycle (retained for compatibility + tests)
    # ------------------------------------------------------------------

    def describe_pipeline(self) -> tuple[str, ...]:
        return tuple(c.value for c in self.pipeline_order)

    def validate_components(self) -> dict[str, bool]:
        return {c.value: self._component_states[c].initialized for c in self.pipeline_order}

    def initialize_components(self) -> dict[str, ComponentState]:
        for c in self.pipeline_order:
            self._component_states[c] = ComponentState(
                name=c, initialized=True, responsibility=_RESPONSIBILITIES[c]
            )
        return {c.value: self._component_states[c] for c in self.pipeline_order}

    def initialize_component(
        self,
        component: AgenticComponent,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if component not in self._component_states:
            raise ValueError(f"Unknown component: {component}")
        self._component_states[component] = ComponentState(
            name=component,
            initialized=True,
            responsibility=_RESPONSIBILITIES[component],
            metadata=metadata or {},
        )

    def is_fully_initialized(self) -> bool:
        return all(s.initialized for s in self._component_states.values())

    def generate_validation_report(self) -> dict[str, Any]:
        validation_results = self.validate_components()
        return {
            "checks": {
                "validation_results": validation_results,
                "component_checks": dict(validation_results),
                "all_checks_passed": all(validation_results.values()),
            }
        }

    def run_all_checks(self) -> bool:
        return all(self.validate_components().values())

    def overview(self) -> dict[str, Any]:
        return {
            "pipeline_order": self.describe_pipeline(),
            "config": {
                "max_iterations": self.config.max_iterations,
                "timeout_seconds": self.config.timeout_seconds,
                "max_tools_per_iteration": self.config.max_tools_per_iteration,
                "stall_limit": self.config.stall_limit,
                "max_tokens": self.config.max_tokens,
                "streaming": self.config.streaming,
            },
            "tools_registered": [t.name for t in self._tools],
            "initialized_components": self.validate_components(),
        }
