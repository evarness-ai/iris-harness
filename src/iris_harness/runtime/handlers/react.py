"""The governed ReAct handler, the tools it runs with, and the helpers only it uses.

Release gate 1: `bootstrap.py` is the composition root only. Building a handler is not
composition -- `runtime/handlers/` is where the other six builders already live, and
this was the largest one still outside it. The tool builders came with it because they
exist to feed this handler; splitting them would have left a cross-module private edge
for no gain.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from iris_harness.foundation.process_state import track_globals
from iris_harness.runtime.handlers.general_support import _normalize_handler_result

if TYPE_CHECKING:
    from iris_harness.agent.agentic_core import ResumeSeed
    from iris_harness.kernel.governance import HookDecision
import math
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from iris_harness.agent.agent_executor import (
    ActivityChunk,
    AgentTask,
    HandlerResult,
    StreamChunk,
    TraceChunk,
)
from iris_harness.llm.tier_router import TierRouter, governance_tier_for_intent
from iris_harness.memory.knowledge.wiki_engine import WikiEngine
from iris_harness.memory.semantic_index import SemanticIndex
from iris_harness.memory.state import CheckpointStore
from iris_harness.memory.store import MemoryStore
from iris_harness.runtime.brief_tools import (
    make_brief_render_tool,
    make_brief_runner,
)
from iris_harness.runtime.capabilities import capability_line, capability_report
from iris_harness.runtime.client_config import (
    MULTI_STEP_ROUTING_INTENT,
    config_from_profile,
    resolve_routing_intent,
)
from iris_harness.runtime.react_tools import (
    agent_self_management_enabled,
    builtin_react_tools,
)
from iris_harness.runtime.routine_authoring import (
    _get_semantic_router,
)
from iris_harness.runtime.skill_matching import (
    SKILL_MATCH_MIN_SCORE,
    score_skill_package,
)
from iris_harness.runtime.turn_context import set_current_query, set_current_session_id
from iris_harness.services.learning.store import LearningMetricsStore
from iris_harness.tools.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

# Process-wide lazy singleton for the search-synthesis client; it moved out of
# bootstrap.py with its only reader (release gate 1).
_SEARCH_SYNTHESIS_CLIENT: Any = None
_SEARCH_SYNTHESIS_INIT = False


def _is_reasoning_model(model: str) -> bool:
    """True for OpenAI reasoning-model families (GPT-5 / o-series) served via Copilot.

    These reject the ``stop`` parameter and a custom ``temperature`` (only the default is
    allowed), so the search-synthesis path must omit both for them."""
    m = model.strip().lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


# ADR-0110 follow-up: the core names no plugin tool. A skill may not shadow a name
# already in the loop's pool (built-ins + every mounted plugin's tools) — computed
# per turn from the pool itself, not from a list the core would have to keep
# current — and the shortlist keeps the core's own two plus any tool whose manifest
# says ``pinned: true``.


def _cloud_safe_memory_context(ctx: Any) -> Any:
    """Strip ALL user/personal context from a memory context for a cloud-egress prompt.

    Privacy-first: when the search loop runs on a cloud model, the prompt must carry no
    personal data (USER.md profile, user facts, the digital-twin layers, conversation
    history, learned signals). Keep only SOUL.md — the agent's own identity (classified
    internal) — so the cloud model still behaves as IRIS. Returns a fresh MemoryContext
    with every personal field cleared, or None if there was none."""
    if ctx is None:
        return None
    from iris_harness.memory.retriever import MemoryContext

    return MemoryContext(soul=getattr(ctx, "soul", None))


def _build_search_synthesis_client() -> Any:
    """Build the opt-in cloud synthesis client, or None when not configured."""
    provider = os.getenv("IRIS_SEARCH_SYNTHESIS_PROVIDER", "").strip().lower()
    if provider != "copilot":
        return None
    if not os.getenv("IRIS_ENABLE_COPILOT_BACKEND", "").strip():
        logger.warning(
            "IRIS_SEARCH_SYNTHESIS_PROVIDER=copilot but IRIS_ENABLE_COPILOT_BACKEND is "
            "not set — falling back to the local search tier."
        )
        return None
    try:
        from iris_harness.llm.client import CodingLLMClient

        # gpt-5-mini is the cheaper Copilot tier (vs Sonnet's premium multiplier) — the
        # right default for a tool that fires on every search. Override via the env var.
        model = os.getenv("IRIS_SEARCH_SYNTHESIS_MODEL", "gpt-5-mini").strip()
        cfg = config_from_profile("copilot")
        # Reasoning models (GPT-5 / o-series) only accept the default temperature.
        # tier_name labels this role in the session log; it is not an llm_tiers.yaml
        # key, so governance still treats the call by its provider (cloud).
        update: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "tier_name": "search_synthesis",
        }
        if not _is_reasoning_model(model):
            update["temperature"] = 0.4
        cfg = cfg.model_copy(update=update)
        # Governed (NOT handled-upstream): the client's own hooks must fire so the egress
        # gate evaluates this as a cloud call. The search loop strips the user's personal
        # data upstream, so a "personal" classification here is PII inside the fetched
        # PUBLIC web content — downgrade it to internal for the egress gate (secrets stay
        # hard-blocked). This declaration is also the one exemption from the turn's label
        # floor (kernel/governance/turn_label.apply_turn_floor).
        client = CodingLLMClient(
            cfg, governance_agent_type="chat", governance_stripped_public_content=True
        )
        logger.info("search synthesis: using cloud model %s via GitHub Copilot", model)
        return client
    except Exception:  # never break search if Copilot init fails
        logger.warning(
            "search synthesis: Copilot client unavailable, using local tier", exc_info=True
        )
        return None


def _refused_for_its_label(decision: HookDecision) -> bool:
    """True when the egress gate refused the synthesis call for what the step holds.

    The gate refuses the cloud synthesis client when the step's label may not reach
    ``tier_3`` -- a ``secret`` turn (never lowered, even for stripped public content) or an
    unclassified prompt (fail closed). Any other hook's refusal (a prompt guard, a cost
    limit) is not about where the call goes, and moving the step to another model would
    route around it.
    """
    from iris_harness.kernel.governance.plugins.egress_gate import EgressGate

    return decision.decided_by == EgressGate.name


def _audit_synthesis_fallback(decision: HookDecision, *, local_tier: str | None) -> None:
    """One ledger row saying the step left the cloud client for the local tier."""
    import uuid

    from iris_harness.foundation.observability.session_log import current_session_id
    from iris_harness.foundation.paths import audit_db_path
    from iris_harness.kernel.governance.audit import AuditLog

    payload: dict[str, Any] = {
        "fallback": "local_tier",
        "refused_by": decision.decided_by,
        "refusal": decision.outcome,
        "local_tier": local_tier,
    }
    session_id = current_session_id()
    if session_id is not None:
        payload["session_id"] = session_id
    try:
        AuditLog(db_path=audit_db_path()).record(
            run_id=str(uuid.uuid4()),
            step_id=None,
            agent_type="chat",
            hook_point="pre_llm_call",
            plugin="search_synthesis_fallback",
            decision="allow",
            severity="warn",
            reason=(
                "search synthesis: the egress gate refused the cloud call "
                f"({decision.reason}); the step ran on the local tier"
            ),
            classification=decision.audit_metadata.get("classification"),
            tier=local_tier,
            payload=payload,
        )
    except Exception:  # the ledger must never break the turn
        logger.warning("search synthesis: could not audit the local fallback", exc_info=True)


def _search_synthesis_client() -> Any:
    """Process-wide cloud synthesis client (lazy, built once)."""
    global _SEARCH_SYNTHESIS_CLIENT, _SEARCH_SYNTHESIS_INIT
    if not _SEARCH_SYNTHESIS_INIT:
        _SEARCH_SYNTHESIS_INIT = True
        _SEARCH_SYNTHESIS_CLIENT = _build_search_synthesis_client()
    return _SEARCH_SYNTHESIS_CLIENT


def _self_management_tools(runtime_holder: list[Any], session_id: str | None) -> list[Any]:
    """Tools that let the agent act on its OWN state (ADR-0086): check context-window
    pressure, compact its conversation when full, and read how its learning is going.

    The runtime is built after the handler, so it arrives via ``runtime_holder`` (a
    1-slot list filled at the end of ``build_runtime``); the session is captured per turn.
    All three degrade to a plain string if the runtime isn't wired yet.
    """
    from iris_harness.agent.agentic_core import ToolSpec

    sid = session_id or "default"

    def _rt() -> Any:
        return runtime_holder[0] if runtime_holder else None

    def _context_health(_args: dict[str, Any]) -> str:
        rt = _rt()
        if rt is None:
            return "Context health unavailable."
        h = rt.sessions.context_health(sid)
        w, b, s = h["window"], h["budgets"], h["suppression"]
        lc = w.get("last_compaction")
        lc_txt = (
            f" Last compaction archived {lc['archived_count']} turns."
            if lc
            else " No compaction yet this session."
        )
        return (
            f"Context window is {w['fill_pct']:.0%} full "
            f"({w['current_tokens']}/{w['budget_tokens']} tokens"
            f"{', NEAR FULL' if w['near_full'] else ''}).{lc_txt} "
            f"In-loop transcript budget {b['transcript_budget']} tokens. "
            f"{s['active_suppressions']} proactive surface(s) suppressed."
        )

    def _compact_context(_args: dict[str, Any]) -> str:
        rt = _rt()
        if rt is None:
            return "Cannot compact right now."
        r = rt.sessions.compact_now(sid)
        if not r.get("compacted"):
            return f"Nothing to compact ({r.get('reason', 'conversation is short')})."
        return (
            f"Compacted the conversation: summarized {r['archived_count']} older turns, "
            f"reclaiming {r['tokens_before'] - r['tokens_after']} tokens "
            f"({r['tokens_before']} -> {r['tokens_after']})."
        )

    def _learning_status(_args: dict[str, Any]) -> str:
        rt = _rt()
        if rt is None:
            return "Learning status unavailable."
        from iris_harness.services.learning.proposal_quality import (
            build_proposal_quality,
        )

        q = build_proposal_quality(rt.learning_store)
        try:
            flags = rt.learning.learning_flags()
            on = [k for k, v in flags.items() if v["enabled"]]
        except Exception:  # noqa: BLE001
            on = []
        b, i = q.behaviors, q.intentions
        return (
            f"Self-learning: miners {('on: ' + ', '.join(on)) if on else 'all off'}. "
            f"Behaviors {b.accepted}/{b.reviewed} accepted ({b.acceptance_rate:.0%}), "
            f"{b.awaiting} awaiting review. Intentions {i.accepted}/{i.reviewed} "
            f"({i.acceptance_rate:.0%}), {i.awaiting} awaiting."
        )

    return [
        ToolSpec(
            name="context_health",
            description=(
                "Check your OWN context-window usage for this conversation — how full the "
                "window is, whether it's near full, and the last compaction. Use it when "
                "the user asks how much context you have left, or before a long task."
            ),
            call=_context_health,
        ),
        ToolSpec(
            name="compact_context",
            description=(
                "Summarize the older turns of THIS conversation to reclaim context-window "
                "space. Use it when context_health says the window is near full and you need "
                "room. Keeps recent turns; older ones become a summary."
            ),
            call=_compact_context,
        ),
        ToolSpec(
            name="learning_status",
            description=(
                "Check how your own self-learning is going — which miners are on, and the "
                "accept/reject tallies for proposed habits and goals. Use it when the user "
                "asks what you've been learning or how your memory is doing."
            ),
            call=_learning_status,
        ),
    ]


def _skills_to_react_tools(
    skill_registry: SkillRegistry,
    *,
    query: str = "",
    agent_name: str | None = None,
    taken: frozenset[str] = frozenset(),
) -> list[Any]:
    """Adapt loaded skill tools into ``ToolSpec`` entries for the ReAct loop.

    When *query* is non-empty, relevance is gated by the semantic router
    (cosine similarity over manifest text), with the legacy keyword scorer
    as fallback when embeddings are unavailable. Brief skills
    (``kind: brief``) are auto-wrapped as zero-arg ``render_<skill>`` tools
    so the LLM can invoke them without depending on the deterministic
    short-circuit. ``taken`` is every name already in the pool (built-ins and
    mounted plugins' tools); a skill tool with one of those names is skipped, so a
    skill can never shadow a real tool.
    """
    from iris_harness.agent.agentic_core import ToolSpec

    try:
        skill_registry.discover()
        packages = skill_registry.list_packages(only_loadable=True)
    except Exception:
        logger.exception("ReAct adapter: skill discovery failed")
        return []

    explicit_local_helper = bool(
        re.search(r"\b(local helper|local skill|promoted skill)\b", query.lower())
    )
    router = _get_semantic_router()

    def _is_relevant(package: Any) -> bool:
        if not query or explicit_local_helper:
            return True
        if router is not None:
            return bool(router.score(query, package) >= router.threshold)
        return score_skill_package(query, package) >= SKILL_MATCH_MIN_SCORE

    specs: list[Any] = []
    seen: set[str] = set()
    for package in packages:
        if agent_name and package.manifest.name != agent_name:
            continue
        if not _is_relevant(package):
            continue
        for tool_manifest, tool_class in zip(
            package.manifest.tools, package.tool_classes, strict=False
        ):
            name = tool_manifest.name
            if name in taken or name in seen:
                continue
            seen.add(name)

            def _make_call(cls: type[Any]) -> Callable[[dict[str, Any]], str]:
                def _call(args: dict[str, Any]) -> str:
                    return str(cls().invoke(args))

                return _call

            specs.append(
                ToolSpec(
                    name=name,
                    description=tool_manifest.description,
                    call=_make_call(tool_class),
                )
            )
        if package.manifest.kind == "brief" and package.manifest.brief is not None:
            runner = make_brief_runner(skill_registry, package.manifest.name)
            brief_tool_name, _brief_tool_class, brief_description = make_brief_render_tool(
                package.manifest.name, package.manifest.description, runner
            )
            if brief_tool_name in taken or brief_tool_name in seen:
                continue
            seen.add(brief_tool_name)

            def _make_brief_call(
                brief_runner: Callable[[], str],
            ) -> Callable[[dict[str, Any]], str]:
                def _call(args: dict[str, Any]) -> str:
                    return brief_runner()

                return _call

            specs.append(
                ToolSpec(
                    name=brief_tool_name,
                    description=brief_description,
                    call=_make_brief_call(runner),
                )
            )
    return specs


# The core's own always-kept tools: memory_search (recall) and ask_user (the loop's
# only way to confirm before a fan-out; it ranked 19th of 29 on the acceptance query
# before it was pinned). Plugin tools that must stay on the menu say so themselves —
# ``pinned: true`` in their manifest (ADR-0110) — and the shortlist honours it below.
_REACT_CORE_TOOL_NAMES = frozenset({"memory_search", "ask_user"})


def _terms(text: str) -> set[str]:
    """The words of ``text`` for lexical matching: lowercased, split on anything that is
    not a letter or digit (so ``trash_email`` is two words), a plural ``s`` dropped."""
    out: set[str] = set()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.add(word)
    return out


def _deterministic_tool_order(
    tools: list[Any], query: str, domain_tools: frozenset[str]
) -> list[Any]:
    """Rank ``tools`` for the turn without a model (ADR-0077 addendum, 2026-10-01).

    1. The turn's own domain first: tools of the plugin that serves the intent or agent
       the turn was routed to (``PluginRegistry.tools_serving``) — the router has
       already decided what the turn is about, and the plugins' declarations say which
       tools that is.
    2. Then lexical overlap between the query and the tool's name + description (the
       text the embedder ranks), each shared word weighted by how rare it is across
       these tools, so a word every description carries decides nothing. No word list:
       the vocabulary is the tools' own.
    3. Then pool order, so the result is the same for the same inputs.
    """
    texts = [_terms(f"{t.name} {t.description}") for t in tools]
    q_terms = _terms(query)
    n = len(tools)
    weight = {term: math.log((n + 1) / (1 + sum(term in tt for tt in texts))) for term in q_terms}

    def _key(item: tuple[int, Any]) -> tuple[int, float, int]:
        i, tool = item
        score = sum(weight[term] for term in q_terms & texts[i])
        return (0 if tool.name in domain_tools else 1, -score, i)

    return [tool for _i, tool in sorted(enumerate(tools), key=_key)]


def _shortlist_react_tools(
    tools: list[Any],
    query: str,
    *,
    cap: int,
    router: Any,
    domain_tools: frozenset[str] = frozenset(),
) -> tuple[list[Any], list[str]]:
    """Cap the toolset at ``cap``, keeping the core + the most relevant of the rest.

    Returns ``(kept_tools, dropped_names)``. The core (_REACT_CORE_TOOL_NAMES) and any
    tool declared ``pinned`` are always kept; the remaining tools are ranked by cosine
    similarity of "name + description" to the query (reusing the semantic router's
    embedder) and the top ones fill the cap. No-ops when the toolset already fits or
    the query is empty. Dropped names are returned so the caller can log them — no
    silent truncation.

    With no embedder (``router`` None, or a query or tool that will not embed — an
    offline install with no model on disk) the slots are filled by
    ``_deterministic_tool_order``: the turn's ``domain_tools`` first, then lexical
    overlap, then pool order. Pool order alone left every email tool off an email
    turn's menu.
    """
    if cap <= 0 or len(tools) <= cap or not query.strip():
        return tools, []
    core = [t for t in tools if t.name in _REACT_CORE_TOOL_NAMES or getattr(t, "pinned", False)]
    core_names = {t.name for t in core}
    rest = [t for t in tools if t.name not in core_names]
    slots = max(0, cap - len(core))
    if slots <= 0:
        kept = (core + rest)[:cap]
        kept_names = {t.name for t in kept}
        return kept, [t.name for t in tools if t.name not in kept_names]
    ranked = (
        router.rank_texts(query, [(t, f"{t.name} {t.description}") for t in rest])
        if router is not None
        else []
    )
    keep_names = {t.name for t, _ in ranked[:slots]}
    # Whatever the embedder did not rank (all of it, with no model) fills the remaining
    # slots in the deterministic order, so a missing embedding never drops a tool below
    # the cap and never leaves the menu to pool order.
    if len(keep_names) < slots:
        for t in _deterministic_tool_order(rest, query, domain_tools):
            if t.name not in keep_names:
                keep_names.add(t.name)
                if len(keep_names) >= slots:
                    break
    kept_rest = [t for t in rest if t.name in keep_names]
    kept = core + kept_rest
    kept_names = {t.name for t in kept}
    dropped = [t.name for t in tools if t.name not in kept_names]
    return kept, dropped


def _make_budget_observer(
    sink: dict[str, int] | None, config: Any
) -> Callable[[int, int], None] | None:
    """Seed the window-derived budget split into ``sink`` and return an AgenticCore
    budget observer that records each turn's latest prompt size + transcript eviction
    (ADR-0081 context-health). Returns None when no sink is wired (legacy/test path)."""
    if sink is None:
        return None
    if config.history_token_budget is not None:
        sink["transcript_budget"] = config.history_token_budget
    if config.memory_token_budget is not None:
        sink["memory_budget"] = config.memory_token_budget

    def _observer(context_tokens: int, transcript_evicted: int) -> None:
        sink["last_context_tokens"] = context_tokens
        sink["last_transcript_evicted"] = transcript_evicted

    return _observer


def _resume_seed_for(
    store_factory: Callable[[], CheckpointStore], task: AgentTask
) -> ResumeSeed | None:
    """The seed for continuing a halted run, or None to run the task normally.

    ADR-0106 Tier B (M5.C5c). The classify stage sets the resume point on the task
    when this session owed an answer to a paused run; everything downstream of that
    decision is here, so both react handlers agree on it.

    **This has to resolve before the core is built, not after.** ``_core_for``
    assembles the tool pool *against the query* — skills, domain tools and the
    relevance shortlist all key off it — and on a resume the query is the user's
    reply ("use Stardog"), not the task the paused run was working ("pick a graph
    database"). Building the core from the reply would shortlist away the very tools
    the resumed run is mid-way through using. So the seed is decoded from the
    checkpoint store directly and its ``query`` is what the core is built for.

    ``store_factory`` rather than a store: the store is lazily constructed, and an
    ordinary turn must not open ``checkpoints.db`` just to find out it is not
    resuming.

    ``task.resume_reply`` — not ``task.query`` — is what gets injected. A Tier B turn
    sets it to the user's answer; an approved *governance* halt leaves it None, because
    there the paused step's observation is a real tool result and overwriting it would
    destroy the work the resume exists to continue.

    None means "start a fresh run", and it is reached by every ordinary turn as well
    as by the degradations: no resume point, a checkpoint past its 7-day TTL, a
    payload that will not decode. The user's reply is then answered as its own turn —
    which is what the harness did before Tier B existed, so the fallback is the
    previously-shipped behaviour rather than an error path.
    """
    if task.resume_run_id is None or task.resume_step_id is None:
        return None
    from iris_harness.agent.agentic_core import resume_seed_from_checkpoint

    try:
        checkpoint = store_factory().get(run_id=task.resume_run_id, step_id=task.resume_step_id)
        seed = resume_seed_from_checkpoint(checkpoint, user_reply=task.resume_reply)
    except Exception:  # an expired or undecodable checkpoint is not an error
        logger.info(
            "tier B: no resumable checkpoint for run %s step %s; answering as a fresh turn",
            task.resume_run_id,
            task.resume_step_id,
            exc_info=True,
        )
        return None
    logger.info("tier B: resuming run %s at step %s", task.resume_run_id, seed.start_iteration)
    return seed


def _make_react_handler(
    tier_router: TierRouter,
    skill_registry: SkillRegistry,
    *,
    semantic_index: SemanticIndex | None = None,
    wiki: WikiEngine | None = None,
    repo_root: Path | None = None,
    memory_store: MemoryStore | None = None,
    learning_store: LearningMetricsStore | None = None,
    data_dir: Path | None = None,
    fallback_handlers: dict[str, Callable[[AgentTask], HandlerResult]] | None = None,
    budget_sink: dict[str, int] | None = None,
    runtime_holder: list[Any] | None = None,
) -> tuple[Callable[[AgentTask], HandlerResult], Callable[[AgentTask], Iterator[StreamChunk]]]:
    """Build sync + streaming handlers that drive the ``AgenticCore`` ReAct loop.

    ADR-0077 (governed-agentic-loop convergence): this unified loop also carries the
    domain capabilities as TOOLS — a single entity-aware ``finance_lookup`` (wrapping
    the deterministic finance digest) and the email tools (``search_inbox`` etc.) —
    so one turn can reason across finance data AND the inbox (e.g. "find my
    store-card dues" → finance_lookup finds nothing local → search_inbox the email). The
    model selects; intent only biases tier/prompt. ``fallback_handlers`` maps an intent
    (``finance`` / ``calendar`` / ``planner``) to its deterministic handler — the
    degrade path when the loop fails on a turn of that intent, so the worst case is
    never worse than today's digest.
    """
    from iris_harness.agent.agentic_core import (
        ASK_USER_TOOL,
        UNGROUNDED_ANSWER,
        UNGROUNDED_REASON,
        AgenticCore,
        AgenticCoreConfig,
    )
    from iris_harness.kernel.governance import kernel_from_env

    governance_kernel = kernel_from_env()
    # Stop sequences for the ReAct LLM call. Prevent the model from
    # hallucinating fake "Observation: ..." (which the orchestrator
    # supplies after running tools) or a fake next "User: ..." turn.
    # Both tokens are unambiguous — the LLM should never legitimately
    # generate them. Belt-and-suspenders with the tightened
    # _FINAL_ANSWER_RE regex in agentic_core.py.
    _REACT_STOP_SEQUENCES = ("\nUser:", "\nObservation:")

    def _llm_call(prompt: str, routing_intent: str) -> str:
        # Opt-in cloud synthesis: the "search" loop runs on a cloud model (Copilot's
        # Claude) when IRIS_SEARCH_SYNTHESIS_PROVIDER=copilot — it follows ReAct cleanly
        # and synthesizes web results far better than the local 7B. Governed (the client's
        # own egress hooks fire), so the cloud call is egress-evaluated.
        if routing_intent == "search":
            synthesis_client = _search_synthesis_client()
            if synthesis_client is not None:
                # The full ReAct prompt is in `prompt`; cloud APIs (Copilot/Claude) reject
                # an EMPTY system message, so pass a minimal non-empty one. Reasoning
                # models (GPT-5 / o-series) also reject the `stop` parameter — omit it.
                syn_model = getattr(getattr(synthesis_client, "config", None), "model", "")
                syn_kwargs: dict[str, Any] = {}
                if not _is_reasoning_model(str(syn_model)):
                    syn_kwargs["stop"] = _REACT_STOP_SEQUENCES
                from iris_harness.llm.client import GovernanceBlockedError

                try:
                    return str(
                        synthesis_client.invoke(
                            system_prompt="You are a precise research assistant. Follow the "
                            "instructions in the message exactly.",
                            user_prompt=prompt,
                            **syn_kwargs,
                        )
                    )
                except GovernanceBlockedError as blocked:
                    # The gate refused this step the cloud (a secret turn): nothing was
                    # sent. The step runs on the local tier below -- the model the loop
                    # would use without synthesis, which the loop's own PRE_LLM_CALL has
                    # already governed for this prompt at that tier. Any other refusal
                    # stands.
                    if not _refused_for_its_label(blocked.decision):
                        raise
                    local_tier = (
                        governance_tier_for_intent(tier_router, routing_intent)
                        if governance_kernel is not None
                        else None
                    )
                    logger.info(
                        "search synthesis: egress gate refused the cloud call; "
                        "the step runs on the local tier (%s)",
                        local_tier,
                    )
                    _audit_synthesis_fallback(blocked.decision, local_tier=local_tier)
        # Tier follows the turn's routing intent (search / multi-step escalate to
        # tier-2) instead of being pinned to "general" — a small tier-1 model on a
        # tool-heavy ReAct loop fails to call tools and just refuses.
        cfg = tier_router.get_llm_config(routing_intent)
        from iris_harness.llm.client import CodingLLMClient, CodingLLMConfig

        # AgenticCore._loop fires PRE_CLASSIFY + PRE_LLM_CALL before every
        # call to this function (enforced by the AST tests in
        # tests/security); the client-internal hooks would fire the same
        # pair AGAIN on the same prompt — measured as 2x audit inflation
        # and a duplicate classifier run per ReAct step (Phase 1 journal).
        client = CodingLLMClient(
            CodingLLMConfig(**vars(cfg)),
            governance_handled_upstream=True,
        )
        return client.invoke(
            system_prompt="",
            user_prompt=prompt,
            stop=_REACT_STOP_SEQUENCES,
        )

    # ADR-0077 P3: derive the per-turn context budget from the model's window via the
    # ContextBudgetController, so admission (memory block) + eviction (transcript) scale
    # automatically when the model is swapped for a bigger-context one (harness stays
    # model-agnostic). The loop runs at the tier-2 window; budget_for reserves room for
    # the response. Env overrides win for tuning.
    from iris_harness.agent.context_budget import ContextBudgetController
    from iris_harness.llm.budget import budget_for

    try:
        _t2_num_ctx = getattr(tier_router.get_tier("communication"), "num_ctx", None) or 8192
    except Exception:  # noqa: BLE001 — partial/stub routers (tests) → default window
        _t2_num_ctx = 8192
    _ctx_ctrl = ContextBudgetController(budget_for(int(_t2_num_ctx)))
    config = AgenticCoreConfig(
        max_iterations=int(os.getenv("IRIS_REACT_MAX_ITERATIONS", "10")),
        timeout_seconds=int(os.getenv("IRIS_REACT_TIMEOUT_SECONDS", "120")),
        streaming=True,
        # ADR-0106 Tier B. The conversational loop may stop and ask, because this is
        # the loop with a conversation to answer into: `record` turns the pause into a
        # continuation and `classify` routes the reply back. Opt out with
        # IRIS_REACT_ASK_USER=0 if an agent starts asking where it should decide.
        allow_ask_user=os.getenv("IRIS_REACT_ASK_USER", "1").strip().lower()
        not in {"0", "false", "no", "off"},
        # Transcript eviction budget + memory-block admission budget. Both come from the
        # controller (window-derived); env vars override for tuning.
        history_token_budget=int(
            os.getenv("IRIS_REACT_HISTORY_TOKEN_BUDGET", str(_ctx_ctrl.transcript_budget))
        ),
        memory_token_budget=int(
            os.getenv("IRIS_REACT_MEMORY_TOKEN_BUDGET", str(_ctx_ctrl.memory_budget))
        ),
    )

    # ADR-0081 context-health: seed the static window-derived budget split and install an
    # observer that records the latest in-loop budget snapshot into the shared sink.
    _budget_observer = _make_budget_observer(budget_sink, config)

    def routing_intent_for(task: AgentTask) -> str:
        intent = str(task.params.get("intent", "general"))
        return resolve_routing_intent(intent, bool(task.params.get("is_multi_step")))

    # One store for every core this runtime builds. Constructing it runs the schema
    # DDL, and `_core_for` runs per turn, so this is built once and lazily — a
    # profile whose turns never halt still never opens checkpoints.db.
    _checkpoint_store_holder: list[CheckpointStore] = []

    def _checkpoints() -> CheckpointStore:
        if not _checkpoint_store_holder:
            _checkpoint_store_holder.append(CheckpointStore())
        return _checkpoint_store_holder[0]

    _approval_queue_holder: list[Any] = []

    def _approval_queue() -> Any:
        """The queue the evaluator's approvals land in, for writing the resume link.

        Lazily built and shared, like the checkpoint store beside it. Both halves of
        the link are SQLite-backed at fixed paths, so this instance and the kernel's
        see the same rows.
        """
        if not _approval_queue_holder:
            from iris_harness.kernel.governance.approvals import ApprovalQueue
            from iris_harness.kernel.governance.audit import AuditLog

            _approval_queue_holder.append(ApprovalQueue(audit_log=AuditLog()))
        return _approval_queue_holder[0]

    def _link_approval_checkpoint(approval_id: str, checkpoint_id: str) -> None:
        _approval_queue().set_checkpoint(approval_id, checkpoint_id)

    def _read_approval(approval_id: str) -> tuple[str, list[tuple[str, dict[str, Any]]]] | None:
        """(status, pinned calls) for the resumed run to settle (ADR-0118)."""
        row = _approval_queue().get(approval_id)
        if row is None:
            return None
        return row.status, [(item.tool, item.args) for item in (row.items or ())]

    def _core_for(
        query: str,
        routing_intent: str,
        session_id: str | None = None,
        origin_channel: str = "console",
        read_first: bool = False,
        serving: tuple[str, ...] = (),
    ) -> AgenticCore:
        # Publish the turn's question before the pool is assembled, so a PLUGIN tool
        # (registered once at setup, with nothing to close over) still reads the
        # question it is answering — the fallback core-built tools get for free from
        # their closure (OSS plan M4.2).
        set_current_query(query)
        set_current_session_id(session_id or "")

        # `iris_doc("CAPABILITIES")` reads the live registry through this closure, so
        # the long form can never drift from what is actually loaded.
        def _capabilities_doc() -> str | None:
            runtime = runtime_holder[0] if runtime_holder else None
            if runtime is None:
                return None
            return capability_report(runtime, [t.name for t in tools])

        tools = builtin_react_tools(
            semantic_index=semantic_index,
            wiki=wiki,
            repo_root=repo_root,
            memory_store=memory_store,
            learning_store=learning_store,
            capabilities=_capabilities_doc,
        )
        # OSS plan M1: plugin tools join the same governed pool (fault-bounded by
        # the registry; executed through the same PRE_TOOL_USE path as builtins).
        if runtime_holder and runtime_holder[0] is not None:
            tools.extend(runtime_holder[0].plugin_registry.tools())
        tools.extend(
            _skills_to_react_tools(
                skill_registry, query=query, taken=frozenset(t.name for t in tools)
            )
        )
        # ADR-0086: self-management tools (context-health / compact / learning-status) so
        # the agent can act on its own state. Opt-in; participates in the shortlist below,
        # so it only surfaces on self-referential turns.
        # Consult the runtime's live state (override > env) so an experiment-console toggle
        # takes effect next turn; fall back to the env gate when no runtime is wired (tests).
        _sm_on = (
            runtime_holder[0].learning.self_management_enabled()
            if (runtime_holder and runtime_holder[0] is not None)
            else agent_self_management_enabled()
        )
        if runtime_holder is not None and _sm_on:
            tools.extend(_self_management_tools(runtime_holder, session_id))
        # ADR-0077's domain tools are all plugin-registered as of M6.1b: the email
        # tools left with their library (decision 2), as finance_lookup, daily_plan,
        # calendar_lookup and configure_brief did before them. They arrive in the
        # pool above, through `plugin_registry.tools()`.
        # ADR-0077 P2: relevance-shortlist + hard cap so a small local model never faces
        # a 20-tool menu as the pool grows (universal surfacing). The router's embedder
        # ranks tools by the turn; the core recall/search tools are always kept.
        if config.allow_ask_user:
            # Appended after the shortlist inputs are gathered but before the cap is
            # applied below — asking a question must not be the tool the shortlist
            # drops when the pool is full.
            tools.append(ASK_USER_TOOL)
        cap = int(os.getenv("IRIS_REACT_TOOL_CAP", "12"))
        full_pool = list(tools)
        # The turn's own domain: what ranks first when there is no embedder to rank by.
        domain_tools = (
            runtime_holder[0].plugin_registry.tools_serving(serving)
            if runtime_holder and runtime_holder[0] is not None
            else frozenset()
        )
        tools, dropped = _shortlist_react_tools(
            tools, query, cap=cap, router=_get_semantic_router(), domain_tools=domain_tools
        )
        if dropped:
            logger.debug("react tool shortlist kept %d, dropped: %s", len(tools), dropped)
        # ADR-0110 follow-up: the shortlist trims the MENU, not the pool. A tool it
        # dropped is still callable — a plugin's own guidance or an observation may
        # name it ("call read_email on one of those") — and the loop admits it on
        # first call instead of answering "unknown tool", which sent gpt-4o into a
        # retry loop on the acceptance query.
        reserve = [t for t in full_pool if t.name in set(dropped)]
        # ADR-0077 P2: load-aware tier escalation — a tier-1 intent facing a large tool
        # menu is exactly where the weak model fumbles tool calls. Borrow the tier-2
        # instruct model (via the multi-step routing intent) when the menu is big.
        effective_intent = routing_intent
        bump_threshold = int(os.getenv("IRIS_REACT_TIER_BUMP_TOOL_COUNT", "6"))
        if (
            governance_kernel is not None
            and len(tools) > bump_threshold
            and governance_tier_for_intent(tier_router, routing_intent) == "tier_1"
        ):
            effective_intent = MULTI_STEP_ROUTING_INTENT
            logger.debug(
                "react tier bump: %s -> %s (%d tools)",
                resolve_routing_intent,
                effective_intent,
                len(tools),
            )
        target_tier = (
            governance_tier_for_intent(tier_router, effective_intent)
            if governance_kernel is not None
            else None
        )
        # Rebuilt per turn (the tool menu is shortlisted per turn), and a copy: the
        # shared config object outlives this call.
        runtime_for_caps = runtime_holder[0] if runtime_holder else None
        core_config = (
            replace(
                config, capabilities_line=capability_line(runtime_for_caps, [t.name for t in tools])
            )
            if runtime_for_caps is not None
            else config
        )
        if read_first:
            core_config = replace(core_config, read_first=True)
        return AgenticCore(
            config=core_config,
            llm_call=lambda prompt: _llm_call(prompt, effective_intent),
            tools=tools,
            reserve_tools=reserve,
            kernel=governance_kernel,
            target_tier=target_tier,
            agent_type="system",
            # ADR-0106 C5a. The Phase-3 checkpoint spine has been complete and
            # disconnected since it shipped: nothing in production ever handed the
            # core a store, so `_write_checkpoint` returned on its first line and no
            # evaluator halt was ever resumable. Wiring it here writes a checkpoint
            # when — and only when — the PostStep evaluator halts a run, which is
            # what `iris run inspect <run_id>` was built to read.
            checkpoint_store=_checkpoints(),
            session_id=session_id,
            # Where a halt raised here should be delivered back to, and how to point the
            # approval it raises at the checkpoint that resumes it. Both were missing:
            # the queue's `channel` column only ever held its "cli" default, and its
            # `checkpoint_id` only ever held NULL.
            origin_channel=origin_channel,
            link_approval_checkpoint=_link_approval_checkpoint,
            read_approval=_read_approval,
            budget_observer=_budget_observer,
            # The governance judge reviews this run on the route it ran on (§9.2).
            review_route=effective_intent,
        )

    def _profile_recall_answer(task: AgentTask) -> str | None:
        # Deterministic recall for targeted "do you know my X" profile questions —
        # answer from stored facts instead of the small model under-grounding on its
        # own context (issue 0023).
        if task.params.get("intent") != "profile_query":
            return None
        ctx = task.memory_context
        facts = ctx.user_facts if ctx else ()
        from iris_harness.memory.profile_recall import build_profile_recall

        return build_profile_recall(task.query, facts)

    def _run_memory_context(task: AgentTask) -> Any:
        # Cloud search synthesis must not egress personal data — strip it from the prompt
        # (governance otherwise gates personal→cloud). Local runs keep full context.
        ctx = task.memory_context
        if routing_intent_for(task) == "search" and _search_synthesis_client() is not None:
            return _cloud_safe_memory_context(ctx)
        return ctx

    def _deterministic_fallback(task: AgentTask) -> HandlerResult | None:
        # Degrade a failed/empty loop turn to its intent's deterministic digest
        # (ADR-0077: finance/calendar/planner) so the worst case is never worse than
        # today — and never a web-search stall.
        if not fallback_handlers:
            return None
        fb = fallback_handlers.get(str(task.params.get("intent") or ""))
        return fb(task) if fb is not None else None

    def _serving(task: AgentTask) -> tuple[str, ...]:
        """The intent and the agent the turn was routed to, as plugins declare them."""
        return (str(task.params.get("intent") or ""), task.agent_type)

    def _read_first(task: AgentTask) -> bool:
        """The turn asks about the user's own data: a plugin lists its intent (or the
        agent it was routed to) under ``read_first_intents`` in its manifest."""
        runtime = runtime_holder[0] if runtime_holder else None
        if runtime is None:
            return False
        declared = runtime.plugin_registry.read_first_intents()
        return bool({str(task.params.get("intent") or ""), task.agent_type} & declared)

    def _ungrounded_fallback(task: AgentTask) -> HandlerResult:
        """The loop answered twice without reading: the intent's deterministic digest,
        found by intent or by the agent it was routed to, else an honest refusal."""
        fb = _deterministic_fallback(task)
        if fb is None and fallback_handlers and task.agent_type in fallback_handlers:
            fb = fallback_handlers[task.agent_type](task)
        if fb is not None:
            return fb
        return UNGROUNDED_ANSWER, {"agentic_core": True, "reason": UNGROUNDED_REASON}

    def handler(task: AgentTask) -> HandlerResult:
        recall = _profile_recall_answer(task)
        if recall is not None:
            return recall, {"profile_recall": True}
        try:
            # ADR-0106 Tier B, sync path. `chat()` drains `chat_stream()` so the
            # streaming handler below serves the conversational turn, but the wave
            # executor and every non-streaming caller land here, and a resume must not
            # depend on which of the two answered.
            seed = _resume_seed_for(_checkpoints, task)
            core = _core_for(
                seed.query if seed is not None else task.query,
                routing_intent_for(task),
                task.session_id,
                task.origin_channel,
                read_first=_read_first(task),
                serving=_serving(task),
            )
            memory_context = _run_memory_context(task)
            if seed is not None:
                trace = core.run_from_seed(seed, memory_context=memory_context)
            else:
                trace = core.run(task.query, memory_context=memory_context)
        except Exception:  # degrade, never 500 the turn
            logger.exception("agentic ReAct loop failed")
            fb = _deterministic_fallback(task)
            if fb is not None:
                return fb
            raise
        meta: dict[str, object] = {
            "agentic_core": True,
            "tools_offered": len(core.tools),  # ADR-0077 P2: post-shortlist tool count
            "iterations": trace.iterations,
            "stall_count": trace.stall_count,
            "success": trace.success,
            "elapsed_ms": trace.elapsed_ms,
            # ADR-0118 decision 5: what this run changed, and whether it continued a
            # paused run. Escalation reads both before it would re-run the turn.
            "effects_executed": list(trace.effects_executed),
            "resumed": seed is not None,
            "pending_approval_id": trace.pending_approval_id,
            "trace": [
                {
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation,
                    "final_answer": s.final_answer,
                    "is_terminal": s.is_terminal,
                }
                for s in trace.steps
            ],
        }
        if trace.ungrounded:
            return _ungrounded_fallback(task)
        answer = (trace.final_answer or "").strip()
        if not (trace.success and answer):
            fb = _deterministic_fallback(task)
            if fb is not None:
                return fb
        return trace.final_answer, meta

    def stream_handler(task: AgentTask) -> Iterator[StreamChunk]:
        recall = _profile_recall_answer(task)
        if recall is not None:
            yield recall
            return
        try:
            # ADR-0106 Tier B: the pipeline resolved this turn to an answer the paused
            # run was waiting for, so continue that run instead of starting one. A seed
            # of None (expired or undecodable checkpoint) falls through to a fresh run
            # on the reply, which is the safe degradation — an answer the user did not
            # expect beats a turn that fails because a checkpoint aged out.
            seed = _resume_seed_for(_checkpoints, task)
            core = _core_for(
                seed.query if seed is not None else task.query,
                routing_intent_for(task),
                task.session_id,
                task.origin_channel,
                read_first=_read_first(task),
                serving=_serving(task),
            )
            if seed is not None:
                yield {"resumed": True}  # ADR-0118 decision 5: never escalate a resume
            stream = core.run_stream(
                task.query, memory_context=_run_memory_context(task), resume=seed
            )
            for item in stream:
                if isinstance(item, (ActivityChunk, TraceChunk)):
                    yield item
                elif isinstance(item, dict) and item.get("reason") == UNGROUNDED_REASON:
                    # The loop yielded no text: answer from the digest instead.
                    text, fmeta = _normalize_handler_result(_ungrounded_fallback(task))
                    yield text
                    yield {"agentic_core": True, **item, **fmeta}
                elif isinstance(item, dict):
                    yield {"agentic_core": True, **item}
                else:
                    yield str(item)
        except Exception:  # degrade a finance turn to the digest
            logger.exception("agentic ReAct stream failed")
            fb = _deterministic_fallback(task)
            if fb is None:
                raise
            text, fmeta = _normalize_handler_result(fb)
            yield text
            yield fmeta

    return handler, stream_handler


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_SEARCH_SYNTHESIS_CLIENT", "_SEARCH_SYNTHESIS_INIT")
