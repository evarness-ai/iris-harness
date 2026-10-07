"""IRIS runtime composition root.

Builds and wires every subsystem (memory, agents, missions, heartbeats, channels,
wiki) into a single ``IrisRuntime`` that exposes a ``chat()`` entry point.

The runtime is intentionally synchronous — async wrappers belong in the FastAPI
service layer.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from iris_harness.runtime.plugin_host.loader import InProcessPlugin


# Imported for its side effect: it registers the eval-runtime factory the learning
# layer asks for (iris_harness.services.learning.eval_runtime).
import iris_harness.runtime.eval_runtime

# Imported for its side effect: it registers the owner-identity sources every guard
# reads (kernel/governance/identity_redaction.py); build_runtime points the facts
# source at its own memory store.
import iris_harness.runtime.identity_redaction
from iris_harness.agent.agent_executor import (
    AgentExecutor,
    AgentTask,
    HandlerResult,
    StreamChunk,
)
from iris_harness.agent.intent_router import (
    IntentRouter,
)
from iris_harness.agent.response_curator import (
    ResponseCurator,
)
from iris_harness.agent.run_review import install_run_reviewer
from iris_harness.agent.task_planner import SubTask, TaskPlanner
from iris_harness.foundation.env import env_flag as _env_flag
from iris_harness.foundation.paths import config_dir as resolve_config_dir
from iris_harness.foundation.paths import data_dir as resolve_data_dir
from iris_harness.foundation.settings import SETTINGS_DB_NAME, SettingsStore
from iris_harness.kernel.governance import kernel_from_env
from iris_harness.kernel.governance.caller_policy import register_caller_policy
from iris_harness.kernel.governance.plugin_egress import (
    bind_egress_kernel,
    register_egress_policy,
)
from iris_harness.kernel.governance.unmask_grants import register_unmask_policy
from iris_harness.llm.budget import budget_for
from iris_harness.llm.client import CodingLLMConfig, GovernedPromptCall
from iris_harness.llm.narrate import make_narrative_llm_call
from iris_harness.llm.tier_router import TierRouter
from iris_harness.memory.compactor import ConversationCompactor, summary_config
from iris_harness.memory.identity import (
    bootstrap_identity_files,
    list_episodic_patterns,
)
from iris_harness.memory.identity.loader import iris_home
from iris_harness.memory.knowledge.wiki_engine import WikiEngine
from iris_harness.memory.log_archive import LogArchive
from iris_harness.memory.retention import RetentionService
from iris_harness.memory.retriever import MemoryRetriever
from iris_harness.memory.semantic_index import SemanticIndex
from iris_harness.memory.store import MemoryStore
from iris_harness.runtime.channel_wiring import (
    _load_channels,
    _resolve_default_channel,
    _runtime_telegram_poller_enabled,
    _telegram_approval_commands,
)
from iris_harness.runtime.classifiers import _build_intent_classifier
from iris_harness.runtime.confirmations import Confirmations
from iris_harness.runtime.egress_access import compile_egress_policy
from iris_harness.runtime.facade import IrisRuntime
from iris_harness.runtime.governance_judge import build_governance_judge
from iris_harness.runtime.handlers.general import (
    _make_clarify_handler,
    _make_coding_handler,
    _make_general_handler,
)
from iris_harness.runtime.handlers.react import _make_react_handler
from iris_harness.runtime.intercept_dispatch import InterceptDispatch
from iris_harness.runtime.judges import (
    build_curator_escalation_client,
    build_curator_faithfulness_client,
    build_curator_grounding_client,
    build_curator_leak_client,
    build_curator_output_safety_client,
    curator_output_safety_timeout_s,
)
from iris_harness.runtime.learning_builders import (
    _build_crystallize_preflight,
    _build_skill_synthesizer,
)
from iris_harness.runtime.learning_controls import LearningControls
from iris_harness.runtime.mission_proposals import MissionProposals
from iris_harness.runtime.replies import DeterministicReplies
from iris_harness.runtime.rollout_flags import (
    _agentic_core_rollout_mode,
    _filemanager_agent_enabled,
    _warn_if_lane_serves_ungoverned,
)
from iris_harness.runtime.routine_authoring import (
    RoutineAuthoring,
    _capability_package_for_template,
)
from iris_harness.runtime.self_learning_loop import SelfLearningLoop
from iris_harness.runtime.session_memory import SessionMemory
from iris_harness.runtime.shadow import _make_shadow_handlers
from iris_harness.runtime.tool_access import compile_caller_policy, compile_unmask_policy
from iris_harness.runtime.tool_service import ToolService
from iris_harness.runtime.turn_capture import (
    TurnCapture,
)
from iris_harness.runtime.turn_context import current_query, current_session_id
from iris_harness.runtime.types import ChatResult
from iris_harness.runtime.welcome import FirstChatWelcome
from iris_harness.services.channels.connectors.telegram import TelegramConnector
from iris_harness.services.channels.connectors.telegram_poller import (
    TelegramPoller,
    allowed_user_ids_from_env,
)

# FileManager turn parsing lives with the domain, not the composition root
# (OSS plan M2.3). These leave the core entirely with the filemanager plugin;
# until then the intercept bodies in this file still call them.
from iris_harness.services.digest.settings import iris_timezone
from iris_harness.services.heartbeat import (
    HeartbeatScheduler,
)
from iris_harness.services.heartbeat.run_store import DB_NAME as HEARTBEAT_RUNS_DB
from iris_harness.services.heartbeat.run_store import HeartbeatRunStore
from iris_harness.services.learning.crystallizer import SkillCrystallizer
from iris_harness.services.learning.engine import LearningEngine
from iris_harness.services.learning.experiment_loop import ExperimentLoop
from iris_harness.services.learning.lesson_capture import LessonCapture
from iris_harness.services.learning.signals import LearningSignalCollector
from iris_harness.services.learning.store import LearningMetricsStore
from iris_harness.services.learning.strategies import load_strategies
from iris_harness.services.missions import MissionEngine, MissionStore
from iris_harness.services.routines import (
    RoutineStore,
)
from iris_harness.tools.skills.hot_reload import SkillHotReloader
from iris_harness.tools.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

# Sandbox pre-flight bounds (ADR-0070 s2) — each query runs baseline+variant on a
# local model, so keep the workload small and repeats modest.


# ---------------------------------------------------------------------------
# Public response shape
# ---------------------------------------------------------------------------


# ChatResult / StreamEvent / WarmupResult were extracted to
# iris_harness.runtime.types (Phase 2) and are re-exported at the top of this module.


# NLU parsing (reminder/meeting/confirmation/time-date) was extracted to
# iris_harness.runtime.nlu_parsing (Phase 2) and re-exported at the top of this module.


# ---------------------------------------------------------------------------
# ReAct-format tool-use (for local models that don't support native function
# calling reliably).  The suffix is appended to the regular system prompt so
# the same base instructions still apply.
# ---------------------------------------------------------------------------


# Cloud search-synthesis client (opt-in). When IRIS_SEARCH_SYNTHESIS_PROVIDER=copilot,
# the AgenticCore "search" loop runs on a cloud model (GitHub Copilot's Claude) instead
# of the local tier — it follows the ReAct format cleanly and synthesizes web results
# far better than the local 7B. EGRESS: the search query + crawled page content leave the
# box to Copilot, so the client is GOVERNED (its own egress hooks fire, seeing provider
# = copilot/cloud). Off by default; the local tier is unchanged when unset.


# Explicit ephemeral-scope qualifiers: the user is asking the assistant to
# treat a statement as conversation-local, not durable. Kept deliberately
# tight — only unambiguous markers, so genuine durable facts ("my name is X")
# are never suppressed. Phase 3 multiturn finding: these were being ignored,
# promoting "for this conversation, my favorite color is teal" to a global
# user fact recalled in fresh sessions.


# Interrogatives + request verbs that open a QUESTION or COMMAND, not a durable
# self-statement. Fact extraction skips these (issue 0021) — mining facts from
# "what's apple stock worth?" or "do you know my blog site?" pollutes the store and
# clobbers real facts. Declarative captures ("I write blogs at X") still run.


# On-demand brief request — "what's my daily brief", "send my morning brief", "show me
# my portfolio brief". Detection is two-part: the word "brief" PLUS a request cue
# (possessive / request verb), which tolerates a brief NAME between them ("my portfolio
# brief"). Matched as an early-intercept so it renders the brief instead of falling
# through to the email agent (issue 0030). The EXCLUDE pattern keeps authoring /
# scheduling / "routine" phrasing ("set up a morning brief every day") with the routine
# handlers, which run after this one.
# ADR-0077 P2: tools that stay on the loop regardless of relevance shortlisting — the
# minimal recall/search core every turn may need. Everything else (memory mutation,
# code_exec, most domain tools, skill tools) competes for the remaining slots by
# relevance, so a small local model never faces a 20-tool menu.
#
# Keep ``search_inbox`` in the core set: finance_lookup explicitly directs the model to
# call it when a named institution is absent locally. If relevance shortlisting drops it,
# the next step becomes an "unknown tool" loop even though the model followed guidance.


def _p4_universal_surfacing_enabled() -> bool:
    """ADR-0077 P4: also route the calendar + planner intents through the one
    governed loop (their digests are already in-loop tools). Opt-in (default off) so
    converging them is reversible independently of the base rollout, which defaults
    on — flipping this must not silently change calendar/planner behaviour. Requires
    the base loop to be available (rollout on/shadow)."""
    return os.getenv("IRIS_AGENTIC_CORE_P4", "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Heartbeat handlers
# ---------------------------------------------------------------------------


# Backwards-compatible alias retained until call sites migrate.
_brief_package_for_template = _capability_package_for_template


# ---------------------------------------------------------------------------
# Channel loader
# ---------------------------------------------------------------------------


# Sentinel for the lazily-built correction detector: "built and disabled".


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _mount_plugins(runtime: IrisRuntime, in_process: Sequence[InProcessPlugin] = ()) -> None:
    """Resolve the profile and load its plugins into ``runtime`` (OSS plan M1).

    ``in_process`` plugins join the profile after its own rows (``add_in_process``).
    """
    from iris_harness.runtime.harness_services import HarnessServices
    from iris_harness.runtime.plugin_host import add_in_process, load_plugins, load_profile
    from iris_harness.runtime.plugin_host.harness_topics import GuardedEventBus
    from iris_harness.services.health.service import register_check_provider
    from iris_harness.services.heartbeat.diagnostics import diagnose_heartbeats

    runtime.profile = add_in_process(load_profile(runtime.config_dir), in_process)
    runtime.tool_service = ToolService(
        tools=lambda: runtime.plugin_registry.tools(),
        kernel=lambda: runtime.governance_kernel,
        events=lambda: runtime._event_bus,
        deliver_in_chat=runtime._activity_notices().inject_system_notice,
    )
    services = HarnessServices(
        config_dir=runtime.config_dir,
        data_dir=runtime.data_dir,
        tier_router=runtime.tier_router,
        agent_executor=runtime.agent_executor,
        heartbeats=runtime.heartbeats,
        channels=runtime.channels,
        deterministic_reply=runtime.replies.system_chat_result,
        # Plugins get a guarded view: harness-owned topics are refused on every verb.
        events=GuardedEventBus(runtime._event_bus),
        react_handler=runtime.react_handler,
        react_stream_handler=runtime.react_stream_handler,
        # Bound late on purpose: `activities()` builds the runner + store on first
        # use, so a profile whose plugins never submit one never opens activities.db.
        submit_activity=lambda **kw: runtime._activity_notices().activities().submit(**kw),
        current_query=current_query,
        current_session_id=current_session_id,
        deliver_in_chat=runtime._activity_notices().inject_system_notice,
        embed=(runtime.semantic_index.embed if runtime.semantic_index is not None else None),
        lessons=runtime.lesson_capture,
        continuations=runtime.continuations,
        # Late-bound: the router is rebuilt when the semantic flag flips at runtime.
        classify_intent=lambda message: runtime.intent_router.classifier.classify(message),
        skill_registry=runtime.skill_registry,
        conversation_in_flight=runtime.routines.has_active_routine_conversation,
        default_channel=lambda: runtime.default_channel,
        tools=runtime.tool_service,
        heartbeat_diagnostics=lambda: diagnose_heartbeats(
            runtime.heartbeats.list_definitions(),
            runtime.heartbeats.runs(),
            created_at=runtime.heartbeats.created_at(),
        ),
    )
    # Capability calls are governed by the runtime's kernel (plugin-capabilities §4); read
    # per call, like the tool service's, and unbound they fail closed.
    runtime.plugin_registry.bind_kernel(lambda: runtime.governance_kernel)
    load_plugins(
        runtime.profile,
        services=services,
        registry=runtime.plugin_registry,
        in_process=in_process,
    )
    _reassert_loop_intents(runtime)
    # A heartbeat whose handler a plugin registers is only a fault when that plugin is
    # mounted and still did not register it (or no plugin owns it); if the plugin is
    # absent from this install it is not.
    runtime.heartbeats.bind_plugin_gap(runtime.plugin_registry.unmounted_reason)
    # The permission contract (plugin-capabilities §4): compiled from the mounted
    # manifests' `uses: tools` and the operator's tool-access.yaml, now that every plugin
    # and its tools are known. The kernel's CallerPolicyHook enforces it.
    register_caller_policy(
        compile_caller_policy(runtime.plugin_registry, config_dir=runtime.config_dir)
    )
    # Issue #103: which hosts each mounted plugin may contact through the governed HTTP
    # client, from the same manifests (`egress:`). Unregistered, every host is denied.
    register_egress_policy(compile_egress_policy(runtime.plugin_registry))
    bind_egress_kernel(lambda: runtime.governance_kernel)
    # ADR-0125: which owner-identity kinds each capability consumer sees unmasked, from
    # the same manifests; unregistered, capability masking grants nothing.
    register_unmask_policy(compile_unmask_policy(runtime.plugin_registry))
    # The governance judge (§9.2) reviews every finished loop run that used tools — the
    # core's and the plugins' — on the route it ran on. Always installed; it reads
    # IRIS_GOVERNANCE_JUDGE_ENABLED per run, so turning it on needs no restart.
    install_run_reviewer(
        build_governance_judge(tier_router=runtime.tier_router, channels=runtime.channels).review
    )
    register_check_provider("plugins", runtime.plugin_registry.health_checks)
    # Issue #136: the model guard off, or on and unable to run, while a mounted tool returns
    # third-party text. Silent when the posture is fine.
    from iris_harness.runtime.external_tools import mounted_external_tools
    from iris_harness.services.health.governance import (
        audit_writes_provider,
        model_guard_provider,
        unrecorded_egress_provider,
    )

    register_check_provider(
        "model_guard", model_guard_provider(lambda: mounted_external_tools(runtime))
    )
    # Issue #175: a governed request whose outcome row could not be written is counted.
    register_check_provider("egress_ledger", unrecorded_egress_provider)
    # Issue #134: audit rows the ledger would not take (spooled, or kept nowhere).
    register_check_provider("audit_writes", audit_writes_provider)


def _reassert_loop_intents(runtime: IrisRuntime) -> None:
    """Put the governed loop back on top of any intent it owns, after plugins mount.

    A domain that left the core brought its deterministic handler with it and
    registers it for its own intent (``finance_workflows`` → ``finance``). Plugin
    load runs after ``build_runtime``'s registrations, so that registration would
    otherwise displace the ReAct loop the rollout flag put there. It never should:
    the deterministic digest has been the loop's DEGRADE path since ADR-0077, not
    its replacement. So the plugin's handler moves into the loop's fallback dict
    and the loop is re-registered on the executor.

    A plugin that calls ``api.register_loop_intent(intent, fallback=...)`` claims the
    intent itself: the loop goes on top of it and the declared fallback is the degrade
    path, whatever the plugin registered as the intent's handler.

    With the loop off there is no loop to put back, and the plugin's lane stands —
    which is the rollout-off behaviour, unchanged.
    """
    if runtime.react_handler is None:
        return
    plugin_handlers = runtime.plugin_registry.intent_handlers()
    # Intents a plugin put on the loop itself (`api.register_loop_intent`), each with
    # the degrade path it declared. The core names no plugin's intent for these.
    claimed = runtime.plugin_registry.loop_intents()
    for intent in sorted(runtime._loop_intents | claimed.keys()):
        fallback = claimed.get(intent)
        if fallback is None:
            pair = plugin_handlers.get(intent)
            if pair is None:
                continue
            fallback = pair[0]
        runtime._react_fallbacks[intent] = fallback
        runtime.agent_executor.register(intent, runtime.react_handler)
        if runtime.react_stream_handler is not None:
            runtime.agent_executor.register_stream(intent, runtime.react_stream_handler)
        logger.info("loop intent %r re-asserted; plugin handler is its degrade path", intent)
    runtime._loop_intents = runtime._loop_intents | claimed.keys()


def _tier_llm(tier_router: TierRouter, intent: str) -> GovernedPromptCall:
    """A governed prompt->text call on the tier that ``intent`` routes to.

    Lazy: the client is built per call, so composing a runtime never reaches for a model.
    The client governs the call at the tier it goes to; the compactor fires nothing of its
    own around it (``GovernedPromptCall``).
    """
    return GovernedPromptCall(
        lambda: cast("CodingLLMConfig", tier_router.get_llm_config(intent)),
        agent_type="memory_compactor",
        system_prompt="You compress conversations accurately and invent nothing.",
    )


def build_runtime(
    *,
    config_dir: Path | None = None,
    data_dir: Path | None = None,
    llm_call: Callable[[str], str] | None = None,
    use_background_scheduler: bool = True,
    in_process_plugins: Sequence[InProcessPlugin] = (),
) -> IrisRuntime:
    """Compose every subsystem into an ``IrisRuntime``.

    Parameters
    ----------
    config_dir, data_dir
        Override the default config (``foundation.paths.config_dir()``) and ``./data``
        locations (used by tests).
    llm_call
        Optional LLM callable. When ``None``, intent classification falls back to
        keywords and the task planner produces single-task plans.
    use_background_scheduler
        When ``False``, no APScheduler is created — heartbeats can still be
        triggered manually via ``HeartbeatScheduler.trigger_now``.
    in_process_plugins
        Plugins supplied by ``setup`` instead of discovery, mounted after the profile's
        own (``plugin_host.loader.InProcessPlugin``; ``iris_harness.testing.harness``).
    """
    cfg = (config_dir or resolve_config_dir()).resolve()
    # ``$IRIS_DATA_DIR`` relocates every local data store (memory.db, chroma,
    # tasks.db, missions.db, ...) so an eval / sandbox instance is fully
    # isolated from the real user's data. The explicit ``data_dir`` arg (used by
    # tests) still wins when provided.
    data = (data_dir or resolve_data_dir()).resolve()
    data.mkdir(parents=True, exist_ok=True)
    # Seed os.environ from the owner's saved setting changes (ADR-0120; the curated
    # agent toggles of ADR-0074 §4 among them) BEFORE any subsystem reads its behaviour
    # flags. The servers already applied them at import; this covers a runtime built
    # directly (CLI, tests, playground) and imports a legacy override file once.
    from iris_harness.runtime.agent_settings_store import apply_overrides_to_env

    apply_overrides_to_env(cfg, store=SettingsStore(db_path=data / SETTINGS_DB_NAME))

    # Persistent stores are the floor under everything. The memory store is essential —
    # if it can't open, fail with a CLEAR reason rather than an opaque stack trace mid-
    # construction (the API turns a build_runtime failure into a 503). The semantic index
    # (ChromaDB) is NOT essential: on failure we degrade to keyword-only recall instead of
    # aborting startup (MemoryRetriever / the memory tools / WikiEngine all accept None).
    memory_store = MemoryStore(db_path=data / "memory.db")
    try:
        memory_store.ensure_schema()
    except Exception:
        logger.exception("FATAL: memory store could not be opened at %s", data / "memory.db")
        raise
    # The owner's confirmed facts are an identity source for the guards (ADR-0125).
    iris_harness.runtime.identity_redaction.use_memory_db(memory_store.db_path)
    bootstrap_identity_files(memory_store=memory_store)

    semantic_index: SemanticIndex | None
    try:
        # Constructing the index just opens ChromaDB (fast); the embedding work
        # — re-syncing facts/signals, which loads the ~80 MB ONNX model on the
        # first upsert — is deferred off the startup hot path below.
        semantic_index = SemanticIndex(persist_dir=data / "chroma")
    except Exception:  # degrade, never abort startup on a vector-store fault
        logger.exception(
            "semantic index unavailable at %s; degrading to keyword-only memory recall",
            data / "chroma",
        )
        semantic_index = None

    if semantic_index is not None:
        _index = semantic_index

        def _memory_sync() -> None:
            try:
                _index.sync_from_store(memory_store)
                _index.sync_episodic_patterns(
                    [(p.pattern_id, p.text) for p in list_episodic_patterns()]
                )
            except Exception:
                logger.exception("memory sync failed; recall degrades to keyword ranking")

        # Default: sync on a daemon thread so build_runtime returns immediately —
        # existing persisted embeddings already serve queries; new rows land a
        # moment later. When the index isn't ready (e.g. IRIS_TEST_NULL_EMBEDDINGS)
        # the sync is a cheap no-op, so run it inline for determinism. Set
        # IRIS_SYNC_MEMORY_BLOCKING=1 to force the old synchronous behavior.
        if _index.is_ready and not _env_flag("IRIS_SYNC_MEMORY_BLOCKING", default=False):
            threading.Thread(target=_memory_sync, name="iris-memory-sync", daemon=True).start()
        else:
            _memory_sync()
    memory_retriever = MemoryRetriever(store=memory_store, index=semantic_index)
    # ``compactor`` is built window-aware below, once ``tier_router`` is loaded.

    wiki_root = data / "wiki"
    wiki_root.mkdir(parents=True, exist_ok=True)
    wiki = WikiEngine(wiki_root=wiki_root, llm_call=llm_call, semantic_index=semantic_index)

    # Phase 1 Track 1K — wiki-side consumer for the ingestion bridge per
    # ADR-0025 §2. The email-side translator is the email-workflows plugin's
    # (OSS plan M3.2); this consumer turns any producer's WikiIngestEvent into
    # entity pages via WikiEngine.ingest, so the wiki still ingests with that
    # plugin unmounted.
    from iris_harness.memory.knowledge.event_subscribers import subscribe_wiki_ingest_consumer

    subscribe_wiki_ingest_consumer(wiki)

    def _csv_env(name: str) -> frozenset[str]:
        raw = os.getenv(name, "").strip()
        if not raw:
            return frozenset()
        return frozenset(item.strip() for item in raw.split(",") if item.strip())

    try:
        _lesson_top_k = int(os.getenv("IRIS_LESSON_TOP_K", "3"))
    except ValueError:
        _lesson_top_k = 3

    lesson_capture = LessonCapture(
        memory_store=memory_store,
        wiki=wiki,
        enabled=os.getenv("IRIS_LESSON_CAPTURE_ENABLED", "1") not in {"0", "false", "False"},
        top_k=max(0, _lesson_top_k),
        domain_allow=_csv_env("IRIS_LESSON_DOMAIN_ALLOW"),
        domain_block=_csv_env("IRIS_LESSON_DOMAIN_BLOCK"),
    )

    # The owner's tier edits (ADR-0120) come from this runtime's own data dir.
    tier_router = TierRouter.load_from_yaml(
        cfg / "llm_tiers.yaml", settings=SettingsStore(db_path=data / SETTINGS_DB_NAME)
    )
    if not os.environ.get("IRIS_DISABLE_ARBITER"):
        try:
            from iris_harness.llm.arbiter import OllamaArbiter, ResourceGovernor
            from iris_harness.llm.tier_router import _ollama_base_url

            adaptive = os.environ.get("IRIS_ADAPTIVE_TIERS", "0") not in {"", "0", "false", "False"}
            tier_router.governor = ResourceGovernor(
                arbiter=OllamaArbiter(base_url=_ollama_base_url()),
                adaptive=adaptive,
            )
        except Exception:
            logger.debug("ResourceGovernor init failed; continuing without it", exc_info=True)

    # Window-aware conversation auto-compaction (ADR-0079). The compactor summarizes the
    # oldest turns once the running history reaches ~80% of the chat model's prompt budget
    # (the tier the conversation history is replayed into = "communication"/Tier 2), so a
    # long or token-heavy session compacts before it overruns the window — the same
    # window-derived discipline as the P3 in-loop ContextBudgetController, applied across
    # turns. Env overrides win for tuning; threshold stays a count floor for tiny chats.
    try:
        _conv_num_ctx = getattr(tier_router.get_tier("communication"), "num_ctx", None) or 8192
    except Exception:  # noqa: BLE001 — partial/stub routers (tests) → default window
        _conv_num_ctx = 8192
    # The summarizer. Every production caller builds the runtime with llm_call=None, so
    # for months the compactor fell back to "first 3 turns cut to 80 chars" and called it
    # a summary. Build one from the tier router the same way fact extraction does, and
    # keep the caller's llm_call when one is passed (tests).
    summary_llm = llm_call or _tier_llm(tier_router, summary_config().get("tier_intent", "general"))
    compactor = ConversationCompactor(
        compaction_threshold=int(os.getenv("IRIS_COMPACTION_THRESHOLD", "20")),
        keep_recent=int(os.getenv("IRIS_COMPACTION_KEEP_RECENT", "10")),
        llm_call=summary_llm,
        token_budget=int(
            os.getenv("IRIS_COMPACTION_TOKEN_BUDGET", str(budget_for(int(_conv_num_ctx))))
        ),
        compaction_ratio=float(os.getenv("IRIS_COMPACTION_RATIO", "0.8")),
    )

    # The one path that deletes conversation memory (SQLite rows and their vectors
    # together). `sessions` is attached after the runtime exists — the closing roll
    # needs it, and it lives on the runtime.
    retention = RetentionService(
        memory_store,
        semantic_index,
        logs_dir=iris_home() / "logs",
        history_path=memory_store.db_path.parent / "housekeeping_runs.jsonl",
        # Session logs past their live window go here, encrypted, instead of being
        # deleted; the key is made in the keychain on first use, not at startup.
        archive=LogArchive(iris_home() / "archive" / "logs"),
    )

    intent_router = IntentRouter(
        classifier=_build_intent_classifier(tier_router, llm_call=llm_call, config_dir=cfg)
    )
    task_planner = TaskPlanner(llm_call=llm_call)
    from iris_harness.kernel.governance.audit import AuditLog

    (
        _output_safety_judge,
        _output_safety_enforce,
        _output_safety_log_only,
    ) = build_curator_output_safety_client(cfg_dir=cfg)
    _escalation_judge, _escalation_config = build_curator_escalation_client(
        tier_router=tier_router,
        llm_call=llm_call,
        config_dir=cfg,
    )
    # One kernel for the curator's PRE_RESPONSE check and the pipeline's PRE_TURN screen.
    governance_kernel = kernel_from_env()
    response_curator = ResponseCurator(
        audit_log=AuditLog(),
        kernel=governance_kernel,
        faithfulness_judge=build_curator_faithfulness_client(
            tier_router=tier_router,
            llm_call=llm_call,
        ),
        leak_judge=build_curator_leak_client(
            tier_router=tier_router,
            llm_call=llm_call,
        ),
        output_safety_judge=_output_safety_judge,
        output_safety_enforce=_output_safety_enforce,
        output_safety_log_only=_output_safety_log_only,
        output_safety_timeout_s=curator_output_safety_timeout_s(),
        grounding_judge=build_curator_grounding_client(
            tier_router=tier_router,
            llm_call=llm_call,
        ),
        escalation_judge=_escalation_judge,
        escalation_config=_escalation_config,
    )

    repo_root = cfg.parent
    skill_registry = SkillRegistry(repo_root=repo_root)

    # Built early so the ReAct handler can expose the learning_intelligence tool
    # over it (the engine/collector that also use it are wired further down).
    learning_store = LearningMetricsStore(db_path=data / "learning.db")
    learning_store.ensure_schema()

    agent_executor = AgentExecutor()
    # ADR-0086: a 1-slot holder for the runtime, filled after IrisRuntime is built so
    # handlers built before it (the general lane's plugin tools, the ReAct loop's
    # self-management tools) can reach it late.
    runtime_holder: list[Any] = []
    # `code_exec` is a reference plugin as of M4.6: it registers its own tool and
    # agent, and decides for itself whether the sandbox is reachable.
    general_handler, general_stream_handler = _make_general_handler(
        tier_router,
        repo_root=repo_root,
        skill_registry=skill_registry,
        semantic_index=semantic_index,
        wiki=wiki,
        runtime_holder=runtime_holder,
    )
    # Narrative LLM (tier-routed, governed) used by the deterministic-first agents and
    # as the finance_lookup narrator on the unified loop. Built before the rollout
    # block so the ReAct handler can take the deterministic finance handler as its
    # degrade fallback (ADR-0077). ``llm_call`` is injected only by tests.
    narrative_llm = make_narrative_llm_call(tier_router) if llm_call is None else llm_call
    # P4 (ADR-0077): the calendar + planner deterministic handlers, built up front so
    # they can serve as the loop's per-intent degrade fallbacks (and the rollout-off /
    # P4-off path). Their digests are also in-loop tools (calendar_lookup, daily_plan).
    # The loop's per-intent degrade path, filled entirely by the plugins that own the
    # intents: `_reassert_loop_intents` moves each plugin's handler in here when it
    # mounts. The core seeded `calendar` until M6.1b, when the calendar library left
    # with its plugin (OSS plan M6, decision 2) — the planner shape, now for all three.
    react_fallbacks: dict[str, Callable[[AgentTask], HandlerResult]] = {}
    # ADR-0081 context-health: a shared sink the handler factories seed (budget split) and
    # the AgenticCore writes into each turn (prompt size + transcript eviction). Assigned to
    # the runtime below so `context_health` reads the latest in-loop budget pressure.
    last_react_budget: dict[str, int] = {}

    react_handler: Callable[[AgentTask], HandlerResult] | None = None
    react_stream_handler: Callable[[AgentTask], Iterator[StreamChunk]] | None = None
    rollout_mode = _agentic_core_rollout_mode()
    _warn_if_lane_serves_ungoverned(rollout_mode, governance_kernel)
    if rollout_mode in {"on", "shadow"}:
        react_handler, react_stream_handler = _make_react_handler(
            tier_router,
            skill_registry,
            semantic_index=semantic_index,
            wiki=wiki,
            repo_root=repo_root,
            memory_store=memory_store,
            learning_store=learning_store,
            data_dir=data,
            fallback_handlers=react_fallbacks,
            budget_sink=last_react_budget,
            runtime_holder=runtime_holder,
        )
        if rollout_mode == "shadow":
            shadow_handler, shadow_stream_handler = _make_shadow_handlers(
                legacy_handler=general_handler,
                legacy_stream_handler=general_stream_handler,
                candidate_handler=react_handler,
            )
            agent_executor.register("system", shadow_handler)
            agent_executor.register_stream("system", shadow_stream_handler)
            logger.info("AgenticCore shadow mode enabled for 'system' agent")
        else:
            agent_executor.register("system", react_handler)
            agent_executor.register_stream("system", react_stream_handler)
            logger.info("AgenticCore ReAct handler enabled for 'system' agent")
    else:
        agent_executor.register("system", general_handler)
        agent_executor.register_stream("system", general_stream_handler)
    agent_executor.register("coding_agent", _make_coding_handler(tier_router))
    # A plugin's intent joins the loop through `api.register_loop_intent` (the email
    # plugin's does): the loop answers it and the plugin's declared fallback is its
    # degrade path (`_reassert_loop_intents`). With the loop off, the plugin's lane stands.
    # Clarify (issue 0002) — when the router can't tell what the user wants it
    # routes here to ask one grounded clarifying question instead of guessing.
    clarify_handler, clarify_stream_handler = _make_clarify_handler(narrative_llm)
    agent_executor.register("clarify", clarify_handler)
    agent_executor.register_stream("clarify", clarify_stream_handler)
    # Finance (ADR-0077) — when the agentic loop is on, finance turns run the SAME
    # unified ReAct handler as system, with finance_lookup + email tools available, so
    # a question like "find my the store card dues" can check local statements AND search
    # email in one turn. The deterministic per-currency digest (built above) becomes
    # the loop's degrade fallback and the rollout-off path. ``intent="finance"`` still
    # biases tier-2 + the finance/email prompt nudge.
    loop_intents: set[str] = set()
    if rollout_mode in {"on", "shadow"} and react_handler is not None:
        agent_executor.register("finance", react_handler)
        agent_executor.register_stream(
            "finance", cast("Callable[[AgentTask], Iterator[StreamChunk]]", react_stream_handler)
        )
        loop_intents.add("finance")
        logger.info("AgenticCore ReAct handler enabled for 'finance' agent (ADR-0077)")
    # With the loop off there is no core lane to register: the deterministic finance
    # digest left with its domain, and the `finance_workflows` plugin registers it.
    # Planner + calendar (ADR-0077 P4) — when universal surfacing is enabled (and the
    # base loop is available), these intents run the SAME unified ReAct handler as
    # finance/system, selecting daily_plan / calendar_lookup from the query; their
    # deterministic handlers (built above) become the loop's degrade fallbacks. When
    # P4 is off they stay deterministic lanes (the historical behaviour). Calendar
    # WRITES are unaffected either way — the calendar plugin's meeting_creation
    # intercept short-circuits create/invite requests upstream, before agent
    # dispatch (ADR-0075/0076).
    p4_loop = react_handler is not None and _p4_universal_surfacing_enabled()
    if p4_loop:
        rs = cast("Callable[[AgentTask], Iterator[StreamChunk]]", react_stream_handler)
        for _intent in ("planner", "calendar"):
            agent_executor.register(
                _intent, cast("Callable[[AgentTask], HandlerResult]", react_handler)
            )
            agent_executor.register_stream(_intent, rs)
            # The planner plugin registers a handler for its intent when it mounts;
            # naming the intent here is what lets `_reassert_loop_intents` put the loop
            # back on top and keep that handler as the degrade path (the finance shape).
            loop_intents.add(_intent)
        logger.info("AgenticCore ReAct handler enabled for 'planner'+'calendar' (ADR-0077 P4)")
    # With P4 off there is no core lane to register for either intent: `planner` and,
    # since M6.1b, `calendar` both register their deterministic handler from their
    # plugin — it stands when the loop is off and becomes the degrade path when on.
    # FileManager + research agents (FMX5/FMX9) — first-class personas on the SAME
    # unified ReAct loop (tools surface from their skill packs via
    # _skills_to_react_tools; intent biases tier + prompt, per ADR-0077). No
    # deterministic fallback lanes exist: with the flag off the agent stays
    # unregistered and `files`/`search` routing falls back exactly as before
    # (allowed_agent_types gates on registered_agents()).
    # The `filemanager` persona is CORE (OSS plan M2.6): every agent needs file
    # access, so the agent that answers `files` is part of the harness. Its tools
    # come from the file-read / file-compose / file-vault skill packs, which are
    # core too; the file_organizer plugin adds its own packs and intercepts on top.
    # The `research` persona left at M4.7 — it mounts THIS loop through
    # HarnessServices.react_handler, which is what that seam is for.
    if rollout_mode in {"on", "shadow"} and react_handler is not None:
        _fm_stream = cast("Callable[[AgentTask], Iterator[StreamChunk]]", react_stream_handler)
        if _filemanager_agent_enabled():
            agent_executor.register("filemanager", react_handler)
            agent_executor.register_stream("filemanager", _fm_stream)
            logger.info("AgenticCore ReAct handler enabled for 'filemanager' agent (FMX5)")

    mission_store = MissionStore(db_path=data / "missions.db")
    mission_engine = MissionEngine(store=mission_store)
    routine_store = RoutineStore(db_path=data / "routines.db")

    if use_background_scheduler:
        from apscheduler.schedulers.background import BackgroundScheduler

        scheduler = BackgroundScheduler(daemon=True, timezone=iris_timezone())
    else:
        scheduler = None
    # The owner's schedule edits live on the data volume, not in the read-only config
    # mount, so they survive restarts and deploys (ADR-0120). Cron times are read in
    # IRIS_TZ, and every run a job keeps lands in heartbeat_runs.db (loop-proof D13).
    heartbeats = HeartbeatScheduler(
        scheduler=scheduler,
        settings=SettingsStore(db_path=data / SETTINGS_DB_NAME),
        run_store=HeartbeatRunStore(db_path=data / HEARTBEAT_RUNS_DB),
        timezone=iris_timezone(),
    )

    channels, default_channel = _load_channels(cfg)

    experiment_loop = ExperimentLoop()
    strategies = load_strategies(cfg / "learning" / "strategies.yaml")
    learning_engine = LearningEngine(
        loop=experiment_loop,
        store=learning_store,
        strategies=strategies,
    )
    signal_collector = LearningSignalCollector(learning_store)

    skill_crystallizer = SkillCrystallizer(
        store=learning_store,
        repo_root=repo_root,
        synthesizer=_build_skill_synthesizer(tier_router=tier_router),
        preflight=_build_crystallize_preflight(repo_root=repo_root),
    )
    skill_hot_reloader = SkillHotReloader(registry=skill_registry, repo_root=repo_root)

    runtime = IrisRuntime(
        intent_router=intent_router,
        task_planner=task_planner,
        agent_executor=agent_executor,
        response_curator=response_curator,
        memory_store=memory_store,
        memory_retriever=memory_retriever,
        semantic_index=semantic_index,
        compactor=compactor,
        retention=retention,
        lesson_capture=lesson_capture,
        _last_react_budget=last_react_budget,
        wiki=wiki,
        mission_engine=mission_engine,
        routine_store=routine_store,
        heartbeats=heartbeats,
        channels=channels,
        default_channel=default_channel,
        learning_store=learning_store,
        learning_engine=learning_engine,
        signal_collector=signal_collector,
        skill_registry=skill_registry,
        skill_crystallizer=skill_crystallizer,
        skill_hot_reloader=skill_hot_reloader,
        config_dir=cfg,
        data_dir=data,
        tier_router=tier_router,
        governance_kernel=governance_kernel,
    )
    # The optional learning analyst, behavior miner and intention analyst, each built only
    # when its env gate is on — the same point in construction the builders always ran.
    runtime.learning = LearningControls.from_env(runtime)
    runtime.routines = RoutineAuthoring(runtime)
    runtime.learning_loop = SelfLearningLoop(runtime)
    runtime.capture = TurnCapture(runtime)
    runtime.mission_proposals = MissionProposals(runtime)
    runtime.replies = DeterministicReplies(runtime)
    runtime.confirmations = Confirmations(runtime)
    runtime.intercepts = InterceptDispatch(runtime)
    runtime.welcome = FirstChatWelcome(runtime)
    runtime.sessions = SessionMemory(runtime)
    # The closing roll runs through SessionMemory, which exists only now.
    retention._sessions = runtime.sessions
    # ADR-0086: hand the runtime to the self-management tools' late-bound holder.
    runtime_holder.append(runtime)

    # The governed ReAct loop, for plugins that mount a persona on it rather than
    # supply their own handler (OSS plan M2.5). None unless the loop is on/shadow,
    # which is exactly the guard the core applies to its own persona registrations.
    if rollout_mode in {"on", "shadow"}:
        runtime.react_handler = react_handler
        runtime.react_stream_handler = react_stream_handler
    # Which intents the loop owns, and the dict its degrade path reads. Both are
    # consulted by `_reassert_loop_intents` once the plugins have mounted.
    runtime._loop_intents = frozenset(loop_intents)
    runtime._react_fallbacks = react_fallbacks

    # The `rag-ingest` and `filemanager-quarantine` Action Center providers were
    # registered here until M6.1b, as core capabilities that must work with no
    # plugin mounted (OSS plan M2.6). The file domain leaves the core at decision 2
    # and executing either approval runs its code, so the file plugin registers them
    # with its other two, below in _mount_plugins.

    # OSS plan M1: mount the profile's plugins through the public PluginAPI. Every
    # registration is fault-bounded by the registry and lands on the same subsystem
    # entry points the core uses; per-plugin verdicts join the System Health snapshot.
    _mount_plugins(runtime, in_process_plugins)
    # Chat surfaces are plugins now, so the default channel can only be validated
    # once they have registered (OSS plan M4.5).
    _resolve_default_channel(runtime, default_channel)

    # Auto-created missions (iris_harness.services.missions.proposer) bind to the generic
    # "agent_query" handler: each step runs through the governed agent. Registered
    # post-construction so the closure can call runtime.chat.
    from iris_harness.services.missions.handlers import make_agent_query_handler

    mission_engine.register_handler(
        "agent_query",
        make_agent_query_handler(
            lambda q: runtime.chat(q, session_id="mission", channel="mission").response
        ),
    )

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    # Approval delivery off-box: the kernel owns which channel an approval goes to and
    # what it says, but not a transport that talks to Telegram (M6.2, decision 6). It
    # asks its registry; the composition root is what fills it. Registered even without
    # a poller: delivering an approval and reading replies are separate paths, and the
    # channel answers is_configured() False with no token, exactly as before.
    from iris_harness.kernel.governance.approvals.channels import (
        register_remote_channel,
    )
    from iris_harness.services.channels.approval_delivery import (
        TelegramApprovalChannel,
    )

    _approval_connector = channels.get("telegram") if "telegram" in channels.channels() else None
    register_remote_channel(
        TelegramApprovalChannel(
            connector=(
                _approval_connector if isinstance(_approval_connector, TelegramConnector) else None
            ),
            chat_id=telegram_chat_id or None,
        )
    )

    if not _runtime_telegram_poller_enabled():
        logger.info("runtime telegram poller skipped; channel gateway Telegram inbound is enabled")
    elif bot_token and not telegram_chat_id:
        # Fail closed: without an allowlist any Telegram user who finds the bot
        # would drive the full agent. Refuse to start the poller instead.
        logger.warning(
            "telegram inbound poller not started: TELEGRAM_CHAT_ID is not set — "
            "the poller refuses to run without an allowlist"
        )
    elif bot_token and "telegram" in channels.channels():
        telegram_connector = channels.get("telegram")
        if isinstance(telegram_connector, TelegramConnector):
            runtime.poller = TelegramPoller(
                bot_token=bot_token,
                connector=telegram_connector,
                chat_handler=lambda text, sid, audience: runtime.chat(
                    text, session_id=sid, audience=audience
                ).response,
                confirmation_options=runtime.confirmations.pending_options,
                command_handler=_telegram_approval_commands(runtime),
                opener=runtime.welcome.new_text_for("telegram"),
                allowed_chat_ids=frozenset([telegram_chat_id]),
                allowed_user_ids=allowed_user_ids_from_env(),
            )
            logger.info("telegram inbound poller configured")
        else:
            logger.warning(
                "telegram channel registered but connector is not TelegramConnector — poller skipped"
            )
    else:
        logger.debug(
            "telegram poller not configured (TELEGRAM_BOT_TOKEN missing or channel not registered)"
        )

    return runtime


# Re-export ``SubTask`` so adapters can construct extra tasks without a deeper import.
__all__ = ["ChatResult", "IrisRuntime", "SubTask", "build_runtime"]
