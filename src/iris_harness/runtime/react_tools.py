"""The built-in ``ToolSpec`` set the governed ReAct loop always offers.

Extracted from ``runtime/bootstrap.py`` (OSS plan M5, gate 1). ``builtin_react_tools``
builds the canonical tools — memory search, wiki, ``iris_doc``, stock quote,
sandbox skill proposals, and the ADR-0086 self-management set behind
``IRIS_AGENT_SELF_MANAGEMENT`` — from the stores the runtime hands it. Plugins add
their own tools through ``PluginAPI.register_tool``; nothing here knows a plugin.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from iris_harness.foundation.paths import data_dir
from iris_harness.memory.graph_context import memory_graph_description, memory_graph_tool
from iris_harness.memory.identity import (
    load_agents_md,
    load_harness_md,
    load_soul,
    load_soul_extended,
    load_user_md,
)
from iris_harness.memory.knowledge.wiki_engine import WikiEngine
from iris_harness.memory.retention import is_ephemeral_session
from iris_harness.memory.semantic_index import SemanticIndex
from iris_harness.memory.store import MemoryStore
from iris_harness.runtime.turn_context import current_session_id
from iris_harness.services.learning.store import LearningMetricsStore

logger = logging.getLogger(__name__)


def _log_tool_failure(tool: str, exc: BaseException) -> None:
    """Log a tool's failure: the agent sees the error text, operators see this.

    Tool name and exception type only — never the query, key or value the agent sent.
    """
    logger.warning("react tool %s failed (%s)", tool, type(exc).__name__, exc_info=True)


def builtin_react_tools(
    *,
    semantic_index: SemanticIndex | None,
    wiki: WikiEngine | None,
    repo_root: Path | None,
    memory_store: MemoryStore | None = None,
    learning_store: LearningMetricsStore | None = None,
    capabilities: Callable[[], str | None] | None = None,
) -> list[Any]:
    """Build the canonical built-in ``ToolSpec`` set always offered to the ReAct loop."""
    from iris_harness.agent.agentic_core import ToolSpec
    from iris_harness.tools.propose_skill_from_sandbox import (
        propose_skill_from_sandbox,
    )
    from iris_harness.tools.stock_quote import stock_quote

    def _stock_quote(args: dict[str, Any]) -> str:
        symbol = str(args.get("symbol") or args.get("ticker") or args.get("input") or "").strip()
        if not symbol:
            return "Error: stock_quote requires a 'symbol' argument, e.g. AAPL."
        return stock_quote(symbol)

    _MEMORY_SCOPES = ("facts", "patterns", "behaviors", "sessions", "all")

    def _search_facts(query: str, n: int) -> list[str]:
        if semantic_index is None:
            return []
        # A relevance cutoff keeps an off-topic query from dumping the nearest
        # unrelated facts — including the user's email — into chat (issue 0032). The
        # index holds only owner-confirmed facts, so an unreviewed proposal cannot
        # surface here either.
        max_dist = float(os.getenv("IRIS_MEMORY_RECALL_MAX_DISTANCE", "1.45"))
        return [
            f"fact — {f}"
            for f in semantic_index.query_facts_text(query, n=max(1, n), max_distance=max_dist)
        ]

    def _search_patterns(query: str, n: int) -> list[str]:
        if semantic_index is None:
            return []
        return [f"pattern — {p}" for p in semantic_index.query_episodic(query, n=max(1, n))]

    def _search_behaviors(query: str, n: int) -> list[str]:
        from iris_harness.memory.identity import (
            list_behaviors,
            score_behaviors,
        )

        scored = score_behaviors("", query)
        behaviors = [b for b, _s in scored] or list_behaviors()
        return [f"lesson — {b.name}: {b.headline}" for b in behaviors[: max(1, n)]]

    def _recallable(session_id: str) -> bool:
        """A playground or test run is not the owner's past (retention.yaml), unless it is
        the conversation asking."""
        return not is_ephemeral_session(session_id) or session_id == current_session_id()

    def _overfetch(n: int) -> int:
        # Test runs are filtered AFTER the query, so ask for more than we keep: the
        # owner's box held 1,420 test turns to 2,310 real ones.
        return max(n * 5, 50)

    def _semantic_turns(query: str, n: int, session_id: str) -> list[tuple[str, str, str]]:
        """``(session_id, role, content)`` of past turns nearest ``query`` by meaning.

        Removed conversations never come back (ADR-0119), and an unreadable removal
        ledger returns nothing (fail closed, as the retriever does); playground and test
        runs are left out. Empty when there is no ready semantic index.
        """
        if semantic_index is None or not getattr(semantic_index, "is_ready", False):
            return []
        try:
            removed = memory_store.removed_session_ids() if memory_store is not None else set()
        except Exception as exc:  # noqa: BLE001 — fail closed: no semantic hits
            _log_tool_failure("recall removed-session ledger", exc)
            return []
        from iris_harness.memory.retriever import recall_max_distance

        found = semantic_index.query_turns_detailed(
            query,
            only_session=session_id or None,
            exclude_session=None if session_id else (current_session_id() or None),
            n=_overfetch(n) if not session_id else n,
            max_distance=recall_max_distance(),
        )
        return [
            (t.session_id, t.role, t.content)
            for t in found
            if t.session_id not in removed and _recallable(t.session_id)
        ][:n]

    def _search_sessions(query: str, n: int) -> list[str]:
        if memory_store is None:
            return []
        n = max(1, n)
        out: list[str] = []
        seen: set[str] = set()
        # Conversations whose turns are near the query by meaning, then summaries that
        # contain it literally (an exact name or number).
        for sid, _role, content in _semantic_turns(query, _overfetch(n), ""):
            if sid in seen:
                continue
            seen.add(sid)
            try:
                summary = memory_store.load_conversation_summary(sid)
            except Exception as exc:  # noqa: BLE001 — the matched turn still says what it was
                _log_tool_failure("memory_search session summary", exc)
                summary = ""
            label = " ".join((summary or content).split())[:240]
            out.append(f"past session {sid} — {label}")
            if len(out) >= n:
                return out
        hits = memory_store.search_summaries(query, limit=_overfetch(n))
        for sid, text in hits:
            if len(out) >= n:
                break
            if _recallable(sid) and sid not in seen:
                seen.add(sid)
                out.append(f"past session {sid} — {' '.join(text.split())[:240]}")
        return out

    def _memory_search(args: dict[str, Any]) -> str:
        query = str(args.get("query") or args.get("input") or "").strip()
        if not query:
            return "Error: memory_search requires a 'query' argument."
        scope = str(args.get("scope") or "all").strip().lower()
        if scope not in _MEMORY_SCOPES:
            return f"Error: scope must be one of {', '.join(_MEMORY_SCOPES)}."
        try:
            n = int(args.get("n") or args.get("limit") or 5)
        except (TypeError, ValueError):
            n = 5
        try:
            lines: list[str] = []
            if scope in ("facts", "all"):
                lines += _search_facts(query, n)
            if scope in ("patterns", "all"):
                lines += _search_patterns(query, n)
            if scope in ("behaviors", "all"):
                lines += _search_behaviors(query, n)
            if scope in ("sessions", "all"):
                lines += _search_sessions(query, n)
        except Exception as exc:  # noqa: BLE001 — the agent reads the failure
            _log_tool_failure("memory_search", exc)
            return f"memory_search failed: {exc}"
        return "\n".join(f"- {line}" for line in lines) or "No stored memory matched that query."

    def _recall_conversation(args: dict[str, Any]) -> str:
        """Earlier exchanges, word for word — what a summary had to leave out."""
        if memory_store is None:
            return "recall_conversation unavailable (memory store not configured)."
        session_id = str(args.get("session_id") or args.get("session") or "").strip()
        query = str(args.get("query") or args.get("input") or "").strip()
        try:
            n = int(args.get("n") or args.get("limit") or 10)
        except (TypeError, ValueError):
            n = 10
        n = max(1, min(n, 40))
        try:
            if query:
                # By meaning first (a question never matches the stored words
                # literally: "which hotel am I in?" vs "Booked Taj Fort Aguada"), then
                # the literal matches, which still find exact names and numbers.
                rows = _semantic_turns(query, n, session_id)
                seen = {(sid, content) for sid, _role, content in rows}
                hits = memory_store.search_turns(query, limit=_overfetch(n))
                if session_id:
                    hits = [h for h in hits if h[1] == session_id]
                for h in hits:
                    if len(rows) >= n:
                        break
                    if _recallable(h[1]) and (h[1], h[3]) not in seen:
                        rows.append((h[1], h[2], h[3]))
                        seen.add((h[1], h[3]))
                if not rows and session_id and _recallable(session_id):
                    # A pointer named this conversation as related: read it back
                    # rather than answer "nothing matches".
                    rows = [
                        (session_id, role, content)
                        for role, content in memory_store.load_recent_turns(session_id, limit=n)
                    ]
            else:
                sid = session_id or current_session_id()
                if not sid:
                    return (
                        "Error: recall_conversation needs a 'query' to search for, or a "
                        "'session_id' to read back."
                    )
                rows = [
                    (sid, role, content)
                    for role, content in memory_store.load_recent_turns(sid, limit=n)
                ]
        except Exception as exc:  # noqa: BLE001 — the agent reads the failure
            _log_tool_failure("recall_conversation", exc)
            return f"recall_conversation failed: {exc}"
        if not rows and query:
            # Cooling a session REPLACES its turns with its summary (see
            # MemoryStore.cool_session), so an older conversation has no raw text left
            # to quote. Answering "nothing matches" would deny what the system
            # deliberately kept: fall back to the summaries and say the words are the
            # summary's, not the user's.
            try:
                summaries = memory_store.search_summaries(query, limit=_overfetch(n))
            except Exception as exc:  # noqa: BLE001 — the empty raw result still stands
                _log_tool_failure("recall_conversation summary fallback", exc)
                summaries = []
            summaries = [s for s in summaries if _recallable(s[0])][:n]
            if summaries:
                if session_id:
                    summaries = [s for s in summaries if s[0] == session_id]
            if summaries:
                lines = "\n".join(
                    f"- [{sid}] summary: {' '.join(text.split())[:400]}" for sid, text in summaries
                )
                return (
                    "The original turns were cooled to their summary, so this is the "
                    "summary rather than word for word:\n" + lines
                )
        if not rows:
            return "Nothing stored matches that."
        return "\n".join(
            f"- [{sid}] {role}: {' '.join(content.split())[:400]}" for sid, role, content in rows
        )

    # --- Conversational memory curation (all reversible + audited via history) ---
    # The user can correct/forget facts in chat ("forget that I live in Berlin",
    # "my blog is actually X"). Writes go through the same reversible store ops as the
    # `iris facts` CLI, so a mistaken correction is undoable. Guard hard against empty
    # / hallucinated args (a local model must never delete a fact on a blank arg).

    def _memory_forget(args: dict[str, Any]) -> str:
        if memory_store is None:
            return "memory_forget unavailable (memory store not configured)."
        key = str(args.get("key") or args.get("fact") or args.get("name") or "").strip()
        if not key:
            return "Error: memory_forget requires a 'key' (the fact name, e.g. 'location')."
        try:
            from iris_harness.memory.coordinator import FactCoordinator

            deleted = FactCoordinator(memory_store, semantic_index).forget(key)
        except Exception as exc:  # noqa: BLE001 — the agent reads the failure
            _log_tool_failure("memory_forget", exc)
            return f"memory_forget failed: {exc}"
        if not deleted:
            return f"No stored fact '{key}' to forget. Use memory_search to find the right key."
        if learning_store is not None:
            learning_store.record_user_behavior_signal("fact_forgotten", subject=key)
        return f"Forgot '{key}'. This is reversible — say 'restore {key}' to bring it back."

    def _memory_correct(args: dict[str, Any]) -> str:
        if memory_store is None:
            return "memory_correct unavailable (memory store not configured)."
        key = str(args.get("key") or args.get("fact") or args.get("name") or "").strip()
        value = str(args.get("value") or args.get("new_value") or args.get("to") or "").strip()
        if not key or not value:
            return "Error: memory_correct requires a 'key' and a 'value' (e.g. key=location, value='New York')."
        try:
            from iris_harness.memory.coordinator import FactCoordinator

            replaced = FactCoordinator(memory_store, semantic_index).correct(key, value)
        except Exception as exc:  # noqa: BLE001 — the agent reads the failure
            _log_tool_failure("memory_correct", exc)
            return f"memory_correct failed: {exc}"
        if learning_store is not None:
            learning_store.record_user_behavior_signal("fact_corrected", subject=key, detail=value)
        verb = "Updated" if replaced else "Recorded"
        return f"{verb} '{key}' = {value}. This is reversible — say 'restore {key}' to undo."

    def _memory_restore(args: dict[str, Any]) -> str:
        if memory_store is None:
            return "memory_restore unavailable (memory store not configured)."
        key = str(args.get("key") or args.get("fact") or args.get("name") or "").strip()
        if not key:
            return "Error: memory_restore requires a 'key' (the fact name to restore)."
        try:
            from iris_harness.memory.coordinator import FactCoordinator

            restored = FactCoordinator(memory_store, semantic_index).restore(key)
        except Exception as exc:  # noqa: BLE001 — the agent reads the failure
            _log_tool_failure("memory_restore", exc)
            return f"memory_restore failed: {exc}"
        if restored is None:
            return f"Nothing to restore for '{key}' (no prior value in history)."
        return f"Restored '{key}' = {restored}."

    def _wiki_search(args: dict[str, Any]) -> str:
        if wiki is None:
            return "Wiki unavailable."
        question = str(args.get("query") or args.get("question") or args.get("input") or "").strip()
        if not question:
            return "Error: wiki_search requires a 'query' argument."
        try:
            return wiki.query(question)
        except Exception as exc:  # noqa: BLE001 — the agent reads the failure
            _log_tool_failure("wiki_search", exc)
            return f"wiki_search failed: {exc}"

    # iris_doc: on-demand loader for the workspace + in-repo operational
    # docs (SOUL.md, USER.md, AGENTS.md, iris-harness.md). The model
    # calls this when it needs the registry or the operational manual.
    # Always-loaded content stays in the system prompt (soul, user
    # profile via MemoryContext); these are heavier docs pulled in only
    # when the LLM asks. See SOUL.md "Operational primer" section.
    _IRIS_DOC_LOADERS: dict[str, Callable[[], str | None]] = {
        "SOUL": load_soul,
        "USER": load_user_md,
        "AGENTS": load_agents_md,
        "HARNESS": load_harness_md,
        # The soul sections held out of the per-turn prompt (config/identity/
        # soul_layers.yaml): the internals primer, style notes, the full tool policy.
        "OPERATING": load_soul_extended,
        # What this install actually has, read off the live registry — never a
        # hand-written list, which is how SOUL.md ended up naming two models that
        # are not the ones configured.
        "CAPABILITIES": (capabilities or (lambda: None)),
    }

    def _iris_doc(args: dict[str, Any]) -> str:
        raw_name = str(args.get("name") or args.get("doc") or args.get("input") or "").strip()
        if not raw_name:
            return (
                "Error: iris_doc requires a 'name' argument. Valid names: "
                + ", ".join(sorted(_IRIS_DOC_LOADERS))
                + "."
            )
        name = raw_name.upper()
        loader_fn = _IRIS_DOC_LOADERS.get(name)
        if loader_fn is None:
            return (
                f"Error: iris_doc unknown name {raw_name!r}. Valid names: "
                + ", ".join(sorted(_IRIS_DOC_LOADERS))
                + "."
            )
        try:
            body = loader_fn()
        except Exception as exc:  # noqa: BLE001 — the agent reads the failure
            _log_tool_failure("iris_doc", exc)
            return f"iris_doc({name}) failed: {exc}"
        if body is None:
            return f"iris_doc({name}): file not present on this install."
        return body

    # `system_health` moved to the `system` reference plugin (OSS plan M1).

    def _pending_actions(args: dict[str, Any]) -> str:
        # Action Center over chat (ADR-0073/0074): surface what needs the user's
        # attention — blocked finance statements, unknown institutions, failed
        # extractions, plus health alerts — so the capability isn't web-only. Reads
        # the task store + cached health snapshot; never raises into the loop.

        from iris_harness.runtime.action_center import (
            collect_pending_actions,
            render_pending_actions,
        )
        from iris_harness.services.tasks import TaskStore
        from iris_harness.services.tasks.pending_actions import reconcile_all

        try:
            dd = data_dir()
            ts = TaskStore(db_path=dd / "tasks.db")
            ts.ensure_schema()
            # Keep provider-backed actions in sync before rendering the unified
            # Action Center list. The core does not name the providers: each domain
            # registers its own (OSS plan M2.6), and one failing provider never
            # stops the others.
            reconcile_all(ts)
            snapshot = None
            try:
                from iris_harness.services.health.service import current_snapshot

                snapshot = current_snapshot()
            except Exception as exc:  # noqa: BLE001 — health is optional; still show task actions
                _log_tool_failure("pending_actions health snapshot", exc)
                snapshot = None
            return render_pending_actions(collect_pending_actions(ts, snapshot))
        except Exception as exc:  # noqa: BLE001 — the agent reads the failure
            _log_tool_failure("pending_actions", exc)
            return f"pending_actions failed: {exc}"

    def _invoke_pending_action(args: dict[str, Any]) -> str:
        # Action Center execution over chat: run one SAFE pending action by id so
        # the assistant can ask for user confirmation and then action it in-line
        # (same provider seam as the dashboard/API).

        from iris_harness.services.tasks import TaskStore

        action_id = str(args.get("id") or args.get("action_id") or "").strip()
        if not action_id:
            return "invoke_pending_action needs 'id' (the pending action id)."
        try:
            dd = data_dir()
            ts = TaskStore(db_path=dd / "tasks.db")
            ts.ensure_schema()
            task = ts.get(action_id)
            if task is None or task.action is None:
                return f"No pending action {action_id!r}."
            if not task.action.safe:
                return "This action is display-only; run its command locally."
            # A review action only shows what the domain already wrote, so answer it
            # here rather than routing to a provider — same as the Web UI Action
            # Center, and it works for a kind whose plugin is not mounted.
            from iris_harness.services.tasks.pending_actions import (
                invoke_and_reconcile,
                provider_for,
                render_review,
            )

            if task.action is not None and task.action.kind == "review":
                return render_review(task)
            # Everything else: whichever domain registered this source_kind owns it.
            # The core no longer names them (OSS plan M2.6).
            registered = provider_for(str(task.source_kind))
            if registered is not None:
                choice = str(args.get("choice") or "").strip() or None
                option = str(args.get("option") or "").strip() or None
                if task.action.choices and choice is None:
                    # A choice card: say what the owner can answer instead of guessing.
                    answers = ", ".join(f"{c.value} ({c.label})" for c in task.action.choices)
                    picks = (
                        "; option one of: " + ", ".join(v.value for v in task.action.options.values)
                        if task.action.options
                        else ""
                    )
                    return f"This card needs an answer: choice one of {answers}{picks}."
                return invoke_and_reconcile(registered, task, ts, choice=choice, option=option)
            return (
                f"No invoker for source_kind={task.source_kind}. "
                "The plugin that owns it may not be mounted in this profile."
            )
        except Exception as exc:  # noqa: BLE001 — the agent reads the failure
            _log_tool_failure("invoke_pending_action", exc)
            return f"invoke_pending_action failed: {exc}"

    def _record_feedback(args: dict[str, Any]) -> str:
        # Surface-feedback over chat (issue 0028): the user says a proactively
        # surfaced item (a reply-followup, a bill, a health alert) wasn't useful,
        # so IRIS records it and suppresses similar items. Accepts a followup
        # Task id or an 'fb:' surface token. Never raises into the loop.

        from iris_harness.services.learning.suppression import (
            NOT_USEFUL,
            VERDICTS,
            SurfaceFeedbackStore,
            decode_ref,
        )

        verdict = str(args.get("verdict", NOT_USEFUL) or NOT_USEFUL).strip().lower()
        if verdict not in VERDICTS:
            return f"verdict must be one of {list(VERDICTS)}."
        email_sender = str(args.get("email_sender", "") or "").strip()
        ref = str(args.get("ref", "") or "").strip()
        if not ref and not email_sender:
            return (
                "record_feedback needs a 'ref' (a followup task id or an 'fb:' token) "
                "or 'email_sender' (a sender to suppress from email search)."
            )
        try:
            feedback = SurfaceFeedbackStore()
            feedback.ensure_schema()
            if email_sender:
                # Email-search close-the-loop (issue 0006): suppress a sender the user
                # said wasn't what they meant, so future searches downrank them.
                from iris_harness.services.learning.suppression import (
                    EMAIL_SEARCH_SUBSYSTEM,
                    EMAIL_SEARCH_SURFACE,
                    email_search_dims_from_sender,
                )

                dims = email_search_dims_from_sender(email_sender)
                if not dims.get("from_domain"):
                    return f"Couldn't read a sender domain from {email_sender!r}."
                feedback.record(EMAIL_SEARCH_SUBSYSTEM, EMAIL_SEARCH_SURFACE, dims, verdict)
                domain = dims["from_domain"]
                if verdict == NOT_USEFUL:
                    return f"Got it — I'll stop surfacing {domain} in email search results."
                return f"Noted — keeping {domain} in email search results."
            if ref.startswith("fb:"):
                subsystem, surface_kind, dims = decode_ref(ref)
                feedback.record(subsystem, surface_kind, dims, verdict)
                return f"Recorded {verdict} for {subsystem}/{surface_kind}. I'll act on it."

            from iris_harness.services.learning.suppression import (
                email_followup_dims_from,
            )
            from iris_harness.services.tasks import TaskStore

            dd = data_dir()
            ts = TaskStore(db_path=dd / "tasks.db")
            ts.ensure_schema()
            task = ts.get(ref)
            if task is None or task.wait_for is None or task.wait_for.kind != "reply_from":
                return f"No email followup {ref!r} to give feedback on."
            payload = task.wait_for.payload
            dims = email_followup_dims_from(
                str(payload.get("account_id", "")), str(payload.get("from", ""))
            )
            feedback.record("email", "followup", dims, verdict)
            if verdict == NOT_USEFUL:
                ts.drop(task.id)
                return (
                    f"Done — dropped that follow-up and I won't raise replies for "
                    f"{dims['from_domain']} again."
                )
            return f"Noted — keeping follow-ups for {dims['from_domain']}."
        except Exception as exc:
            logger.exception("record_feedback tool failed for ref=%r", ref)
            return f"record_feedback failed: {exc}"

    def _agents(args: dict[str, Any]) -> str:
        # Per-agent console over chat (ADR-0074): list IRIS's agents + what's pending
        # for each, or one agent's settings + pending actions when a name is given.
        # Composes from the catalog + task store (no live runtime needed); never
        # raises into the loop.
        from iris_harness.runtime.agent_console import (
            render_agent_detail,
            render_agents_overview_local,
        )

        name = str(args.get("name") or args.get("agent") or "").strip()
        try:
            return render_agent_detail(name) if name else render_agents_overview_local()
        except Exception as exc:  # noqa: BLE001 — the agent reads the failure
            _log_tool_failure("agents", exc)
            return f"agents failed: {exc}"

    def _learning_intelligence(args: dict[str, Any]) -> str:
        # Harness self-reflection (ADR-0069 #4): measured learning telemetry —
        # signal accuracy/integrity + a per-(intent, tier) outcome matrix — so the
        # model can answer "how am I learning / where am I weak?" from real numbers.
        # Pure read over learning.db; never raises into the loop. Interpretation is
        # the model's job (this tool only measures).
        if learning_store is None:
            return "learning_intelligence unavailable (learning store not configured)."
        from iris_harness.services.learning.intelligence import (
            build_intelligence,
            render_text,
        )

        try:
            days = int(args.get("window_days") or args.get("days") or 7)
        except (TypeError, ValueError):
            days = 7
        try:
            report = build_intelligence(learning_store, window=timedelta(days=max(1, days)))
        except Exception as exc:  # noqa: BLE001 — the agent reads the failure
            _log_tool_failure("learning_intelligence", exc)
            return f"learning_intelligence failed: {exc}"
        return render_text(report)

    def _learning_recommendations(args: dict[str, Any]) -> str:
        # The agentic analyst's latest advisory recommendations (ADR-0069 #4 s2).
        # Reads the persisted analysis (produced by the opt-in learning_analysis
        # heartbeat); does NOT run the LLM inline. Pure read; never raises.
        if learning_store is None:
            return "learning_recommendations unavailable (learning store not configured)."
        from iris_harness.services.learning.analyst import LearningAnalysis
        from iris_harness.services.learning.analyst import (
            render_text as render_analysis,
        )

        try:
            payload = learning_store.latest_analysis()
        except Exception as exc:  # noqa: BLE001 — the agent reads the failure
            _log_tool_failure("learning_recommendations", exc)
            return f"learning_recommendations failed: {exc}"
        if not payload:
            return (
                "No learning recommendations yet. The learning analyst is opt-in "
                "(IRIS_LEARNING_ANALYST) and runs on a schedule; once it has produced an "
                "analysis it appears here. Use learning_intelligence for the raw measured data."
            )
        analysis = LearningAnalysis.from_dict(payload)
        if analysis is None:
            return "learning_recommendations: stored analysis could not be read."
        return render_analysis(analysis)

    def _promote_recommendation(args: dict[str, Any]) -> str:
        # HITL loop-closer (ADR-0069 #4 s3): promote a recommendation to a tracked
        # experiment that captures a baseline and re-measures over time. Does NOT
        # apply the change — that stays a human step. Pure store write; never raises.
        if learning_store is None:
            return "promote_recommendation unavailable (learning store not configured)."
        raw = args.get("index") or args.get("number") or args.get("input")
        try:
            index = int(str(raw))
        except (TypeError, ValueError):
            return (
                "Error: promote_recommendation requires an 'index' (the recommendation's "
                "number, 1-based)."
            )
        from iris_harness.services.learning.promote import promote_recommendation

        try:
            result = promote_recommendation(learning_store, index=index)
        except Exception as exc:  # noqa: BLE001 — the agent reads the failure
            _log_tool_failure("promote_recommendation", exc)
            return f"promote_recommendation failed: {exc}"
        if result is None:
            return (
                f"No recommendation #{index} to promote. Use learning_recommendations to see the "
                "current list (the analyst must have run first)."
            )
        exp = result.experiment
        return (
            f'Tracking recommendation #{index} as experiment {exp.id}: "{exp.hypothesis}". '
            f"{result.note}. Baseline {exp.baseline_metric:.3f}; I'll re-measure it over the next "
            f"{exp.evaluation_window_hours}h and keep or discard it by the outcome. "
            "Note: I have NOT applied the change — that's still your call."
        )

    def _propose_skill(args: dict[str, Any]) -> str:
        if repo_root is None:
            return "propose_skill_from_sandbox unavailable (no repo_root)."
        script = str(args.get("script") or "").strip()
        intent = str(args.get("intent") or args.get("query") or "").strip()
        narrative = str(args.get("narrative") or args.get("description") or "").strip()
        if not script or not intent or not narrative:
            return (
                "Error: propose_skill_from_sandbox requires 'script', 'intent', "
                "'narrative' (and optional 'slug')."
            )
        slug = args.get("slug")
        try:
            result = propose_skill_from_sandbox(
                repo_root,
                script=script,
                intent=intent,
                narrative=narrative,
                slug=str(slug) if isinstance(slug, str) else None,
            )
        except Exception as exc:  # noqa: BLE001 — the agent reads the failure
            _log_tool_failure("propose_skill_from_sandbox", exc)
            return f"propose_skill_from_sandbox failed: {exc}"
        return str(result)

    specs: list[Any] = [
        ToolSpec(
            name="stock_quote",
            description=(
                "Get the CURRENT/live price of a stock or ETF by ticker. Use this "
                "(not research) for 'what is X stock worth' / market-price questions — "
                "web search snippets do not contain live prices. "
                'Args: {"symbol": "AAPL"}.'
            ),
            call=_stock_quote,
        ),
        ToolSpec(
            name="memory_search",
            description=(
                "Search what IRIS remembers. Scopes: 'facts' (confirmed facts about the "
                "user), 'patterns' (habits noticed), 'behaviors' (lessons and recipes "
                "learned), 'sessions' (summaries of past conversations), 'all' (default). "
                'Args: {"query": str, "scope": str, "n": int}.'
            ),
            call=_memory_search,
        ),
        ToolSpec(
            name="memory_graph",
            description=memory_graph_description(memory_store),
            call=lambda args: memory_graph_tool(memory_store, args),
        ),
        ToolSpec(
            name="recall_conversation",
            description=(
                "Read earlier exchanges WORD FOR WORD — use when the summary of this "
                "conversation is not enough, or to check exactly what was said before. "
                'Args: {"query": str, "session_id": str, "n": int}.'
            ),
            call=_recall_conversation,
        ),
        ToolSpec(
            name="memory_correct",
            effect="write",
            confirm="never",
            description=(
                "Correct a stored fact about the user when they say it's wrong or changed "
                "(e.g. 'I moved to New York', 'my blog is actually X'). Overrides the value; "
                "reversible. If unsure of the exact key, memory_search first. "
                'Args: {"key": str, "value": str}.'
            ),
            call=_memory_correct,
        ),
        ToolSpec(
            name="memory_forget",
            effect="write",
            confirm="never",
            description=(
                "Forget a stored fact when the user asks you to (e.g. 'forget that I live in "
                "Berlin', 'forget my location'). Reversible (kept in history). If unsure of the "
                'key, memory_search first. Args: {"key": str}.'
            ),
            call=_memory_forget,
        ),
        ToolSpec(
            name="memory_restore",
            effect="write",
            confirm="never",
            description=(
                "Undo the last correction/forget of a fact, re-activating its prior value "
                "(e.g. the user says 'actually, put that back'). "
                'Args: {"key": str}.'
            ),
            call=_memory_restore,
        ),
        ToolSpec(
            name="wiki_search",
            description=(
                "Search the curated knowledge-base wiki for cross-referenced answers. "
                'Args: {"query": str}.'
            ),
            call=_wiki_search,
            # The wiki is compiled from ingested documents -- text third parties wrote --
            # so its results are marked untrusted and tripwire-scanned (the floor).
            content="external",
        ),
        ToolSpec(
            name="iris_doc",
            description=(
                "AUTHORITATIVE source for facts about IRIS itself and the "
                "user. CALL THIS FIRST — do not answer from memory or "
                "training — when the user asks about anything in the "
                "trigger lists below. The system-prompt primer has an "
                "OUTLINE; iris_doc has the SPECIFICS. If a question needs "
                "exact values, file paths, model names, config numbers, or "
                "detailed behavior, fetch the doc.\n"
                'Args: {"name": "SOUL" | "USER" | "AGENTS" | "HARNESS" | '
                '"OPERATING" | "CAPABILITIES"}.\n'
                "\n"
                "HARNESS — call when the user asks about any of: "
                "AgenticCore / pipeline / stages / IntentRouter / TaskPlanner / "
                "ReActLoop / AgentExecutor / ResponseCurator / governance hooks / "
                "PreClassify / PreLLMCall / PreToolUse / PostToolUse / PostStep / "
                "PreResponse / tier routing / Tier 1 / Tier 2 / Tier 3 / model names / "
                "keep_alive / num_ctx / tool taxonomy / memory model / 4 layers / "
                "ChromaDB / ReAct format / Thought / Action / Observation / "
                "escalation / HITL / approval / failure handling / "
                "context bloat / kernel down / model swap / onboarding / "
                "checkpoint / resume / classification (public/internal/personal/secret).\n"
                "\n"
                "AGENTS — call when the user asks about any of: "
                "which agents / agent registry / agent list / handoff / "
                "coding agent / iris-code / personas (orchestrator, analyst, "
                "architect, developer, tester, sm, ux-designer) / routine "
                "executor / heartbeat / memory compactor / response curator / "
                "skills / MCP agents.\n"
                "\n"
                "USER — call when the user asks about themselves: "
                "name / role / profession / focus / stack / languages / "
                "preferences / goals / interests / background / "
                "communication style / 'what do you know about me'.\n"
                "\n"
                "SOUL — call when the user asks you to quote your own "
                "invariants / mission / behavioral rules / reasoning style / "
                "security policy verbatim.\n"
                "\n"
                "CAPABILITIES — call when the user asks what you can do / which "
                "plugins, tools or models you have / 'can you ...' about a capability. "
                "Read off the live registry, so it is current.\n"
                "\n"
                "OPERATING — call for your own operating detail held out of the "
                "prompt: the harness primer, reasoning style, the full tool policy.\n"
                "\n"
                "If unsure between HARNESS and AGENTS for an internals "
                "question, call HARNESS first. Never invent IRIS specifics "
                "(model names, keep_alive values, persona names, tier numbers) "
                "from training — fetch them."
            ),
            call=_iris_doc,
        ),
        ToolSpec(
            name="pending_actions",
            description=(
                "List what needs the USER's attention right now — pending actions "
                "raised by IRIS's background agents (a finance statement blocked on a "
                "missing password, an unrecognised institution to register, a failed "
                "extraction to retry) plus any health alerts. CALL THIS when the user "
                "asks 'what needs my attention', 'what's pending', 'any actions / "
                "anything blocked', 'what do I need to do', or 'is anything waiting on "
                "me'. Returns each item with the exact fix (a command to run, or where "
                "to resolve it). No arguments."
            ),
            call=_pending_actions,
        ),
        ToolSpec(
            name="invoke_pending_action",
            effect="write",
            confirm="once",
            description=(
                "Execute one SAFE pending action from the Action Center by id. "
                "Use only after the user clearly approves. Typical flow: call "
                "pending_actions, ask which item they want to run, then call this "
                "tool with that id. Rejects display-only actions. A card with "
                'choices ("Yes, it\'s mine" / "Ignore") needs the owner\'s choice, '
                "and the option they picked when the choice carries one. "
                'Args: {"id": str, "choice": str (for a choice card), "option": str}.'
            ),
            call=_invoke_pending_action,
        ),
        ToolSpec(
            name="record_feedback",
            effect="write",
            confirm="never",
            description=(
                "Record that a proactively-surfaced item was NOT useful (or was "
                "useful) so IRIS suppresses similar items in future. CALL THIS when "
                "the user dismisses something IRIS raised — 'that follow-up isn't "
                "useful', 'stop reminding me to reply to X', 'this isn't a real "
                "follow-up', 'don't surface this bill / alert again', or conversely "
                "'yes, keep reminding me about this'. Pass the item's id: a followup "
                "Task id (from the daily brief / pending actions) or an 'fb:' surface "
                "token shown next to an item. ALSO: when the user says a SENDER's "
                "emails aren't what they meant in search ('ignore X in search', 'stop "
                "showing me results from X'), pass that sender (a domain or address) as "
                "`email_sender` to downrank them in future email searches. "
                'Args: {"ref"?: str, "email_sender"?: str, '
                '"verdict": "not_useful" | "useful"} (verdict defaults to not_useful).'
            ),
            call=_record_feedback,
        ),
        ToolSpec(
            name="agents",
            description=(
                "List IRIS's own agents and what each is doing, or describe ONE agent. "
                "CALL THIS when the user asks 'what agents do you have', 'what can you "
                "do / your capabilities', 'show me the finance/email agent', 'what are "
                "<agent>'s settings', or 'how is the <agent> agent configured'. With no "
                "argument: every agent + its pending-action count. With "
                '{"name": "finance"}: that agent\'s settings (LLM tier, toggles, '
                "schedule) and its pending actions. Reads local config + stores."
            ),
            call=_agents,
        ),
        ToolSpec(
            name="learning_intelligence",
            description=(
                "Report how IRIS is LEARNING and performing over recent traffic, "
                "from its own measured telemetry. CALL THIS when the user asks 'how "
                "are you learning', 'how are you performing', 'where are you weak / "
                "what's underperforming', 'how accurate are your learning signals', "
                "'which intents or tiers do badly', or 'what patterns have you found'. "
                "Returns signal accuracy/integrity (escalation-judge precision, signal "
                "drop rate) and a per-(intent, tier) outcome matrix (task-completion "
                "rate, user-correction rate, downstream-reuse, average token cost). "
                "These are raw measured numbers — interpret them yourself. Optional "
                'arg: {"window_days": int} (default 7).'
            ),
            call=_learning_intelligence,
        ),
        ToolSpec(
            name="learning_recommendations",
            description=(
                "Return the learning analyst's latest RANKED, advisory recommendations "
                "for improving IRIS — each with the measured finding behind it, a "
                "concrete reversible action, and a confidence. CALL THIS when the user "
                "asks 'what should I improve / change', 'what do you recommend', 'what "
                "have you learned that I should act on', or 'any suggestions to make you "
                "better'. These are proposals for the user to approve, never auto-applied. "
                "(For the raw measured data behind them, use learning_intelligence.) "
                "No arguments."
            ),
            call=_learning_recommendations,
        ),
        ToolSpec(
            name="promote_recommendation",
            effect="write",
            confirm="once",
            description=(
                "Start TRACKING a learning recommendation as an experiment so its effect "
                "can be measured. CALL THIS when the user says to test/track/try a "
                "recommendation ('test recommendation 2', 'let's track that one', 'try the "
                "first suggestion'). It captures the current baseline of the metric the "
                "recommendation targets and re-measures it over time, keeping or discarding "
                "it by the outcome. It does NOT apply the change itself — that stays the "
                'user\'s decision. Args: {"index": int} — the recommendation number (1-based) '
                "from learning_recommendations."
            ),
            call=_promote_recommendation,
        ),
    ]
    specs.append(
        ToolSpec(
            name="propose_skill_from_sandbox",
            effect="write",
            confirm="once",
            description=(
                "Promote a successful sandbox script into a proposed code-first skill. "
                'Args: {"script": str, "intent": str, "narrative": str, '
                '"slug"?: str}.'
            ),
            call=_propose_skill,
        )
    )
    return specs


def agent_self_management_enabled() -> bool:
    """ADR-0086: let the agent introspect + manage its own brain via ReAct tools.
    Off by default — it adds three tools to the loop and one of them mutates the
    conversation (compaction). Opt-in via ``IRIS_AGENT_SELF_MANAGEMENT``."""
    return os.getenv("IRIS_AGENT_SELF_MANAGEMENT", "").strip().lower() in {"1", "true", "yes", "on"}


__all__ = ["agent_self_management_enabled", "builtin_react_tools"]
