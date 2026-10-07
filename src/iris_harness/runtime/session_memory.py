"""Session memory — the per-session conversation window and its compaction.

What a conversation remembers between turns lives here: the in-memory history per
session (``conversations``), the summary of what compaction archived, the lazy reload of
both from the memory store on first access, the turn record that persists and indexes
each exchange, the compaction that keeps the window inside its token budget (reactive at
the turn boundary and on demand through ``compact_now``, ADR-0084), and the context-health
snapshot the CLI, API, web and the self-management tools read (ADR-0081). The two views
the turn pipeline builds from it — the routing transcript for the classifier (issue 0002)
and the memory context for the route stage — are here too.

Carved out of ``IrisRuntime`` at OSS plan M5.7 track C slice 16 as ``SessionMemory(host)``,
held as ``runtime.sessions``. The state only this code wrote moved with it: the
conversations, summaries, last-compaction telemetry, the compaction archive buffer
(ADR-0082), and the loaded and ephemeral session sets. :class:`SessionMemoryHost` declares
the seven runtime members read; the host is read **at call time**, not captured.
``conversations`` is public because activity notices append to it and the API reads it;
``drain_compaction_archive`` because behavior mining consumes the buffer.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

from iris_harness.foundation.observability.session_log import bind_context, current_turn_id
from iris_harness.kernel.governance.reentry import reenter_many, reenter_text
from iris_harness.llm.budget import estimate_tokens
from iris_harness.memory.compactor import CompactedHistory, ConversationTurn, summary_flag
from iris_harness.memory.retention import is_ephemeral_session
from iris_harness.memory.retriever import MemoryContext
from iris_harness.runtime.turn_capture import _is_conversation_scoped

if TYPE_CHECKING:
    from iris_harness.memory.compactor import ConversationCompactor
    from iris_harness.memory.graph_context import Linked
    from iris_harness.memory.retriever import MemoryRetriever
    from iris_harness.memory.semantic_index import SemanticIndex
    from iris_harness.memory.store import MemoryStore
    from iris_harness.runtime.learning_controls import LearningControls
    from iris_harness.services.learning.store import LearningMetricsStore

logger = logging.getLogger(__name__)

# "Learn before you forget" (ADR-0082): the compaction archive buffer is bounded.
_COMPACTION_ARCHIVE_BUFFER_MAX = 200

# How much of the live window we hand to the prompt builder, which then fills its own
# token budget from the newest end. A ceiling, not the prompt's slice: the compactor
# keeps the window near half the compaction trigger, so this rarely binds.
_MEMORY_CONTEXT_MAX_TURNS = 40

# How many sessions keep their window in memory (IRIS_SESSION_MEMORY_MAX_SESSIONS). The
# server is long-lived and every chat, channel thread and playground run is a session, so
# the least recently used ones beyond this are dropped from memory. A persisted session
# comes back from the memory store on its next turn, exactly as after a restart.
_DEFAULT_MAX_SESSIONS = 256


def _max_sessions() -> int:
    raw = os.getenv("IRIS_SESSION_MEMORY_MAX_SESSIONS", "")
    try:
        value = int(raw) if raw.strip() else _DEFAULT_MAX_SESSIONS
    except ValueError:
        logger.warning("IRIS_SESSION_MEMORY_MAX_SESSIONS=%r is not a number; using default", raw)
        value = _DEFAULT_MAX_SESSIONS
    return max(1, value)


# Ceiling on rows pulled back per session at reload; the window budget then decides how
# many of them are kept.
_RELOAD_MAX_TURNS = 60
# The closing summary reads at most this many of an idle session's newest turns.
_CLOSE_MAX_TURNS = 200


class SessionMemoryHost(Protocol):
    """The seven runtime members session memory reaches.

    ``_last_react_budget`` is declared as-is, underscore included, for the reason
    ``TurnHost`` gives: renaming a member is its own change. It is the ADR-0081 budget
    sink the AgenticCore writes each turn; ``context_health`` only reads it.
    """

    compactor: ConversationCompactor
    learning: LearningControls
    learning_store: LearningMetricsStore
    memory_retriever: MemoryRetriever
    memory_store: MemoryStore
    semantic_index: SemanticIndex | None
    _last_react_budget: dict[str, int]


class SessionMemory:
    """The conversation window, its compaction and its views for one runtime. See the
    module docstring."""

    # Routing context budget. The Router tier (llama3.2:3b) has a tiny num_ctx
    # (config/llm_tiers.yaml); the router system prompt is already ~700 tokens, so
    # the context MUST be hard-capped or a busy conversation overflows the window
    # and Ollama silently truncates the system instructions. Routing only needs
    # the immediately-preceding topic, so a small, recency-first budget is plenty
    # AND bounded regardless of how long the conversation grows (issue 0002).
    _ROUTING_CONTEXT_MAX_CHARS = 600
    _ROUTING_CONTEXT_MAX_TURNS = 4
    _ROUTING_CONTEXT_PER_TURN_CHARS = 200

    def __init__(self, host: SessionMemoryHost) -> None:
        self._host = host
        self.conversations: dict[str, list[ConversationTurn]] = {}
        self._session_summaries: dict[str, str] = {}
        # The provenance flag of each cached summary (#145): True built from external-origin
        # turns, False known not, None unknown. A summary with no entry reads as unknown.
        self._summary_flags: dict[str, bool | None] = {}
        # Context-health telemetry (ADR-0081): the last conversation-compaction event per
        # session. In-memory; transient by nature.
        self._last_compaction: dict[str, dict[str, Any]] = {}
        # "Learn before you forget" (ADR-0082): turns archived by conversation compaction
        # are buffered here so the behavior-mining heartbeat mines that about-to-be-evicted
        # span specifically — before it scrolls out of the miner's recency window.
        # Bounded; only filled when the miner is enabled. In-memory.
        self._compaction_archive_buffer: list[ConversationTurn] = []
        self._loaded_sessions: set[str] = set()
        # Phase 3 follow-up: sessions the user has marked conversation-scoped
        # ("for this conversation, ..."). Once a session is ephemeral, NONE of
        # its turns are added to the cross-session semantic index — not the
        # scoped statement and not the follow-up Q&A that would otherwise echo
        # it back out. In-session recall is unaffected (history is in memory).
        self._ephemeral_sessions: set[str] = set()
        # Sessions with a summary roll in flight. Summarization is an LLM call, so it
        # runs AFTER the reply is sent; the verbatim window is only swapped once the new
        # summary lands, so nothing is dropped if the call fails or the process exits.
        self._compacting: set[str] = set()
        self._compaction_lock = threading.Lock()
        # Sessions in memory, least recently used first; see _DEFAULT_MAX_SESSIONS.
        self._max_sessions = _max_sessions()
        self._recent: OrderedDict[str, None] = OrderedDict()
        self._recent_lock = threading.Lock()

    def _touch(self, session_id: str) -> None:
        """Mark a session as just used, and drop the least recently used ones beyond the
        cap. An active session is never the one dropped: it was just touched."""
        with self._recent_lock:
            recent = self._recent
            recent[session_id] = None
            recent.move_to_end(session_id)
            # A window someone else opened (an activity notice) counts as least recent.
            for other in self.conversations.keys() - recent.keys():
                recent[other] = None
                recent.move_to_end(other, last=False)
            excess = len(recent) - self._max_sessions
            if excess <= 0:
                return
            for victim in list(recent):
                if excess <= 0:
                    break
                if victim == session_id or victim in self._compacting:
                    continue
                del recent[victim]
                self._forget(victim)
                excess -= 1

    def _forget(self, session_id: str) -> None:
        """Drop a session's in-memory state; the next access reloads what was persisted."""
        self.conversations.pop(session_id, None)
        self._session_summaries.pop(session_id, None)
        self._summary_flags.pop(session_id, None)
        self._last_compaction.pop(session_id, None)
        self._loaded_sessions.discard(session_id)
        # A playground/eval session is re-marked ephemeral on its next turn; a session the
        # user scoped to "this conversation" is not, so that mark is kept.
        if is_ephemeral_session(session_id):
            self._ephemeral_sessions.discard(session_id)

    def _load_session_if_needed(self, session_id: str) -> None:
        """Lazy-load persisted conversation history for a session on first access per process."""
        self._touch(session_id)
        if session_id in self._loaded_sessions:
            return
        self._loaded_sessions.add(session_id)
        if session_id in self.conversations:
            return  # already active (e.g. in tests)
        try:
            summary = self._host.memory_store.load_conversation_summary(session_id)
            if summary:
                self._session_summaries[session_id] = summary
                self._summary_flags[session_id] = (
                    self._host.memory_store.load_conversation_summary_flags([session_id]).get(
                        session_id
                    )
                )
            # Reload what the window can hold, not a fixed 10 rows: with the prompt's
            # transcript now filled by token budget, a restart used to hand it a third
            # of the history it had a moment earlier.
            raw = self._host.memory_store.load_recent_turns_with_origin(
                session_id, limit=_RELOAD_MAX_TURNS
            )
            if raw:
                turns = [
                    ConversationTurn(role=role, content=content, origin=origin)
                    for role, content, origin in raw
                ]
                self.conversations[session_id] = self._fit_reloaded(turns)
        except Exception:
            logger.exception("failed to reload session history for %s", session_id)

    def _fit_reloaded(self, turns: list[ConversationTurn]) -> list[ConversationTurn]:
        """Keep the newest reloaded turns that fit the compaction window."""
        budget = getattr(self._host.compactor, "token_budget", None)
        if not budget:
            return turns[-10:]
        keep_budget = max(1, int(budget * 0.5))
        kept: list[ConversationTurn] = []
        used = 0
        for turn in reversed(turns):
            cost = estimate_tokens(f"{turn.role}: {turn.content}")
            if kept and used + cost > keep_budget:
                break
            kept.append(turn)
            used += cost
        return list(reversed(kept))

    def context_health(self, session_id: str = "default") -> dict[str, Any]:
        """One snapshot of how the harness is managing its context (ADR-0081).

        Composes the conversation-window fill + last compaction (compactor), the in-loop
        budget split + latest eviction (P3 controller via the budget sink), and the
        surface-feedback suppression roll-up — the three context-bounding mechanisms — into
        a single view for the CLI / API / web. Pure read; never raises.
        """
        from iris_harness.agent.context_health import BudgetSplit, ContextHealth, WindowHealth
        from iris_harness.services.learning.suppression import SurfaceFeedbackStore

        self._load_session_if_needed(session_id)
        history = self.conversations.get(session_id, [])
        summary = self._session_summaries.get(session_id, "")
        compactor = self._host.compactor
        current = compactor._history_tokens(history) + (estimate_tokens(summary) if summary else 0)
        window = WindowHealth(
            budget_tokens=compactor.token_budget or 0,
            current_tokens=current,
            compaction_ratio=compactor.compaction_ratio,
            last_compaction=self._last_compaction.get(session_id),
        )
        sink = self._host._last_react_budget
        budgets = BudgetSplit(
            transcript_budget=sink.get("transcript_budget", 0),
            memory_budget=sink.get("memory_budget", 0),
            last_context_tokens=sink.get("last_context_tokens"),
            last_transcript_evicted=sink.get("last_transcript_evicted"),
        )
        try:
            db_path = getattr(self._host.learning_store, "db_path", None)
            suppression = SurfaceFeedbackStore(db_path=db_path).summary().as_dict()
        except Exception:  # health view degrades, never crashes
            logger.debug("suppression summary failed for context-health", exc_info=True)
            suppression = {"total_feedback": 0, "active_suppressions": 0, "by_subsystem": {}}
        return ContextHealth(
            session_id=session_id, window=window, budgets=budgets, suppression=suppression
        ).as_dict()

    def build_memory_context(
        self,
        message: str,
        *,
        session_id: str,
        intent: str = "",
    ) -> MemoryContext:
        self._load_session_if_needed(session_id)
        summary = self._session_summaries.get(session_id, "")
        history = self.conversations.get(session_id, [])
        # The summary used to be the FIRST element of recent_turns, which the prompt's
        # `recent_turns[-3:]` tail then cut as soon as a session had two lines: written
        # on every compaction, rendered never. It travels in its own field now.
        #
        # We hand over the whole in-memory window (the compactor already bounds it) and
        # let the prompt builder fill its transcript budget from the newest end, instead
        # of pre-cutting to 6 lines here and 3 lines there.
        window = history[-_MEMORY_CONTEXT_MAX_TURNS:]
        # The one place the stored window and summary become prompt text for every reader
        # (the loop, the general lane, code_exec, escalation actions, the reloaded session
        # after a restart): assistant turns and the summary are scanned, the owner's own
        # turns are not (#145). The stored rows and the in-memory window stay as written.
        shown = reenter_many(
            [(t.role, t.content) for t in window],
            reader="session_window",
            origin="transcript",
            origins=[t.origin for t in window],
        )
        recent = tuple(f"{t.role}: {r.text}" for t, r in zip(window, shown, strict=True))
        if summary:
            # A summary that absorbed external-origin turns comes back inside the envelope,
            # as well as scanned (#145); an unknown or known-not one is scanned only.
            summary = reenter_text(
                summary,
                reader="session_summary",
                origin="summary",
                role="summary",
                turn_origin="external" if self._summary_flags.get(session_id) is True else None,
            ).text
        pointers: list[str] = []
        if summary:
            pointers.append(
                "Earlier exchanges in this session are condensed in the summary above, "
                "not repeated in full below. Use recall_conversation to read them word "
                "for word."
            )
        try:
            context = self._host.memory_retriever.build_context(
                query=message,
                recent_turns=recent,
                session_id=session_id,
                intent=intent,
            )
        except Exception:
            logger.exception("memory retriever failed; using empty context")
            context = MemoryContext(recent_turns=recent)
        linked = self._linked(message)
        if linked is not None and linked.pointer:
            pointers.append(linked.pointer)
        return replace(
            context,
            summary=summary or None,
            pointers=tuple(pointers) + context.pointers,
            linked=(linked.text or None) if linked is not None else None,
        )

    def _linked(self, message: str) -> Linked | None:
        """The graph around the names in ``message`` (ADR-0115 decision 9), or None."""
        if not (message or "").strip():
            return None
        try:
            from iris_harness.memory.graph_context import (
                graph_context,
                linking_max_tokens,
            )

            return graph_context(self._host.memory_store).linked(
                message, max_tokens=linking_max_tokens()
            )
        except Exception:  # linking must never cost a turn
            logger.exception("memory graph linking failed; the turn goes on without it")
            return None

    def format_recent_context(self, session_id: str) -> str | None:
        """Recent conversation turns as a compact transcript for the intent
        classifier (issue 0002), so a follow-up is routed by topic, not isolated
        keywords. Hard-capped (see ``_ROUTING_CONTEXT_MAX_CHARS``) and built
        most-recent-first so it never grows with conversation length. None on the
        first turn (no history)."""
        history = self.conversations.get(session_id, [])
        if not history:
            return None
        lines: list[str] = []
        used = 0
        # Most recent first — the latest exchange matters most for routing — then
        # stop as soon as the total budget is hit.
        window = history[-self._ROUTING_CONTEXT_MAX_TURNS :]
        # Stored text going into the router's prompt: assistant turns are scanned whole,
        # before the per-turn cut so a phrase cannot hide at it; the owner's are not (#145).
        shown = reenter_many(
            [(getattr(t, "role", ""), getattr(t, "content", "") or "") for t in window],
            reader="router_context",
            origin="transcript",
            origins=[getattr(t, "origin", None) for t in window],
            limit=self._ROUTING_CONTEXT_PER_TURN_CHARS,
        )
        for turn, scanned in zip(reversed(window), reversed(shown), strict=True):
            text = (scanned.text or "").strip().replace("\n", " ")
            if not text:
                continue
            role = "User" if getattr(turn, "role", "") == "user" else "Assistant"
            # An enveloped turn was cut inside its envelope already; cutting it again would
            # drop the closing tag.
            line = (
                f"{role}: {text}"
                if scanned.enveloped
                else f"{role}: {text[: self._ROUTING_CONTEXT_PER_TURN_CHARS]}"
            )
            if used + len(line) > self._ROUTING_CONTEXT_MAX_CHARS:
                break
            lines.append(line)
            used += len(line) + 1  # +1 for the newline join
        lines.reverse()  # back to chronological order for the prompt
        return "\n".join(lines) if lines else None

    def record_turn(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        *,
        origin: str | None = None,
    ) -> None:
        """Remember one exchange. ``origin`` is the assistant turn's ``turn_origin`` (#145):
        ``"external"`` when the run read third-party text before answering, None when the
        producer did not say (only the loop does, today)."""
        self._touch(session_id)
        history = self.conversations.setdefault(session_id, [])
        history.append(ConversationTurn(role="user", content=user_msg))
        history.append(ConversationTurn(role="assistant", content=assistant_msg, origin=origin))
        # Honor an explicit conversation-scope qualifier ("for this
        # conversation, ...", "just for now", ...): mark the whole session
        # ephemeral so neither the scoped statement nor the follow-up Q&A
        # that echoes it gets added to the cross-session semantic index.
        # In-session history still works. Phase 3 multiturn finding.
        if _is_conversation_scoped(user_msg):
            self._ephemeral_sessions.add(session_id)
        # Playground / eval / test sessions are not the user's memory. 1,420 of 3,730
        # stored turns came from runs like these, and cross-session recall served them
        # back as "your past conversations" (config/memory/retention.yaml). They keep a
        # live window — the run still needs its own history — but nothing is persisted.
        persist = not is_ephemeral_session(session_id)
        if not persist:
            self._ephemeral_sessions.add(session_id)
        cross_session_ok = session_id not in self._ephemeral_sessions
        try:
            row_ids = (
                self._host.memory_store.save_conversation_turns_and_get_ids(
                    session_id,
                    [("user", user_msg), ("assistant", assistant_msg)],
                    assistant_origin=origin,
                )
                if persist
                else []
            )
            semantic_index = self._host.semantic_index
            if semantic_index and row_ids and cross_session_ok:
                # Stamp the live turn_id so a later turn that recalls this content
                # can attribute downstream_reuse back to this turn (§4.2).
                tid = current_turn_id()
                semantic_index.index_turn(row_ids[0], session_id, "user", user_msg, turn_id=tid)
                semantic_index.index_turn(
                    row_ids[1], session_id, "assistant", assistant_msg, turn_id=tid
                )
                semantic_index._save_watermark(row_ids[-1])
        except Exception:
            logger.exception("failed to persist conversation turns for session %s", session_id)
        compactor = self._host.compactor
        if compactor.needs_compaction(history):
            self._schedule_compaction(session_id)

    def _schedule_compaction(self, session_id: str) -> None:
        """Roll the summary on a worker thread, after this turn's reply has gone out.

        Compaction used to run inline in ``record_turn``, which is inside the chat
        finalizer — with a real summarizer wired that would add seconds to the turn the
        user is waiting on. One roll per session at a time; the window is swapped only
        when the new summary is in hand.
        """
        with self._compaction_lock:
            if session_id in self._compacting:
                return
            self._compacting.add(session_id)

        # ThreadPoolExecutor/Thread do not carry context vars, so the summarizer's
        # llm_call would be logged against no session (PR #491).
        worker = bind_context(self._run_compaction)
        threading.Thread(
            target=worker,
            args=(session_id,),
            name=f"iris-compaction-{session_id[:24]}",
            daemon=True,
        ).start()

    def _run_compaction(self, session_id: str) -> None:
        try:
            history = list(self.conversations.get(session_id, []))
            if not history:
                return
            compacted = self._host.compactor.compact(
                history,
                previous_summary=self._session_summaries.get(session_id, ""),
            )
            if compacted.trigger == "none":
                return
            # Turns recorded while the summary was being written are not in `compacted`;
            # keep them, or a turn taken during the roll would vanish from the window.
            live = self.conversations.get(session_id, [])
            appended = live[len(history) :]
            self._apply_compaction(
                session_id,
                replace(compacted, recent_turns=tuple(list(compacted.recent_turns) + appended)),
            )
        except Exception:  # a failed roll must never break the session
            logger.exception("background conversation compaction failed for %s", session_id)
        finally:
            with self._compaction_lock:
                self._compacting.discard(session_id)

    def compaction_in_flight(self, session_id: str) -> bool:
        """True while a summary roll is running (telemetry + tests)."""
        with self._compaction_lock:
            return session_id in self._compacting

    def wait_for_compaction(self, session_id: str, *, timeout: float = 30.0) -> bool:
        """Block until this session's summary roll finishes. Returns False on timeout.

        For callers that need the settled state — tests, and any control that reports
        the window right after a turn. Normal turns never call this: the point of the
        roll being asynchronous is that the reply does not wait for it.
        """
        deadline = time.monotonic() + timeout
        while self.compaction_in_flight(session_id):
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)
        return True

    def _apply_compaction(self, session_id: str, compacted: CompactedHistory) -> None:
        """Apply a CompactedHistory to live + durable state: swap the in-memory history,
        stamp the last-compaction telemetry (ADR-0081), buffer the archived span for the
        miner (ADR-0082), and persist the summary. Shared by the reactive turn-boundary
        path and the on-demand ``compact_now`` control (ADR-0084)."""
        if compacted.trigger == "none":
            return
        self.conversations[session_id] = list(compacted.recent_turns)
        logger.info(
            "conversation compaction fired: session=%s trigger=%s archived=%d "
            "tokens %d->%d kept_turns=%d",
            session_id,
            compacted.trigger,
            compacted.archived_count,
            compacted.tokens_before,
            compacted.tokens_after,
            len(compacted.recent_turns),
        )
        self._last_compaction[session_id] = {
            "trigger": compacted.trigger,
            "archived_count": compacted.archived_count,
            "tokens_before": compacted.tokens_before,
            "tokens_after": compacted.tokens_after,
            "kept_turns": len(compacted.recent_turns),
        }
        # Learn before you forget (ADR-0082): preserve the archived span for the behavior
        # miner so a long session's oldest turns aren't lost to the miner's recency window.
        # Cheap (no LLM here); only when the miner is enabled. Bounded.
        if self._host.learning.behavior_miner is not None and compacted.archived_turns:
            buf = self._compaction_archive_buffer
            buf.extend(compacted.archived_turns)
            if len(buf) > _COMPACTION_ARCHIVE_BUFFER_MAX:
                del buf[:-_COMPACTION_ARCHIVE_BUFFER_MAX]
        if compacted.summary:
            # Whether the rolled summary absorbed third-party text: the previous flag, rolled
            # forward over the turns folded into it (sticky, #145).
            previous_summary = self._session_summaries.get(session_id, "")
            flag = summary_flag(
                self._summary_flags.get(session_id) if previous_summary else None,
                had_summary=bool(previous_summary),
                folded=compacted.archived_turns,
            )
            self._session_summaries[session_id] = compacted.summary
            self._summary_flags[session_id] = flag
            try:
                self._host.memory_store.save_conversation_summary(
                    session_id, compacted.summary, has_external=flag
                )
            except Exception:
                logger.exception("failed to persist summary for session %s", session_id)

    def drain_compaction_archive(self) -> list[ConversationTurn]:
        """Hand the archived span to the behavior miner and empty the buffer (ADR-0082).

        Consumed once: the miner mines it as its own pass, so a second drain before the
        next compaction returns nothing."""
        archived = self._compaction_archive_buffer
        self._compaction_archive_buffer = []
        return archived

    def close_session(self, session_id: str) -> dict[str, Any]:
        """Write the closing summary of an idle session — over ALL of its turns.

        The retention pass used ``compact_now`` for this, which summarizes only what is
        older than the kept window: a short conversation returned "nothing to compact",
        no summary was written, and the session was counted closed anyway (391 were
        waiting on the owner's box, 8 had summaries). Reads the stored turns without
        loading the session into the live cache; a summary already stored is rolled
        forward, not replaced. ``closed`` is True only when a summary was saved.
        """
        store = self._host.memory_store
        previous = store.load_conversation_summary(session_id) or ""
        raw = store.load_recent_turns_with_origin(session_id, limit=_CLOSE_MAX_TURNS)
        turns = [
            ConversationTurn(role=role, content=content, origin=origin)
            for role, content, origin in raw
        ]
        if not turns:
            return {"closed": False, "reason": "no turns", "session_id": session_id}
        summary = self._host.compactor.summarize_all(turns, previous_summary=previous)
        if not summary.strip() or summary == previous:
            return {"closed": False, "reason": "no summary produced", "session_id": session_id}
        # The previous flag comes from the store: this reads a session that is not loaded.
        previous_flag = store.load_conversation_summary_flags([session_id]).get(session_id)
        flag = summary_flag(previous_flag, had_summary=bool(previous), folded=turns)
        store.save_conversation_summary(session_id, summary, has_external=flag)
        if session_id in self._session_summaries:
            self._session_summaries[session_id] = summary
            self._summary_flags[session_id] = flag
        return {"closed": True, "session_id": session_id, "turns": len(turns)}

    def compact_now(self, session_id: str = "default") -> dict[str, Any]:
        """Force-compact a session's conversation on demand (ADR-0084) — the action half of
        the context-health control loop. Summarizes the older span even below the auto
        trigger, as long as there's something to summarize; a short conversation is a no-op.
        Returns ``{compacted: bool, ...telemetry}``."""
        self._load_session_if_needed(session_id)
        history = self.conversations.get(session_id, [])
        compacted = self._host.compactor.compact(
            history,
            force=True,
            previous_summary=self._session_summaries.get(session_id, ""),
        )
        if compacted.trigger == "none":
            return {"compacted": False, "reason": "nothing to compact", "session_id": session_id}
        self._apply_compaction(session_id, compacted)
        return {
            "compacted": True,
            "session_id": session_id,
            "trigger": compacted.trigger,
            "archived_count": compacted.archived_count,
            "tokens_before": compacted.tokens_before,
            "tokens_after": compacted.tokens_after,
            "kept_turns": len(compacted.recent_turns),
        }
