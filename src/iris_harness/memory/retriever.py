"""Memory retrieval — semantic when ChromaDB is available, keyword fallback otherwise."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, TypeVar

from iris_harness.foundation.process_state import track_globals
from iris_harness.llm.budget import trim_text

from .store import LearningSignal, MemoryStore, UserFact

if TYPE_CHECKING:
    from .semantic_index import RetrievedTurn, SemanticIndex

logger = logging.getLogger(__name__)


RECALL_MODE_ENV = "IRIS_MEMORY_RECALL_MODE"
RECALL_MODES = ("push", "pointer")


def recall_mode() -> str:
    """``IRIS_MEMORY_RECALL_MODE``, read per turn so a switch in the app applies at once.

    Anything but ``pointer`` is ``push``, today's behaviour, so a typo never changes it.
    """
    value = (os.environ.get(RECALL_MODE_ENV) or "").strip().lower()
    return value if value in RECALL_MODES else "push"


_REFERS_BACK: tuple[re.Pattern[str], ...] | None = None


def _refers_back_patterns() -> tuple[re.Pattern[str], ...]:
    """The ``refers_back`` phrases of ``config/memory/recall.yaml``, compiled once."""
    global _REFERS_BACK
    if _REFERS_BACK is not None:
        return _REFERS_BACK
    from iris_harness.foundation.paths import config_path, default_config_dir

    path = config_path("memory", "recall.yaml")
    if not path.exists():
        path = default_config_dir() / "memory" / "recall.yaml"
    phrases: list[str] = []
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
        phrases = [str(p) for p in (loaded or {}).get("refers_back") or [] if str(p).strip()]
    except Exception as exc:  # noqa: BLE001 — no vocabulary means no backstop, never a crash
        logger.warning("could not read %s (%s); recall backstop off", path, type(exc).__name__)
    _REFERS_BACK = tuple(
        re.compile(r"\b" + r"\s+".join(map(re.escape, p.lower().split())) + r"\b") for p in phrases
    )
    return _REFERS_BACK


def refers_back(message: str) -> bool:
    """Whether ``message`` clearly asks about an earlier conversation (recall.yaml)."""
    text = (message or "").lower()
    return any(p.search(text) for p in _refers_back_patterns())


def recall_max_distance() -> float:
    """``IRIS_MEMORY_RECALL_MAX_DISTANCE``: the cutoff for recalling past turns by meaning.

    One reader for the retriever's pushed turns and the recall tools' search, so the two
    cannot drift apart.
    """
    return _env_float("IRIS_MEMORY_RECALL_MAX_DISTANCE", 1.45)


def _env_float(name: str, default: float) -> float:
    """Read a float env override, falling back to ``default`` on unset/invalid."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class MemoryContext:
    """Context payload injected into prompts."""

    recent_turns: tuple[str, ...] = ()
    user_facts: tuple[UserFact, ...] = ()
    episodic_patterns: tuple[str, ...] = ()
    summary: str | None = None
    soul: str | None = None
    user_profile: str | None = None
    active: str | None = None
    episodic_digest: str | None = None
    behavior: str | None = None
    behavior_name: str | None = None
    # Prior turns whose content was semantically recalled into THIS turn's
    # context — the observable behind the downstream_reuse outcome (§4.2).
    reused_turn_refs: tuple[RetrievedTurn, ...] = ()
    # Excerpts recalled from OTHER sessions. Their own field, because they used to
    # be prepended into `recent_turns`, where the prompt's `[-3:]` tail cut them
    # after a session's second line — retrieved every turn, shown almost never.
    related_turns: tuple[str, ...] = ()
    # L1 of the context layers: one-line notices about memory that exists but is
    # NOT in this prompt (e.g. exchanges the transcript budget dropped). Cheap to
    # carry, and the model would otherwise have no way to know the history is there.
    pointers: tuple[str, ...] = ()
    # Pointer recall mode (IRIS_MEMORY_RECALL_MODE=pointer): the earlier conversations a
    # recall note named, and whether the refers-back backstop pushed turns instead.
    # Telemetry only (recall_pointer_offered / recall_backstop signals).
    recall_pointer_sessions: tuple[str, ...] = ()
    recall_backstop: bool = False
    # The memory graph around the names this message mentions (memris PR 6): their
    # current one-hop statements, confirmed only, within learning.yaml's linking budget.
    linked: str | None = None


@dataclass
class MemoryRetriever:
    """Build memory context using semantic search (ChromaDB) with keyword fallback.

    When a ``SemanticIndex`` is attached the retriever:
      1. Ranks user facts by embedding similarity to the current query.
      2. Ranks learning signals by embedding similarity.
      3. Appends a block of cross-session turns that are semantically relevant
         to the query — supplementing the caller-supplied chronological turns.

    When ``index`` is None or not ready, every step falls back to keyword
    overlap ranking so the system degrades gracefully.
    """

    store: MemoryStore = field(default_factory=MemoryStore)
    index: SemanticIndex | None = None
    max_facts: int = 10
    max_signals: int = 5
    max_cross_session_turns: int = 4
    max_episodic: int = 5
    # Recall quality filter (clean-context). Facts below `min_fact_confidence` are
    # DROPPED from recall (a 0.3 mis-extraction shouldn't be injected as authoritative);
    # facts below `uncertain_below` are kept but marked `uncertain` so the prompt renderer
    # flags them "(unconfirmed)". Static thresholds (no time-decay yet) per the agreed
    # first cut. Default-on; tune/disable via env (set min to 0.0 to stop dropping).
    min_fact_confidence: float = field(
        default_factory=lambda: _env_float("IRIS_MEMORY_MIN_FACT_CONFIDENCE", 0.35)
    )
    uncertain_below: float = field(
        default_factory=lambda: _env_float("IRIS_MEMORY_UNCERTAIN_BELOW", 0.6)
    )
    # Per-blob char caps prevent identity markdown from dominating tier1/tier2 contexts.
    # 2000 chars ≈ 500 tokens — fits 4× blobs inside a 2K-token budget with room to spare.
    max_identity_chars: int = 2000
    # USER.md is the one identity blob the user curates by hand, and 2000 chars cut a
    # real profile mid-sentence. It gets its own, larger cap; only the curated head is
    # injected (the `## Auto-detected` block is projected data, not a curated profile).
    max_profile_chars: int = field(
        default_factory=lambda: int(_env_float("IRIS_MEMORY_MAX_PROFILE_CHARS", 8000))
    )
    # Cross-session recall had NO relevance cutoff: the nearest 4 turns came back
    # whatever their distance. Same default as the memory_search tool (issue 0032).
    cross_session_max_distance: float = field(default_factory=lambda: recall_max_distance())
    # Pointer mode (IRIS_MEMORY_RECALL_MODE=pointer): how many earlier conversations the
    # note names, and how many turns the refers-back backstop still pushes.
    max_recall_pointers: int = 2
    backstop_turns: int = 2

    def build_context(
        self,
        *,
        query: str,
        recent_turns: Sequence[str] = (),
        session_id: str = "",
        intent: str = "",
    ) -> MemoryContext:
        """Return prompt context with facts and signals ranked by relevance."""
        if self.index is not None and self.index.is_ready:
            context = self._build_semantic(query, recent_turns, session_id)
        else:
            context = self._build_keyword(query, recent_turns)
        return self._attach_identity_context(context, query=query, intent=intent)

    # ------------------------------------------------------------------
    # Semantic path (ChromaDB available)
    # ------------------------------------------------------------------

    def _build_semantic(
        self,
        query: str,
        recent_turns: Sequence[str],
        session_id: str,
    ) -> MemoryContext:
        assert self.index is not None  # guarded by caller

        # --- Facts ---
        fact_keys = self.index.query_facts(query, n=self.max_facts)
        facts: list[UserFact] = []
        seen_keys: set[str] = set()
        for key in fact_keys:
            fact = self._safe_fetch_fact(key)
            if fact:
                facts.append(fact)
                seen_keys.add(key)
        # Backfill with keyword-ranked facts not already included
        if len(facts) < self.max_facts:
            remaining = self.max_facts - len(facts)
            all_facts = self._safe_fetch_all_facts()
            kw_ranked = _rank_by_relevance(query, all_facts, key=lambda f: f"{f.key} {f.value}")
            for f in kw_ranked:
                if f.key not in seen_keys and remaining > 0:
                    facts.append(f)
                    remaining -= 1

        # Learning signals are NOT fetched here any more. They were read out of the
        # whole signals table on every turn and then dropped by the prompt builder
        # unless an opt-in flag was set, which it never was. Lessons reach the model
        # two ways now: an approved one is a behavior (matched and injected), and
        # `memory_search(scope="behaviors")` reaches the rest on demand.

        # --- Cross-session turns (semantic recall from other sessions) ---
        cross_turns = self.index.query_turns_detailed(
            query,
            exclude_session=session_id or None,
            n=self.max_cross_session_turns,
            max_distance=self.cross_session_max_distance,
        )
        removed = self._removed_sessions()
        if removed is None:
            # The removal ledger could not be read, so any of these turns might come
            # from a session the owner removed (ADR-0119). Recall none rather than risk
            # putting one back into a prompt.
            cross_turns = []
        else:
            cross_turns = [t for t in cross_turns if t.session_id not in removed]
        pointers: tuple[str, ...] = ()
        pointed: tuple[str, ...] = ()
        backstop = False
        if recall_mode() == "pointer":
            if refers_back(query):
                # The owner is asking about the past: don't leave it to the model to go
                # and look (small local models often won't). Push the nearest few.
                cross_turns = cross_turns[: self.backstop_turns]
                backstop = bool(cross_turns)
            else:
                pointers = self._recall_pointers(cross_turns)
                if pointers:
                    pointed = tuple(
                        dict.fromkeys(t.session_id for t in cross_turns if t.session_id)
                    )[: self.max_recall_pointers]
                cross_turns = []
        related = tuple(f"{t.role}: {t.content}" for t in cross_turns)

        # --- Episodic patterns (semantic-only — no keyword fallback) ---
        episodic = tuple(self.index.query_episodic(query, n=self.max_episodic))

        return MemoryContext(
            recent_turns=tuple(recent_turns),
            user_facts=tuple(self._filter_facts(facts)[: self.max_facts]),
            episodic_patterns=episodic,
            reused_turn_refs=tuple(cross_turns),
            related_turns=related,
            pointers=pointers,
            recall_pointer_sessions=pointed,
            recall_backstop=backstop,
        )

    def _recall_pointers(self, turns: Sequence[RetrievedTurn]) -> tuple[str, ...]:
        """One note naming up to ``max_recall_pointers`` earlier conversations.

        Only a title, a date and the ``session_id`` the model passes to
        ``recall_conversation``: not their text, so an unrelated conversation costs a few
        tokens and never an answer.
        """
        named: list[str] = []
        seen: set[str] = set()
        for turn in turns:
            if not turn.session_id or turn.session_id in seen:
                continue
            seen.add(turn.session_id)
            title = self._conversation_title(turn)
            try:
                ts = self.store.conversation_row_ts(turn.row_id) or ""
            except Exception as exc:  # noqa: BLE001 — a pointer without its date is fine
                _log_degraded("read a recalled turn's date", exc, "naming it without a date")
                ts = ""
            when = f"{ts[:10]}, " if ts else ""
            named.append(f'"{title}" ({when}session_id={turn.session_id})')
            if len(named) >= self.max_recall_pointers:
                break
        if not named:
            return ()
        return (
            "Earlier conversations that may be related, NOT loaded: "
            + "; ".join(named)
            + ". If the user needs one, call recall_conversation with its session_id and a"
            " query.",
        )

    def _conversation_title(self, turn: RetrievedTurn) -> str:
        """A short label: the start of that conversation's summary, else the matched turn."""
        try:
            summary = self.store.load_conversation_summary(turn.session_id)
        except Exception as exc:  # noqa: BLE001 — fall back to the matched turn
            _log_degraded("read a recalled conversation's summary", exc, "titling it by turn")
            summary = ""
        text = " ".join((summary or turn.content or "").split())
        # 120, not 60: the A/B (PR #688) cut "…ticket R-58213, pickup is on Friday"
        # before the fact the question needed.
        return text if len(text) <= 120 else text[:117].rstrip() + "..."

    # ------------------------------------------------------------------
    # Keyword fallback path (no ChromaDB)
    # ------------------------------------------------------------------

    def _build_keyword(
        self,
        query: str,
        recent_turns: Sequence[str],
    ) -> MemoryContext:
        all_facts = self._safe_fetch_all_facts()
        ranked_facts = _rank_by_relevance(query, all_facts, key=lambda f: f"{f.key} {f.value}")
        top_facts = tuple(self._filter_facts(ranked_facts)[: self.max_facts])
        return MemoryContext(
            recent_turns=tuple(recent_turns),
            user_facts=top_facts,
        )

    # ------------------------------------------------------------------
    # Recall quality filter (clean-context)
    # ------------------------------------------------------------------

    def _filter_facts(self, facts: list[UserFact]) -> list[UserFact]:
        """Drop polluting low-confidence facts; mark the uncertain band.

        Static confidence thresholds only (no time-decay in this first cut). Facts below
        ``min_fact_confidence`` are dropped from recall entirely; facts in
        ``[min_fact_confidence, uncertain_below)`` are kept but flagged ``uncertain`` so the
        renderer marks them "(unconfirmed)". This is applied ONLY to extracted store facts —
        the hand-curated USER.md identity block is attached separately and stays
        authoritative.
        """
        out: list[UserFact] = []
        for f in facts:
            # Owner-confirmed only. Confidence is not consent: this store held
            # `name=ollama` at 1.0 and `employer=Department of Justice` at 0.9, both
            # mined from content the user was merely discussing. Unconfirmed facts
            # wait in the review queue instead of reaching a prompt.
            if not f.confirmed:
                continue
            if f.confidence < self.min_fact_confidence:
                continue
            out.append(replace(f, uncertain=True) if f.confidence < self.uncertain_below else f)
        return out

    # ------------------------------------------------------------------
    # Safe store accessors
    # ------------------------------------------------------------------

    def _safe_fetch_all_facts(self) -> list[UserFact]:
        try:
            # Recall reads the confirmed set; the rest is a review queue, not memory.
            return self.store.fetch_all_user_facts(confirmed_only=True)
        except Exception as exc:  # noqa: BLE001 — recall degrades, the turn goes on
            _log_degraded("fetch confirmed facts", exc, "recalling no stored facts")
            return []

    def _removed_sessions(self) -> set[str] | None:
        """Sessions the owner removed (ADR-0119): never recalled into another one.

        None when the ledger cannot be read: the caller then recalls no cross-session
        turn at all (fail closed), since it cannot tell which ones were removed.
        """
        try:
            return set(self.store.removed_session_ids())
        except Exception as exc:  # noqa: BLE001 — recall degrades, the turn goes on
            _log_degraded("read removed sessions", exc, "recalling no cross-session turns")
            return None

    def _safe_fetch_fact(self, key: str) -> UserFact | None:
        """A fact for recall — None unless the owner confirmed it.

        The semantic index still holds keys for unconfirmed facts until the next
        reindex, so this gate is what keeps them out of a prompt.
        """
        try:
            fact = self.store.fetch_user_fact(key)
        except Exception as exc:  # noqa: BLE001 — recall degrades, the turn goes on
            _log_degraded("fetch a fact", exc, "skipping it")
            return None
        return fact if (fact is not None and fact.confirmed) else None

    def _safe_fetch_signals(self) -> list[LearningSignal]:
        try:
            return self.store.fetch_learning_signals()
        except Exception as exc:  # noqa: BLE001 — recall degrades, the turn goes on
            _log_degraded("fetch learning signals", exc, "recalling none")
            return []

    def _attach_identity_context(
        self,
        context: MemoryContext,
        *,
        query: str,
        intent: str,
    ) -> MemoryContext:
        """Attach user-editable markdown memory layers to ``context``."""
        try:
            from iris_harness.memory.identity import (
                load_active_md,
                load_episodic_digest,
                load_soul_core,
                load_user_md,
                match_behavior,
                other_matching_behaviors,
            )

            matched = match_behavior(intent, query)
            # L1: name the lessons that also look relevant, one line each, instead of
            # spending their whole recipes' tokens. The agent reaches them with
            # memory_search(scope="behaviors").
            others = other_matching_behaviors(intent, query)
            pointers = list(context.pointers)
            if others:
                headlines = "; ".join(f"{b.name} ({b.headline})" for b in others)
                pointers.append(
                    f"{len(others)} more lesson(s) may apply — {headlines}. "
                    'Use memory_search with scope="behaviors" to read one.'
                )
            cap = self.max_identity_chars
            profile = _curated_profile(load_user_md() or "")
            return replace(
                context,
                soul=load_soul_core(),
                user_profile=trim_text(profile, max_chars=self.max_profile_chars) or None,
                active=trim_text(load_active_md() or "", max_chars=cap) or None,
                episodic_digest=trim_text(load_episodic_digest() or "", max_chars=cap) or None,
                behavior=trim_text(matched.body, max_chars=cap) if matched is not None else None,
                behavior_name=matched.name if matched is not None else None,
                pointers=tuple(pointers),
            )
        except Exception as exc:  # noqa: BLE001 — recall degrades, the turn goes on
            _log_degraded("attach identity context", exc, "prompt goes without SOUL/USER/lessons")
            return context


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _log_degraded(operation: str, exc: BaseException, fallback: str) -> None:
    """Say that recall degraded, so a broken store is not read as "no memory".

    Names the operation and the exception type only — never the query or any memory
    content (the traceback carries the store's own error message).
    """
    logger.warning(
        "memory recall: %s failed (%s); %s",
        operation,
        type(exc).__name__,
        fallback,
        exc_info=True,
    )


_AUTO_BLOCK_RE = re.compile(r"^##\s+Auto-detected\s*$", re.MULTILINE)


def _curated_profile(user_md: str) -> str:
    """Return only the hand-curated head of USER.md.

    Everything below ``## Auto-detected`` is the fact store's projection, written
    without the confidence filter recall applies, so it does not belong in the
    prompt as if the user had written it. Facts reach the prompt through the ranked
    `user_facts` path instead.
    """
    match = _AUTO_BLOCK_RE.search(user_md)
    head = user_md[: match.start()] if match else user_md
    return head.strip()


_T = TypeVar("_T")


def _rank_by_relevance(
    query: str,
    items: list[_T],
    key: Callable[[_T], str],
) -> list[_T]:
    """Sort items by keyword overlap with query (higher = more relevant)."""
    query_tokens = set(re.findall(r"\w+", query.lower()))
    if not query_tokens:
        return items

    def score(item: _T) -> int:
        text_tokens = set(re.findall(r"\w+", key(item).lower()))
        return len(query_tokens & text_tokens)

    return sorted(items, key=score, reverse=True)


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_REFERS_BACK")
