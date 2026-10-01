"""Per-turn learning capture — user facts, the chat learning signal, turn metrics and
cross-turn outcome attribution (§4.2).

Extracted from ``IrisRuntime`` as ``FactCaptureMixin`` in Phase 2 (byte-identical
bodies, inherited). OSS plan M5.7 track C slice 12 made it the last mixin to become a
collaborator: ``TurnCapture(host)``, held as ``runtime.capture``. After it
``IrisRuntime`` inherits nothing.

The state only this code used moved with it: the prior turn's outcome descriptor per
session and the lazily-built correction detector and outcome config. The three members
something outside the class calls went public — ``extract_and_store_facts`` and
``record_signal`` (the curate step; ``record_signal`` also from routine authoring) and
``evaluate_prior_turn_outcome`` (the turn pipeline's intercept stage, through
``TurnHost.capture``). :class:`TurnCaptureHost` declares the five runtime members read;
the host is read **at call time**, not captured.

A few general predicates live here at module level and are imported by bootstrap and
the escalation actions; they will move to declarative config in a later phase.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

from iris_harness.agent.intent_router import IntentResult
from iris_harness.agent.outcomes import (
    CORRECTION_JUDGE_SYSTEM_PROMPT,
    OutcomeConfig,
    build_correction_user_prompt,
    load_outcome_config,
    looks_like_correction,
    parse_correction_verdict,
)
from iris_harness.agent.response_curator import CuratedResponse
from iris_harness.foundation.observability.session_log import current_session_id, current_turn_id
from iris_harness.foundation.observability.tracer import current_trace_ids
from iris_harness.llm.tier_router import TierRouter
from iris_harness.memory.fact_keys import (
    ASK,
    AUTO,
    ask_max_per_session,
    canonical_key,
    classify_capture,
    is_self_statement,
    repeat_confirms_after,
    subject_max_hops,
)
from iris_harness.memory.fact_statements import Link, SubjectError
from iris_harness.memory.ontology import normalise_fact_key
from iris_harness.memory.semantic_index import SemanticIndex
from iris_harness.memory.store import MemoryStore
from iris_harness.memory.triage import MemoryDestination, triage_memory_item
from iris_harness.services.learning.signals import LearningSignalCollector

logger = logging.getLogger(__name__)


_CONVERSATION_SCOPE_RE = re.compile(
    r"\b("
    r"for (?:this|the current) (?:conversation|chat|session|thread)|"
    r"in (?:this|the current) (?:conversation|chat|session|thread)|"
    r"(?:just|only) for (?:now|today|this (?:chat|conversation|session))|"
    r"temporarily|"
    r"don'?t (?:remember|save|store|keep) this"
    r")\b",
    re.IGNORECASE,
)


def _about(candidate: tuple[Any, ...]) -> str | None:
    """Who an extracted fact is about, when not the owner (its optional fifth field)."""
    return candidate[4] if len(candidate) > 4 and candidate[4] else None


def _is_conversation_scoped(message: str) -> bool:
    """True when the user explicitly scoped a statement to this conversation."""
    return bool(_CONVERSATION_SCOPE_RE.search(message or ""))


_QUESTION_OR_COMMAND_RE = re.compile(
    r"^\s*(?:what|whats|what's|who|whom|whose|how|why|when|where|which|"
    r"do|does|did|is|are|am|was|were|can|could|will|would|should|shall|may|might|"
    r"please|show|list|tell|find|get|give|fetch|search|look|summari[sz]e|read|open|"
    r"check|compare|convert|calculate|explain|help|any|anything)\b",
    re.IGNORECASE,
)


def _looks_like_question_or_command(message: str) -> bool:
    """True for question/command turns that shouldn't yield durable user facts."""
    m = (message or "").strip()
    if not m:
        return True
    return m.endswith("?") or bool(_QUESTION_OR_COMMAND_RE.match(m))


_OUTCOME_DISABLED: Any = object()


def _build_correction_detector(
    *,
    tier_router: TierRouter,
    config_dir: Path,
) -> Callable[..., Any] | None:
    """Build the model-driven user-correction detector (§4.2), or None if off.

    Opt-in via ``config/outcomes.yaml`` (``user_correction.enabled``) or the
    ``IRIS_LEARNING_CORRECTION_JUDGE`` env flag — default off, since it costs a
    governed LLM call on turns that look like corrections. Returns a sync
    callable ``detect(prior_query, prior_response, new_message) ->
    CorrectionVerdict | None``; the runtime gates it behind a cheap pre-filter so
    only candidate turns pay. None when disabled or unbuildable (no signal
    emitted — we never substitute a heuristic for the measurement).
    """
    cfg = load_outcome_config(config_dir)
    if not cfg.user_correction_enabled:
        return None
    try:
        from iris_harness.llm.client import CodingLLMClient

        client_cfg = cast(Any, tier_router.get_llm_config("general"))
        client_cfg = client_cfg.model_copy(update={"temperature": 0.0, "max_tokens": 128})
        client = CodingLLMClient(client_cfg, governance_agent_type="chat")
    except Exception:
        logger.debug("correction detector disabled: could not init LLM client", exc_info=True)
        return None

    def detect(*, prior_query: str, prior_response: str, new_message: str) -> Any:
        user_prompt = build_correction_user_prompt(
            prior_query=prior_query, prior_response=prior_response, new_message=new_message
        )
        raw = client.invoke(system_prompt=CORRECTION_JUDGE_SYSTEM_PROMPT, user_prompt=user_prompt)
        return parse_correction_verdict(raw)

    return detect


def _coerce_total_tokens(value: Any) -> int | None:
    """Best-effort int for the curated total_tokens metadata; None if absent/zero."""
    try:
        tokens = int(value)
    except (TypeError, ValueError):
        return None
    return tokens if tokens > 0 else None


def _extract_escalation_verdict(curated: CuratedResponse) -> dict[str, Any] | None:
    """Pull the shadow escalation verdict out of the curator's judge bundle, if any."""
    bundle = curated.metadata.get("judge_bundle")
    if not isinstance(bundle, dict):
        return None
    for signal in bundle.get("signals", []):
        if not isinstance(signal, dict) or signal.get("name") != "escalation":
            continue
        meta = signal.get("metadata")
        if isinstance(meta, dict):
            verdict = meta.get("escalation")
            if isinstance(verdict, dict):
                return verdict
    return None


_YES = {"yes", "y", "yep", "yeah", "sure", "ok", "okay", "please do", "go ahead", "correct"}
_NO = {"no", "nope", "don't", "dont", "no thanks", "nah", "wrong", "incorrect"}


def _parse_yes_no(message: str) -> bool | None:
    """True/False for a plain answer, None when the turn is about something else."""
    text = " ".join((message or "").strip().lower().split()).strip(".!?")
    if not text or len(text.split()) > 4:
        return None
    if text in _YES:
        return True
    if text in _NO:
        return False
    return None


class TurnCaptureHost(Protocol):
    """The five runtime members the per-turn capture reaches.

    ``semantic_index`` is back: a plainly-stated fact is confirmed during the turn, and
    a confirmed fact belongs in the recall index immediately — waiting for the next
    startup resync would mean IRIS knew something it could not recall.
    """

    config_dir: Path
    memory_store: MemoryStore
    semantic_index: SemanticIndex | None
    signal_collector: LearningSignalCollector
    tier_router: TierRouter


class TurnCapture:
    """Per-turn learning capture for one runtime: user facts, the chat learning signal,
    the turn's metrics, and the prior turn's measured outcome. See the module docstring."""

    def __init__(self, host: TurnCaptureHost) -> None:
        self._host = host
        # §4.2 cross-turn outcomes: the prior turn's descriptor per session, so the
        # NEXT turn can attribute a measured outcome (user_correction) back to the
        # turn that produced it. In-memory; outcomes are conversational.
        self._last_turn_outcome: dict[str, dict[str, Any]] = {}
        # Lazily-built model-driven correction detector (§4.2). Sentinel-cached:
        # None = not built, _OUTCOME_DISABLED = built-and-off, else the callable.
        self._correction_detector_cache: Any = None
        # What this turn learned outright, to be said in the reply ("Noted: ...").
        self._notices: list[str] = []
        # The one open "should I remember this?" per session: {session: (id, key, value)}.
        self._pending_question: dict[str, tuple[str, str, str]] = {}
        # Or the one open "is X the same as Y?" per session: {session: decision id}. Both
        # kinds share one budget (ADR-0115 decision 4): at most one question a conversation.
        self._pending_lookalike: dict[str, str] = {}
        self._asked_this_session: set[str] = set()
        # Whose fact the open question is about, when not the owner's: {session: name}.
        self._question_subject: dict[str, str] = {}
        self._outcome_config_cache: OutcomeConfig | None = None

    def extract_and_store_facts(self, user_msg: str, session_id: str = "") -> None:
        """Extract user facts from a message and persist to SQLite + ChromaDB + user.md.

        Mode is controlled by ``IRIS_FACT_EXTRACTION_MODE`` (default ``llm``).
        The LLM extractor understands sentence structure ("I'm working on Aur"
        is a project, not a name); the brittle ``regex`` path remains available
        as an explicit opt-in fallback.  Either way, every candidate fact must
        clear ``is_plausible_fact`` before it can be persisted.
        """
        from iris_harness.memory.fact_validation import (
            is_durable_fact,
            is_fact_grounded,
            is_plausible_fact,
        )

        # Explicitly conversation-scoped statements ("for this conversation,
        # my favorite color is teal") must not become durable user facts.
        self._notices = []
        if _is_conversation_scoped(user_msg):
            logger.info("fact extraction skipped: message is conversation-scoped")
            return
        # Don't mine durable facts from QUESTIONS or commands — that's how the store
        # got polluted (greeting=hello, stock=apple stock) and, worse, how asking
        # "do you know my blog site?" extracted blog="site" and clobbered the real
        # blog=web3notes.example. The LLM extractor only runs on declarative statements;
        # the deterministic declarative capture below still runs either way (it only
        # fires on grounded "I … at <url>" / "my <noun> is <url>" patterns). (issue 0021)
        # A fact about the user has to come from the user talking about themselves.
        # `employer=Department of Justice` (0.9) was mined from a pasted news article;
        # `name=ollama` (1.0) from a message about configuring a model.
        if not is_self_statement(user_msg):
            logger.info("fact extraction skipped: message is not a first-person statement")
            return
        mode = os.getenv("IRIS_FACT_EXTRACTION_MODE", "llm").strip().lower()
        # (key, value, confidence, source[, about]) — `about` names someone one hop away.
        extracted: list[tuple[Any, ...]]
        if _looks_like_question_or_command(user_msg):
            extracted = []
        elif mode == "regex":
            extracted = self._extract_facts_via_regex(user_msg)
        else:
            extracted = self._extract_facts_via_llm(user_msg)

        # Deterministic high-signal declaratives ("I write blogs at <url>", "my site
        # is <url>") are a reliable floor under the extractor — the tier-1 LLM keeps
        # missing them or mangling them (e.g. blog="site"). They WIN for their keys:
        # grounded, verbatim from the message, so they replace any same-key LLM guess
        # (issue 0020).
        declarative = self._extract_facts_declarative(user_msg)
        if declarative:
            decl_keys = {f[0] for f in declarative}
            extracted = [t for t in extracted if t[0] not in decl_keys or _about(t)] + list(
                declarative
            )
        # Facts about the owner first: a relation they state ("my wife Petra") is what
        # brings its object one hop away into scope for the facts about her.
        extracted = sorted(extracted, key=lambda t: _about(t) is not None)

        persisted = False
        links: list[Link] = []
        for candidate in extracted:
            raw_key, value, confidence, source = candidate[:4]
            about = _about(candidate)
            # Key allowlist (the fact mappings in config/memory/mappings.yaml): the extractor
            # invents keys for whatever the conversation is ABOUT — topic, error, source,
            # number — and those are not properties of the user. A key the ontology lacks
            # is not stored: once through every gate below it is COUNTED (memris PR 7),
            # and only a key that keeps coming up is learned (learning.yaml).
            key = canonical_key(raw_key)
            unknown = key is None
            if key is None:
                key = normalise_fact_key(raw_key)
                if not key:
                    continue
            # Validation gate: a durable identity fact can never be a sentence
            # fragment, regardless of which extractor produced it. Rejections
            # are logged so the decision stays auditable (logs are the record).
            ok, reason = is_plausible_fact(key, value)
            if not ok:
                logger.warning(
                    "fact write REJECTED by validation gate: key=%s value=%r source=%s reason=%s",
                    key,
                    value,
                    source,
                    reason,
                )
                continue
            # Durability gate: reject ephemeral/conversational tokens (greeting=Hello,
            # day=tomorrow, task=plan, reminder_time="6 pm tomorrow") that aren't durable
            # user attributes — a floor under the extractor's precision.
            durable, durable_reason = is_durable_fact(key, value)
            if not durable:
                logger.warning(
                    "fact write REJECTED — not durable: key=%s value=%r source=%s reason=%s",
                    key,
                    value,
                    source,
                    durable_reason,
                )
                continue
            # Grounding gate: the value must be supported by THIS message, so the
            # LLM extractor can't bleed its own few-shot example ("teacher")
            # or otherwise invent a fact the user never stated.
            if not is_fact_grounded(value, user_msg):
                logger.warning(
                    "fact write REJECTED — ungrounded in message: key=%s value=%r source=%s",
                    key,
                    value,
                    source,
                )
                continue
            subject_class = None
            if about is not None:
                # One hop (ADR-0115 decision 6): the person must be named in this message
                # and linked to the owner — by a relation stated here or one confirmed.
                if not is_fact_grounded(about, user_msg):
                    logger.warning(
                        "fact write REJECTED — subject ungrounded in message: key=%s about=%r",
                        key,
                        about,
                    )
                    continue
                subject_class = self._one_hop_class(about, links)
                if subject_class is None:
                    logger.info(
                        "fact write REJECTED — subject not one hop from the owner: key=%s about=%r",
                        key,
                        about,
                    )
                    continue
            triage = triage_memory_item(f"{key}: {value}", signal_type="extracted_fact")
            if triage.destination != MemoryDestination.USER_MD:
                logger.info(
                    "fact write skipped by memory triage: key=%s destination=%s reason=%s",
                    key,
                    triage.destination,
                    triage.reason,
                )
                continue
            if unknown:
                learned = self._learn_key(key, value, subject_class, session_id)
                if learned is None:
                    continue
                key = learned
            self._persist_fact(
                key=key,
                value=value,
                confidence=confidence,
                source=source,
                evidence=user_msg.strip(),
                message=user_msg,
                session_id=session_id,
                about=about,
                subject_class=subject_class,
                known=True,
            )
            links.append(Link(about, key, value))
            persisted = True
        if persisted:
            self._consider_lookalike(session_id)

    def _learn_key(
        self, key: str, value: str, subject_class: str | None, session_id: str
    ) -> str | None:
        """A key the ontology lacks: count it; return the key to store the fact with now
        (an alias's existing key, or an active learned one), or None — only counted."""
        from iris_harness.memory.resolution import current_episode
        from memris.ontology.compiler import XSD

        store = self._host.memory_store
        try:
            vocabulary = store.vocabulary()
            domain = subject_class or vocabulary.graph.ontology.owner_class or ""
            learned = vocabulary.observe(
                key,
                domain=domain,
                datatype=f"{XSD}string",
                example=value,
                episode=session_id or current_episode(),
            )
        except Exception:
            logger.exception("failed to count the unknown fact key %r", key)
            return None
        seen = learned.observation
        if learned.fact_key is None:
            logger.info(
                "fact key not known yet — counted, not stored: key=%s outcome=%s seen=%s",
                key,
                seen.outcome if seen else "ignored",
                seen.term.observations if seen else 0,
            )
            return None
        logger.info(
            "fact key learned: %s → %s (%s)", key, learned.fact_key, seen.outcome if seen else ""
        )
        return learned.fact_key

    def _one_hop_class(self, name: str, links: list[Link]) -> str | None:
        """The class of ``name`` when it is within the configured hops of the owner."""
        try:
            return self._host.memory_store.subject_in_scope(
                name, links, max_hops=subject_max_hops()
            )
        except Exception:
            logger.exception("failed to check the subject scope of %r", name)
            return None

    def _learned_keys(self) -> tuple[str, ...]:
        """Keys memory learned (memris PR 7), offered to the extractor with the declared ones."""
        try:
            return tuple(self._host.memory_store.learned_fact_keys())
        except Exception:
            logger.exception("failed to read the learned fact keys")
            return ()

    def _extract_facts_declarative(self, user_msg: str) -> list[tuple[str, str, float, str]]:
        """Deterministic high-signal declaratives (blog/site URLs, "my X is Y"),
        run for every turn regardless of extraction mode (issue 0020)."""
        from iris_harness.memory.fact_extractor import extract_declarative_facts

        return [
            (f.key, f.value, f.confidence, "conversation:declarative")
            for f in extract_declarative_facts(user_msg)
        ]

    def _extract_facts_via_regex(self, user_msg: str) -> list[tuple[str, str, float, str]]:
        import re

        # The LLM extractor is the default; this regex path is an explicit
        # fallback (IRIS_FACT_EXTRACTION_MODE=regex). Captures are intentionally
        # strict — Capitalized tokens only, bounded to a few words, stopping at
        # the first conjunction — so a lowercase predicate ("working on a
        # project called Aur") can never be captured as a name/location. The
        # validation gate is the final backstop for whatever slips through.
        tok = r"[A-Z][a-zA-Z'\-]*"  # one Capitalized name/place token
        patterns = [
            (rf"(?i:my name is)\s+({tok}(?:[ '\-]{tok}){{0,2}})", "name"),
            (rf"(?i:i am|i'm|call me)\s+({tok}(?:[ '\-]{tok}){{0,2}})", "name"),
            (
                rf"(?i:i (?:work|am working) (?:at|for))\s+({tok}(?:[ ,&.]+{tok}){{0,3}})",
                "employer",
            ),
            (rf"(?i:i live in)\s+({tok}(?:[ ,]+{tok}){{0,3}})", "location"),
            (r"(?i:my (?:email|email address) is)\s+([\w.@+\-]+)", "email"),
        ]
        out: list[tuple[str, str, float, str]] = []
        for pattern, key in patterns:
            m = re.search(pattern, user_msg)
            if not m:
                continue
            value = m.group(1).strip().rstrip(".,!?")
            if not value:
                continue
            # Graduated confidence: the regex fallback is less trustworthy than
            # the LLM extractor, so it never claims more than 0.8.
            out.append((key, value, 0.8, "conversation:regex"))
        return out

    def _extract_facts_via_llm(self, user_msg: str) -> list[tuple[Any, ...]]:
        try:
            from iris_harness.llm.client import CodingLLMClient
            from iris_harness.memory.fact_extractor import extract_facts_with_llm

            cfg = self._host.tier_router.get_llm_config("intent_classification")
            client = CodingLLMClient(cfg)  # type: ignore[arg-type]
            facts = extract_facts_with_llm(user_msg, client=client, extra_keys=self._learned_keys())
        except Exception:
            logger.exception("LLM fact extraction setup failed — skipping turn")
            return []
        return [(f.key, f.value, f.confidence, "conversation:llm", f.about) for f in facts]

    def _persist_fact(
        self,
        *,
        key: str,
        value: str,
        confidence: float,
        source: str,
        evidence: str = "",
        message: str = "",
        session_id: str = "",
        about: str | None = None,
        subject_class: str | None = None,
        known: bool | None = None,
    ) -> None:
        """Queue the fact for review — extraction no longer writes what IRIS believes.

        Confidence does not confirm anything: this store held `name=ollama` at 1.0 and
        `employer=Department of Justice` at 0.9. A proposal reaches the prompt only
        after the owner approves it (`iris facts approve`, the review queue, or an
        explicit "remember that ..." which is captured as confirmed elsewhere).
        """
        tier = classify_capture(message or evidence, key, value, confidence, known=known)
        if about is not None and tier == AUTO:
            # The plain-statement shapes are the owner speaking about themselves; a fact
            # about someone else is asked about instead (or queued, below the bar).
            tier = ASK
        if tier == AUTO:
            # Plainly stated about themselves: remember it now and say so. A queue the
            # owner has to work is not learning — that was the first cut's mistake.
            try:
                self._coordinator().record(key, value, confidence, source, confirmed=True)
            except Exception:
                logger.exception("failed to store confirmed fact key=%s", key)
                return
            self._notices.append(f"{key.replace('_', ' ')}: {value}")
            logger.info("fact confirmed from a plain self-statement: %s=%r", key, value)
            return

        try:
            proposal_id = self._host.memory_store.add_fact_proposal(
                key=key,
                value=value,
                confidence=confidence,
                source=source,
                evidence=evidence,
                subject=about,
                subject_class=subject_class,
            )
        except SubjectError as exc:
            logger.info("fact write REJECTED — %s (about=%r)", exc, about)
            return
        except Exception:
            logger.exception("failed to queue fact proposal key=%s", key)
            return
        if proposal_id is None:
            logger.info("fact already confirmed; re-confirmed key=%s", key)
            return

        proposal = self._host.memory_store.fetch_fact_proposal(proposal_id)
        # Repetition confirms: people repeat what is true about them.
        if proposal is not None and proposal.seen_count >= repeat_confirms_after():
            try:
                self._coordinator().approve_proposal(proposal_id)
                self._notices.append(f"{key.replace('_', ' ')}: {value}")
                logger.info("fact confirmed by repetition: %s=%r", key, value)
                return
            except Exception:
                logger.exception("failed to confirm repeated fact key=%s", key)

        if (
            tier == ASK
            and session_id
            and session_id not in self._asked_this_session
            and session_id not in self._pending_question
            and len(self._asked_this_session) < ask_max_per_session() * 1_000
        ):
            self._pending_question[session_id] = (proposal_id, key, value)
            if about is not None:
                self._question_subject[session_id] = about
            else:
                self._question_subject.pop(session_id, None)
            self._asked_this_session.add(session_id)
            logger.info("will ask about fact: %s=%r (proposal %s)", key, value, proposal_id)
            return
        logger.info(
            "fact proposed for review: id=%s key=%s value=%r source=%s",
            proposal_id,
            key,
            value,
            source,
        )

    def _consider_lookalike(self, session_id: str) -> None:
        """If this conversation's budget is unspent, ask about a look-alike it turned up.

        Resolution records a pair of similar names ("Fabrikam Meridian" / "Fabrikam Meridien")
        as a candidate with the conversation as evidence. Only a pair seen in THIS
        conversation is asked about — the owner can answer while the name is fresh — and
        only a pair never asked before; unanswered, evidence keeps gathering toward the
        automatic merge.
        """
        # Asking either kind of question spends the session: one a conversation.
        if not session_id or session_id in self._asked_this_session:
            return
        try:
            graph = self._host.memory_store.memory_graph()
            candidate = next(
                (
                    d
                    for d in graph.open_candidates()
                    if d.asked_at is None and session_id in d.evidence
                ),
                None,
            )
            if candidate is None:
                return
            graph.mark_asked(candidate.id)
        except Exception:
            logger.exception("failed to pick a look-alike to ask about")
            return
        self._pending_lookalike[session_id] = candidate.id
        self._asked_this_session.add(session_id)
        logger.info("will ask about look-alike entities (decision %s)", candidate.id)

    def _lookalike_names(self, decision_id: str) -> tuple[Any, str, str] | None:
        graph = self._host.memory_store.memory_graph()
        decision = graph.get_decision(decision_id)
        if decision is None or decision.decision != "candidate":
            return None
        pair = [e for i in (decision.a, decision.b) if (e := graph.get_entity(i)) is not None]
        if len(pair) != 2:
            return None
        older, newer = sorted(pair, key=lambda e: (e.created_at, e.id))
        return graph, newer.label, older.label

    def _coordinator(self) -> Any:
        from iris_harness.memory.coordinator import FactCoordinator

        return FactCoordinator(self._host.memory_store, self._host.semantic_index)

    # ------------------------------------------------------------------
    # What the reply should say about what was learned
    # ------------------------------------------------------------------

    def take_notices(self) -> list[str]:
        """Facts confirmed during this turn, for the reply to mention once."""
        notices, self._notices = self._notices, []
        return notices

    def pending_question(self, session_id: str) -> tuple[str, str, str] | None:
        return self._pending_question.get(session_id)

    def take_question(self, session_id: str) -> str | None:
        """The one "should I remember this?" line for this session, if any."""
        pending = self._pending_question.get(session_id)
        if pending is None:
            return self._lookalike_question(session_id)
        _pid, key, value = pending
        return f"Should I remember that {self._whose(session_id, key)} is {value}?"

    def _whose(self, session_id: str, key: str) -> str:
        """ "your employer" — or "Petra's employer" for a fact one hop away."""
        subject = self._question_subject.get(session_id)
        whose = f"{subject}'s" if subject else "your"
        return f"{whose} {key.replace('_', ' ')}"

    def _lookalike_question(self, session_id: str) -> str | None:
        decision_id = self._pending_lookalike.get(session_id)
        if decision_id is None:
            return None
        try:
            named = self._lookalike_names(decision_id)
        except Exception:
            logger.exception("failed to read look-alike %s", decision_id)
            named = None
        if named is None:  # decided elsewhere meanwhile (CLI, Memory page, evidence)
            self._pending_lookalike.pop(session_id, None)
            return None
        _graph, new_name, old_name = named
        return f"Is {new_name} the same as {old_name}?"

    def resolve_question(self, session_id: str, message: str) -> str | None:
        """Consume a yes/no answer to the open question. Returns what to reply, or None."""
        pending = self._pending_question.get(session_id)
        if pending is None:
            return self._resolve_lookalike(session_id, message)
        decision = _parse_yes_no(message)
        if decision is None:
            return None
        proposal_id, key, value = pending
        whose = self._whose(session_id, key)
        self._pending_question.pop(session_id, None)
        subject = self._question_subject.pop(session_id, None)
        coordinator = self._coordinator()
        if decision is False:
            coordinator.reject_proposal(proposal_id)
            if subject is not None:
                return f"Okay — I won't remember {whose}."
            return f"Okay — I won't remember that {key.replace('_', ' ')}."
        coordinator.approve_proposal(proposal_id)
        return f"Got it — remembered that {whose} is {value}."

    def _resolve_lookalike(self, session_id: str, message: str) -> str | None:
        """A yes merges the pair (undo: `iris memory unmerge`); a no keeps it apart for good."""
        decision_id = self._pending_lookalike.get(session_id)
        if decision_id is None:
            return None
        answer = _parse_yes_no(message)
        if answer is None:
            return None
        self._pending_lookalike.pop(session_id, None)
        try:
            named = self._lookalike_names(decision_id)
            if named is None:
                return "That one was already settled — nothing to change."
            graph, new_name, old_name = named
            if answer:
                graph.accept_candidate(decision_id, decided_by="owner")
                return f"Got it — {new_name} and {old_name} are the same; I'll treat them as one."
            graph.reject_candidate(decision_id, decided_by="owner")
        except Exception:
            logger.exception("failed to record the answer about look-alike %s", decision_id)
            return "Sorry — I couldn't record that. It's still on the Memory page to decide."
        return f"Okay — I'll keep {new_name} and {old_name} apart."

    def record_signal(
        self,
        intent: IntentResult,
        curated: CuratedResponse,
        *,
        latency_ms: float = 0.0,
        model: str = "",
        provider: str = "",
        query: str = "",
    ) -> None:
        # Stamp the correlation chain so every learning signal is auditable back
        # to its turn + span (learning-observability.md §4.1). trace/span ids are
        # populated only when tracing is live (an OTLP endpoint set); learning never
        # depends on them being present (D1). resolved_tier is the tier of the
        # model that actually answered — not the requested one (D4).
        trace_id, span_id = current_trace_ids()
        resolved_tier = self._host.tier_router.tier_name_for_model(model) if model else None
        if resolved_tier is None and model in {"deterministic", ""}:
            resolved_tier = "deterministic" if model == "deterministic" else None
        session_id = current_session_id()
        turn_id = current_turn_id()
        self._host.signal_collector.record_response(
            intent=intent.intent,
            agent_type=intent.agent_type,
            intent_confidence=intent.confidence,
            has_errors=curated.has_errors,
            latency_ms=latency_ms,
            metadata={"sources": list(curated.sources), "model": model, "provider": provider},
            session_id=session_id,
            turn_id=turn_id,
            trace_id=trace_id,
            span_id=span_id,
            resolved_tier=resolved_tier,
            resolved_agent=intent.agent_type,
        )
        # Per-turn "answered it cleanly" signals are gone: they were written on every
        # successful turn, carried no information, and were never read into a prompt.
        # A lesson needs evidence now — see memory/lessons.py.
        # Shadow escalation judge (ADR-0068 L2): if the curator emitted a
        # would-be decision, record it as its own correlated learning signal —
        # the flagship consumer of the L1 outcome signals. Acting is L3.
        verdict = _extract_escalation_verdict(curated)
        if verdict is not None:
            self._host.signal_collector.record_escalation_shadow(
                verdict=verdict,
                has_errors=curated.has_errors,
                intent=intent.intent,
                latency_ms=latency_ms,
                session_id=session_id,
                turn_id=turn_id,
                trace_id=trace_id,
                span_id=span_id,
                resolved_tier=resolved_tier,
                resolved_agent=intent.agent_type,
            )

        # §4.2 deterministic measured outcomes (always on, pure observables):
        #   task_completed — the turn ran cleanly; turn_tokens — per-tier cost.
        correlation = {
            "session_id": session_id,
            "turn_id": turn_id,
            "trace_id": trace_id,
            "span_id": span_id,
            "resolved_tier": resolved_tier,
            "resolved_agent": intent.agent_type,
        }
        self._host.signal_collector.record_metric(
            metric_name="task_completed",
            value=0.0 if curated.has_errors else 1.0,
            success=not curated.has_errors,
            latency_ms=latency_ms,
            metadata={"intent": intent.intent, "model": model},
            **correlation,
        )
        total_tokens = _coerce_total_tokens(curated.metadata.get("total_tokens"))
        if total_tokens is not None:
            self._host.signal_collector.record_metric(
                metric_name="turn_tokens",
                value=float(total_tokens),
                success=not curated.has_errors,
                latency_ms=latency_ms,
                metadata={"intent": intent.intent, "model": model, "provider": provider},
                **correlation,
            )

        # Stash this turn's descriptor so the NEXT turn can attribute a measured
        # user_correction back to it (cross-turn, §4.2).
        if session_id:
            self._last_turn_outcome[session_id] = {
                "turn_id": turn_id,
                "trace_id": trace_id,
                "span_id": span_id,
                "resolved_tier": resolved_tier,
                "resolved_agent": intent.agent_type,
                "intent": intent.intent,
                "query": query,
                "response": curated.text,
                "has_errors": curated.has_errors,
            }

    def evaluate_prior_turn_outcome(self, session_id: str, new_message: str) -> None:
        """At a turn's start, judge whether it corrects the prior answer (§4.2).

        Emits ``user_correction`` correlated to the PRIOR turn — the turn that
        produced the outcome — when the model-driven judge (gated by a cheap
        pre-filter) confirms a correction above the confidence floor. Best-effort:
        learning telemetry must never break the chat path.
        """
        try:
            desc = self._last_turn_outcome.pop(session_id, None)
            if not desc:
                return
            detector = self._correction_detector()
            if detector is None:
                return
            if not looks_like_correction(new_message):
                return
            verdict = detector(
                prior_query=str(desc.get("query") or ""),
                prior_response=str(desc.get("response") or ""),
                new_message=new_message,
            )
            if verdict is None:
                return
            cfg = self._outcome_config()
            is_correction = bool(
                verdict.is_correction and verdict.confidence >= cfg.correction_confidence_floor
            )
            if is_correction:
                # Evidence, not telemetry: the user just told IRIS the answer was
                # wrong. That is one of the three things worth learning from, so it
                # goes to the review queue as a proposed lesson (the others are a tool
                # failure the run recovered from, and an evaluator halt).
                self._propose_correction_lesson(desc, new_message, verdict)
            self._host.signal_collector.record_metric(
                metric_name="user_correction",
                value=1.0 if is_correction else 0.0,
                success=not is_correction,
                metadata={
                    "confidence": verdict.confidence,
                    "reason": verdict.reason,
                    "intent": desc.get("intent"),
                },
                session_id=session_id,
                turn_id=desc.get("turn_id"),
                trace_id=desc.get("trace_id"),
                span_id=desc.get("span_id"),
                resolved_tier=desc.get("resolved_tier"),
                resolved_agent=desc.get("resolved_agent"),
            )
        except Exception:  # outcome telemetry never breaks the turn
            logger.debug("prior-turn outcome evaluation failed", exc_info=True)

    def _propose_correction_lesson(
        self, desc: dict[str, Any], new_message: str, verdict: Any
    ) -> None:
        """Queue what the correction implies, for the owner to approve or reject."""
        try:
            from iris_harness.memory.lessons import (
                SOURCE_CORRECTION,
                LessonCurator,
            )

            prior_query = str(desc.get("query") or "").strip()
            if not prior_query:
                return
            correction = " ".join(new_message.split())
            LessonCurator(self._host.memory_store).propose(
                trigger=prior_query[:200],
                lesson=f"the user corrected this with: {correction[:240]}",
                source=SOURCE_CORRECTION,
                evidence=(
                    f"asked: {prior_query[:300]}\n"
                    f"answered: {str(desc.get('response') or '')[:300]}\n"
                    f"corrected: {correction[:300]}\n"
                    f"judge: {getattr(verdict, 'reason', '')}"
                ),
            )
        except Exception:  # learning capture never breaks a turn
            logger.debug("correction lesson proposal failed", exc_info=True)

    def _correction_detector(self) -> Callable[..., Any] | None:
        """Lazily build the model-driven correction detector (§4.2); cache result."""
        if self._correction_detector_cache is None:
            self._correction_detector_cache = (
                _build_correction_detector(
                    tier_router=self._host.tier_router, config_dir=self._host.config_dir
                )
                or _OUTCOME_DISABLED
            )
        if self._correction_detector_cache is _OUTCOME_DISABLED:
            return None
        return cast("Callable[..., Any]", self._correction_detector_cache)

    def _outcome_config(self) -> OutcomeConfig:
        if self._outcome_config_cache is None:
            self._outcome_config_cache = load_outcome_config(self._host.config_dir)
        return self._outcome_config_cache
