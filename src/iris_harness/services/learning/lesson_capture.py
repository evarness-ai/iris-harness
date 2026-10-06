"""Capture, redact, persist, and retrieve reusable code_exec lessons.

A *lesson* is a structured note recorded after a successful sandbox-backed
``code_exec`` run. The planner LLM emits a fenced ```lesson JSON block at
the end of its final prose answer; this module parses it, scrubs PII /
local-path noise, and dual-writes to:

1. ``MemoryStore.append_learning_signal`` — queryable signals table.
2. ``WikiEngine.ingest`` — markdown wiki page for human browsing.

A lightweight Jaccard-keyword retriever (``find_similar``) surfaces past
lessons to the planner before each new run so it can reuse known
strategies. Embedding-backed RAG is intentionally deferred.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from iris_harness.kernel.governance.external_content import redact_text
from iris_harness.memory.knowledge.models import WikiIngestEvent
from iris_harness.memory.knowledge.wiki_engine import WikiEngine
from iris_harness.memory.store import LearningSignal, MemoryStore

logger = logging.getLogger(__name__)


# Fenced ```lesson { ... } ``` block (case-insensitive language tag).
_LESSON_BLOCK_RE = re.compile(
    r"```\s*lesson\s*\n(?P<body>.*?)\n```",
    re.IGNORECASE | re.DOTALL,
)

# Redaction patterns (applied in order on every text field).
_HOME_PATH_RE = re.compile(r"/(?:Users|home)/[^/\s]+")
_TILDE_PATH_RE = re.compile(r"~/[^\s]*")
_SANDBOX_HASH_RE = re.compile(r"sandbox/[0-9a-f]{8,}")
_ENV_ASSIGN_RE = re.compile(r"\b([A-Z][A-Z0-9_]{3,})=\S+")
_LONG_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_-]{32,}\b")  # API keys, JWTs, etc.

LESSON_SIGNAL_TYPE = "code_exec_lesson"


@dataclass(frozen=True)
class Lesson:
    """A parsed (un-persisted) lesson extracted from an LLM final answer."""

    category: str
    summary: str
    tools: tuple[str, ...] = field(default_factory=tuple)
    sources: tuple[str, ...] = field(default_factory=tuple)
    scripts: tuple[str, ...] = field(default_factory=tuple)


def _redact(text: str) -> str:
    """Strip user-identifying paths, env-var assignments, and long opaque tokens."""
    if not text:
        return text
    out = _HOME_PATH_RE.sub("/<home>", text)
    out = _TILDE_PATH_RE.sub("~/<…>", out)
    out = _SANDBOX_HASH_RE.sub("sandbox/<id>", out)
    out = _ENV_ASSIGN_RE.sub(r"\1=<redacted>", out)
    out = _LONG_TOKEN_RE.sub("<token>", out)
    return out


def _tripwire(text: str) -> str:
    """The external-content floor's tripwire (the kernel's ``redact_text``: honours the floor
    setting, one counts-only ledger row per match).

        A lesson is a note the planner wrote about a run whose output can be derived from
        third-party data, and it is stored and re-injected into later planner prompts, so
        instruction-like spans are redacted on the way in and on the way out (issue #140).
    """
    return redact_text(text, source="lesson", caller="core:lesson_capture")


def _redact_lesson(lesson: Lesson) -> Lesson:
    return Lesson(
        category=_tripwire(lesson.category),
        summary=_tripwire(_redact(lesson.summary)),
        tools=tuple(_tripwire(_redact(t)) for t in lesson.tools),
        sources=tuple(_tripwire(_redact(s)) for s in lesson.sources),
        scripts=tuple(_tripwire(_redact(s)) for s in lesson.scripts),
    )


def extract_from_answer(answer: str) -> tuple[Lesson | None, str]:
    """Parse a fenced ``lesson`` JSON block from ``answer``.

    Returns ``(lesson_or_none, cleaned_answer)``: ``cleaned_answer`` has the
    lesson block removed so the user never sees the raw JSON.
    """
    if not isinstance(answer, str) or "lesson" not in answer.lower():
        return None, answer

    match = _LESSON_BLOCK_RE.search(answer)
    if match is None:
        return None, answer

    body = match.group("body").strip()
    try:
        loaded = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        logger.debug("lesson block was not valid JSON: %r", body[:200])
        cleaned = (answer[: match.start()] + answer[match.end() :]).rstrip()
        return None, cleaned
    if not isinstance(loaded, dict):
        cleaned = (answer[: match.start()] + answer[match.end() :]).rstrip()
        return None, cleaned

    category = str(loaded.get("category") or "general").strip().lower() or "general"
    summary = str(loaded.get("summary") or "").strip()
    if not summary:
        cleaned = (answer[: match.start()] + answer[match.end() :]).rstrip()
        return None, cleaned

    def _as_tuple(key: str) -> tuple[str, ...]:
        raw = loaded.get(key)
        if isinstance(raw, list):
            return tuple(str(x).strip() for x in raw if str(x).strip())
        if isinstance(raw, str) and raw.strip():
            return (raw.strip(),)
        return ()

    lesson = Lesson(
        category=category,
        summary=summary,
        tools=_as_tuple("tools"),
        sources=_as_tuple("sources"),
        scripts=_as_tuple("scripts"),
    )
    cleaned = (answer[: match.start()] + answer[match.end() :]).rstrip()
    return lesson, cleaned


def _format_wiki_content(lesson: Lesson, query: str) -> str:
    parts = [
        f"# Lesson — {lesson.category}",
        "",
        f"**Original query:** {_redact(query)}",
        "",
        f"**Summary:** {lesson.summary}",
        "",
    ]
    if lesson.tools:
        parts.append("**Tools used:** " + ", ".join(lesson.tools))
    if lesson.sources:
        parts.append("**Sources:**")
        parts.extend(f"- {s}" for s in lesson.sources)
    if lesson.scripts:
        parts.append("**Scripts:**")
        parts.extend(f"- {s}" for s in lesson.scripts)
    return "\n".join(parts)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


@dataclass
class LessonCapture:
    """Application service for capturing and retrieving code_exec lessons."""

    memory_store: MemoryStore
    wiki: WikiEngine | None = None
    enabled: bool = True
    top_k: int = 3
    """Default number of prior lessons to inject. Override per-call via ``find_similar(k=...)``."""
    domain_allow: frozenset[str] = field(default_factory=frozenset)
    """If non-empty, only lessons whose ``domain`` (category) is in this set are returned."""
    domain_block: frozenset[str] = field(default_factory=frozenset)
    """Lessons whose ``domain`` is in this set are excluded. Applied after ``domain_allow``."""

    def handle(
        self,
        *,
        query: str,
        answer: str,
        artifacts: list[str] | tuple[str, ...] = (),
        session_id: str | None = None,
        iterations: int = 0,
        all_succeeded: bool = True,
    ) -> tuple[Lesson | None, str]:
        """Extract, redact, and persist a lesson if conditions are met.

        Always returns ``(lesson_or_none, cleaned_answer)`` so callers can
        substitute the cleaned answer (lesson JSON stripped) for display.
        """
        lesson, cleaned = extract_from_answer(answer)
        if lesson is None or not self.enabled:
            return lesson, cleaned

        # Only persist when the run produced something concrete.
        if iterations < 1 or not all_succeeded:
            return None, cleaned

        redacted = _redact_lesson(lesson)
        try:
            self._record(redacted, query=query, session_id=session_id, artifacts=tuple(artifacts))
        except Exception:
            logger.exception("lesson persistence failed; continuing")
        return redacted, cleaned

    def _record(
        self,
        lesson: Lesson,
        *,
        query: str,
        session_id: str | None,
        artifacts: tuple[str, ...],
    ) -> None:
        signal = LearningSignal(
            id=str(uuid.uuid4()),
            signal_type=LESSON_SIGNAL_TYPE,
            domain=lesson.category,
            agent_type="code_exec",
            query=_redact(query)[:1000],
            context=json.dumps(
                {
                    "tools": list(lesson.tools),
                    "scripts": list(lesson.scripts),
                    "artifacts": [_redact(a) for a in artifacts],
                    "session_id": session_id,
                },
                sort_keys=True,
            ),
            outcome=lesson.summary[:2000],
            improvement_hint=("\n".join(lesson.sources)[:1000] or None),
            timestamp=datetime.now(UTC),
        )
        self.memory_store.append_learning_signal(signal)

        if self.wiki is not None:
            content = _format_wiki_content(lesson, query)
            try:
                self.wiki.ingest(
                    WikiIngestEvent(
                        source_agent="code_exec",
                        source_id=signal.id,
                        content=content,
                        entities_hint=[],
                        metadata={
                            "category": lesson.category,
                            "tools": list(lesson.tools)[:5],
                        },
                    )
                )
            except Exception:
                logger.exception("wiki ingest failed for lesson %s", signal.id)

    def find_similar(self, query: str, k: int | None = None) -> list[Lesson]:
        """Return up to ``k`` past lessons whose query overlaps ``query``.

        Uses simple Jaccard similarity over word tokens — fast, no embeddings.
        ``k`` defaults to ``self.top_k``. Domain allow/block filters are applied
        before ranking so they cannot be bypassed by a high overlap score.
        """
        effective_k = self.top_k if k is None else k
        if not self.enabled or effective_k <= 0:
            return []
        try:
            signals = self.memory_store.fetch_learning_signals({"signal_type": LESSON_SIGNAL_TYPE})
        except Exception:
            logger.exception("fetch_learning_signals failed")
            return []
        if not signals:
            return []

        # Domain allow/block filtering (applied before ranking).
        if self.domain_allow:
            signals = [s for s in signals if s.domain in self.domain_allow]
        if self.domain_block:
            signals = [s for s in signals if s.domain not in self.domain_block]
        if not signals:
            return []

        q_tokens = _tokens(query)
        if not q_tokens:
            return []

        scored: list[tuple[float, LearningSignal]] = []
        for signal in signals:
            s_tokens = _tokens(signal.query + " " + signal.outcome)
            if not s_tokens:
                continue
            inter = len(q_tokens & s_tokens)
            if inter == 0:
                continue
            union = len(q_tokens | s_tokens)
            scored.append((inter / union, signal))
        scored.sort(key=lambda x: x[0], reverse=True)

        out: list[Lesson] = []
        for _score, signal in scored[:effective_k]:
            ctx: dict[str, Any] = {}
            try:
                ctx = json.loads(signal.context)
            except (json.JSONDecodeError, ValueError):
                ctx = {}
            tools = ctx.get("tools") if isinstance(ctx, dict) else None
            scripts = ctx.get("scripts") if isinstance(ctx, dict) else None
            sources = (signal.improvement_hint or "").splitlines()
            out.append(
                Lesson(
                    category=signal.domain,
                    summary=signal.outcome,
                    tools=tuple(tools) if isinstance(tools, list) else (),
                    sources=tuple(s for s in sources if s.strip()),
                    scripts=tuple(scripts) if isinstance(scripts, list) else (),
                )
            )
        return out

    def render_prior_lessons(self, lessons: list[Lesson]) -> str:
        """Format past lessons for inclusion in the planner prompt."""
        if not lessons:
            return ""
        lines = ["PRIOR LESSONS (from past similar tasks):"]
        for ls in lessons:
            tools = ", ".join(ls.tools) if ls.tools else "—"
            srcs = ", ".join(ls.sources[:2]) if ls.sources else "—"
            lines.append(
                f"- [{ls.category}] {ls.summary[:200]}" f"\n  tools: {tools} | sources: {srcs}"
            )
        # Scanned again on the way out: rows stored before the store-time tripwire existed
        # (or written by another path) must not re-enter a prompt raw.
        return _tripwire("\n".join(lines))
