"""Behavior-pattern mining — surface the user's recurring habits from their history.

A pure, model-driven analyzer: it reads a transcript of recent turns and asks a local
LLM to name RECURRING behavior patterns (habits, workflow preferences) — distinct from
durable identity facts, which the memory subsystem handles. It only PROPOSES; the runtime
records proposals for human review (HITL), and an approved pattern becomes a durable
episodic pattern. Off by default (``IRIS_BEHAVIOR_MINER``).

Mirrors ``learning/analyst.py``: ``invoke(system, user) -> str`` is the governed LLM call
supplied by the runtime; parsing is defensive and never raises into the caller.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from iris_harness.services.learning.dedup import cosine, dedupe_by_text

# Cosine ≥ this means two pattern texts say the same thing — calibrated on real mined
# pairs (paraphrased duplicates ~0.74; genuinely distinct habits ≤ ~0.41 with MiniLM).
_SEMANTIC_DUP_THRESHOLD = 0.70

logger = logging.getLogger(__name__)

BEHAVIOR_MINING_SYSTEM_PROMPT = """You identify the user's recurring BEHAVIOR PATTERNS \
from a transcript of their recent conversations with an assistant.

A behavior pattern is a HABIT or recurring way the user works — something that shows up
across MULTIPLE turns or days and would help the assistant anticipate them. Examples:
- "Asks for a summary of their inbox most mornings"
- "Reviews the calendar before scheduling anything"
- "Prefers short, direct answers without preamble"
- "Works on the project-iris codebase on weekday evenings"

Do NOT output:
- One-off requests or anything that happened only once.
- Durable identity facts (name, location, employer, email) — those are handled elsewhere.
- A restatement of a single message; a pattern must RECUR.

Return ONLY a JSON array, no prose or code fences. Each element is an object:
{"pattern": "<one concise sentence, present tense>",
 "confidence": "low" | "medium" | "high",
 "evidence": ["<short quote or paraphrase>", ...]}
confidence = how strongly the transcript supports this recurring. If nothing genuinely
recurs, return [].
"""

USER_PROMPT_TEMPLATE = (
    "Recent activity:\n{activity}\n\nReturn the JSON array of recurring behavior patterns now."
)

_CONFIDENCE = {"low", "medium", "high"}
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass(frozen=True)
class BehaviorPattern:
    """A proposed recurring behavior pattern, pending human review."""

    pattern_id: str  # stable 16-char sha256 of the normalized text
    text: str
    confidence: str  # low | medium | high
    evidence: tuple[str, ...]


def pattern_id_for(text: str) -> str:
    """Stable id from the normalized pattern text (so the same habit dedups)."""
    norm = " ".join(text.strip().lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def render_activity(turns: Sequence[tuple[str, str]], *, max_chars: int = 6000) -> str:
    """Render (role, content) turns into a compact transcript for the miner."""
    lines = [f"{role}: {content}".strip() for role, content in turns if content.strip()]
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text


def mine_behavior_patterns(
    turns: Sequence[tuple[str, str]],
    *,
    invoke: Callable[[str, str], str],
    min_turns: int = 6,
) -> list[BehaviorPattern]:
    """Propose recurring behavior patterns from recent turns. Never raises.

    ``turns`` are ``(role, content)`` pairs (oldest first). Returns ``[]`` when there's
    too little history to mine, the model returns nothing parseable, or anything fails.
    """
    if len(turns) < min_turns:
        return []
    activity = render_activity(turns)
    if not activity:
        return []
    try:
        raw = invoke(BEHAVIOR_MINING_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE.format(activity=activity))
    except Exception:  # mining is best-effort, never breaks the heartbeat
        logger.debug("behavior miner: LLM invocation failed", exc_info=True)
        return []
    return parse_behavior_patterns(raw)


def parse_behavior_patterns(raw: str) -> list[BehaviorPattern]:
    """Defensively parse the model's raw output into BehaviorPattern objects."""
    text = (raw or "").strip()
    if not text:
        return []
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []

    out: list[BehaviorPattern] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        pattern = str(item.get("pattern") or item.get("text") or "").strip()
        if not pattern:
            continue
        confidence = str(item.get("confidence", "low")).strip().lower()
        if confidence not in _CONFIDENCE:
            confidence = "low"
        raw_evidence = item.get("evidence")
        evidence = (
            tuple(str(e).strip() for e in raw_evidence if str(e).strip())
            if isinstance(raw_evidence, list)
            else ()
        )
        pid = pattern_id_for(pattern)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(
            BehaviorPattern(pattern_id=pid, text=pattern, confidence=confidence, evidence=evidence)
        )
    return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return cosine(a, b)


def dedupe_semantically(
    patterns: Sequence[BehaviorPattern],
    existing_texts: Sequence[str] = (),
    *,
    embed: Callable[[list[str]], list[list[float]]],
    threshold: float = _SEMANTIC_DUP_THRESHOLD,
) -> list[BehaviorPattern]:
    """Drop candidates that are semantic near-duplicates of each other or of an existing
    pattern (cosine ≥ ``threshold``). Exact-id dedup catches re-phrasings of identical
    text; this catches paraphrases ("daily weather in London" vs "checks London weather").

    Thin wrapper over the shared :func:`iris_harness.services.learning.dedup.dedupe_by_text`, keyed on the
    pattern text; the intention rollup uses the same core keyed on the intention title.
    """
    return dedupe_by_text(
        patterns,
        key=lambda p: p.text,
        existing_texts=existing_texts,
        embed=embed,
        threshold=threshold,
    )


__all__ = [
    "BEHAVIOR_MINING_SYSTEM_PROMPT",
    "BehaviorPattern",
    "dedupe_semantically",
    "mine_behavior_patterns",
    "parse_behavior_patterns",
    "pattern_id_for",
    "render_activity",
]
