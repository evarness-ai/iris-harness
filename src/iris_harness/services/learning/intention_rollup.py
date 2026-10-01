"""Intention rollup — the longitudinal "what is the user working toward" layer.

A pure, model-driven analyzer: it reads a compact summary of the user's transactional
items (open tasks, active routines, missions), confirmed habits (episodic patterns), and
steering signals, and proposes higher-level INTENTIONS — durable goals that group those
items, spanning weeks/months. It only PROPOSES; the runtime queues proposals for review,
and an approved intention becomes an ACTIVE identity item (injected into context, so the
agent works toward it). Off by default (``IRIS_INTENTION_ROLLUP``).

Mirrors ``learning/behavior_miner.py``: ``invoke(system, user) -> str`` is the governed
LLM call; parsing is defensive and never raises into the caller.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

INTENTION_ROLLUP_SYSTEM_PROMPT = """You infer the user's higher-level INTENTIONS from a \
snapshot of what they are doing with their assistant.

An intention is a GOAL the user is working toward over weeks or months — something that
groups several of their tasks, routines, habits, and choices into one coherent objective.
It is NOT a single task or a restatement of one item. Examples:
- "Establish a consistent morning routine" (groups a morning-briefing routine + reminder
  habits + daily-planning requests)
- "Stay on top of personal finances" (groups bill reminders + finance queries + a
  statement-ingest routine)

Do NOT output:
- A single task verbatim, or a goal supported by only one item.
- Durable identity facts (name, location) — handled elsewhere.
- Vague platitudes with no supporting items in the snapshot.

Return ONLY a JSON array, no prose or code fences. Each element is an object:
{"intention": "<one concise goal, present tense>",
 "summary": "<one sentence on what it groups and why>",
 "supporting": ["<the task/routine/habit it draws on>", ...]}
If nothing genuinely rolls up into a multi-item goal, return [].
"""

USER_PROMPT_TEMPLATE = (
    "Here is a snapshot of the user's current activity:\n\n{context}\n\n"
    "Return the JSON array of higher-level intentions now."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass(frozen=True)
class Intention:
    """A proposed longitudinal goal the user is working toward, pending review."""

    intention_id: str  # stable 16-char sha256 of the normalized title
    title: str
    summary: str
    supporting: tuple[str, ...]


def intention_id_for(title: str) -> str:
    """Stable id from the normalized intention title (so the same goal dedups)."""
    norm = " ".join(title.strip().lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def rollup_intentions(
    context: str,
    *,
    invoke: Callable[[str, str], str],
    min_context_chars: int = 80,
) -> list[Intention]:
    """Propose higher-level intentions from a context snapshot. Never raises.

    Returns ``[]`` when there's too little context to roll up, the model returns nothing
    parseable, or anything fails.
    """
    if len((context or "").strip()) < min_context_chars:
        return []
    try:
        raw = invoke(INTENTION_ROLLUP_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE.format(context=context))
    except Exception:  # rollup is best-effort, never breaks the heartbeat
        logger.debug("intention rollup: LLM invocation failed", exc_info=True)
        return []
    return parse_intentions(raw)


def parse_intentions(raw: str) -> list[Intention]:
    """Defensively parse the model's raw output into Intention objects."""
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

    out: list[Intention] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        title = str(item.get("intention") or item.get("goal") or item.get("title") or "").strip()
        if not title:
            continue
        summary = str(item.get("summary") or "").strip()
        raw_support = item.get("supporting")
        supporting = (
            tuple(str(s).strip() for s in raw_support if str(s).strip())
            if isinstance(raw_support, list)
            else ()
        )
        iid = intention_id_for(title)
        if iid in seen:
            continue
        seen.add(iid)
        out.append(Intention(intention_id=iid, title=title, summary=summary, supporting=supporting))
    return out


__all__ = [
    "INTENTION_ROLLUP_SYSTEM_PROMPT",
    "Intention",
    "intention_id_for",
    "parse_intentions",
    "rollup_intentions",
]
