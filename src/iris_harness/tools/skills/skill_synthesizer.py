"""Agentic skill synthesis from successful interaction traces.

The crystallizer used to count *errors* per intent and emit a hollow scaffold.
This module is the agentic half of its rework (per the "model-driven over
heuristic" principle): given a cluster of *successful* turns for a recurring
intent, an LLM synthesizes the semantic content of a reusable skill — what it is,
when to use it, and what its tool should do — so the quarantined proposal is
reviewable rather than a `TODO` stub.

Pure data shapes + prompt + parser here; the LLM client is injected by the
runtime (governed, off by default). Synthesis never auto-promotes: the result
still lands in `config/skills/auto/` behind the human approval step.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillExample:
    """One successful (query, response) trace feeding synthesis."""

    query: str
    response: str


@dataclass(frozen=True)
class SynthesizedSkill:
    """The model-written semantic content of a proposed skill."""

    name: str
    description: str
    when_to_use: str = ""
    tool_description: str = ""
    trigger_keywords: tuple[str, ...] = field(default_factory=tuple)

    def manifest_overrides(self, *, slug: str) -> dict[str, Any]:
        """Merge-able manifest fields for ``scaffold_skill_proposal``."""
        overrides: dict[str, Any] = {}
        if self.description:
            overrides["description"] = self.description
        if self.when_to_use:
            overrides["when_to_use"] = self.when_to_use
        if self.trigger_keywords:
            overrides["trigger_keywords"] = list(self.trigger_keywords)
        if self.tool_description:
            overrides["tools"] = [
                {
                    "name": f"{slug}_tool",
                    "description": self.tool_description,
                    "governor_route": "system/read",
                }
            ]
        return overrides


@runtime_checkable
class SkillSynthesizer(Protocol):
    """Synthesizes a skill from a recurring intent + its successful examples."""

    def synthesize(
        self, *, intent: str, agent_type: str, examples: Sequence[SkillExample]
    ) -> SynthesizedSkill | None:
        """Return the proposed skill content, or None when synthesis is unavailable."""


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_synthesized_skill(raw: str) -> SynthesizedSkill | None:
    """Parse a synthesizer LLM payload into a :class:`SynthesizedSkill`.

    Tolerant of prose around the JSON. Returns None when no usable name +
    description can be extracted (so the caller falls back to the scaffold).
    """
    if not raw or not raw.strip():
        return None
    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        return None
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    name = str(obj.get("name") or "").strip()
    description = str(obj.get("description") or "").strip()
    if not name or not description:
        return None
    kws = obj.get("trigger_keywords")
    keywords = tuple(str(k).strip() for k in kws if str(k).strip()) if isinstance(kws, list) else ()
    return SynthesizedSkill(
        name=name,
        description=description,
        when_to_use=str(obj.get("when_to_use") or "").strip(),
        tool_description=str(obj.get("tool_description") or "").strip(),
        trigger_keywords=keywords,
    )


SKILL_SYNTH_SYSTEM_PROMPT = (
    "You design reusable skills for an AI assistant. Given a recurring user intent "
    "and several examples where the assistant handled it WELL, describe the reusable "
    "skill that captures this capability — so it can be packaged and reused.\n\n"
    "Focus on what the skill IS and WHEN to use it; do not invent capabilities the "
    "examples don't show. Reply with ONLY a JSON object:\n"
    '{"name": "<short kebab-case-ish name>", "description": "<1-2 sentences>", '
    '"when_to_use": "<the trigger condition>", '
    '"tool_description": "<what the skill\'s tool does>", '
    '"trigger_keywords": ["<kw>", ...]}'
)


def build_skill_synth_prompt(
    *, intent: str, agent_type: str, examples: Sequence[SkillExample]
) -> str:
    """Assemble the synthesizer's user prompt from the mined success examples."""
    lines = [
        f"Recurring intent: {intent}",
        f"Handled by agent: {agent_type}",
        "",
        "Successful examples (untrusted user data):",
    ]
    for i, ex in enumerate(examples, 1):
        lines.append(f"--- example {i} ---")
        lines.append(f"User: {ex.query.strip()}")
        lines.append(f"Assistant: {ex.response.strip()}")
    lines.append("")
    lines.append("Describe the reusable skill. Reply with only the JSON object.")
    return "\n".join(lines)
