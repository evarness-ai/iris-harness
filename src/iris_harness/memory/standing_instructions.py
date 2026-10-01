"""Standing-instruction capture — teach a response-shape preference in chat.

When the user *teaches a rule* rather than asks a question — "when I ask how is
my day, show my top 10 emails and any dues, to-do's and calendar" — that is a
standing instruction: a trigger (the phrase the user will say again) bound to a
response shape (what IRIS should do when they say it). This module turns such an
utterance into a structured :class:`StandingInstruction` so the runtime can
persist it as a behavior recipe (``~/.iris/behaviors/*.md``) and honor it on the
*next* matching turn via the existing ``match_behavior`` injection path — with no
nightly batch mine in between.

Detection is deterministic (a directive-shape pre-filter), extraction is
model-driven (an :data:`LLMCaller` returns the trigger keywords + the instruction
body) with a regex fallback so a downed LLM still captures something usable. The
caller owns persistence and the in-chat confirmation; this module is pure.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

LLMCaller = Callable[[str], str]

# A standing instruction *teaches a rule keyed on a future utterance*: "when/whenever/
# every time/from now on (when) I ask/say X, do Y". This is deliberately narrower than
# triage's ``_WHEN_DO_RE`` — we only fire on a conditional bound to the user *asking* or
# *saying* something, which is what makes it matchable as a trigger next turn. A bare
# preference ("keep answers short") is a behavior for the miner, not a triggered rule.
_DIRECTIVE_RE = re.compile(
    r"\b(?:when|whenever|every\s+time|any\s+time|each\s+time)\b[^.?!]*?"
    r"\bI\s+(?:ask|say|said|request|mention|type|tell\s+you)\b",
    re.IGNORECASE | re.DOTALL,
)
# "from now on, ... my brief ..." style — a standing rule without an explicit "when I ask".
_FROM_NOW_ON_RE = re.compile(r"\bfrom\s+now\s+on\b", re.IGNORECASE)

# Pull the trigger phrase out of "when I ask <PHRASE>, ..." for the regex fallback.
_TRIGGER_CAPTURE_RE = re.compile(
    r"\bI\s+(?:ask|say|said|request|mention|type|tell\s+you)\b\s*(?:me\s+)?"
    r"(?:you\s+)?[\"'`]?(?P<trigger>[^,.;:?!\"'`]{3,80})",
    re.IGNORECASE,
)
# Trimmed off the EDGES of a captured trigger only. "my" is deliberately absent —
# it's usually part of the phrase the user repeats ("my status", "my day").
_STOPWORDS = frozenset(
    {"a", "an", "the", "me", "is", "are", "do", "you", "just", "please", "to", "for"}
)


@dataclass(frozen=True)
class StandingInstruction:
    """A user-taught rule: when the trigger is said, perform the instruction.

    ``trigger_keywords`` are lower-cased substrings matched against a future
    query (the same contract as ``Behavior.match_keywords``); ``instruction`` is
    the response shape rendered into the system prompt when one matches.
    """

    name: str
    trigger_keywords: tuple[str, ...]
    instruction: str

    def summary(self) -> str:
        """One-line gloss for the in-chat confirmation."""
        trigger = self.trigger_keywords[0] if self.trigger_keywords else self.name
        body = " ".join(self.instruction.split())
        if len(body) > 140:
            body = body[:137].rstrip() + "..."
        return f"when you ask “{trigger}” I'll {body}"


def looks_like_standing_instruction(text: str) -> bool:
    """Cheap pre-filter: does this utterance *teach a triggered rule*?

    Gates the (more expensive) LLM extraction so ordinary questions and
    statements never pay for it. A trailing "?" alone is fine — "when I ask how
    is my day ... can you show me X?" is still teaching a rule.
    """
    t = (text or "").strip()
    if len(t) < 12:
        return False
    return bool(_DIRECTIVE_RE.search(t) or _FROM_NOW_ON_RE.search(t))


_EXTRACTION_PROMPT = """\
The user is teaching you a STANDING INSTRUCTION: a rule that says when they ASK or \
SAY a certain thing in future, you should respond a certain way. Extract it.

Return ONLY a JSON object on a single line with these keys:
  "name": a short kebab-case slug naming the rule (e.g. "day-overview")
  "trigger_keywords": a list of 1-3 short lower-case phrases the user will say to \
trigger it (e.g. ["how is my day", "how is my day today"]). Use the user's own words; \
omit filler like "looks like".
  "instruction": one imperative sentence telling you what to do when triggered \
(e.g. "Show the top 10 emails from both accounts plus today's dues, to-dos, and \
calendar events or reminders for today and this week.")

If the message is NOT teaching such a rule, return exactly: {"name": null}

Message:
%(message)s
"""


def extract_standing_instruction(text: str, llm_caller: LLMCaller) -> StandingInstruction | None:
    """Extract a :class:`StandingInstruction`, model-driven with a regex fallback.

    Returns ``None`` when the message isn't actually teaching a triggered rule
    (pre-filter miss, or the LLM declines with ``{"name": null}`` and the regex
    fallback can't recover a trigger).
    """
    if not looks_like_standing_instruction(text):
        return None
    try:
        raw = llm_caller(_EXTRACTION_PROMPT % {"message": text.strip()})
        parsed = _parse_llm_json(raw)
        if parsed is not None:
            return parsed
    except Exception:  # never let capture break a turn
        logger.exception("standing-instruction LLM extraction failed; using regex fallback")
    return _extract_via_regex(text)


def _parse_llm_json(raw: str) -> StandingInstruction | None:
    """Parse the extractor's JSON; tolerate fences/prose around the object."""
    if not raw:
        return None
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("name"):
        return None
    keywords = _clean_keywords(data.get("trigger_keywords"))
    instruction = " ".join(str(data.get("instruction") or "").split()).strip()
    if not keywords or not instruction:
        return None
    return StandingInstruction(
        name=_slugify(str(data["name"])),
        trigger_keywords=keywords,
        instruction=instruction,
    )


def _extract_via_regex(text: str) -> StandingInstruction | None:
    """Deterministic floor: recover trigger + instruction without an LLM.

    Splits on the first clause boundary after the trigger so the trigger phrase
    and the "do Y" remainder land in the right fields. Lower precision than the
    LLM, but it still produces a matchable rule when the model is unavailable.
    """
    trigger_match = _TRIGGER_CAPTURE_RE.search(text)
    if trigger_match is None:
        return None
    trigger = _normalise_trigger(trigger_match.group("trigger"))
    if not trigger:
        return None
    # Instruction = everything after the trigger clause (the comma that ends "when I ask X").
    tail = text[trigger_match.end() :]
    instruction = re.sub(r"^[\s,.;:-]+", "", tail).strip()
    instruction = re.sub(
        r"^(?:just\s+|please\s+|can\s+you\s+|i\s+want\s+you\s+to\s+)",
        "",
        instruction,
        flags=re.IGNORECASE,
    )
    instruction = " ".join(instruction.split())
    if len(instruction) < 4:
        return None
    return StandingInstruction(
        name=_slugify(trigger),
        trigger_keywords=(trigger,),
        instruction=instruction[0].upper() + instruction[1:],
    )


def _normalise_trigger(phrase: str) -> str:
    """Trim filler ("looks like", trailing stopwords) off a captured trigger."""
    t = " ".join((phrase or "").lower().split())
    t = re.sub(r"\b(?:looks?\s+like|today)\b", "", t).strip()
    words = [w for w in t.split() if w]
    while words and words[-1] in _STOPWORDS:
        words.pop()
    while words and words[0] in _STOPWORDS:
        words.pop(0)
    return " ".join(words)


def _clean_keywords(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    out: list[str] = []
    for item in raw:
        kw = " ".join(str(item).lower().split()).strip()
        if kw and kw not in out:
            out.append(kw)
    return tuple(out[:3])


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug or "standing-instruction"
