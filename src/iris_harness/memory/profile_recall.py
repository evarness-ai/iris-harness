"""Deterministic profile recall (issue 0023).

For a targeted self-question — "do you know my blog?", "what's my email?" — answer
straight from the stored user facts instead of relying on the small profile model
to notice the fact in its context (which it does inconsistently). Broad profile
asks ("what do you know about me?") are left to the LLM, which has the full USER.md.
"""

from __future__ import annotations

import re
from typing import Any

# A targeted recall: a "do you know / what's / tell me" trigger followed by
# "my <attribute>" at the end of the question.
_RECALL_RE = re.compile(
    r"(?:do\s+you\s+(?:know|have|remember)|what'?s|what\s+is|tell\s+me|recall)\b"
    r".*?\bmy\s+([a-z0-9][a-z0-9 /_-]*?)\s*[?.!]*\s*$",
    re.IGNORECASE,
)

# Generic nouns that don't themselves identify the attribute ("blog site" → blog).
_GENERIC = {
    "site",
    "address",
    "details",
    "detail",
    "info",
    "information",
    "id",
    "the",
    "a",
    "an",
    "is",
    "are",
    "please",
    "value",
    "name",
}
# NB: "name" is generic as a trailing noun ("what's my name" handles via the key
# token "name" matching directly — see below), but "my account name" shouldn't bind.


def build_profile_recall(query: str, facts: Any) -> str | None:
    """Deterministic answer for a targeted "my <attribute>" question, or None to
    defer to the LLM. ``facts`` is a sequence of objects with ``.key`` / ``.value``."""
    facts = list(facts or [])
    m = _RECALL_RE.search(query or "")
    if not m:
        return None
    asked_raw = re.findall(r"[a-z0-9]+", m.group(1).lower())
    if not asked_raw:
        return None
    # Keep meaningful tokens, but if the whole phrase is a single generic word that
    # IS also a fact key (e.g. "name"), keep it so "what's my name" still resolves.
    asked = {t for t in asked_raw if t not in _GENERIC} or set(asked_raw)

    for f in facts:
        key_tokens = set(str(f.key).lower().split("_"))
        if key_tokens & asked:
            label = str(f.key).replace("_", " ")
            return f"Yes — your {label} is {f.value}."

    # Recognised as a recall but nothing on file: be honest + invite the fact,
    # rather than letting the model guess (which web-searched a stranger's blog).
    asked_phrase = " ".join(asked_raw)
    return f"I don't have your {asked_phrase} on file yet. Tell me and I'll remember it."
