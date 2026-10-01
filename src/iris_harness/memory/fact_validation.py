"""Plausibility gate for durable user facts.

Even with the LLM extractor as the default path (``IRIS_FACT_EXTRACTION_MODE``),
a deterministic validator runs at the single persistence choke point so a
sentence fragment can *never* be written as an identity fact again.

This is the backstop behind a real regression: an eval prompt
("I'm working on a project called Aur" / "I live in Berlin and I prefer very
concise answers") was mis-parsed by the greedy regex extractor and stored as
the user's ``name`` and ``location`` at 0.9 confidence, corrupting their real
profile.  The gate rejects such fragments regardless of which extractor (LLM or
regex fallback) produced them, and every rejection is logged so the decision is
auditable.

The heuristics are deliberately conservative: it is better to drop a borderline
fact than to corrupt the user's identity.  Genuine atomic facts pass
("Robin", "Springfield, Illinois", "teacher", "Spanish, French").
"""

from __future__ import annotations

import re

# A durable fact VALUE should never contain a first-person marker — that is the
# signature of a captured sentence ("Berlin and I prefer ...") rather than a fact.
_FIRST_PERSON = frozenset(
    {"i", "me", "my", "mine", "myself", "we", "our", "ours", "i'm", "im", "i've", "i'd", "i'll"}
)

# Words that, found inside a captured "name", prove the capture is really a
# sentence/predicate, not a personal name.
_NAME_DISQUALIFIERS = frozenset(
    {
        "working",
        "work",
        "works",
        "live",
        "living",
        "stay",
        "staying",
        "going",
        "trying",
        "doing",
        "based",
        "located",
        "project",
        "called",
        "prefer",
        "prefers",
        "like",
        "likes",
        "on",
        "in",
        "at",
        "the",
        "a",
        "an",
        "to",
        "of",
        "and",
        "or",
        "but",
        "with",
        "for",
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "from",
        "by",
        "as",
        "my",
        "name",
    }
)

# Conversational clause markers — the value is a fragment, not an atomic fact.
_CLAUSE_MARKERS = (
    " and i ",
    " but i ",
    " so i ",
    " because ",
    " i prefer",
    " i like",
    " and prefer",
    " i am ",
    " i'm ",
    " i work",
    " i live",
    " i stay",
)

# Keys whose values are short proper nouns — held to a tighter word budget.
_PLACE_LIKE_KEYS = frozenset(
    {"location", "employer", "city", "region", "country", "company", "hometown", "nationality"}
)

_MAX_VALUE_CHARS = 80
_MAX_VALUE_WORDS = 6
_MAX_NAME_WORDS = 3
_MAX_PLACE_WORDS = 5

# A single name token: starts with a letter, then letters / apostrophes / hyphens / dots.
_NAME_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z'\-.]*$")


def is_plausible_fact(key: str, value: str) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for a candidate durable fact.

    ``ok=False`` means the caller must NOT persist the fact.  ``reason`` is a
    short human-readable explanation suitable for an audit log line.
    """
    key = (key or "").strip().lower()
    v = (value or "").strip()

    if not v:
        return False, "empty value"
    if len(v) > _MAX_VALUE_CHARS:
        return False, f"value too long ({len(v)}>{_MAX_VALUE_CHARS} chars) — looks like a sentence"

    words = v.split()
    if len(words) > _MAX_VALUE_WORDS:
        return False, f"value has {len(words)} words (>{_MAX_VALUE_WORDS}) — looks like a sentence"

    padded = f" {v.lower()} "
    for marker in _CLAUSE_MARKERS:
        if marker in padded:
            return False, f"value contains a conversational clause ({marker.strip()!r})"

    tokens = {w.strip(".,!?;:()").lower() for w in words}
    if tokens & _FIRST_PERSON:
        return False, "value contains a first-person pronoun — likely a sentence, not a fact"

    if key == "name":
        name_words = [w.strip(",") for w in words if w.strip(",")]
        if len(name_words) > _MAX_NAME_WORDS:
            return (
                False,
                f"name has {len(name_words)} words (>{_MAX_NAME_WORDS}) — not a personal name",
            )
        disqualifiers = tokens & _NAME_DISQUALIFIERS
        if disqualifiers:
            return (
                False,
                f"name contains non-name word(s) {sorted(disqualifiers)} — likely a sentence",
            )
        if not all(_NAME_TOKEN_RE.match(w) for w in name_words):
            return False, "name has non-alphabetic tokens"

    if key in _PLACE_LIKE_KEYS and len(words) > _MAX_PLACE_WORDS:
        return False, f"{key} has {len(words)} words (>{_MAX_PLACE_WORDS}) — looks like a sentence"

    return True, ""


# --- Durability gate (extractor-precision floor) -----------------------------
# The LLM extractor sometimes mints ephemeral/conversational tokens as "facts"
# (greeting=Hello, day=tomorrow, task=plan, reminder_time="6 pm tomorrow"). These
# are not durable user attributes; this deterministic gate rejects them as a floor
# under the model, mirroring is_plausible_fact / is_fact_grounded.

# Keys that name a transient/conversational artifact, never a durable user attribute.
_EPHEMERAL_KEYS = frozenset(
    {
        "greeting",
        "day",
        "date",
        "today",
        "tomorrow",
        "time",
        "task",
        "current_task",
        "reminder",
        "reminder_time",
        "mood",
        "weather",
        "lunch",
        "now",
        "status",
        "request",
        # System / conversational metadata mis-captured as user facts (issue 0032):
        # "action: fetch market indexes" (a query echo), "model: gpt-5-mini",
        # "version: 5.4" — about the request or the assistant, never the user.
        "action",
        "command",
        "intent",
        "query",
        "model",
        "version",
    }
)

# Inherently ephemeral VALUES, regardless of the key they're filed under.
_GREETING_VALUES = frozenset(
    {"hello", "hi", "hey", "yo", "hiya", "thanks", "thank you", "ok", "okay", "bye"}
)
_RELATIVE_TIME_VALUES = frozenset(
    {
        "today",
        "tomorrow",
        "yesterday",
        "tonight",
        "now",
        "later",
        "soon",
        "this morning",
        "this afternoon",
        "this evening",
        "this week",
        "next week",
        "last week",
        "next month",
        "last month",
    }
)
# A clock time like "6 pm", "6pm tomorrow", "11:30 am". Requires an am/pm marker so
# bare numbers (e.g. age=12) are NOT swallowed.
_CLOCK_TIME_RE = re.compile(
    r"^\d{1,2}(:\d{2})?\s*[ap]m"
    r"(\s+(today|tomorrow|tonight|yesterday|morning|afternoon|evening))?$",
    re.IGNORECASE,
)


def is_durable_fact(key: str, value: str) -> tuple[bool, str]:
    """Return ``(ok, reason)`` — False for ephemeral, non-durable 'facts'.

    Rejects greetings, relative dates/times, clock times, and the current task/request
    so they never become durable user facts. A deterministic floor under the LLM
    extractor; ``reason`` is suitable for an audit log line.
    """
    k = (key or "").strip().lower()
    v = (value or "").strip().lower().strip(".,!?;:")
    if k in _EPHEMERAL_KEYS:
        return False, f"ephemeral key {k!r} — not a durable user fact"
    if v in _GREETING_VALUES:
        return False, "value is a greeting, not a durable fact"
    if v in _RELATIVE_TIME_VALUES:
        return False, "value is a relative date/time, not a durable fact"
    if _CLOCK_TIME_RE.match(v):
        return False, "value is a clock time, not a durable fact"
    return True, ""


# Words too common to anchor a fact to a message — they appear in almost any
# turn, so they can't ground a value on their own.
_GROUNDING_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "my",
        "your",
        "is",
        "am",
        "are",
        "was",
        "in",
        "at",
        "on",
        "to",
        "for",
        "with",
        "as",
        "by",
        "from",
        "this",
        "that",
        "it",
        "me",
        "you",
        "i",
        "we",
        "our",
        "today",
        "now",
    }
)
_GROUNDING_TOKEN_RE = re.compile(r"[a-z0-9]+")


def is_fact_grounded(value: str, message: str) -> bool:
    """True when the fact *value* is supported by the user's *message*.

    Every significant token of the value (alphanumeric, length >= 3, not a
    stopword) must appear in the message. This rejects facts the extractor
    invented or echoed from its own few-shot example — e.g. an LLM emitting
    ``profession = "teacher"`` when the user only asked to see their daily
    brief. Deliberately conservative: a fact the model *normalised* away from the
    user's wording ("NYC" -> "New York City") may be rejected, and that is the
    right trade for profile integrity — the user can always restate it.
    """
    msg = (message or "").lower()
    if not msg:
        return False
    tokens = [
        t
        for t in _GROUNDING_TOKEN_RE.findall((value or "").lower())
        if len(t) >= 3 and t not in _GROUNDING_STOPWORDS
    ]
    if not tokens:
        # Value too short to tokenise meaningfully (e.g. a 2-letter name) —
        # require the trimmed value to appear verbatim.
        v = (value or "").strip().lower()
        return bool(v) and v in msg
    return all(t in msg for t in tokens)
