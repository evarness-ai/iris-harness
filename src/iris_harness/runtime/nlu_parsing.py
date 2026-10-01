"""Deterministic natural-language parsing the harness's own mechanisms need.

Pure functions — no runtime state, no I/O — for the two things the core still
reads off a raw message: the in-chat approve/reject decision that resolves a
pending confirmation (a governance mechanism, ADR-0076) and the direct clock/date
reply the ``system`` reference plugin answers with. Extracted verbatim from
``bootstrap.py`` (Phase 2 decomposition).

The reminder and meeting parsers that used to live beside these belong to the
calendar plugin (``plugins_builtin/calendar/nlu.py``) since OSS plan M5.7
track A, and their vocabulary is that plugin's ``nlu.yaml`` — the core carries no
domain's trigger words.
"""

from __future__ import annotations

import re
from datetime import datetime

from iris_harness.foundation.clock import local_now

# Trailing temporal qualifiers ("today", "now", "right now", "currently") are
# allowed so "what time is it today?" / "what's the date right now?" still match
# and get a deterministic, correctly-grounded reply instead of falling through to
# the LLM (which has no reliable clock and will hallucinate the date/timezone).
_TEMPORAL_SUFFIX = r"(?:\s+(?:right\s+now|now|today|currently))?"
# Polite lead-ins ("could you tell me …", "please tell me …") carry no
# semantics; without them "could you tell me what time it is" fell through to
# the agent loop (~10 s + an unreliable clock) instead of the 110 ms intercept.
_POLITE_PREFIX = r"(?:(?:could|can|would|will)\s+you\s+)?(?:please\s+)?(?:tell\s+me\s+)?"
# The tail covers both word orders: "what time is it" / "…what time it is".
_QUERY_TAIL = r"(?:\s+(?:is\s+it|it\s+is|is))?"
_TIME_QUERY_RE = re.compile(
    r"^\s*"
    + _POLITE_PREFIX
    + r"(?:what(?:'s|s| is)?\s+)?(?:the\s+)?(?:current\s+)?time"
    + _QUERY_TAIL
    + _TEMPORAL_SUFFIX
    + r"\??\s*$",
    re.IGNORECASE,
)
_DATE_QUERY_RE = re.compile(
    r"^\s*"
    + _POLITE_PREFIX
    + r"(?:what(?:'s|s| is)?\s+)?(?:the\s+)?(?:current\s+|today'?s\s+)?date"
    + _QUERY_TAIL
    + _TEMPORAL_SUFFIX
    + r"\??\s*$",
    re.IGNORECASE,
)
_DAY_QUERY_RE = re.compile(
    r"^\s*"
    + _POLITE_PREFIX
    + r"(?:what(?:'s|s| is)?\s+)?day"
    + _QUERY_TAIL
    + _TEMPORAL_SUFFIX
    + r"\??\s*$",
    re.IGNORECASE,
)


def _current_local_datetime() -> datetime:
    return local_now().replace(tzinfo=None)


# In-chat approve/reject for a pending consequential action. Anchored at the start
# so it only resolves a deliberate decision, not prose that merely contains "no".
# Answer vocabulary for a pending question. Both consumers
# (`_resolve_pending_confirmation` and `reads_as_answer`) only reach here when
# something is already pending, so these words are read in the one context where
# they unambiguously mean "yes": a question is on the table.
#
# `proceed` / `continue` / `carry on` mirror the offer vocabulary the harness
# itself asks with (`_PROCEED_OFFER_RE` in runtime/continuations.py). IRIS asked
# "Would you like to proceed with this script?" and then could not read
# "proceed with this script" as the yes it plainly was, so the continuation
# stayed open and the turn routed on its keywords into code_exec with no script
# to run. Keep the two vocabularies in step: a phrase we offer with is a phrase
# users echo back.
_CONFIRM_APPROVE_RE = re.compile(
    r"^\s*(?:approve|approved|yes|yeah|yep|y|confirm|confirmed|ok|okay|"
    r"do it|go ahead|go for it|send it|sounds good|sure|"
    r"proceed|continue|carry on|please do|let'?s do it)\b",
    re.IGNORECASE,
)
_CONFIRM_REJECT_RE = re.compile(
    r"^\s*(?:reject|rejected|no|nope|n|cancel|cancelled|don'?t|stop|"
    r"never\s*mind|nevermind|discard)\b",
    re.IGNORECASE,
)


def _parse_confirmation_decision(message: str) -> str | None:
    """'approve' / 'reject' / None for a pending in-chat confirmation."""
    if _CONFIRM_APPROVE_RE.match(message):
        return "approve"
    if _CONFIRM_REJECT_RE.match(message):
        return "reject"
    return None


def _deterministic_time_date_reply(message: str) -> str | None:
    """Return a deterministic local time/date reply for direct clock/calendar prompts."""
    lowered = message.strip().lower()
    now = local_now()
    if _TIME_QUERY_RE.match(lowered):
        return f"Current local time: {now.strftime('%H:%M %Z')}."
    if _DATE_QUERY_RE.match(lowered):
        return f"Today's date: {now.strftime('%A, %B %d, %Y')}."
    if _DAY_QUERY_RE.match(lowered):
        return f"Today is {now.strftime('%A')}."
    return None
