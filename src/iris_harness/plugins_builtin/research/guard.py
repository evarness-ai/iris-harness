"""Egress guards on the ``research`` tool (OSS plan M4.7).

Two refusals, both lifted verbatim from ``bootstrap.py``:

* **ADR-0102** — a question about the user's OWN money can only be answered from
  local finance + inbox data. The web cannot know their accounts, so a model that
  searches anyway fabricates a provider (a real Telegram session web-searched the
  user's email and invented an insurance agency). Refuse and redirect.
* **issue 0031** — never hand a search provider the user's name or email. The
  identifiers are *derived* from the identity files rather than hardcoded, so the
  guard tracks whatever the profile records.

The dues vocabulary these read is core (``iris_harness.foundation.data.dues_vocabulary``): it
is shared with ``configure_brief``'s dues-section narrowing, which answers with
no plugin mounted at all.
"""

from __future__ import annotations

import re

# Possessive framing ("my", "I owe") + a money noun = a question only the local
# finance + inbox data can answer.
_PERSONAL_FRAME_RE = re.compile(r"\b(my|mine|our|i|i'm|i am|me|we)\b", re.IGNORECASE)
_FINANCE_NOUN_RE = re.compile(
    r"\b(insurance|premiums?|polic(?:y|ies)|emis?|loans?|mortgages?|credit\s*cards?|"
    r"banks?|statements?|balances?|net\s*worth|dues?|bills?)\b",
    re.IGNORECASE,
)

REDIRECT_TO_LOCAL_TOOLS = (
    "I won't web-search your personal finances — the web doesn't have your "
    "accounts. Answer from the user's OWN data: their local finance tools for "
    "what they owe / dues / balances, and search_inbox to find the bill, "
    "premium, or policy in their email. "
    'e.g. Action: search_inbox  Action Input: {"query": "insurance premium due"}'
)


def is_personal_finance_web_query(text: str) -> bool:
    """True when a web query is really about the user's OWN finances (ADR-0102).

    Matches a possessive frame ("my", "I owe") plus a money noun
    (insurance/premium/dues/...), or the existing dues vocabulary. Excludes
    informational asks ("how does term insurance work", "what is an EMI").
    """
    from iris_harness.sdk.identity import DUES_EXCLUDE_RE, is_dues_query

    low = " ".join(text.lower().split())
    if not low or DUES_EXCLUDE_RE.search(low):
        return False
    if is_dues_query(low) and _PERSONAL_FRAME_RE.search(low):
        return True
    return bool(_FINANCE_NOUN_RE.search(low) and _PERSONAL_FRAME_RE.search(low))


def owner_identity_identifiers() -> list[str]:
    """The user's personal identifiers (name(s) + email(s)) from the identity files.

    USER.md for names, USER.md + SOUL.md for emails. Returns ``[]`` when the
    profile records nothing.
    """
    from iris_harness.sdk.identity import load_soul, load_user_md

    from .privacy import extract_personal_identifiers

    return extract_personal_identifiers(load_user_md(), load_soul())
