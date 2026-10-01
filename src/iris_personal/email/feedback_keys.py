"""Email's "not useful" keys: which surface a verdict is about, and what it hides.

The owner answers "not useful" on three email surfaces -- a search result, a
reply-followup and a line of the digest's Focus section -- and the verdict lands in the
harness's surface-feedback store (``iris_harness.sdk.learning.SurfaceFeedbackStore``)
under ``(subsystem, surface, dims)``. The store is the harness's; the vocabulary is
email's, so it lives here, and email's readers and writers take it from here:

* search results are keyed on the sender's domain (a suppressed sender is downranked);
* a followup on ``account`` + sender domain (who it is from, not the subject);
* a Focus line on the sender's full address (a bank's marketing sender and its
  relationship manager share a domain; hiding one must not hide the other).

**Until PR 7** (core/SDK boundary plan, "email feedback surfaces") three of the paths
that *record* a verdict are still the core's -- the ``record_feedback`` ReAct tool,
``iris feedback`` and the digest's ``/not-useful`` route -- and they build the same
keys from ``iris_harness.services.learning.suppression``'s copy. A verdict recorded
there must hide what is read here, so the two are pinned equal by
``tests/unit/iris_personal/test_email/test_feedback_keys.py``; PR 7 moves those
recorders to email and deletes the core's copy.
"""

from __future__ import annotations

#: The subsystem every email surface records under.
EMAIL_SUBSYSTEM = "email"
EMAIL_SEARCH_SURFACE = "search_result"
EMAIL_FOLLOWUP_SURFACE = "followup"
#: A line of the morning digest's Focus section.
EMAIL_FOCUS_SURFACE = "focus"


def _bare(value: str) -> str:
    """The part inside ``<...>`` of a 'Name <addr>' pair, else the value, stripped."""
    s = value.strip()
    if "<" in s and ">" in s:
        s = s[s.rfind("<") + 1 : s.rfind(">")]
    return s.strip()


def _domain_of(value: str) -> str:
    """The domain in a bare domain, an address, or a 'Name <addr>' pair."""
    s = _bare(value)
    return (s.rsplit("@", 1)[-1] if "@" in s else s).strip().lower()


def email_search_dims(from_domain: str) -> dict[str, str]:
    """Sender-scoped suppression key for an email search result."""
    return {"from_domain": (from_domain or "").strip().lower()}


def email_focus_dims(sender: str) -> dict[str, str]:
    """Suppression key for a line in the digest's Focus section (loop-proof D17): the
    sender's full address, lowercased. Accepts a bare address or 'Name <addr>'."""
    return {"sender": _bare(sender).lower()}


def email_followup_dims_from(account_id: str, from_value: str) -> dict[str, str]:
    """Suppression-key dimensions for an email followup, from raw fields.

    Keyed on ``account · from_domain`` -- who it is from, not the per-message subject or
    topical category -- so a "not useful" on one newsletter blast suppresses the rest,
    and the key can be rebuilt from a followup Task's ``wait_for`` payload (which carries
    account + from, not the category).
    """
    domain = _domain_of(from_value) if "@" in from_value else ""
    return {"account": account_id, "from_domain": domain}


__all__ = [
    "EMAIL_FOCUS_SURFACE",
    "EMAIL_FOLLOWUP_SURFACE",
    "EMAIL_SEARCH_SURFACE",
    "EMAIL_SUBSYSTEM",
    "email_focus_dims",
    "email_followup_dims_from",
    "email_search_dims",
]
