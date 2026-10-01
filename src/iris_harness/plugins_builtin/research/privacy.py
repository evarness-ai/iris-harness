"""Strip the user's personal identifiers from outbound web-search queries.

A privacy-first guard (issue 0031): the agent must never leak the user's identity to a
third-party search provider — e.g. building ``"latest market performance of <name>'s
portfolio"`` into a SearXNG/Tavily query. The model sometimes folds the user's name (read
from the identity profile) into a research query; this removes it before egress.

The identifiers are NOT hardcoded — they are derived from the user's identity files
(USER.md / SOUL.md) by :func:`extract_personal_identifiers`, so the guard tracks whatever
the profile actually says. Names come only from the USER profile (never SOUL.md's *agent*
name); emails are unambiguous PII and are pulled from either file.

Pure and deterministic so it is unit-testable in isolation.

What it shares with the harness's owner-identity matchers (ADR-0125, PR 3), and what it
keeps: a name is matched with the SDK's ``name_pattern`` (whole words, case-insensitive,
single spaces -- exactly this module's rule before the share). The rest is this guard's own,
on purpose: it strips ANY email address, not only the owner's (an address learned from inbox
context is still PII), strips each word of a name of 3+ characters, and reads names from the
USER profile's ``Name:`` line. The owner matchers match only the owner's own literals, so
they cannot stand in for either.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# A trailing possessive left behind after a name is removed ("<name>'s portfolio").
_POSSESSIVE = re.compile(r"['’]s\b", re.IGNORECASE)
# A "Name:" / "- **name**:" line in an identity markdown file.
_NAME_LINE = re.compile(r"(?im)^\s*[-*]?\s*\**\s*name\s*\**\s*[:\-]\s*(.+)$")
# An email address anywhere in an identity file (no trailing punctuation captured).
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def extract_personal_identifiers(user_md: str | None, soul_md: str | None = None) -> list[str]:
    """Personal identifiers (name(s) + email(s)) to strip from web queries, derived from
    the user's identity files.

    Names are taken ONLY from the USER profile (``user_md``) — never from SOUL.md, whose
    ``name:`` is the *agent's* own name (e.g. "IRIS"), which must not be stripped. Emails
    are unambiguous PII and are collected from either file. Order: names first (longest
    matched first downstream), then emails; de-duplicated case-insensitively.
    """
    identifiers: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        candidate = value.strip()
        if candidate and candidate.lower() not in seen:
            seen.add(candidate.lower())
            identifiers.append(candidate)

    if user_md:
        for match in _NAME_LINE.finditer(user_md):
            name = match.group(1).strip().strip("*").strip()
            # Drop trailing parentheticals / HTML comments / role suffixes.
            name = re.split(r"[(\[,;<]| - ", name, maxsplit=1)[0].strip()
            if name:
                _add(name)
    for text in (user_md, soul_md):
        if text:
            for email in _EMAIL.findall(text):
                _add(email)
    return identifiers


def owner_name_variants(names: Iterable[str]) -> list[str]:
    """Full names plus their individual word tokens (>= 3 chars), longest first.

    Longest-first so a multi-word full name is removed before its component tokens,
    and the >= 3 floor avoids stripping short initials / common words.
    """
    variants: set[str] = set()
    for name in names:
        normalised = " ".join((name or "").split())
        if not normalised:
            continue
        variants.add(normalised)
        for token in normalised.split():
            if len(token) >= 3:
                variants.add(token)
    return sorted(variants, key=len, reverse=True)


def strip_emails(query: str) -> tuple[str, bool]:
    """Remove ANY email address from an outbound web query (defense-in-depth).

    :func:`strip_owner_identifiers` only knows the identifiers listed in the identity
    files. But the model can fold in an address it learned from *inbox context* — e.g. a
    secondary Gmail account the profile never mentions — and that is still PII that must
    never reach a search provider. This is a blanket guard: no email-shaped token egresses
    to the web, whether or not we've "seen" it before. Returns ``(cleaned, was_stripped)``.
    """
    if not query:
        return query, False
    cleaned = _EMAIL.sub(" ", query)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, cleaned != query


def strip_owner_identifiers(query: str, names: Iterable[str]) -> tuple[str, bool]:
    """Remove the user's name(s) (and any trailing possessive) from ``query``.

    Returns ``(sanitised_query, was_stripped)``. Case-insensitive, word-boundary
    matched, so "the owner's portfolio vs Nifty" → "portfolio vs Nifty". A query with no
    owner identifier is returned unchanged with ``was_stripped=False``.
    """
    from iris_harness.sdk.identity import name_pattern

    variants = owner_name_variants(names)
    if not query or not variants:
        return query, False
    cleaned = query
    for variant in variants:
        # Remove the name and an immediately-following possessive in one pass.
        cleaned = re.sub(
            rf"{name_pattern(variant, any_space=False)}(?:['’]s)?",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
    cleaned = _POSSESSIVE.sub("", cleaned)  # any orphaned "'s" left behind
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)  # tidy space-before-punct
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, cleaned != query
