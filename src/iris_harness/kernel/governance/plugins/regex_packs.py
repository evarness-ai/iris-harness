"""Regex packs for Stage 1 of the DataClassifier.

Each pack is a list of ``(name, compiled_pattern, classification)``
tuples. The classifier scans inbound text against every pack and takes
the highest-severity match (severity order: secret > personal >
internal > public).

Design notes:

- Precision over recall. False positives cause spurious cloud blocks
  and erode user trust; missed PII is partly covered by other layers
  (redaction filter, prompt-injection detector, vault).
- Stage 1B (Presidio NER) will land as a follow-up to catch
  unstructured person names / addresses; that's a separate plugin
  gated by config and an optional install extra.
- Patterns should be readable. If a pattern needs comments to justify
  itself, prefer adding a focused test over silent cleverness.
"""

from __future__ import annotations

import re
from typing import Final

from iris_harness.kernel.governance.hooks.types import DataClassification

# (name, compiled_regex, classification)
RegexEntry = tuple[str, re.Pattern[str], DataClassification]


CREDENTIAL_PATTERNS: Final[list[RegexEntry]] = [
    # Cloud LLM provider API keys
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "secret"),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"), "secret"),
    ("openrouter_key", re.compile(r"\bsk-or-[A-Za-z0-9_-]{20,}\b"), "secret"),
    # GitHub PATs (classic, OAuth, server-to-server, user-to-server, refresh)
    ("github_pat_classic", re.compile(r"\bghp_[A-Za-z0-9]{36,}\b"), "secret"),
    ("github_pat_oauth", re.compile(r"\bgho_[A-Za-z0-9]{36,}\b"), "secret"),
    ("github_pat_server", re.compile(r"\bghs_[A-Za-z0-9]{36,}\b"), "secret"),
    ("github_pat_user", re.compile(r"\bghu_[A-Za-z0-9]{36,}\b"), "secret"),
    ("github_pat_refresh", re.compile(r"\bghr_[A-Za-z0-9]{36,}\b"), "secret"),
    # GitHub fine-grained PAT (exp-007: previously unclassified). Format is
    # `github_pat_` + ~82 chars of [A-Za-z0-9_]; matched permissively from 50.
    ("github_pat_finegrained", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b"), "secret"),
    # AWS access key id (the secret key is harder to pattern-match deterministically)
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "secret"),
    # JWT (3 base64url segments separated by dots)
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
        "secret",
    ),
    # PEM private key headers (RSA, EC, OpenSSH, or generic)
    (
        "pem_private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |)PRIVATE KEY-----"),
        "secret",
    ),
]


PII_PATTERNS: Final[list[RegexEntry]] = [
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "personal",
    ),
    # US Social Security Number (no real validation, just the format)
    ("ssn_us", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "personal"),
    # US phone in explicit format: requires separator (avoids matching
    # raw 10-digit numbers which are often not phones)
    (
        "phone_us_strict",
        re.compile(r"\b\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
        "personal",
    ),
    # International with leading + and country code, then 6-14 digits with
    # optional separators between each digit (so "+44 20 7946 0958" works).
    (
        "phone_intl_plus",
        re.compile(r"\+\d{1,4}(?:[\s.-]?\d){6,14}\b"),
        "personal",
    ),
]


IRIS_SPECIFIC_PATTERNS: Final[list[RegexEntry]] = [
    # Telegram bot tokens look like 12345678:AAH...alphanum (35-46 chars)
    (
        "telegram_bot_token",
        re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b"),
        "secret",
    ),
    # vault://handle references in prompts are a leak signal —
    # tools should resolve handles, never the LLM
    (
        "vault_handle",
        re.compile(r"\bvault://[A-Za-z0-9_/-]+"),
        "internal",
    ),
    # IRIS voice transcript markers — anything inside is from the mic
    (
        "voice_transcript_marker",
        re.compile(r"\[voice_transcript:[^\]]*\]"),
        "personal",
    ),
]


ALL_PACKS: Final[dict[str, list[RegexEntry]]] = {
    "credentials": CREDENTIAL_PATTERNS,
    "pii": PII_PATTERNS,
    "iris_specific": IRIS_SPECIFIC_PATTERNS,
}
