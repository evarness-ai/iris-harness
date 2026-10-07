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
- Every entry must run in time linear in the text. The classifier scans
  untrusted text (a document, a tool result, a file) with every entry, so a
  pattern that backtracks quadratically is a denial-of-service lever (issue
  #156: 200 KB of ``"a."*100000 + "@b."`` took 20 s). A regex whose start can
  re-enter the same long run stays a regex only if a ``\b`` or a literal prefix
  stops that; ``email`` and ``voice_transcript_marker`` could not be made
  linear that way, so they are small matchers with the same meaning.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final, Protocol

from iris_harness.kernel.governance.hooks.types import DataClassification

# (name, compiled_regex, classification)
RegexEntry = tuple[str, re.Pattern[str], DataClassification]


class Searchable(Protocol):
    """What the classifier needs of a pattern: does the text contain a match.

    A compiled regex satisfies it; so do the linear matchers below. Only the credential
    patterns are also used to *replace* text (redaction), and those stay compiled regexes.
    """

    def search(self, text: str, /) -> object | None: ...


# (name, searchable, classification): the classifier's view of an entry.
ClassifierEntry = tuple[str, Searchable, DataClassification]


class _EmailMatcher:
    """``\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b``, in linear time.

    The regex starts a scan at every word boundary inside a run of local-part characters, and
    each scan runs to the end of the run: quadratic on ``"a."*N`` or ``"+1-"*N``. The meaning
    does not depend on where in the run it starts, so this looks at each maximal run once: a
    run immediately followed by ``@`` is a local part when some position in it is a word
    boundary (the regex's leading ``\\b``), and the match needs a domain after the ``@``.
    Each character is visited a constant number of times (a run, then a domain run, and the
    two never overlap a third), so the cost is linear in the text.
    """

    _RUN = re.compile(r"[A-Za-z0-9._%+-]+")
    _BOUNDARY_START = re.compile(r"\b[A-Za-z0-9._%+-]")
    _DOMAIN = re.compile(r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

    def search(self, text: str, /) -> object | None:
        if "@" not in text:
            return None
        for run in self._RUN.finditer(text):
            end = run.end()
            if text.startswith("@", end) and self._is_local_part(text, run.start(), end):
                if self._DOMAIN.match(text, end + 1):
                    return True
        return None

    def _is_local_part(self, text: str, start: int, end: int) -> bool:
        # A ``\b`` somewhere in [start, end): ``pos``/``endpos`` keep the character before
        # ``start`` in view, so a boundary at ``start`` itself is judged against it.
        return self._BOUNDARY_START.search(text, start, end) is not None


class _VoiceMarkerMatcher:
    """``\\[voice_transcript:[^\\]]*\\]``, in linear time.

    The regex re-scans to the end of the text from every ``[voice_transcript:`` when no ``]``
    follows (quadratic). A match exists exactly when a ``]`` follows the first occurrence of
    the prefix: a later occurrence has fewer characters after it, and the prefix itself
    contains no ``]``.
    """

    _PREFIX = "[voice_transcript:"

    def search(self, text: str, /) -> object | None:
        start = text.find(self._PREFIX)
        if start == -1:
            return None
        return True if text.find("]", start + len(self._PREFIX)) != -1 else None


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


PII_PATTERNS: Final[list[ClassifierEntry]] = [
    ("email", _EmailMatcher(), "personal"),
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


IRIS_SPECIFIC_PATTERNS: Final[list[ClassifierEntry]] = [
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
    ("voice_transcript_marker", _VoiceMarkerMatcher(), "personal"),
]


ALL_PACKS: Final[Mapping[str, Sequence[ClassifierEntry]]] = {
    "credentials": CREDENTIAL_PATTERNS,
    "pii": PII_PATTERNS,
    "iris_specific": IRIS_SPECIFIC_PATTERNS,
}
