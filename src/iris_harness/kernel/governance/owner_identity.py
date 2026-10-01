"""What the owner's identity is, as literals every guard reads the same way.

One extractor for the owner's identity, shared by every guard that protects it (owner-PII
masking, ADR-0125). Pure: text and declared literals in, kind-tagged literals out; no I/O,
no cache, standard library only. Where the text and the declarations come from, and the one
cached corpus built from them, is the seam's (`identity_redaction.owner_identity()`).

The kinds:

- ``secret``: secret-shaped tokens out of the identity documents (14+ characters of
  ``[A-Za-z0-9_.+/-]`` with a letter and a digit) that are not URL-, domain- or
  email-shaped. The response check halts on these and capability masking masks them.
  Only the documents produce it: a secret is recognised by its shape, never declared.
- ``link``: the secret-shaped tokens that are URL-, domain- or email-shaped. A blog like
  ``www.web3notes.example`` has a digit but is not a credential; halting an answer on it
  blocked the owner's own profile answers (issue 0022). The egress guard still refuses to
  send either kind. The documents produce it, and the owner's confirmed ``blog`` and
  ``website`` facts (``identity.yaml`` ``ontology_kinds``) declare it.
- ``name``, ``email``, ``phone``, ``address``, ``handle``: declared by the sources (the
  USER.md ``identity:`` block, the owner's confirmed facts, plugin account addresses).
  Free text adds ``email`` and ``phone`` shapes only -- no names from prose, no NER.

Each guard asks for the kinds it acts on (``OwnerIdentity.of``), so adding a kind changes
no guard that does not ask for it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal

IdentityKind = Literal["secret", "link", "name", "email", "phone", "address", "handle"]

KINDS: tuple[IdentityKind, ...] = (
    "secret",
    "link",
    "name",
    "email",
    "phone",
    "address",
    "handle",
)

# The owner's personal identifiers: what a plugin may provide (``identity: provides``),
# what a capability consumer sees as a pseudonym, and what a manifest may unmask.
OWNER_PII_KINDS: tuple[IdentityKind, ...] = ("name", "email", "phone", "address", "handle")

# Every kind a source may declare by value: the PII kinds, and ``link`` -- the owner's
# confirmed ``blog`` and ``website`` facts join the documents' links, which egress refuses
# to send (ADR-0125 amendment 6). Only the composition root's own sources may declare a
# link; a plugin provides PII kinds only. Never ``secret``: it is a shape found in the
# documents, and nothing may name a literal a secret (or, through ``never_match``, take
# one away).
DECLARABLE_KINDS: tuple[IdentityKind, ...] = (*OWNER_PII_KINDS, "link")

# The key a source uses, beside its kinds, for the strings that are never the owner's.
NEVER_MATCH: Final = "never_match"
# The kinds ``never_match`` never touches: the ones a guard acts on today. A secret is a
# credential-shaped token; a link is what egress refuses to send. Neither may be switched
# off by a profile entry (owner decision, ADR-0125 amendment 5).
NEVER_MATCH_SPARES: tuple[IdentityKind, ...] = ("secret", "link")

_SECRET_SHAPED = re.compile(r"[A-Za-z0-9_.+/-]{14,}")
_URL_EMAIL_SHAPE_RE = re.compile(r"@|^https?://|^www\.|\.[a-z]{2,}(?:[/:?#]|$)", re.IGNORECASE)

_EMAIL_RE = re.compile(
    r"(?<![\w.%+-])[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"
)

# A phone candidate: an optional ``+``, then 2-7 digit groups (an area code may sit in
# parentheses) joined by at most one space, hyphen or dot. Not next to a word character or
# to the punctuation that makes a date, time, path or version (``/``, ``:``, ``.5``).
_PHONE_GROUP = r"(?:\(\d{1,5}\)|\d{1,5})"
_PHONE_RE = re.compile(
    rf"(?<![\w+/.:,-])(\+?{_PHONE_GROUP}(?:[ .-]?{_PHONE_GROUP}){{1,6}})(?![\w/:]|[.,-]\d)"
)
_DIGIT_RUN = re.compile(r"\d+")


def is_url_or_email_shaped(token: str) -> bool:
    """A URL, domain or email address: never an identity *secret*, digits or not."""
    return bool(_URL_EMAIL_SHAPE_RE.search(token))


def _secret_shaped_tokens(texts: Iterable[str]) -> set[str]:
    return {
        tok
        for text in texts
        for tok in _SECRET_SHAPED.findall(text)
        if any(c.isalpha() for c in tok) and any(c.isdigit() for c in tok)
    }


def _is_phone(candidate: str) -> bool:
    """Whether a phone-shaped candidate is a phone number, conservatively.

    - 8 to 15 digits (E.164 allows at most 15).
    - Without a leading ``+``: at least 10 digits in at least two groups. A bare digit
      run is an account number, a timestamp or an amount as often as a phone.
    - One separator style after the first: ``+1 555-123-4567`` and ``(555) 123-4567``
      pass, ``555-123-4567 2026`` (a number, then a year) does not.
    - Not thousands-grouped (``1 234 567 890``, ``192.168.100.200``): after a first group
      of at most three digits, every group of exactly three digits is an amount or an
      address, not a phone.
    """
    groups = _DIGIT_RUN.findall(candidate)
    digits = sum(len(g) for g in groups)
    if not 8 <= digits <= 15:
        return False
    if not candidate.startswith("+") and (digits < 10 or len(groups) < 2):
        return False
    separators = _DIGIT_RUN.split(candidate)[1:-1]  # what joins each pair of groups
    if len(set(separators[1:])) > 1:
        return False
    return not (len(groups) >= 3 and len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:]))


def _emails(texts: Iterable[str]) -> set[str]:
    return {m.group(0) for text in texts for m in _EMAIL_RE.finditer(text)}


def _phones(texts: Iterable[str]) -> set[str]:
    return {
        m.group(1).strip()
        for text in texts
        for m in _PHONE_RE.finditer(text)
        if _is_phone(m.group(1).strip())
    }


def _frozen(
    literals: Mapping[IdentityKind, frozenset[str]],
) -> Mapping[IdentityKind, frozenset[str]]:
    return MappingProxyType({kind: literals.get(kind, frozenset()) for kind in KINDS})


@dataclass(frozen=True)
class OwnerIdentity:
    """The owner's identity literals, by kind. Build with :func:`extract` or :func:`declared`."""

    literals: Mapping[IdentityKind, frozenset[str]] = field(
        default_factory=lambda: MappingProxyType({kind: frozenset() for kind in KINDS})
    )

    def of(self, *kinds: IdentityKind) -> frozenset[str]:
        """The literals of ``kinds``, together."""
        out: frozenset[str] = frozenset()
        for kind in kinds:
            out |= self.literals.get(kind, frozenset())
        return out


EMPTY = OwnerIdentity()


def extract(texts: Sequence[str]) -> OwnerIdentity:
    """The owner's identity literals out of the identity documents' free text.

    Secret-shaped tokens split into ``secret`` and ``link`` (as before PR 2), plus the
    ``email`` and ``phone`` shapes. Nothing else: free text never yields a name.
    """
    tokens = _secret_shaped_tokens(texts)
    links = frozenset(tok for tok in tokens if is_url_or_email_shaped(tok))
    return OwnerIdentity(
        literals=_frozen(
            {
                "secret": frozenset(tokens - links),
                "link": links,
                "email": frozenset(_emails(texts)),
                "phone": frozenset(_phones(texts)),
            }
        )
    )


def declared(literals: Mapping[str, Iterable[str]]) -> OwnerIdentity:
    """An identity out of kind-tagged literals a source declared.

    Keys that are not a declarable kind (``secret``, ``never_match``, anything unknown) are
    not literals and are ignored here; the seam decides what a source may declare and says
    so when it drops one. Values are stripped; empty ones are skipped.
    """
    out: dict[IdentityKind, frozenset[str]] = {}
    for key, values in literals.items():
        if key not in DECLARABLE_KINDS:
            continue
        if isinstance(values, str):  # one literal, not its characters
            values = (values,)
        cleaned = frozenset(v.strip() for v in values if isinstance(v, str) and v.strip())
        out[key] = out.get(key, frozenset()) | cleaned
    return OwnerIdentity(literals=_frozen(out))


def merge(*identities: OwnerIdentity) -> OwnerIdentity:
    """Every literal of every identity, kind by kind."""
    return OwnerIdentity(
        literals=_frozen(
            {kind: frozenset().union(*(i.of(kind) for i in identities)) for kind in KINDS}
        )
    )


def apply_never_match(identity: OwnerIdentity, never_match: Iterable[str]) -> OwnerIdentity:
    """``identity`` without the strings the owner says are never theirs.

    Exact strings, compared case-insensitively (a shared first name, a contact's address).
    Every kind but :data:`NEVER_MATCH_SPARES`: an entry in a profile must not be able to
    switch off what a guard acts on today (owner decision, ADR-0125 amendment 5).
    """
    drop = {s.strip().casefold() for s in never_match if isinstance(s, str) and s.strip()}
    if not drop:
        return identity
    return OwnerIdentity(
        literals=_frozen(
            {
                kind: (
                    identity.of(kind)
                    if kind in NEVER_MATCH_SPARES
                    else frozenset(v for v in identity.of(kind) if v.casefold() not in drop)
                )
                for kind in KINDS
            }
        )
    )


__all__ = [
    "DECLARABLE_KINDS",
    "EMPTY",
    "KINDS",
    "NEVER_MATCH",
    "NEVER_MATCH_SPARES",
    "OWNER_PII_KINDS",
    "IdentityKind",
    "OwnerIdentity",
    "apply_never_match",
    "declared",
    "extract",
    "is_url_or_email_shaped",
    "merge",
]
