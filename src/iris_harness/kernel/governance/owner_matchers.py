"""Where the owner's identity literals occur in a text, kind by kind (ADR-0125, PR 3).

``owner_identity`` says *what* the owner's identity is; this module finds it in text.
Pure: an :class:`~iris_harness.kernel.governance.owner_identity.OwnerIdentity` and a text
in, kind-tagged spans out. Standard library only, no I/O, no cache beyond the compiled
patterns an :class:`OwnerMatcher` holds for one corpus.

Each kind is matched the way it is written, and conservatively (no partial words):

- ``secret`` and ``link``: the exact literal, as every guard has matched them so far.
- ``email``: the address, case-insensitively, not inside a longer address or word.
- ``phone``: the digit sequence, whatever the formatting -- spaces, hyphens, dots,
  parentheses, a ``+`` or ``00`` international prefix, a national trunk ``0``. A number
  written with its country code matches the same number written without one (the code is
  one to three digits), and the match never starts or ends inside a digit group.
- ``name``: the whole name, case-insensitively, on word boundaries, any whitespace between
  its words; never inside an email address, a handle or a hyphenated word. A name is
  matched as declared: ``"Robin Example"`` does not match ``"Robin"`` alone -- a first
  name is its own literal when a source declares it.
- ``handle``: case-insensitively, with or without a leading ``@``, and as a path segment
  of a URL (``github.com/robin-gh``). A handle declared as a URL is matched by its last
  path segment.
- ``address``: the address's words in order, case-insensitively, with any punctuation or
  whitespace between them (``1 Example Street, Springfield`` matches the same address
  over two lines).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from iris_harness.kernel.governance.owner_identity import KINDS, IdentityKind, OwnerIdentity

# A phone literal needs this many digits to be matched at all: fewer is an extension or a
# PIN, and would match amounts and dates.
MIN_PHONE_DIGITS = 7
# A country code is one to three digits (E.164).
_MAX_COUNTRY_CODE = 3


@dataclass(frozen=True)
class Match:
    """One occurrence: ``text[start:end]`` is the owner's ``literal`` of ``kind``.

    ``literal`` is the corpus literal, never the text's spelling of it, so two spellings
    of one phone number are one literal.
    """

    start: int
    end: int
    kind: IdentityKind
    literal: str

    def overlaps(self, other: Match) -> bool:
        return self.start < other.end and other.start < self.end


def name_pattern(literal: str, *, any_space: bool = True) -> str:
    r"""A whole-word, case-insensitive regex source for a name (compile with ``re.I``).

    The words of ``literal`` bounded by ``\b``. ``any_space`` lets any run of whitespace
    stand between the words (a name over a line break); without it exactly one space
    does. Shared by the owner matchers and the research plugin's query scrub (which keeps
    single spaces, its behaviour before the share).
    """
    words = literal.split()
    joiner = r"\s+" if any_space else " "
    return r"\b" + joiner.join(re.escape(w) for w in words) + r"\b"


# -- phones ------------------------------------------------------------------------------

# A run of digit groups joined by spaces, hyphens, dots or parentheses, maybe after a
# ``+``. Not next to a word character, or to the punctuation that makes a path, a time or
# a version (a colon only after a digit, so ``tel:+1 ...`` is a phone and ``10:30`` not).
_PHONE_RUN = re.compile(r"(?<![\w+/.\-])(?<!\d:)\+?[(\d][\d() .\-]*\d\)?(?![\w/:])")
_DIGITS = re.compile(r"\d+")


@dataclass(frozen=True)
class _Phone:
    international: bool
    digits: str


def _parse_phone(written: str) -> _Phone | None:
    """What a written number is: its digits, and whether they start with a country code.

    ``None`` below :data:`MIN_PHONE_DIGITS`. A ``(0)`` right after the country code is the
    national trunk prefix written inside an international number (``+44 (0)20 ...``); it
    is not dialled, so it is not part of the digits.
    """
    groups = _DIGITS.findall(written)
    if sum(len(g) for g in groups) < MIN_PHONE_DIGITS:
        return None
    around = _DIGITS.split(written)  # around[i] is the text before group i
    plus = written.lstrip().startswith("+")
    digits = "".join(
        g
        for i, g in enumerate(groups)
        if not (
            plus and i == 1 and g == "0" and around[1].endswith("(") and around[2].startswith(")")
        )
    )
    if plus:
        return _Phone(True, digits)
    if digits.startswith("00") and len(digits) > 2:
        return _Phone(True, digits[2:])
    return _Phone(False, digits)


def _national(digits: str) -> str:
    """A national number without its trunk ``0``."""
    return digits[1:] if digits.startswith("0") else digits


def _same_phone(a: _Phone, b: _Phone) -> bool:
    if a.international == b.international:
        if a.international:
            return a.digits == b.digits
        return _national(a.digits) == _national(b.digits)
    intl, nat = (a, b) if a.international else (b, a)
    national = _national(nat.digits)
    extra = len(intl.digits) - len(national)
    return (
        len(national) >= MIN_PHONE_DIGITS
        and 1 <= extra <= _MAX_COUNTRY_CODE
        and intl.digits.endswith(national)
    )


# Where a run may hold a second number: spaces between two groups, maybe then a ``(``.
_SPACE_SEP = re.compile(r" +\(?")


def _phone_matches(text: str, phones: Sequence[tuple[str, _Phone]]) -> list[Match]:
    """Every span of a phone-shaped run in ``text`` that is one of ``phones``.

    A run can hold more than one number (``call 555 0100 0199 2026``): a span may start or
    end inside a run only where spaces separate two groups, never a hyphen or a dot, so a
    span is never part of one written number -- and never starts inside a run that opens
    with a country code. The longest span from each start wins.
    """
    out: list[Match] = []
    if not phones:
        return out
    for run in _PHONE_RUN.finditer(text):
        written = run.group(0)
        groups = list(_DIGITS.finditer(written))
        seps = [written[: groups[0].start()]] + [
            written[groups[k - 1].end() : groups[k].start()] for k in range(1, len(groups))
        ]
        gaps = {k for k in range(1, len(groups)) if _SPACE_SEP.fullmatch(seps[k])}
        # A run that opens with a country code (``+44 ...``, ``0044 ...``) is one number
        # to its end, spaces and all: nothing inside it starts another.
        international = written.startswith("+") or written.startswith("00")
        starts = {0} if international else {0} | gaps
        ends = {k - 1 for k in gaps} | {len(groups) - 1}
        i = 0
        while i < len(groups):
            hit: Match | None = None
            if i in starts:
                lead = seps[i].lstrip(" ")
                begin = groups[i].start() - len(lead)
                for j in range(len(groups) - 1, i - 1, -1):
                    if j not in ends:
                        continue
                    stop = groups[j].end()
                    span = written[begin:stop]
                    if written[stop : stop + 1] == ")" and span.count("(") > span.count(")"):
                        stop += 1
                    key = _parse_phone(written[begin:stop])
                    if key is None:
                        continue
                    literal = next((lit for lit, p in phones if _same_phone(key, p)), None)
                    if literal is not None:
                        hit = Match(run.start() + begin, run.start() + stop, "phone", literal)
                        i = j + 1
                        break
            if hit is None:
                i += 1
            else:
                out.append(hit)
    return out


# -- the other kinds ---------------------------------------------------------------------


def _handle_core(literal: str) -> str:
    """A handle without its ``@``; a handle declared as a URL is its last path segment."""
    value = literal.strip()
    if "/" in value:
        segments = [s for s in value.split("?", 1)[0].split("/") if s]
        value = segments[-1] if segments else ""
    return value.lstrip("@")


def _pattern(kind: IdentityKind, literal: str) -> re.Pattern[str] | None:
    """The compiled pattern for one literal of ``kind``; ``None`` when it cannot match."""
    if kind == "email":
        return re.compile(rf"(?<![\w.%+\-]){re.escape(literal)}(?![\w\-]|\.\w)", re.IGNORECASE)
    if kind == "name":
        if not literal.split():
            return None
        # Not inside an address, a handle, a hyphenated word or a dotted token.
        return re.compile(
            rf"(?<![@\-/+])(?<!\w\.){name_pattern(literal)}(?![@\-]|[./]\w)", re.IGNORECASE
        )
    if kind == "handle":
        core = _handle_core(literal)
        if not core:
            return None
        return re.compile(rf"(?<![\w.\-@])@?{re.escape(core)}(?![\w\-]|\.\w)", re.IGNORECASE)
    if kind == "address":
        words = re.findall(r"\w+", literal)
        if not words:
            return None
        body = r"[\W_]{1,6}".join(re.escape(w) for w in words)
        return re.compile(rf"\b{body}\b", re.IGNORECASE)
    return None


def _canonical(kind: IdentityKind, literal: str) -> str:
    """One key per literal however a source spelled it (used to number pseudonyms)."""
    if kind == "phone":
        phone = _parse_phone(literal)
        if phone is not None:
            return ("+" if phone.international else "") + phone.digits
    if kind == "handle":
        return _handle_core(literal).casefold()
    if kind == "address":
        return " ".join(re.findall(r"\w+", literal)).casefold()
    if kind in ("email", "name"):
        return " ".join(literal.split()).casefold()
    return literal


def canonical(kind: IdentityKind, literal: str) -> str:
    """The canonical key of ``literal``: two spellings of one phone number share one."""
    return _canonical(kind, literal)


_EXACT_KINDS: tuple[IdentityKind, ...] = ("secret", "link")
_PATTERN_KINDS: tuple[IdentityKind, ...] = ("email", "name", "handle", "address")


class OwnerMatcher:
    """The compiled matchers for one corpus. Build once per corpus; ``find`` is pure."""

    def __init__(self, identity: OwnerIdentity) -> None:
        self.identity = identity
        self._exact: dict[IdentityKind, tuple[str, ...]] = {
            kind: tuple(sorted(identity.of(kind), key=len, reverse=True)) for kind in _EXACT_KINDS
        }
        self._patterns: dict[IdentityKind, list[tuple[str, re.Pattern[str]]]] = {}
        for kind in _PATTERN_KINDS:
            compiled = []
            for literal in sorted(identity.of(kind), key=len, reverse=True):
                pattern = _pattern(kind, literal)
                if pattern is not None:
                    compiled.append((literal, pattern))
            self._patterns[kind] = compiled
        self._phones: list[tuple[str, _Phone]] = [
            (literal, key)
            for literal in sorted(identity.of("phone"))
            if (key := _parse_phone(literal)) is not None
        ]

    def find_all(self, text: str, kinds: Iterable[IdentityKind] = KINDS) -> list[Match]:
        """Every occurrence of every literal of ``kinds``, overlapping ones included."""
        out: list[Match] = []
        if not text:
            return out
        for kind in kinds:
            if kind in self._exact:
                for literal in self._exact[kind]:
                    at = text.find(literal)
                    while at != -1:
                        out.append(Match(at, at + len(literal), kind, literal))
                        at = text.find(literal, at + 1)
            elif kind == "phone":
                out.extend(_phone_matches(text, self._phones))
            else:
                for literal, pattern in self._patterns.get(kind, ()):
                    out.extend(
                        Match(m.start(), m.end(), kind, literal) for m in pattern.finditer(text)
                    )
        return sorted(out, key=lambda m: (m.start, -(m.end - m.start), KINDS.index(m.kind)))

    def find(self, text: str, kinds: Iterable[IdentityKind] = KINDS) -> list[Match]:
        """Non-overlapping occurrences: the earliest, then the longest, then by kind order."""
        return non_overlapping(self.find_all(text, kinds))


def non_overlapping(matches: Iterable[Match]) -> list[Match]:
    """``matches`` in the order given, dropping any that overlaps one already kept."""
    kept: list[Match] = []
    for match in matches:
        if not any(match.overlaps(k) for k in kept):
            kept.append(match)
    return sorted(kept, key=lambda m: m.start)


def is_first_name_alone(match: Match) -> bool:
    """A ``name`` literal of one word: a first name or nickname on its own (ADR-0125 §4)."""
    return match.kind == "name" and len(match.literal.split()) == 1


__all__ = [
    "MIN_PHONE_DIGITS",
    "Match",
    "OwnerMatcher",
    "canonical",
    "is_first_name_alone",
    "name_pattern",
    "non_overlapping",
]
