"""What each guard does with the owner's identity in a text (ADR-0125, PR 3).

Three pieces, each pure apart from the pseudonym book:

- :func:`decide` -- given a text, a guard (and, for an answer, its audience), every
  occurrence of an owner literal with the action ``identity.yaml``'s table assigns it.
  Capability masking acts on it; the shadow hook (``plugins/owner_pii_shadow.py``, PR 4)
  audits what the other guards would do with it. The egress guard, the response check and
  the LLM path keep acting on ``secret``/``link`` exactly as before until PR 5.
- :func:`pseudonym` -- ``[owner:<kind>#<n>]``, ``n`` stable per literal within a process,
  so a consumer can still tell two of the owner's addresses apart.
- :func:`redact_capability_text` -- the capability column applied: ``mask`` becomes
  :data:`MASK`, ``pseudonym`` a pseudonym unless the consumer's manifest grants the kind.
"""

from __future__ import annotations

import threading
from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

from iris_harness.foundation.process_state import track_globals
from iris_harness.kernel.governance.hooks.response_payload import Audience
from iris_harness.kernel.governance.identity_config import (
    BLOCKING,
    Action,
    GuardColumn,
    GuardTable,
)
from iris_harness.kernel.governance.owner_identity import IdentityKind, OwnerIdentity
from iris_harness.kernel.governance.owner_matchers import (
    Match,
    OwnerMatcher,
    canonical,
    is_first_name_alone,
    non_overlapping,
)

Guard = Literal["egress", "answer", "capability", "tier3", "web_search"]

MASK = "[redacted: identity]"

# When two occurrences overlap, the stricter action keeps the span.
_RANK: dict[str, int] = {
    "deny": 5,
    "halt": 5,
    "placeholder": 4,
    "mask": 4,
    "pseudonym": 3,
    "log": 2,
    "pass": 1,
}


@dataclass(frozen=True)
class Decision:
    """One occurrence of an owner literal, and what the guard does with it."""

    start: int
    end: int
    kind: IdentityKind
    literal: str
    action: Action

    @property
    def blocks(self) -> bool:
        return self.action in BLOCKING


def column_for(guard: Guard, audience: Audience = "owner") -> GuardColumn:
    """The table column a guard reads; an answer's depends on who reads it."""
    if guard == "answer":
        return "answer_owner" if audience == "owner" else "answer_other"
    return guard


_matcher_lock = threading.Lock()
_matcher: OwnerMatcher | None = None


def matcher_for(identity: OwnerIdentity) -> OwnerMatcher:
    """The compiled matchers for ``identity``, reused while the corpus is the same object."""
    global _matcher
    with _matcher_lock:
        if _matcher is not None and _matcher.identity is identity:
            return _matcher
    built = OwnerMatcher(identity)
    with _matcher_lock:
        _matcher = built
    return built


def decide(
    text: str,
    *,
    guard: Guard,
    identity: OwnerIdentity,
    table: GuardTable,
    audience: Audience = "owner",
    destination: str | None = None,
) -> tuple[Decision, ...]:
    """Every occurrence of an owner literal in ``text`` and the action ``guard`` takes.

    Occurrences that overlap keep the stricter action's span (a ``pass`` link never hides
    an email inside it). ``pass`` occurrences are included, so a caller can audit them.
    ``destination`` is the network tool at ``egress`` (``log_only_destinations``).
    """
    column = column_for(guard, audience)
    ranked: list[tuple[Match, Action]] = []
    for match in matcher_for(identity).find_all(text):
        action = table.action(
            match.kind,
            column,
            first_name=is_first_name_alone(match),
            destination=destination,
        )
        ranked.append((match, action))
    ranked.sort(key=lambda ma: (-_RANK[ma[1]], -(ma[0].end - ma[0].start), ma[0].start))
    actions = {id(m): a for m, a in ranked}
    kept = non_overlapping(m for m, _ in ranked)
    return tuple(Decision(m.start, m.end, m.kind, m.literal, actions[id(m)]) for m in kept)


# -- pseudonyms --------------------------------------------------------------------------

_book_lock = threading.Lock()
_book: dict[tuple[IdentityKind, str], int] = {}
_next: dict[IdentityKind, int] = {}


def pseudonym(kind: IdentityKind, literal: str) -> str:
    """``[owner:<kind>#<n>]``: the same ``n`` for the same literal for the process's life.

    Keyed by the literal's canonical form, so two spellings of one phone number share a
    number and two different addresses never do.
    """
    key = (kind, canonical(kind, literal))
    with _book_lock:
        n = _book.get(key)
        if n is None:
            n = _next.get(kind, 0) + 1
            _next[kind] = n
            _book[key] = n
    return f"[owner:{kind}#{n}]"


def reset_pseudonyms() -> None:
    """Forget every number (tests only; production numbers live as long as the process)."""
    with _book_lock:
        _book.clear()
        _next.clear()


@dataclass(frozen=True)
class Redacted:
    """A capability text after the capability column: the text, and what it held."""

    text: str
    kinds: frozenset[IdentityKind]  # every kind of owner literal found, unmasked or not
    changed: bool


def redact_capability_text(
    text: str,
    *,
    identity: OwnerIdentity,
    table: GuardTable,
    grants: Collection[str],
) -> Redacted:
    """``text`` with the capability column applied for a consumer holding ``grants``.

    ``mask`` -> :data:`MASK`; ``pseudonym`` -> :func:`pseudonym` unless the kind is in
    ``grants`` (then the literal stays); ``pass`` -> unchanged. ``secret`` is masked
    whatever ``grants`` says: the table cannot make it a pseudonym, and a grant only
    unmasks pseudonyms.
    """
    decisions = decide(text, guard="capability", identity=identity, table=table)
    out: list[str] = []
    at = 0
    kinds: set[IdentityKind] = set()
    for d in decisions:
        if d.action == "pass":
            continue
        kinds.add(d.kind)
        out.append(text[at : d.start])
        if d.action == "pseudonym" and d.kind in grants:
            out.append(text[d.start : d.end])
        elif d.action == "pseudonym":
            out.append(pseudonym(d.kind, d.literal))
        else:
            out.append(MASK)
        at = d.end
    out.append(text[at:])
    result = "".join(out)
    return Redacted(result, frozenset(kinds), result != text)


__all__ = [
    "MASK",
    "Audience",
    "Decision",
    "Guard",
    "Redacted",
    "column_for",
    "decide",
    "matcher_for",
    "pseudonym",
    "redact_capability_text",
    "reset_pseudonyms",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_matcher", "_book", "_next")
