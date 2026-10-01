"""The owner's identity, supplied from above: named sources and the one cached corpus.

The exfiltration guard (`plugins/network_egress.py`, exp-007 GAP-14) denies any
network tool call whose arguments carry a secret-shaped literal out of the user's
identity documents. To do that it needs the text of those documents -- and it used
to read them itself, by importing the identity loader.

That import was the last thing keeping `identity/` a layer of its own: the kernel
sits below memory, the loader is read by memory, and a package read from both
sides cannot be folded into either. The kernel asks for the text now, and the
composition root supplies it (`runtime/identity_redaction.py`).

**The registry's failure mode, and why it is not silent.** An inversion like this
can be left unregistered, and an unregistered exfiltration guard redacts nothing
while reporting success -- which is worse than the import it replaced. So:

- `identity_texts()` returns ``None`` when nothing is registered, distinct from
  ``[]`` (registered, but the documents are empty). The guard treats the two
  differently and logs the first, once.
- `tests/unit/iris_harness/kernel/test_governance/test_identity_redaction.py`
  asserts a built runtime HAS a provider registered. A regression that drops the
  registration fails the suite rather than quietly widening egress.

**One corpus for every guard.** `owner_identity()` merges every source into the owner's
kind-tagged literals (`owner_identity`) and every guard reads that one copy: the egress
guard, the response check and capability masking can no longer disagree about what the
owner's identity is (ADR-0125).

**Sources (ADR-0125, PR 2).** The identity documents are the ``documents`` source
(`register_identity_text_provider`); it alone decides whether the seam is registered at
all. Everything else is a named source (`register_owner_identity_source`) returning
``{kind: literals}`` plus, optionally, ``never_match``. A source registered with ``kinds``
may return only those; anything else it returns is dropped and reported (a plugin's
account addresses, bound to its manifest). A source that raises contributes nothing,
with a warning, and never breaks a guard.

**Invalidation.** In-process writers call `invalidate_owner_identity()` and the next read
rebuilds. A write from another process is caught by the sources' fingerprints: cheap
probes (file stats, SQLite's change counter) checked at most once per
:data:`FINGERPRINT_INTERVAL_S` seconds, and only the sources whose fingerprint moved are
read again. A source without a fingerprint is read once and then only after an
invalidation.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

from iris_harness.foundation.process_state import track_globals
from iris_harness.kernel.governance.owner_identity import (
    DECLARABLE_KINDS,
    EMPTY,
    NEVER_MATCH,
    OWNER_PII_KINDS,
    IdentityKind,
    OwnerIdentity,
    apply_never_match,
    declared,
    extract,
    merge,
)

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

IdentityTextProvider = Callable[[], Sequence[str]]
# ``{kind: literals}``, and optionally ``{"never_match": strings}``.
OwnerIdentitySource = Callable[[], Mapping[str, Iterable[str]]]
Fingerprint = Callable[[], Hashable]

DOCUMENTS = "documents"
# ADR-0125 decision 4: the cross-process corpus is at most 5 s stale.
FINGERPRINT_INTERVAL_S = 5.0


@dataclass(frozen=True)
class _Source:
    provider: OwnerIdentitySource
    fingerprint: Fingerprint | None
    # The keys this source may return (kinds, and never_match); None = any declarable kind.
    allowed: frozenset[str] | None
    on_undeclared: Callable[[frozenset[str]], None] | None


@dataclass(frozen=True)
class _Part:
    """One source's contribution, and the fingerprint it was read at."""

    identity: OwnerIdentity
    never_match: frozenset[str]
    fingerprint: Hashable


_lock = threading.Lock()
# Serialises rebuilds so concurrent guards do not all read every source at once.
_build_lock = threading.RLock()
_provider: IdentityTextProvider | None = None
_documents_fingerprint: Fingerprint | None = None
_sources: dict[str, _Source] = {}
_parts: dict[str, _Part] = {}
_corpus: OwnerIdentity | None = None
_checked_at: float | None = None
# Bumped by every registration and invalidation: a rebuild that started before one does
# not cache what it read.
_generation = 0
_clock: Callable[[], float] = time.monotonic
_UNREAD = object()  # a fingerprint no probe returns: "read this source"


def _drop_cache() -> None:
    """Forget every source's part and the corpus. Callers hold ``_lock``."""
    global _corpus, _checked_at, _generation
    _parts.clear()
    _corpus = None
    _checked_at = None
    _generation += 1


def _fresh_corpus() -> OwnerIdentity | None:
    """The cached corpus while it is younger than the interval. Callers hold ``_lock``."""
    if _corpus is None or _checked_at is None:
        return None
    return _corpus if _clock() - _checked_at < FINGERPRINT_INTERVAL_S else None


# -- registration --------------------------------------------------------------------


def register_identity_text_provider(
    provider: IdentityTextProvider, *, fingerprint: Fingerprint | None = None
) -> None:
    """Supply the identity documents: the ``documents`` source every guard depends on."""
    global _provider, _documents_fingerprint
    with _lock:
        _provider = provider
        _documents_fingerprint = fingerprint
        _drop_cache()


def clear_identity_text_provider() -> None:
    """Forget the documents provider. For tests; production registers once at import."""
    global _provider, _documents_fingerprint
    with _lock:
        _provider = None
        _documents_fingerprint = None
        _drop_cache()


def has_identity_text_provider() -> bool:
    """Whether anything has supplied the guard with documents to redact against."""
    with _lock:
        return _provider is not None


def register_owner_identity_source(
    name: str,
    provider: OwnerIdentitySource,
    *,
    fingerprint: Fingerprint | None = None,
    kinds: Iterable[IdentityKind] | None = None,
    on_undeclared: Callable[[frozenset[str]], None] | None = None,
) -> None:
    """Add (or replace) the source ``name``.

    ``kinds`` binds the source to what it declared: anything else it returns --
    another kind, or ``never_match`` -- is dropped, logged and passed to
    ``on_undeclared``. A bound source (a plugin's) may declare only the owner's PII kinds
    (``OWNER_PII_KINDS``). Without ``kinds`` (the composition root's own sources) any
    declarable kind -- ``link`` included, for the confirmed ``blog`` and ``website`` -- and
    ``never_match`` are accepted. No source may declare ``secret``.
    """
    if name == DOCUMENTS:
        raise ValueError(f"{DOCUMENTS!r} is the identity documents' source; use another name")
    allowed: frozenset[str] | None = None
    if kinds is not None:
        allowed = frozenset(kinds)
        bad = sorted(allowed - set(OWNER_PII_KINDS))
        if bad:
            raise ValueError(f"identity source {name!r} cannot declare {', '.join(bad)}")
    with _lock:
        _sources[name] = _Source(provider, fingerprint, allowed, on_undeclared)
        _drop_cache()


def unregister_owner_identity_source(name: str) -> None:
    """Remove the source ``name``; unknown names are ignored."""
    with _lock:
        if _sources.pop(name, None) is not None:
            _drop_cache()


def owner_identity_sources() -> tuple[str, ...]:
    """The registered sources, ``documents`` first when it is registered."""
    with _lock:
        return ((DOCUMENTS,) if _provider is not None else ()) + tuple(sorted(_sources))


def set_owner_identity_clock(clock: Callable[[], float] | None) -> None:
    """Replace the monotonic clock the fingerprint interval is measured on (tests)."""
    global _clock
    with _lock:
        _clock = clock or time.monotonic
        _drop_cache()


# -- reading -------------------------------------------------------------------------


def identity_texts() -> list[str] | None:
    """The identity documents, or ``None`` when no provider is registered.

    ``None`` and ``[]`` are different answers on purpose: the first means the guard
    does not know what to redact, the second means there is nothing to redact. Only
    the first is a problem, and only the first is logged.
    """
    with _lock:
        provider = _provider
    if provider is None:
        return None
    try:
        return [text for text in provider() if text]
    except Exception:  # advisory guard; never break an egress check
        logger.warning(
            "identity text provider raised; egress redaction has no corpus", exc_info=True
        )
        return []


def _probe(name: str, fingerprint: Fingerprint | None) -> Hashable:
    """A source's fingerprint now; a failing probe reads as changed."""
    if fingerprint is None:
        return None
    try:
        return fingerprint()
    except Exception:
        logger.debug("owner identity: fingerprint of %r raised; re-reading it", name, exc_info=True)
        return _UNREAD


def _read_documents(fingerprint: Hashable) -> _Part:
    return _Part(extract(identity_texts() or []), frozenset(), fingerprint)


def _read_source(name: str, source: _Source, fingerprint: Hashable) -> _Part:
    try:
        raw = source.provider()
    except Exception:  # one broken source costs its own literals, never the corpus
        logger.warning(
            "owner identity source %r raised; it contributes nothing", name, exc_info=True
        )
        return _Part(EMPTY, frozenset(), fingerprint)
    allowed = (
        source.allowed
        if source.allowed is not None
        else frozenset(DECLARABLE_KINDS) | {NEVER_MATCH}
    )
    dropped = frozenset(str(key) for key in raw if key not in allowed)
    if dropped:
        logger.warning(
            "owner identity source %r returned undeclared %s; dropped",
            name,
            ", ".join(sorted(dropped)),
        )
        if source.on_undeclared is not None:
            try:
                source.on_undeclared(dropped)
            except Exception:
                logger.debug("owner identity: on_undeclared for %r raised", name, exc_info=True)
    kept = {key: values for key, values in raw.items() if key in allowed}
    never = kept.pop(NEVER_MATCH, ())
    if isinstance(never, str):
        never = (never,)
    return _Part(
        declared(kept),
        frozenset(s for s in never if isinstance(s, str)),
        fingerprint,
    )


def owner_identity() -> OwnerIdentity | None:
    """The owner's identity literals, from every source; ``None`` when unregistered.

    ``None`` is not cached, so a documents provider registered later takes effect. Any
    answer from a registered seam is, including the empty one failing sources degrade to,
    until an invalidation or a moved fingerprint (checked at most every
    :data:`FINGERPRINT_INTERVAL_S` seconds).
    """
    global _corpus, _checked_at
    with _lock:
        if _provider is None:
            return None
        cached = _fresh_corpus()
        if cached is not None:
            return cached
    with _build_lock:
        with _lock:
            if _provider is None:
                return None
            cached = _fresh_corpus()
            if cached is not None:
                return cached
            now = _clock()
            generation = _generation
            documents_fp = _documents_fingerprint
            sources = dict(_sources)
            parts = dict(_parts)
            corpus = _corpus
        fresh: dict[str, _Part] = {}
        changed = corpus is None
        for name, fingerprint in [
            (DOCUMENTS, documents_fp),
            *((n, s.fingerprint) for n, s in sources.items()),
        ]:
            part = parts.get(name)
            if part is not None and fingerprint is None:
                fresh[name] = part  # no probe: read again only after an invalidation
                continue
            probe = _probe(name, fingerprint)
            if part is None or probe is _UNREAD or probe != part.fingerprint:
                part = (
                    _read_documents(probe)
                    if name == DOCUMENTS
                    else _read_source(name, sources[name], probe)
                )
                changed = True
            fresh[name] = part
        if changed or corpus is None:
            never = frozenset().union(*(p.never_match for p in fresh.values()))
            corpus = apply_never_match(merge(*(p.identity for p in fresh.values())), never)
        with _lock:
            # A registration or invalidation while we read wins: this caller gets what was
            # built, but it is not cached over the newer state.
            if _generation == generation:
                _parts.clear()
                _parts.update(fresh)
                _corpus = corpus
                _checked_at = now
        return corpus


def invalidate_owner_identity() -> None:
    """A source changed in this process: the next read rebuilds, at once.

    Called by the in-process writers (the fact store, the USER.md writers). Cheap: it
    drops the cache and reads nothing.
    """
    with _lock:
        _drop_cache()


def invalidates_owner_identity(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """Mark ``fn`` as a writer of something a source reads (USER.md, the fact store).

    It invalidates the corpus when ``fn`` returns or raises, whether or not it got as far
    as writing: a stale corpus is the failure, an extra rebuild is not. Another process's
    write is caught by the source's fingerprint instead.
    """

    @functools.wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return fn(*args, **kwargs)
        finally:
            invalidate_owner_identity()

    return wrapper


def reset_owner_identity() -> None:
    """Forget the cached corpus. For tests, and for a late change to the documents."""
    invalidate_owner_identity()


__all__ = [
    "DOCUMENTS",
    "FINGERPRINT_INTERVAL_S",
    "Fingerprint",
    "IdentityTextProvider",
    "OwnerIdentitySource",
    "clear_identity_text_provider",
    "has_identity_text_provider",
    "identity_texts",
    "invalidate_owner_identity",
    "invalidates_owner_identity",
    "owner_identity",
    "owner_identity_sources",
    "register_identity_text_provider",
    "register_owner_identity_source",
    "reset_owner_identity",
    "set_owner_identity_clock",
    "unregister_owner_identity_source",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(
    __name__,
    "_provider",
    "_documents_fingerprint",
    "_sources",
    "_parts",
    "_corpus",
    "_checked_at",
    "_generation",
    "_clock",
)
