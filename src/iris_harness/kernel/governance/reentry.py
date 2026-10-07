"""Scan stored text on the way back into a prompt (issue #145, step one).

The external-content floor (``plugins/external_content_floor.py``) screens a tool result
the moment it arrives. It cannot see the same words later: the model restates a tool's
output in its answer, the turn is stored, and the stored text comes back into a prompt
through the history window, the session summary, ``recall_conversation`` and
``memory_search``. This module is the second look, at that re-entry.

It is a pure function plus an audit recorder, not a kernel hook: the readers are synchronous
code inside ``memory/`` and tool bodies, ``chat_stream`` runs inside an event loop (where
``fire_sync`` refuses), and a new hook point would change the stable hook vocabulary.

What it does, and does not:

- It reuses :func:`external_content.scan` (the floor's tripwire; the patterns are not
  copied) and the floor's switch (:func:`external_content.floor_enabled`). There is no new
  setting: floor off means no scan here either.
- It scans ``assistant`` turns, ``summary`` text and learned memory (``memory``: the identity
  files ``active.md`` / ``episodic.md``, a lesson's body, #163) only. A ``user`` turn is the owner's own
  words and comes back untouched (an owner's note that says "ignore previous instructions"
  must not be rewritten).
- It redacts, and marks only what is known to be third-party. A turn the loop recorded as
  ``external`` (the run read a ``content: external`` tool's text before answering,
  ``turn_origin`` on the stored row, #145 step two) also comes back inside the
  untrusted-content envelope; every other assistant turn (``internal`` or unknown, which is
  every row written before the column) is scanned as before and not enveloped. Internal is
  not safe: a model's restatement of an internal-declared tool's result is stored as it was
  said, so the scan stays on all of them.
- It is phrase-level. A paraphrase, another language, a homoglyph or an LLM-written summary
  that rewords an instruction passes: the tripwire is thirteen phrase patterns.
- Stored rows are never altered; the redaction is on the copy that goes to the prompt.

Limits: a text over :data:`MAX_ITEM_CHARS` is cut and a visible marker says so; a call that
has used :data:`MAX_CALL_CHARS` replaces the remaining (older) texts with the marker. An
unscanned tail is never passed on.

Audit: one ``audit_log`` row per helper call, written only when something matched or a
limit was hit, with counts and pattern ids and never the text. A failing recorder never
changes the outcome: the redaction still applies and a warning is logged.

The same poisoned turn sitting in the history window is read on every later turn (about two
reads a turn), so a row per read would write a hundred rows for one text (issue #164). Reads
are deduplicated per ``(session, reader, origin, hashes of the matched or capped texts)``,
in this process: the first sighting writes a row, and a later row is written only at the
2nd, 4th, 8th, 16th... sighting, each carrying ``sightings`` (the running count), so the
ledger stays append-only and still shows how often the text was read. A different poisoned
text is a different key, so it always gets its own row; a restart resets the counts.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from iris_harness.kernel.governance.external_content import MARKER, floor_enabled, scan, wrap

logger = logging.getLogger(__name__)

#: The longest text scanned whole; more is cut.
MAX_ITEM_CHARS = 16 * 1024
#: The most characters one reader call scans; the newest texts are scanned first.
MAX_CALL_CHARS = 128 * 1024

CUT_MARKER = "[not scanned: text over the re-entry limit was cut]"
SPENT_MARKER = "[not scanned: this read passed the re-entry limit]"
#: What replaces a matched span in stored text. The floor's own marker says "external content",
#: which is wrong for the model's own earlier turns, so re-entry words it for stored content.
REENTRY_MARKER = "[redacted: instruction-like text in stored content]"

#: ``audit_log.hook_point`` / ``plugin`` of the rows this module writes.
AUDIT_HOOK_POINT = "reentry_scan"
AUDIT_PLUGIN = "reentry"

#: Roles that come back verbatim: the owner's own words.
FIRST_PARTY_ROLES = frozenset({"user"})

#: ``role`` for learned memory read back into a prompt (``active.md``, ``episodic.md``, a
#: lesson): not the owner's own words, so it is scanned (issue #163).
MEMORY_ROLE = "memory"

#: ``turn_origin`` of a stored assistant turn whose run read third-party text (#145 step two).
EXTERNAL_TURN = "external"
#: ``source`` of the envelope a stored external-origin turn comes back in.
ENVELOPE_SOURCE = "stored_transcript"
#: ``source`` of the envelope learned memory comes back in when the scan redacted it (#163).
MEMORY_ENVELOPE_SOURCE = "learned_memory"

_MEMO_MAX = 2048
#: How many ``(session, reader, origin, texts)`` keys the sighting counter remembers; the
#: oldest are forgotten first (their next sighting counts from 1 again, so it writes a row).
_SIGHTINGS_MAX = 4096


@dataclass(frozen=True)
class Reentry:
    """One text after the re-entry scan."""

    text: str
    ids: tuple[str, ...] = ()
    spans: int = 0
    chars: int = 0  # characters actually scanned (0 for a passed-through or refused text)
    capped: bool = False
    cached: bool = False
    enveloped: bool = False  # came back inside the untrusted-content envelope (external origin)
    digest: str = ""  # hash of the text read (empty for a passed-through text), for the dedupe key


@dataclass(frozen=True)
class ReentryAudit:
    """What one helper call did, as counts. Never carries text."""

    reader: str
    origin: str
    role_counts: dict[str, int]
    items: int
    chars_scanned: int
    spans: int
    ids: tuple[str, ...]
    capped_items: int
    cache_hits: int
    enveloped_items: int = 0
    sightings: int = 1  # how many times this same set of texts was read (#164)

    def as_payload(self) -> dict[str, Any]:
        return {
            "reader": self.reader,
            "origin": self.origin,
            "role_counts": dict(self.role_counts),
            "items": self.items,
            "chars_scanned": self.chars_scanned,
            "spans": self.spans,
            "patterns": list(self.ids),
            "capped_items": self.capped_items,
            "cache_hits": self.cache_hits,
            "enveloped_items": self.enveloped_items,
            "sightings": self.sightings,
        }


Recorder = Callable[[ReentryAudit], None]

_lock = threading.Lock()
_memo: OrderedDict[str, tuple[str, tuple[str, ...], int]] = OrderedDict()
_recorder: Recorder | None = None
_sightings: OrderedDict[tuple[str, str, str, tuple[str, ...]], int] = OrderedDict()


def set_reentry_recorder(recorder: Recorder | None) -> None:
    """Install (or, with ``None``, remove) the sink that audits a match or a cap hit."""
    global _recorder
    _recorder = recorder


def audit_recorder(audit_log: Any) -> Recorder:
    """A recorder that writes to ``audit_log`` (an ``AuditLog``), stamped like a hook row.

    The session and trace ids are read when the row is written, so they are the turn's.
    """

    def record(event: ReentryAudit) -> None:
        from iris_harness.foundation.observability.session_log import current_session_id
        from iris_harness.kernel.governance.kernel import _current_trace_id_hex

        payload = event.as_payload()
        trace_id = _current_trace_id_hex()
        if trace_id is not None:
            payload["trace_id"] = trace_id
        session_id = current_session_id()
        if session_id is not None:
            payload["session_id"] = session_id
        audit_log.record(
            run_id=session_id or "reentry",
            step_id=None,
            agent_type="reentry",
            hook_point=AUDIT_HOOK_POINT,
            plugin=AUDIT_PLUGIN,
            decision="transform" if event.spans else "allow",
            severity="warn" if event.spans else "info",
            reason=(
                f"reentry: redacted {event.spans} instruction-like span(s) in stored text"
                if event.spans
                else "reentry: stored text over the scan limit was cut"
            ),
            payload=payload,
        )

    return record


def _clear_memo() -> None:
    with _lock:
        _memo.clear()
        _sightings.clear()


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).hexdigest()


def _scan_cached(text: str) -> tuple[str, tuple[str, ...], int, bool, str]:
    key = _digest(text)
    with _lock:
        hit = _memo.get(key)
        if hit is not None:
            _memo.move_to_end(key)
            return hit[0], hit[1], hit[2], True, key
    result = scan(text)
    redacted = result.text.replace(MARKER, REENTRY_MARKER) if result.spans else result.text
    with _lock:
        _memo[key] = (redacted, result.ids, result.spans)
        if len(_memo) > _MEMO_MAX:
            _memo.popitem(last=False)
    return redacted, result.ids, result.spans, False, key


def _one(
    text: str,
    role: str,
    remaining: int | None,
    turn_origin: str | None = None,
    reader: str = "",
    limit: int | None = None,
    envelope_if_redacted: bool = False,
) -> tuple[Reentry, int | None]:
    """Scan one text under the per-text cap and the call's remaining characters; an
    external-origin turn also comes back inside the envelope."""
    if role in FIRST_PARTY_ROLES or not text:
        return Reentry(text), remaining
    capped = False
    if len(text) > MAX_ITEM_CHARS:
        text, capped = text[:MAX_ITEM_CHARS], True
    if remaining is not None:
        if remaining <= 0:
            return Reentry(SPENT_MARKER, capped=True, digest=_digest(text)), remaining
        if len(text) > remaining:
            text, capped = text[:remaining], True
        remaining -= len(text)
    scanned, ids, spans, cached, digest = _scan_cached(text)
    if capped:
        scanned += CUT_MARKER
    enveloped = turn_origin == EXTERNAL_TURN or (envelope_if_redacted and spans > 0)
    if enveloped:
        if limit is not None:
            # The reader will show a one-line excerpt: cut the TEXT before the envelope goes
            # round it, so the closing tag is never the part that is cut off.
            scanned = " ".join(scanned.split())[:limit]
        scanned = wrap(
            scanned,
            source=MEMORY_ENVELOPE_SOURCE if role == MEMORY_ROLE else ENVELOPE_SOURCE,
            tool=reader or "transcript",
        )
    return Reentry(scanned, ids, spans, len(text), capped, cached, enveloped, digest), remaining


def _sighting(reader: str, origin: str, results: Sequence[tuple[str, Reentry]]) -> int:
    """How many times this read of these texts has now happened in this process (1 = first).

    The key is the session, the reader, the origin and the sorted hashes of the texts that
    matched or were cut, so one poisoned turn read again and again counts up, while a
    different poisoned text (a different hash) starts its own count and is never hidden by
    another's.
    """
    from iris_harness.foundation.observability.session_log import current_session_id

    hashes = tuple(sorted({r.digest for _, r in results if r.spans or r.capped}))
    key = (current_session_id() or "", reader, origin, hashes)
    with _lock:
        count = _sightings.get(key, 0) + 1
        _sightings[key] = count
        _sightings.move_to_end(key)
        if len(_sightings) > _SIGHTINGS_MAX:
            _sightings.popitem(last=False)
    return count


def _emit(reader: str, origin: str, results: Sequence[tuple[str, Reentry]]) -> None:
    """Audit one call, when something matched or a limit was hit. Never raises."""
    spans = sum(r.spans for _, r in results)
    capped = sum(1 for _, r in results if r.capped)
    if spans == 0 and capped == 0:
        return
    recorder = _recorder
    if recorder is None:
        return
    sightings = _sighting(reader, origin, results)
    if sightings & (sightings - 1):  # not a power of two: counted, not written
        return
    roles: dict[str, int] = {}
    ids: list[str] = []
    for role, r in results:
        roles[role] = roles.get(role, 0) + 1
        ids.extend(i for i in r.ids if i not in ids)
    event = ReentryAudit(
        reader=reader,
        origin=origin,
        role_counts=roles,
        items=len(results),
        chars_scanned=sum(r.chars for _, r in results),
        spans=spans,
        ids=tuple(ids),
        capped_items=capped,
        cache_hits=sum(1 for _, r in results if r.cached),
        enveloped_items=sum(1 for _, r in results if r.enveloped),
        sightings=sightings,
    )
    try:
        recorder(event)
    except Exception as exc:  # noqa: BLE001 - an audit failure must not undo the redaction
        logger.warning("reentry: audit write failed for %s (%s): %s", reader, origin, exc)


def reenter_text(
    text: str,
    *,
    reader: str,
    origin: str,
    role: str,
    budget: int | None = None,
    turn_origin: str | None = None,
    envelope_if_redacted: bool = False,
) -> Reentry:
    """``text`` as it should enter a prompt: a ``user`` text unchanged, anything else scanned.

    ``budget`` is the characters this call may scan (default :data:`MAX_CALL_CHARS`).
    ``reader`` names the code reading the text, ``origin`` where it was stored from
    (``"transcript"``, ``"summary"``...); both go to the audit row, with counts only.

    ``envelope_if_redacted`` is for learned memory read back into a prompt (``active.md``,
    ``episodic.md``, a lesson's body: issue #163): text the owner approved is scanned and
    otherwise comes back as it is, but a text the scan had to redact is known to have taken
    third-party words in, so it also comes back inside the untrusted-content envelope.
    """
    if not floor_enabled():
        return Reentry(text)
    result, _ = _one(
        text,
        role,
        MAX_CALL_CHARS if budget is None else budget,
        turn_origin,
        reader,
        envelope_if_redacted=envelope_if_redacted,
    )
    _emit(reader, origin, [(role, result)])
    return result


def reenter_many(
    items: Sequence[tuple[str, str]],
    *,
    reader: str,
    origin: str,
    chronological: bool = True,
    origins: Sequence[str | None] | None = None,
    limit: int | None = None,
    envelope_if_redacted: bool = False,
) -> list[Reentry]:
    """:func:`reenter_text` over ``(role, text)`` pairs, one call budget for all of them.

    ``chronological=True`` (oldest first, a history window) spends the budget from the
    newest end; ``False`` spends it in list order (a ranked result). The result is in the
    order of ``items``. One audit row for the whole call.

    ``origins`` is the stored ``turn_origin`` of each item, in the same order (None for an
    item whose origin is unknown, and when omitted): an ``external`` one comes back inside
    the envelope, as well as scanned. ``limit`` is the length of the one-line excerpt the
    reader will show (see :func:`one_line`): an enveloped text is cut to it BEFORE it is
    wrapped.
    """
    if not floor_enabled():
        return [Reentry(text) for _, text in items]
    out: list[Reentry | None] = [None] * len(items)
    remaining: int | None = MAX_CALL_CHARS
    order = range(len(items) - 1, -1, -1) if chronological else range(len(items))
    for i in order:
        role, text = items[i]
        out[i], remaining = _one(
            text,
            role,
            remaining,
            origins[i] if origins is not None else None,
            reader,
            limit,
            envelope_if_redacted,
        )
    results = [r for r in out if r is not None]
    _emit(reader, origin, [(items[i][0], r) for i, r in enumerate(results)])
    return results


def reenter_memory(text: str, reader: str) -> str:
    """Learned memory (``active.md``, ``episodic.md``, a lesson, an episodic pattern) on its way
    into a prompt or a tool result: scanned, and enveloped when it had to be redacted (#163).

    The owner approved this text, so it comes back as it is unless the scan finds something;
    a lesson is a recipe the model is meant to follow, which is why it is not enveloped by
    default. An empty text comes back empty.
    """
    if not text:
        return text
    return reenter_text(
        text,
        reader=reader,
        origin="learned_memory",
        role=MEMORY_ROLE,
        envelope_if_redacted=True,
    ).text


def reenter_memory_lines(lines: Sequence[str], reader: str) -> tuple[str, ...]:
    """:func:`reenter_memory` over several texts, with one audit row for all of them."""
    results = reenter_many(
        [(MEMORY_ROLE, line) for line in lines],
        reader=reader,
        origin="learned_memory",
        chronological=False,
        envelope_if_redacted=True,
    )
    return tuple(r.text for r in results)


def one_line(result: Reentry, limit: int) -> str:
    """``result`` as one line of at most ``limit`` characters, for a reader that shows an
    excerpt. An enveloped text was already cut to ``limit`` inside its envelope (pass the same
    ``limit`` to :func:`reenter_many`), so it is never cut again: that would drop the closing
    tag. Any other text is collapsed to one line and cut."""
    if result.enveloped:
        return result.text
    return " ".join(result.text.split())[:limit]


__all__ = [
    "AUDIT_HOOK_POINT",
    "CUT_MARKER",
    "REENTRY_MARKER",
    "ENVELOPE_SOURCE",
    "EXTERNAL_TURN",
    "MAX_CALL_CHARS",
    "MAX_ITEM_CHARS",
    "MEMORY_ROLE",
    "SPENT_MARKER",
    "Reentry",
    "ReentryAudit",
    "audit_recorder",
    "one_line",
    "reenter_many",
    "reenter_memory",
    "reenter_memory_lines",
    "reenter_text",
    "set_reentry_recorder",
]
