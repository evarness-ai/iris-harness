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
- It scans ``assistant`` turns and ``summary`` text only. A ``user`` turn is the owner's own
  words and comes back untouched (an owner's note that says "ignore previous instructions"
  must not be rewritten).
- It redacts and does not mark. There is no envelope in this step: marking turns needs to
  know which turns came from third-party text, which is not recorded yet.
- It is phrase-level. A paraphrase, another language, a homoglyph or an LLM-written summary
  that rewords an instruction passes: the tripwire is thirteen phrase patterns.
- Stored rows are never altered; the redaction is on the copy that goes to the prompt.

Limits: a text over :data:`MAX_ITEM_CHARS` is cut and a visible marker says so; a call that
has used :data:`MAX_CALL_CHARS` replaces the remaining (older) texts with the marker. An
unscanned tail is never passed on.

Audit: one ``audit_log`` row per helper call, written only when something matched or a
limit was hit, with counts and pattern ids and never the text. A failing recorder never
changes the outcome: the redaction still applies and a warning is logged.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from iris_harness.kernel.governance.external_content import floor_enabled, scan

logger = logging.getLogger(__name__)

#: The longest text scanned whole; more is cut.
MAX_ITEM_CHARS = 16 * 1024
#: The most characters one reader call scans; the newest texts are scanned first.
MAX_CALL_CHARS = 128 * 1024

CUT_MARKER = "[not scanned: text over the re-entry limit was cut]"
SPENT_MARKER = "[not scanned: this read passed the re-entry limit]"

#: ``audit_log.hook_point`` / ``plugin`` of the rows this module writes.
AUDIT_HOOK_POINT = "reentry_scan"
AUDIT_PLUGIN = "reentry"

#: Roles that come back verbatim: the owner's own words.
FIRST_PARTY_ROLES = frozenset({"user"})

_MEMO_MAX = 2048


@dataclass(frozen=True)
class Reentry:
    """One text after the re-entry scan."""

    text: str
    ids: tuple[str, ...] = ()
    spans: int = 0
    chars: int = 0  # characters actually scanned (0 for a passed-through or refused text)
    capped: bool = False
    cached: bool = False


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
        }


Recorder = Callable[[ReentryAudit], None]

_lock = threading.Lock()
_memo: OrderedDict[str, tuple[str, tuple[str, ...], int]] = OrderedDict()
_recorder: Recorder | None = None


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


def _scan_cached(text: str) -> tuple[str, tuple[str, ...], int, bool]:
    key = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).hexdigest()
    with _lock:
        hit = _memo.get(key)
        if hit is not None:
            _memo.move_to_end(key)
            return hit[0], hit[1], hit[2], True
    result = scan(text)
    with _lock:
        _memo[key] = (result.text, result.ids, result.spans)
        if len(_memo) > _MEMO_MAX:
            _memo.popitem(last=False)
    return result.text, result.ids, result.spans, False


def _one(text: str, role: str, remaining: int | None) -> tuple[Reentry, int | None]:
    """Scan one text under the per-text cap and the call's remaining characters."""
    if role in FIRST_PARTY_ROLES or not text:
        return Reentry(text), remaining
    capped = False
    if len(text) > MAX_ITEM_CHARS:
        text, capped = text[:MAX_ITEM_CHARS], True
    if remaining is not None:
        if remaining <= 0:
            return Reentry(SPENT_MARKER, capped=True), remaining
        if len(text) > remaining:
            text, capped = text[:remaining], True
        remaining -= len(text)
    scanned, ids, spans, cached = _scan_cached(text)
    if capped:
        scanned += CUT_MARKER
    return Reentry(scanned, ids, spans, len(text), capped, cached), remaining


def _emit(reader: str, origin: str, results: Sequence[tuple[str, Reentry]]) -> None:
    """Audit one call, when something matched or a limit was hit. Never raises."""
    spans = sum(r.spans for _, r in results)
    capped = sum(1 for _, r in results if r.capped)
    if spans == 0 and capped == 0:
        return
    recorder = _recorder
    if recorder is None:
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
) -> Reentry:
    """``text`` as it should enter a prompt: a ``user`` text unchanged, anything else scanned.

    ``budget`` is the characters this call may scan (default :data:`MAX_CALL_CHARS`).
    ``reader`` names the code reading the text, ``origin`` where it was stored from
    (``"transcript"``, ``"summary"``...); both go to the audit row, with counts only.
    """
    if not floor_enabled():
        return Reentry(text)
    result, _ = _one(text, role, MAX_CALL_CHARS if budget is None else budget)
    _emit(reader, origin, [(role, result)])
    return result


def reenter_many(
    items: Sequence[tuple[str, str]],
    *,
    reader: str,
    origin: str,
    chronological: bool = True,
) -> list[Reentry]:
    """:func:`reenter_text` over ``(role, text)`` pairs, one call budget for all of them.

    ``chronological=True`` (oldest first, a history window) spends the budget from the
    newest end; ``False`` spends it in list order (a ranked result). The result is in the
    order of ``items``. One audit row for the whole call.
    """
    if not floor_enabled():
        return [Reentry(text) for _, text in items]
    out: list[Reentry | None] = [None] * len(items)
    remaining: int | None = MAX_CALL_CHARS
    order = range(len(items) - 1, -1, -1) if chronological else range(len(items))
    for i in order:
        role, text = items[i]
        out[i], remaining = _one(text, role, remaining)
    results = [r for r in out if r is not None]
    _emit(reader, origin, [(items[i][0], r) for i, r in enumerate(results)])
    return results


__all__ = [
    "AUDIT_HOOK_POINT",
    "CUT_MARKER",
    "MAX_CALL_CHARS",
    "MAX_ITEM_CHARS",
    "SPENT_MARKER",
    "Reentry",
    "ReentryAudit",
    "audit_recorder",
    "reenter_many",
    "reenter_text",
    "set_reentry_recorder",
]
