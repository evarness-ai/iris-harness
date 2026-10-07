"""Marking text a third party wrote before a plugin's own model reads it.

A tool declared ``content: external`` already reaches the loop's model marked and scanned
by the kernel's external-content floor. A plugin that takes such text through ``api.tools``
(or reads a page, a mailbox or a feed itself) and puts it into a prompt of its own gets the
tripwire redaction from the floor but no envelope, because plugin code may also show the
text to the owner. :func:`wrap_external_content` applies the same two things on request:
the tripwire (instruction-like spans replaced by a visible marker) and the
``<external_content ... trust="untrusted">`` envelope. It is the kernel's own
implementation (``kernel/governance/external_content.py``), not a copy, and runs offline.

:func:`redact_external_content` is the tripwire without the envelope, for text a plugin
shows the owner, returns to a channel or logs rather than hands to a model. It honours the
external-content floor setting and writes a counts-only ledger row for the first few matching calls per scope.

    from iris_harness.sdk.content import wrap_external_content

    prompt = f"Summarise this:\n{wrap_external_content(page_text, source='my_plugin')}"
"""

from __future__ import annotations

import contextvars
import logging

from iris_harness.kernel.governance.external_content import (
    floor_enabled,
    redact_text,
    scan,
    wrap_scanned,
)

logger = logging.getLogger(__name__)


def wrap_external_content(text: str, *, source: str, tool: str | None = None) -> str:
    """``text`` with instruction-like spans redacted, inside the untrusted-content envelope.

    ``source`` names where the text came from (your plugin, a site, a mailbox); ``tool``
    the tool or step that fetched it (defaults to ``source``). Never wrapped twice: text
    that is already in an envelope is unwrapped first, scanned again and wrapped once more,
    so the result carries ONE envelope whose ``source`` and ``tool`` are the ones given
    here (an envelope's own claimed source never survives). A literal closing tag inside
    ``text`` is escaped, so the text cannot end the envelope early. No model, no network.
    """
    return wrap_scanned(text, source=source, tool=tool or source)


#: Ledger rows one scope (a run, or a session when nobody opens one) may write through
#: :func:`redact_external_content`; past it the text is still redacted, just not recorded.
_AUDIT_ROWS_PER_SCOPE = 5
# [session id, rows written, cap warning given]
_BUDGET: contextvars.ContextVar[list[object] | None] = contextvars.ContextVar(
    "sdk_content_audit_budget", default=None
)


def _reset_audit_budget() -> None:
    """Open a fresh row budget (a plugin calls this at the start of a run). Not public."""
    _BUDGET.set(None)


def _audit_budget() -> list[object]:
    from iris_harness.foundation.observability.session_log import current_session_id

    key = current_session_id()
    state = _BUDGET.get()
    if state is None or state[0] != key:
        state = [key, 0, False]
        _BUDGET.set(state)
    return state


def redact_external_content(text: str) -> str:
    """``text`` with instruction-like spans redacted, and no envelope.

    The tripwire alone, for text a plugin keeps for itself or shows the owner (a result it
    returns to a channel, a note it stores) rather than hands to a model: the floor's own
    scan, through the kernel's one tripwire-only helper. It honours
    ``IRIS_GOVERNANCE_EXTERNAL_CONTENT_FLOOR``: with the floor off the very same string
    comes back, untouched. A call that matches writes a counts-only ledger row (pattern
    ids and counts, never the text) and a warning, but only the first few per scope (per
    run for a plugin that opens one, else per session): later matches are still redacted
    and only counted in one warning, so a plugin that calls this per chunk cannot flood the
    ledger. Idempotent. No model, no network.
    """
    if not text:
        return text
    state = _audit_budget()
    rows = state[1]
    if isinstance(rows, int) and rows < _AUDIT_ROWS_PER_SCOPE:
        out = redact_text(text, source="sdk:plugin", caller="plugin")
        if out is not text:
            state[1] = rows + 1
        return out
    if not floor_enabled():
        return text
    found = scan(text)
    if not found.matched:
        return text
    if not state[2]:
        state[2] = True
        logger.warning(
            "redact_external_content: further spans redacted; ledger rows capped at %d per scope",
            _AUDIT_ROWS_PER_SCOPE,
        )
    return found.text


__all__ = ["redact_external_content", "wrap_external_content"]
