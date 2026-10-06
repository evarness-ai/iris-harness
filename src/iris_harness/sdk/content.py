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
external-content floor setting and writes one counts-only ledger row per matching call.

    from iris_harness.sdk.content import wrap_external_content

    prompt = f"Summarise this:\n{wrap_external_content(page_text, source='my_plugin')}"
"""

from __future__ import annotations

from iris_harness.kernel.governance.external_content import (
    ENVELOPE_TAG,
    redact_text,
    scan,
    unwrap,
    wrap,
)


def wrap_external_content(text: str, *, source: str, tool: str | None = None) -> str:
    """``text`` with instruction-like spans redacted, inside the untrusted-content envelope.

    ``source`` names where the text came from (your plugin, a site, a mailbox); ``tool``
    the tool or step that fetched it (defaults to ``source``). Never wrapped twice: text
    that is already in an envelope is unwrapped first, scanned again and wrapped once more,
    so the result carries ONE envelope whose ``source`` and ``tool`` are the ones given
    here (an envelope's own claimed source never survives). A literal closing tag inside
    ``text`` is escaped, so the text cannot end the envelope early. No model, no network.
    """
    bare = unwrap(text) if text.lstrip().startswith(f"<{ENVELOPE_TAG} ") else text
    return wrap(scan(bare).text, source=source, tool=tool or source)


def redact_external_content(text: str) -> str:
    """``text`` with instruction-like spans redacted, and no envelope.

    The tripwire alone, for text a plugin keeps for itself or shows the owner (a result it
    returns to a channel, a note it stores) rather than hands to a model: the floor's own
    scan, through the kernel's one tripwire-only helper. It honours
    ``IRIS_GOVERNANCE_EXTERNAL_CONTENT_FLOOR``: with the floor off the very same string
    comes back, untouched. A call that matches writes ONE counts-only ledger row (pattern
    ids and counts, never the text) and a warning, so a plugin that calls it per chunk or
    per line should batch (see the ``code_exec`` plugin). Idempotent. No model, no network.
    """
    return redact_text(text, source="sdk:plugin", caller="plugin")


__all__ = ["redact_external_content", "wrap_external_content"]
