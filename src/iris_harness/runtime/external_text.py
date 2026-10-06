"""The external-content tripwire for code that renders a skill tool's text itself.

A skill tool that declares ``content: external`` is scanned by the always-on floor
(``kernel/governance/plugins/external_content_floor.py``) when it runs under the governed
runner. A brief or digest slot, and a chat turn answered directly from a skill, call the
tool's class in-process instead and hand the text to the owner (the digest store, Telegram,
web push, the chat transcript). This is the floor's tripwire for those sinks: the same
``scan`` (one pattern list, never copied), a matched span replaced by the same marker, and a
ledger row in the floor's shape (pattern ids, tool, source; never the text).

No envelope here, on purpose: the envelope is for text a model reads. An owner-facing
channel gets redaction only, so ``<external_content ...>`` never appears in a message or a
digest. The model-bound path (the ``render_<brief>`` tool the loop calls) is declared
``content: external`` on its ToolSpec and is enveloped by the runner.
"""

from __future__ import annotations

from iris_harness.kernel.governance.external_content import redact_text

#: ``caller`` stamped on the ledger row: code, not a model (the floor's own vocabulary).
_CALLER = "core:skill_render"


def redact_external_text(text: str, *, skill: str, tool: str) -> str:
    """``text`` with instruction-like spans replaced by the floor's marker.

    A skill-flavoured call of the kernel's one tripwire-only helper
    (``kernel.governance.external_content.redact_text``): unchanged (the very same string)
    when the floor is off or nothing matched; a match is logged at WARNING and written to the
    governance ledger as a ``post_tool_use`` row of the ``external_content_floor`` plugin,
    neither carrying the text.
    """
    return redact_text(text, source=f"skill:{skill}", tool=tool, caller=_CALLER)


__all__ = ["redact_external_text"]
