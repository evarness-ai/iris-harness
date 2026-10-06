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

import logging

from iris_harness.kernel.governance.external_content import floor_enabled, scan

logger = logging.getLogger(__name__)

#: ``caller`` stamped on the ledger row: code, not a model (the floor's own vocabulary).
_CALLER = "core:skill_render"


def redact_external_text(text: str, *, skill: str, tool: str) -> str:
    """``text`` with instruction-like spans replaced by the floor's marker.

    Unchanged (the very same string) when the floor is off or nothing matched. A match is
    logged at WARNING and written to the governance ledger as a ``post_tool_use`` row of
    the ``external_content_floor`` plugin; neither carries the text.
    """
    if not text or not floor_enabled():
        return text
    found = scan(text)
    if not found.matched:
        return text
    source = f"skill:{skill}"
    logger.warning(
        "external_content_floor: redacted %d span(s) (%s) in a result from %s",
        found.spans,
        ",".join(found.ids),
        tool,
    )
    _audit(tool=tool, source=source, ids=found.ids, spans=found.spans)
    return found.text


def _audit(*, tool: str, source: str, ids: tuple[str, ...], spans: int) -> None:
    """One ledger row, in the shape the floor hook writes. Never raises."""
    try:
        from iris_harness.foundation.observability.session_log import current_session_id
        from iris_harness.kernel.governance.audit.log import AuditLog

        payload: dict[str, object] = {
            "tool": tool,
            "source": source,
            "patterns": list(ids),
            "spans": spans,
            "marked": False,
            "caller": _CALLER,
        }
        session_id = current_session_id()
        if session_id is not None:
            payload["session_id"] = session_id
        AuditLog().record(
            run_id=session_id or "skill_render",
            step_id=None,
            agent_type="core",
            hook_point="post_tool_use",
            plugin="external_content_floor",
            decision="transform",
            severity="warn",
            reason=f"external_content_floor: redacted {spans} instruction-like span(s)",
            payload=payload,
        )
    except Exception:  # noqa: BLE001 - an audit write never breaks the render
        logger.warning("external_content_floor: could not write the ledger row", exc_info=False)


__all__ = ["redact_external_text"]
