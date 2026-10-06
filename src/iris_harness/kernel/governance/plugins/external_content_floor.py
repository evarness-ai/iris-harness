"""ExternalContentFloorHook -- ``PostToolUse``: the always-on floor under the model guard.

A tool, capability or MCP result that declares ``content: external`` is text a third party
wrote. The model guard (``prompt_guard.PromptGuardRetrievedHook``) is opt-in, needs
classifier weights, and passes the text through unchanged when they are absent. This hook
needs neither, so on a default install an external result is still:

- scanned by the deterministic tripwire (``kernel/governance/external_content.py``): a
  matched span is replaced with a visible marker and the ledger row names the pattern ids,
  the tool and its source -- never the text;
- handed to the model inside an ``<external_content ... trust="untrusted">`` envelope, so
  it reads as data with a named source.

Runs at priority 46, after the model guard (45): it sees the text the guard left, and the
envelope goes on last, so the guard never scores (or redacts) the envelope's own tag. It
is a ``PostToolUse`` hook, so every path that fires that hook -- the agent loop,
``api.tools``, ``iris mcp serve``, the MCP bridge, capability calls -- gets both from here.

What gets the envelope is text that goes to a model. A capability result is a typed value
read by plugin code, redacted field by field (the field map is what the consumer
receives), so it gets the tripwire and not the envelope: wrapping a number or a place name
would break the type. The consumer's own tool that hands that text on declares
``content: external`` itself and is wrapped there. A tool that raised is not wrapped (its
message is the harness's, and the loop reads its ``Error:`` prefix); it is still scanned.

A call by plugin or core *code* (``plugin:<name>`` / ``core:<workflow>``, through
``api.tools``) gets the tripwire and not the envelope: its caller is code that may show the
text to the owner or parse it (the email agent's fallback answers with a tool's output), and
markup there would leak into the answer. The marker is for text a model reads (the loop's
``model:<agent>`` calls and an MCP client's ``mcp:<client>`` calls); code that hands an
external result to a model itself marks it with ``external_content.wrap``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from iris_harness.kernel.governance.external_content import clean_label, scan, wrap
from iris_harness.kernel.governance.external_content_allow import allowed_ids, scope_source
from iris_harness.kernel.governance.hooks.tool_payload import (
    RESULT,
    TOOL_ERROR,
    TOOL_PLUGIN,
    is_external,
    result_of,
    tool_name_of,
)
from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint

logger = logging.getLogger(__name__)

#: ``tool_plugin`` of a tool the core provides; it names no source worth showing.
_SYSTEM = "system"

#: Callers that are code, not a model (``tool_service``: ``plugin:<name>``, ``core:<workflow>``).
_CODE_CALLERS = ("plugin:", "core:")

#: Keys whose string value, inside a structured result, is text a model reads.
_TEXT_KEYS = frozenset({"text", "content", "output", "result", "body"})


class ExternalContentFloorHook:
    """Deterministic tripwire and untrusted-content envelope for external results."""

    name: str = "external_content_floor"
    hook_point: HookPoint = HookPoint.POST_TOOL_USE
    priority: int = 46  # after the model guard (45)

    async def __call__(self, ctx: HookContext) -> HookDecision:
        tool = tool_name_of(ctx.payload)
        if not tool or not is_external(ctx.metadata):
            return HookDecision(
                outcome="allow", reason="external_content_floor: not external content"
            )
        source = _source(ctx.payload, tool)
        # Labels reach the log and the audit row: an MCP-supplied tool name must not carry a newline.
        audit: dict[str, Any] = {"tool": clean_label(tool), "source": clean_label(source)}
        # The patterns the owner allowed for this call's scope (issue #139): the namespaced
        # source the harness stamped (never a name a plugin chose), and the tool.
        allow = allowed_ids(scope_source(ctx.payload.get(TOOL_PLUGIN), tool), tool)

        fields = ctx.payload.get("fields")
        if isinstance(fields, dict) and fields:
            return self._typed(ctx, fields, audit, allow)

        result = result_of(ctx.payload)
        caller = ctx.metadata.get("caller")
        # No envelope for a tool that raised, nor for code that is not a model (see above).
        wrapped = ctx.metadata.get(TOOL_ERROR) is None and not (
            isinstance(caller, str) and caller.startswith(_CODE_CALLERS)
        )
        if isinstance(result, str):
            return self._text(ctx, result, audit, wrap_it=wrapped, allow=allow)
        if isinstance(result, (dict, list)):
            return self._structured(ctx, result, audit, wrap_it=wrapped, allow=allow)
        return HookDecision(outcome="allow", reason="external_content_floor: no text result")

    # -- a capability result: typed fields, redacted field by field ------------------------
    def _typed(
        self,
        ctx: HookContext,
        fields: dict[Any, Any],
        audit: dict[str, Any],
        allow: frozenset[str],
    ) -> HookDecision:
        new_fields: dict[Any, Any] = {}
        ids: list[str] = []
        used: list[str] = []
        spans = 0
        for path, text in fields.items():
            if not isinstance(text, str):
                new_fields[path] = text
                continue
            found = scan(text, allow=allow)
            new_fields[path] = found.text
            spans += found.spans
            ids.extend(i for i in found.ids if i not in ids)
            used.extend(i for i in found.allowed if i not in used)
        audit = _with_allowed(audit, used)
        if spans == 0:
            return HookDecision(
                outcome="allow",
                reason="external_content_floor: no instruction-like text (typed fields)",
                audit_metadata={**audit, "marked": False, "spans": 0},
            )
        joined = "\n".join(str(v) for v in new_fields.values())
        payload = {**ctx.payload, "fields": new_fields, RESULT: joined}
        return _redacted(payload, audit, ids, spans, marked=False)

    # -- a tool's text result --------------------------------------------------------------
    def _text(
        self,
        ctx: HookContext,
        text: str,
        audit: dict[str, Any],
        *,
        wrap_it: bool,
        allow: frozenset[str],
    ) -> HookDecision:
        if not text.strip():
            return HookDecision(outcome="allow", reason="external_content_floor: no text")
        found = scan(text, allow=allow)
        audit = _with_allowed(audit, found.allowed)
        out = (
            wrap(found.text, source=audit["source"], tool=audit["tool"]) if wrap_it else found.text
        )
        if not found.matched and not wrap_it:
            return HookDecision(
                outcome="allow",
                reason="external_content_floor: no instruction-like text",
                audit_metadata={**audit, "marked": False, "spans": 0},
            )
        payload = {**ctx.payload, RESULT: out}
        if found.matched:
            return _redacted(payload, audit, list(found.ids), found.spans, marked=wrap_it)
        return HookDecision(
            outcome="transform",
            reason="external_content_floor: marked untrusted",
            transformed_payload=payload,
            audit_metadata={**audit, "marked": True, "spans": 0},
        )

    # -- an MCP bridge result: structured, its text parts are what a model reads ------------
    def _structured(
        self,
        ctx: HookContext,
        result: Any,
        audit: dict[str, Any],
        *,
        wrap_it: bool,
        allow: frozenset[str],
    ) -> HookDecision:
        ids: list[str] = []
        used: list[str] = []
        spans = 0
        wrapped_any = False

        def walk(node: Any, key: str | None, parent: Any) -> Any:
            nonlocal spans, wrapped_any
            if isinstance(node, str):
                if not node.strip():
                    return node
                found = scan(node, allow=allow)
                spans += found.spans
                ids.extend(i for i in found.ids if i not in ids)
                used.extend(i for i in found.allowed if i not in used)
                text = found.text
                # The text of an MCP content item (``{"type": "text", "text": ...}``), or of a
                # server that answers in a bare ``content`` / ``result`` field, is what a model
                # reads; other strings (ids, types, urls) are left as they are.
                if wrap_it and key in _TEXT_KEYS and isinstance(parent, dict):
                    wrapped_any = True
                    return wrap(text, source=audit["source"], tool=audit["tool"])
                return text
            if isinstance(node, dict):
                return {
                    k: walk(v, k if isinstance(k, str) else None, node) for k, v in node.items()
                }
            if isinstance(node, list):
                return [walk(v, None, node) for v in node]
            return node

        new_result = walk(result, None, None)
        audit = _with_allowed(audit, used)
        if spans == 0 and not wrapped_any:
            return HookDecision(
                outcome="allow",
                reason="external_content_floor: no instruction-like text",
                audit_metadata={**audit, "marked": False, "spans": 0},
            )
        payload = {**ctx.payload, RESULT: new_result}
        if spans:
            return _redacted(payload, audit, ids, spans, marked=wrapped_any)
        return HookDecision(
            outcome="transform",
            reason="external_content_floor: marked untrusted",
            transformed_payload=payload,
            audit_metadata={**audit, "marked": True, "spans": 0},
        )


def _with_allowed(audit: dict[str, Any], used: Sequence[str]) -> dict[str, Any]:
    """``audit`` with the allow-listed pattern ids that matched and were kept, when any did."""
    return {**audit, "allowed": sorted(used)} if used else audit


def _redacted(
    payload: dict[str, Any], audit: dict[str, Any], ids: list[str], spans: int, *, marked: bool
) -> HookDecision:
    """The decision for a result with matches: ids and counts only, never the text."""
    logger.warning(
        "external_content_floor: redacted %d span(s) (%s) in a result from %s",
        spans,
        ",".join(ids),
        audit["tool"],
    )
    return HookDecision(
        outcome="transform",
        reason=f"external_content_floor: redacted {spans} instruction-like span(s)",
        transformed_payload=payload,
        severity="warn",
        audit_metadata={**audit, "patterns": ids, "spans": spans, "marked": marked},
    )


def _source(payload: dict[str, Any], tool: str) -> str:
    """Who produced the text: the owning plugin, ``skill:<name>`` or ``mcp:<server>``; the
    tool's own name for a core tool."""
    plugin = payload.get(TOOL_PLUGIN)
    return plugin if isinstance(plugin, str) and plugin and plugin != _SYSTEM else tool


__all__ = ["ExternalContentFloorHook"]
