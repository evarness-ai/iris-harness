"""CapabilityRedactionHook -- ``PostToolUse``: the owner's identity out of capability results.

A capability call (docs/architecture/plugin-capabilities.md §2, §4) hands one plugin's data
to another plugin's code, or to the core, with no model and no answer check between them.
So its result is redacted in place before the consumer sees it, by the ``capability``
column of ``config/governance/identity.yaml`` (ADR-0125, PR 3), over the seam's one corpus
(``identity_redaction.owner_identity()``):

- ``secret`` -> :data:`MASK`, always; no grant unmasks it.
- ``name``, ``email``, ``phone``, ``address``, ``handle`` -> ``[owner:<kind>#<n>]``, unless
  the consumer's manifest grants the kind for this capability (``unmask_grants``). The
  consumer is the caller the harness stamped on the call, never one the plugin names.
- ``link`` -> unchanged; a first name alone -> unchanged (it is masked only in web-search
  arguments).

A result that held any owner literal is marked ``personal`` (raised, never lowered).

What is redacted is exactly what the method declares: the runner puts the result's
text-bearing fields in ``payload["fields"]`` (``{concrete path: text}``) beside their joined
text in ``payload["result"]``; this hook rewrites both, and the runner writes the field map
back into a copy of the typed result. Only ``capability:`` calls: a plain tool's result is
unchanged (its text reaches the model, which the answer-side checks cover).

Priority 5: first at ``POST_TOOL_USE``, so every later hook, and every audit row after this
one, sees the redacted text.
"""

from __future__ import annotations

import logging
from typing import Final

from iris_harness.foundation.capabilities import CAPABILITY_TOOL_PREFIX
from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.identity_config import GuardTable, guard_table
from iris_harness.kernel.governance.identity_redaction import owner_identity
from iris_harness.kernel.governance.owner_identity import KINDS
from iris_harness.kernel.governance.owner_pii import MASK, redact_capability_text
from iris_harness.kernel.governance.plugins.output_classifier import more_restrictive
from iris_harness.kernel.governance.unmask_grants import unmask_grants

logger = logging.getLogger(__name__)

# With no readable table, every owner literal is masked and nothing is granted: masking
# fails closed, never open.
_ALL_MASKED: Final = GuardTable.model_validate(
    {
        "kinds": {
            kind: {
                "egress": "deny",
                "answer_owner": "halt" if kind == "secret" else "pass",
                "answer_other": "halt",
                "capability": "mask",
                "tier3": "placeholder",
                "web_search": "deny",
            }
            for kind in KINDS
        },
        "first_name_alone": {
            "egress": "log",
            "answer_owner": "pass",
            "answer_other": "pass",
            "capability": "mask",
            "tier3": "pass",
            "web_search": "mask",
        },
    }
)


def _table() -> GuardTable:
    try:
        table = guard_table()
    except Exception:  # a malformed table must not open the masking
        logger.warning(
            "capability_redaction: identity.yaml is unreadable; masking every owner literal",
            exc_info=True,
        )
        return _ALL_MASKED
    if table is None:
        logger.warning(
            "capability_redaction: identity.yaml has no guards table; masking every owner literal"
        )
        return _ALL_MASKED
    return table


class CapabilityRedactionHook:
    """Mask and pseudonymise the owner's identity in a capability result's text fields."""

    name: str = "capability_redaction"
    hook_point: HookPoint = HookPoint.POST_TOOL_USE
    priority: int = 5

    async def __call__(self, ctx: HookContext) -> HookDecision:
        tool = str(ctx.payload.get("tool_name") or "")
        if not tool.startswith(CAPABILITY_TOOL_PREFIX):
            return HookDecision(outcome="allow", reason="capability_redaction: not a capability")
        fields = ctx.payload.get("fields")
        if not isinstance(fields, dict) or not fields:
            return HookDecision(outcome="allow", reason="capability_redaction: no text fields")
        identity = owner_identity()
        if identity is None:
            return HookDecision(outcome="allow", reason="capability_redaction: no identity seam")
        table = _table()
        caller = str(ctx.payload.get("caller") or "")
        capability = str(ctx.payload.get("capability") or "")
        grants = unmask_grants(caller, capability) & table.grantable()
        kinds: set[str] = set()
        masked: dict[str, str] = {}
        for path, text in fields.items():
            redacted = redact_capability_text(
                str(text), identity=identity, table=table, grants=grants
            )
            masked[path] = redacted.text
            kinds |= redacted.kinds
        changed = sorted(path for path in fields if masked[path] != fields[path])
        if not kinds:
            return HookDecision(
                outcome="allow", reason=f"capability_redaction: {tool} carries no identity"
            )
        personal = more_restrictive(ctx.classification, "personal")
        # Paths and kinds, never text: the audit row names where and what kind, not what.
        audit = {"masked_fields": changed, "identity_kinds": sorted(kinds)}
        if not changed:
            return HookDecision(
                outcome="allow",
                reason=f"capability_redaction: {tool} identity unmasked by grant",
                set_classification=personal,
                audit_metadata=audit,
            )
        return HookDecision(
            outcome="transform",
            reason=f"capability_redaction: masked identity in {len(changed)} field(s) of {tool}",
            set_classification=personal,
            audit_metadata=audit,
            transformed_payload={
                **ctx.payload,
                "fields": masked,
                "result": "\n".join(masked.values()),
            },
        )


__all__ = ["MASK", "CapabilityRedactionHook"]
