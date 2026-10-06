"""CredentialBroker — Phase 2 ``PreToolUse`` handle resolver.

Tool arguments sometimes carry a ``vault://<handle>`` reference (typically because the
operator wrote one into a skill manifest or a recipe). This hook walks the call's
``args`` (``kernel/governance/hooks/tool_payload.py``), looks each handle up in the local
vault, and returns a ``transform`` whose ``args`` have every handle replaced by its
value. The runner executes the tool with the final ``args``, so the tool receives the
secret and the model never does: the handle is what the model wrote and what it sees
again. This is the runtime realization of design §7 (vault is a service, not a config;
secrets never enter the LLM prompt path).

Where the secret must never land, and why it does not:

- The audit row: the kernel copies only allow-listed payload keys into it
  (``kernel._AUDITED_PAYLOAD_KEYS``; ``args`` is not one), and this hook's
  ``audit_metadata`` names handles, never values.
- Every other ``PRE_TOOL_USE`` hook: this one runs last (priority 90, after the
  destructive-approval hook at 50), so policy hooks judge -- and the approval queue
  pins -- the call as written, with its handles, and no hook after it sees the value.
- The session timeline: ``tool.invoke.start`` logs the arguments as the caller wrote
  them, not the transformed ones (``GovernedToolRunner.execute``).

Failure modes:

- Unknown handle  → ``deny`` (severity ``error``); audit lists the
  handle so the operator can fix the misconfiguration.
- Vault offline + tool declared ``requires_credentials=True`` →
  ``deny`` (severity ``critical``).
- Vault offline + no required credentials declared → ``allow`` (the
  tool simply gets no resolved values).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from iris_harness.kernel.governance.hooks.tool_payload import ARGS, args_of
from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint

logger = logging.getLogger(__name__)

VAULT_PREFIX = "vault://"


class _SecretLookup(Protocol):
    def get(self, handle: str) -> str | None: ...


class CredentialBroker:
    """Resolve ``vault://`` handles in tool arguments at PreToolUse.

    ``vault`` is optional — when ``None``, the broker is a no-op unless
    a tool declares ``requires_credentials=True``, in which case it
    fails closed.
    """

    name: str = "credential_broker"
    hook_point: HookPoint = HookPoint.PRE_TOOL_USE
    # Last at PRE_TOOL_USE but for the pre-execution ledger row, which never reads the
    # arguments (module docstring): every hook before it, the approval queue included,
    # sees the handle; nothing after it sees the secret.
    priority: int = 90

    def __init__(self, *, vault: _SecretLookup | None = None) -> None:
        self._vault = vault

    async def __call__(self, ctx: HookContext) -> HookDecision:
        tool_args = args_of(ctx.payload)
        declared = ctx.metadata.get("required_credentials") or ()
        declared_handles = tuple(
            handle for handle in declared if isinstance(handle, str) and handle
        )

        found_handles = _find_vault_handles(tool_args) if tool_args is not None else []
        all_handles = list({*found_handles, *declared_handles})

        if not all_handles:
            return HookDecision(
                outcome="allow",
                reason="credential_broker: no vault handles in tool arguments",
            )

        if self._vault is None:
            requires = bool(ctx.metadata.get("requires_credentials"))
            if requires or declared_handles:
                return HookDecision(
                    outcome="deny",
                    reason=(
                        "credential_broker: tool declares required credentials but "
                        "the vault is unavailable (fail-closed)"
                    ),
                    severity="critical",
                    audit_metadata={
                        "handles": all_handles,
                        "vault_available": False,
                    },
                )
            return HookDecision(
                outcome="allow",
                reason="credential_broker: vault unavailable, no required credentials declared",
                severity="warn",
                audit_metadata={"handles": all_handles, "vault_available": False},
            )

        resolved: dict[str, str] = {}
        missing: list[str] = []
        for handle in all_handles:
            value = self._vault.get(handle)
            if value is None:
                missing.append(handle)
            else:
                resolved[handle] = value

        if missing:
            return HookDecision(
                outcome="deny",
                reason=f"credential_broker: vault handle(s) not found: {sorted(missing)}",
                severity="error",
                audit_metadata={
                    "missing_handles": sorted(missing),
                    "resolved_count": len(resolved),
                },
            )

        if not found_handles:
            # Declared and resolvable, but not written into this call's arguments: there
            # is nothing to substitute.
            return HookDecision(
                outcome="allow",
                reason="credential_broker: declared credentials available",
                audit_metadata={"resolved_handles": sorted(resolved)},
            )

        assert tool_args is not None  # found_handles came from it
        transformed: dict[str, Any] = {**ctx.payload, ARGS: _substitute(tool_args, resolved)}
        return HookDecision(
            outcome="transform",
            reason=f"credential_broker: resolved {len(found_handles)} handle(s) in the arguments",
            transformed_payload=transformed,
            audit_metadata={"resolved_handles": sorted(found_handles)},
        )


def _find_vault_handles(value: Any) -> list[str]:
    """Recursively walk ``value`` collecting unique ``vault://`` references.

    Walks dict values, list/tuple/set members, and string leaves. Other
    types are ignored — they cannot carry a handle.
    """
    found: list[str] = []
    _walk(value, found)
    return list(dict.fromkeys(found))  # preserve order, dedupe


def _walk(value: Any, found: list[str]) -> None:
    if isinstance(value, str):
        if value.startswith(VAULT_PREFIX):
            found.append(value)
        return
    if isinstance(value, dict):
        for v in value.values():
            _walk(v, found)
        return
    if isinstance(value, (list, tuple, set)):
        for v in value:
            _walk(v, found)


def _substitute(value: Any, resolved: dict[str, str]) -> Any:
    """A copy of ``value`` with every handle leaf ``_walk`` found replaced by its secret.

    The same leaves ``_walk`` collects (a string that is a handle), so what is resolved
    is exactly what was looked up; containers are rebuilt, never mutated in place, so the
    caller's own arguments keep the handle.
    """
    if isinstance(value, str):
        return resolved.get(value, value) if value.startswith(VAULT_PREFIX) else value
    if isinstance(value, dict):
        return {k: _substitute(v, resolved) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, resolved) for v in value]
    if isinstance(value, tuple):
        return tuple(_substitute(v, resolved) for v in value)
    if isinstance(value, set):
        return {_substitute(v, resolved) for v in value}
    return value
