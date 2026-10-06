"""The credential broker, on payloads built by the tool-payload builders.

The payload is ``tool_payload.pre_tool_payload`` -- exactly what the governed runner and
the MCP bridge send -- so a test here cannot pass on a key no producer writes (the broker
once read ``tool_arguments`` and resolved nothing on any real call). End-to-end through
the runner: ``test_tool_hook_contract_runner.py``.
"""

from __future__ import annotations

from typing import Any

from iris_harness.kernel.governance import HookContext, HookPoint
from iris_harness.kernel.governance.hooks.tool_payload import args_of, pre_tool_payload
from iris_harness.kernel.governance.plugins import CredentialBroker


class _Vault:
    def __init__(self, entries: dict[str, str]) -> None:
        self._entries = entries

    def get(self, handle: str) -> str | None:
        return self._entries.get(handle)


def _ctx(args: dict[str, Any], *, metadata: dict[str, object] | None = None) -> HookContext:
    return HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="run-broker",
        agent_type="chat",
        payload=pre_tool_payload("github_issue_create", args),
        metadata=metadata or {},
    )


async def test_broker_allows_when_no_vault_handles_present() -> None:
    broker = CredentialBroker(vault=_Vault({"vault://github-token": "ghp_abc"}))
    decision = await broker(_ctx({"title": "hello", "body": "world"}))

    assert decision.outcome == "allow"


async def test_broker_rewrites_the_handle_in_the_args() -> None:
    broker = CredentialBroker(vault=_Vault({"vault://github-token": "ghp_real"}))
    args = {"title": "open issue", "auth": {"api_key": "vault://github-token"}}
    decision = await broker(_ctx(args))

    assert decision.outcome == "transform"
    transformed = args_of(decision.transformed_payload or {})
    assert transformed == {"title": "open issue", "auth": {"api_key": "ghp_real"}}
    # The caller's own arguments keep the handle (rebuilt, never mutated in place).
    assert args["auth"] == {"api_key": "vault://github-token"}
    # The audit row names the handle, never the value.
    assert "ghp_real" not in repr(decision.audit_metadata)
    assert decision.audit_metadata["resolved_handles"] == ["vault://github-token"]


async def test_broker_walks_nested_lists_and_dicts() -> None:
    broker = CredentialBroker(
        vault=_Vault(
            {
                "vault://github-token": "ghp_real",
                "vault://openrouter": "sk-or-real",
            }
        )
    )
    args = {
        "calls": [
            {"key": "vault://github-token"},
            {"nested": {"key": "vault://openrouter"}},
        ],
        "plain": "vault-free",
    }
    decision = await broker(_ctx(args))

    assert decision.outcome == "transform"
    assert args_of(decision.transformed_payload or {}) == {
        "calls": [{"key": "ghp_real"}, {"nested": {"key": "sk-or-real"}}],
        "plain": "vault-free",
    }


async def test_broker_denies_on_unknown_handle() -> None:
    broker = CredentialBroker(vault=_Vault({}))
    decision = await broker(_ctx({"key": "vault://does-not-exist"}))

    assert decision.outcome == "deny"
    assert decision.severity == "error"
    assert decision.audit_metadata["missing_handles"] == ["vault://does-not-exist"]


async def test_broker_fails_closed_when_vault_offline_and_tool_requires_credentials() -> None:
    broker = CredentialBroker(vault=None)
    decision = await broker(
        _ctx({"key": "vault://github-token"}, metadata={"requires_credentials": True})
    )

    assert decision.outcome == "deny"
    assert decision.severity == "critical"


async def test_broker_fails_open_when_vault_offline_and_no_requirement_declared() -> None:
    broker = CredentialBroker(vault=None)
    decision = await broker(_ctx({"key": "vault://github-token"}))

    # No declared requirement → allow with warn, so credential-free tools
    # don't get blocked by an unavailable vault.
    assert decision.outcome == "allow"
    assert decision.severity == "warn"
    assert decision.audit_metadata.get("vault_available") is False


async def test_broker_treats_declared_required_credentials_as_requiring_resolution() -> None:
    """A skill that declared required handles in its manifest must have them resolved.

    Even if those handles don't appear in the tool args, the broker must
    deny when the vault is offline.
    """
    broker = CredentialBroker(vault=None)
    decision = await broker(
        _ctx({"title": "hello"}, metadata={"required_credentials": ["vault://github-token"]})
    )

    assert decision.outcome == "deny"
    assert decision.severity == "critical"
    assert "vault://github-token" in decision.audit_metadata["handles"]


async def test_declared_credentials_not_in_the_args_leave_the_call_as_it_is() -> None:
    broker = CredentialBroker(vault=_Vault({"vault://github-token": "ghp_real"}))
    decision = await broker(
        _ctx({"title": "hello"}, metadata={"required_credentials": ["vault://github-token"]})
    )

    assert decision.outcome == "allow"
    assert decision.transformed_payload is None


async def test_broker_runs_last_at_pre_tool_use() -> None:
    """Every other PRE_TOOL_USE hook -- the approval queue that pins a call included --
    judges the call with its handles; nothing after the broker sees the secret. The one hook
    after it, the pre-execution ledger row, never reads the arguments."""
    from iris_harness.kernel.governance.wiring import build_default_kernel

    kernel = build_default_kernel()
    names = kernel.hook_names(HookPoint.PRE_TOOL_USE)
    assert names[-2:] == ("credential_broker", "pre_tool_use_ledger"), names
