"""ToolPolicyHook tests for Phase 1 PreToolUse enforcement."""

from __future__ import annotations

from iris_harness.kernel.governance import HookContext, HookPoint, build_default_kernel
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.plugins import ToolPolicyHook


def _ctx(tool_name: str = "echo") -> HookContext:
    return HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="run-tool-policy",
        agent_type="chat",
        route=f"tool/{tool_name}",
        payload={"tool_name": tool_name, "args": {"text": "hello"}},
    )


async def test_tool_policy_allows_by_default() -> None:
    decision = await ToolPolicyHook()(_ctx("echo"))

    assert decision.outcome == "allow"
    assert decision.audit_metadata["tool_name"] == "echo"


async def test_tool_policy_denies_blocked_tool() -> None:
    decision = await ToolPolicyHook(blocked_tools=frozenset({"echo"}))(_ctx("echo"))

    assert decision.outcome == "deny"
    assert decision.severity == "warn"
    assert "blocked" in decision.reason


async def test_tool_policy_denies_tool_outside_allowlist() -> None:
    decision = await ToolPolicyHook(allowed_tools=frozenset({"memory_search"}))(_ctx("echo"))

    assert decision.outcome == "deny"
    assert "allowlist" in decision.reason


async def test_tool_policy_denies_malformed_context() -> None:
    ctx = HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="run-tool-policy",
        agent_type="chat",
        payload={"args": {}},
    )

    decision = await ToolPolicyHook()(ctx)

    assert decision.outcome == "deny"
    assert decision.severity == "error"


def test_default_kernel_registers_pre_tool_use_policy(tmp_path) -> None:
    kernel = build_default_kernel(audit_log=AuditLog(db_path=tmp_path / "audit.db"))

    # Phase 1: ToolPolicyHook. Phase 2: CredentialBroker. Phase 4
    # (12.gov-4.2): PersonaSurface. 12.gov-4.3: CommandSandbox.
    # 12.gov-4.4: FSJail. 12.gov-4.5: NetworkEgress.
    # 12.gov-4.6: MCPAllowlistHook. ADR-0118: DestructiveApprovalHook.
    # Plugin-capabilities §4: CallerPolicyHook (the permission contract). Issue #73:
    # PreToolUseLedgerHook. Update this count when adding new PreToolUse plugins in
    # build_default_kernel.
    assert kernel.hook_count(HookPoint.PRE_TOOL_USE) == 10
    assert "caller_policy" in kernel.hook_names(HookPoint.PRE_TOOL_USE)


# --- confirm once per run (multi-step loop plan, decision 8) ------------------


def _write_ctx(tool_name: str, *, asked_user: bool | None) -> HookContext:
    metadata = {} if asked_user is None else {"asked_user": asked_user}
    return HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="run-confirm-once",
        agent_type="chat",
        route=f"tool/{tool_name}",
        payload={"tool_name": tool_name, "args": {"task": "pay rent", "date": "2026-09-30"}},
        metadata=metadata,
    )


async def test_confirm_once_tool_is_turned_back_until_the_run_has_asked() -> None:
    hook = ToolPolicyHook(confirm_once_tools=frozenset({"create_reminder"}))

    unasked = await hook(_write_ctx("create_reminder", asked_user=False))
    assert unasked.outcome == "require_approval"
    assert "ask_user" in unasked.reason
    assert unasked.audit_metadata["policy"] == "confirm_once_tools"

    no_evidence = await hook(_write_ctx("create_reminder", asked_user=None))
    assert no_evidence.outcome == "require_approval"


async def test_confirm_once_tool_passes_once_the_run_has_asked() -> None:
    hook = ToolPolicyHook(confirm_once_tools=frozenset({"create_reminder"}))
    decision = await hook(_write_ctx("create_reminder", asked_user=True))
    assert decision.outcome == "allow"


async def test_confirm_once_does_not_touch_other_tools() -> None:
    hook = ToolPolicyHook(confirm_once_tools=frozenset({"create_reminder"}))
    decision = await hook(_write_ctx("calendar_lookup", asked_user=False))
    assert decision.outcome == "allow"


def test_confirm_once_tools_from_env_is_an_override_only(monkeypatch) -> None:
    """ADR-0110: which tools ask first is their manifest's; the env var adds names, and the
    core's default names no plugin tool."""
    from iris_harness.kernel.governance.wiring import _confirm_once_tools_from_env

    monkeypatch.delenv("IRIS_GOVERNANCE_CONFIRM_ONCE_TOOLS", raising=False)
    assert _confirm_once_tools_from_env() == frozenset()

    monkeypatch.setenv("IRIS_GOVERNANCE_CONFIRM_ONCE_TOOLS", "none")
    assert _confirm_once_tools_from_env() == frozenset()

    monkeypatch.setenv("IRIS_GOVERNANCE_CONFIRM_ONCE_TOOLS", "create_reminder, send_email")
    assert _confirm_once_tools_from_env() == frozenset({"create_reminder", "send_email"})


def _declared_ctx(tool_name: str, *, confirm: str, asked_user: bool) -> HookContext:
    return HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="run-declared",
        agent_type="chat",
        route=f"tool/{tool_name}",
        payload={"tool_name": tool_name, "args": {}},
        metadata={"asked_user": asked_user, "tool_effect": "write", "tool_confirm": confirm},
    )


async def test_the_tools_own_declaration_drives_confirm_once() -> None:
    """ADR-0110: no name in the hook's set — the loop carries the manifest's `confirm`."""
    hook = ToolPolicyHook()  # no confirm_once_tools at all
    turned_back = await hook(_declared_ctx("any_write", confirm="once", asked_user=False))
    assert turned_back.outcome == "require_approval"
    assert (
        await hook(_declared_ctx("any_write", confirm="once", asked_user=True))
    ).outcome == "allow"
    assert (
        await hook(_declared_ctx("any_write", confirm="never", asked_user=False))
    ).outcome == "allow"
