"""The side-effect ledger's write-ahead row for high-risk calls (issue #73), in the kernel.

``PreToolUseLedgerHook`` writes a ``pending`` row before a destructive tool or a pinned
write runs, or denies the call when it cannot; ``PostToolUseLedgerHook`` settles that row
(``SideEffectLedger.finalize``). The ledger is on by default; turning it off turns
high-risk calls off. The runner's side is ``tests/unit/iris_harness/agent/test_agent/
test_pre_execution_record.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from iris_harness.kernel.governance import HookPoint, build_default_kernel, kernel_from_env
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.hooks.tool_payload import (
    SIDE_EFFECT_ID,
    TOOL_CALL_ID,
    TOOL_EFFECT,
    TOOL_VERIFY,
    post_tool_payload,
    pre_tool_payload,
    side_effect_id_of,
    tool_post_metadata,
)
from iris_harness.kernel.governance.hooks.types import HookContext
from iris_harness.kernel.governance.plugins.destructive_approval import (
    approved_per_call,
    pinned_by_declaration,
)
from iris_harness.kernel.governance.plugins.post_tool_use_ledger import (
    PARTIAL_STREAM,
    PRE_RECORDED,
    PostToolUseLedgerHook,
)
from iris_harness.kernel.governance.plugins.pre_tool_use_ledger import PreToolUseLedgerHook
from iris_harness.kernel.governance.side_effects import SideEffectLedger
from iris_harness.kernel.governance.side_effects.probes import NO_PROBE, run_probe

KEY = "run-1:2:call-1"


def _pre(
    tool: str = "wipe",
    *,
    effect: str | None = "destructive",
    confirm: str | None = "approval",
    approved_by: str | None = "appr-1",
    verify: str | None = None,
    capability: bool = False,
    per_call_approval: bool = False,
) -> HookContext:
    payload = pre_tool_payload(tool, {"id": "m1"})
    if capability:
        payload["capability"] = "test.shredder"
    return HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="run-1",
        step_id=2,
        agent_type="chat",
        payload=payload,
        metadata={
            TOOL_EFFECT: effect,
            "tool_confirm": confirm,
            TOOL_CALL_ID: "call-1",
            TOOL_VERIFY: verify,
            "approved_by": approved_by,
            "per_call_approval": per_call_approval,
        },
    )


def _post(
    tool: str = "wipe",
    result: Any = "done",
    *,
    error: str | None = None,
    effect: str = "destructive",
    **extra: Any,
) -> HookContext:
    return HookContext(
        hook_point=HookPoint.POST_TOOL_USE,
        run_id="run-1",
        step_id=2,
        agent_type="chat",
        payload=post_tool_payload(tool, result, **extra),
        metadata=tool_post_metadata(
            effect=effect, content="internal", verify=None, tool_call_id="call-1", error=error
        ),
    )


@pytest.fixture
def ledger(tmp_path: Path) -> SideEffectLedger:
    return SideEffectLedger(tmp_path / "side_effects.db")


# ------------------------------------------------------------------- what is high-risk
@pytest.mark.parametrize(
    ("effect", "confirm", "high"),
    [
        ("destructive", "approval", True),
        ("destructive", "never", True),
        ("write", "approval", True),
        ("write", "once", False),
        ("write", "never", False),
        ("read", "never", False),
        (None, None, False),  # an MCP server's tool declares nothing
    ],
)
def test_high_risk_is_the_declaration_that_pins_every_call(
    effect: str | None, confirm: str | None, high: bool
) -> None:
    assert pinned_by_declaration(effect, confirm) is high


def test_a_code_caller_s_confirm_once_write_is_approved_per_call_but_not_high_risk() -> None:
    metadata = {"tool_effect": "write", "tool_confirm": "once", "per_call_approval": True}
    assert approved_per_call(metadata) is True
    assert pinned_by_declaration(metadata["tool_effect"], metadata["tool_confirm"]) is False


# --------------------------------------------------------------------- PreToolUseLedgerHook
async def test_an_approved_high_risk_call_gets_a_pending_row_first(
    ledger: SideEffectLedger,
) -> None:
    decision = await PreToolUseLedgerHook(ledger)(_pre())

    assert decision.outcome == "allow"
    assert side_effect_id_of(decision.audit_metadata) == KEY
    row = ledger.get(KEY)
    assert row is not None
    assert (row.tool, row.status, row.step_id, row.verification_probe) == (
        "wipe",
        "pending",
        2,
        NO_PROBE,
    )
    assert row.probe_metadata == {"subject": KEY, "effect": "destructive", PRE_RECORDED: True}


async def test_the_pre_row_carries_no_probe_even_for_a_declared_one(
    ledger: SideEffectLedger,
) -> None:
    """A crash leaves the row ``pending`` with no probe: ambiguous, never a probe that
    could only report "not landed" for a call that did land."""
    await PreToolUseLedgerHook(ledger)(_pre(verify="mail_trashed"))
    row = ledger.get(KEY)
    assert row is not None and row.verification_probe == NO_PROBE


@pytest.mark.parametrize("tool", ["git_commit", "git_push"])
async def test_a_crashed_declared_probe_tool_resolves_to_ambiguous(
    ledger: SideEffectLedger, tool: str
) -> None:
    await PreToolUseLedgerHook(ledger)(_pre(tool))
    row = ledger.get(KEY)
    assert row is not None and row.status == "pending"
    assert (row.verification_probe, row.probe_subject) == (NO_PROBE, KEY)
    assert run_probe(row.verification_probe, row.side_effect_id, row.probe_metadata) == "ambiguous"


async def test_post_sets_the_declared_probe_once_the_call_returned(
    ledger: SideEffectLedger,
) -> None:
    await PreToolUseLedgerHook(ledger)(_pre(verify="mail_trashed"))
    post = _post()
    post.metadata["tool_verify"] = "mail_trashed"
    await PostToolUseLedgerHook(ledger)(post)
    row = ledger.get(KEY)
    assert row is not None and row.status == "completed"
    assert (row.verification_probe, row.probe_subject) == ("mail_trashed", KEY)


async def test_a_raised_call_keeps_no_probe(ledger: SideEffectLedger) -> None:
    await PreToolUseLedgerHook(ledger)(_pre(verify="mail_trashed"))
    post = _post(error="KeyError")
    post.metadata["tool_verify"] = "mail_trashed"
    await PostToolUseLedgerHook(ledger)(post)
    row = ledger.get(KEY)
    assert row is not None and (row.status, row.verification_probe) == ("error", NO_PROBE)


async def test_a_mapped_tool_waits_for_its_result_for_a_probe(ledger: SideEffectLedger) -> None:
    await PreToolUseLedgerHook(ledger)(_pre("git_commit", verify="something"))
    row = ledger.get(KEY)
    assert row is not None and row.verification_probe == NO_PROBE


@pytest.mark.parametrize(
    "ctx",
    [
        _pre(effect="write", confirm="once"),
        _pre(effect="write", confirm="never"),
        _pre(effect="read", confirm="never"),
        _pre(effect=None, confirm=None),
        # A code caller's confirm-once write: a plain write, recorded after the call.
        _pre(effect="write", confirm="once", per_call_approval=True),
        # Not approved: the runner refuses it, so nothing runs and nothing is recorded.
        _pre(approved_by=None),
    ],
)
async def test_other_calls_are_left_alone(ledger: SideEffectLedger, ctx: HookContext) -> None:
    decision = await PreToolUseLedgerHook(ledger)(ctx)
    assert decision.outcome == "allow" and SIDE_EFFECT_ID not in decision.audit_metadata
    assert ledger.list_by_run("run-1") == []


async def test_a_capability_call_is_recorded_without_an_approval(
    ledger: SideEffectLedger,
) -> None:
    """A capability call carries no approval of its own: allowed this far, it runs."""
    decision = await PreToolUseLedgerHook(ledger)(_pre(approved_by=None, capability=True))
    assert side_effect_id_of(decision.audit_metadata) == KEY


async def test_no_ledger_denies_the_call() -> None:
    decision = await PreToolUseLedgerHook(None)(_pre())
    assert decision.outcome == "deny" and decision.severity == "error"
    assert "no side-effect ledger is configured" in decision.reason
    assert "nothing was run" in decision.reason


async def test_a_failed_write_denies_the_call_naming_only_the_class(tmp_path: Path) -> None:
    class Broken(SideEffectLedger):
        def record(self, **kwargs: Any) -> str:
            raise OSError("/private/owner@example.com/side_effects.db is read-only")

    decision = await PreToolUseLedgerHook(Broken(tmp_path / "s.db"))(_pre())
    assert decision.outcome == "deny" and "OSError" in decision.reason
    assert "owner@example.com" not in decision.reason


# ------------------------------------------------------------- PostToolUseLedgerHook settles
async def test_post_settles_the_pre_row_as_completed(ledger: SideEffectLedger) -> None:
    await PreToolUseLedgerHook(ledger)(_pre())
    decision = await PostToolUseLedgerHook(ledger)(_post())

    assert decision.audit_metadata == {"side_effect_id": KEY, "status": "completed"}
    row = ledger.get(KEY)
    assert row is not None and (row.status, row.error) == ("completed", None)
    assert row.completed_at is not None
    assert len(ledger.list_by_run("run-1")) == 1  # settled, not inserted again


async def test_post_settles_a_raised_call_as_an_error(ledger: SideEffectLedger) -> None:
    await PreToolUseLedgerHook(ledger)(_pre())
    await PostToolUseLedgerHook(ledger)(_post(result="Tool error: boom", error="KeyError"))

    row = ledger.get(KEY)
    assert row is not None and (row.status, row.error, row.completed_at) == (
        "error",
        "KeyError",
        None,
    )


async def test_post_sets_a_mapped_tool_s_subject_on_the_pre_row(
    ledger: SideEffectLedger,
) -> None:
    await PreToolUseLedgerHook(ledger)(_pre("git_commit"))
    await PostToolUseLedgerHook(ledger)(_post("git_commit", '{"commit_sha": "abc123"}'))

    row = ledger.get(KEY)
    assert row is not None
    assert (row.status, row.verification_probe, row.probe_subject) == (
        "completed",
        "git_commit",
        "abc123",
    )
    assert row.probe_metadata[PRE_RECORDED] is True


async def test_a_stream_is_settled_at_its_end_only(ledger: SideEffectLedger) -> None:
    await PreToolUseLedgerHook(ledger)(_pre(capability=True))
    hook = PostToolUseLedgerHook(ledger)
    await hook(_post(result="item", stream_item=0))
    row = ledger.get(KEY)
    assert row is not None and row.status == "pending"

    await hook(_post(result="", stream_end=True, stream_partial=True))
    row = ledger.get(KEY)
    assert row is not None and (row.status, row.error) == ("error", PARTIAL_STREAM)


async def test_a_partial_stream_keeps_the_class_that_stopped_it(
    ledger: SideEffectLedger,
) -> None:
    await PreToolUseLedgerHook(ledger)(_pre(capability=True))
    await PostToolUseLedgerHook(ledger)(
        _post(result="", error="GeneratorExit", stream_end=True, stream_partial=True)
    )
    row = ledger.get(KEY)
    assert row is not None and row.error == "GeneratorExit"


async def test_a_plain_write_is_still_inserted_after_the_call(ledger: SideEffectLedger) -> None:
    await PostToolUseLedgerHook(ledger)(_post("note", effect="write"))
    row = ledger.get(KEY)
    assert row is not None and row.status == "pending" and PRE_RECORDED not in row.probe_metadata


async def test_an_unreadable_ledger_still_tries_the_plain_insert(tmp_path: Path) -> None:
    """Reading for a pre-row must not cost a plain write its record."""

    class Unreadable(SideEffectLedger):
        def get(self, side_effect_id: str) -> Any:
            raise OSError("locked")

    ledger = Unreadable(tmp_path / "s.db")
    decision = await PostToolUseLedgerHook(ledger)(_post("note", effect="write"))
    assert decision.audit_metadata.get("side_effect_id") == KEY
    assert [r.tool for r in ledger.list_by_run("run-1")] == ["note"]


# ------------------------------------------------------------------ SideEffectLedger.finalize
def test_finalize_settles_by_key_and_merges_metadata(ledger: SideEffectLedger) -> None:
    ledger.record(
        side_effect_id=KEY,
        run_id="run-1",
        step_id=2,
        tool="git_commit",
        verification_probe=NO_PROBE,
        probe_metadata={"subject": KEY, PRE_RECORDED: True},
    )
    assert ledger.finalize(
        KEY,
        status="completed",
        verification_probe="git_commit",
        probe_metadata={"subject": "abc", "repo_path": "."},
    )
    row = ledger.get(KEY)
    assert row is not None
    assert row.verification_probe == "git_commit"
    assert row.probe_metadata == {"subject": "abc", "repo_path": ".", PRE_RECORDED: True}


def test_finalize_keeps_the_probe_when_none_is_given(ledger: SideEffectLedger) -> None:
    ledger.record(side_effect_id=KEY, run_id="run-1", step_id=2, tool="t", verification_probe="p")
    ledger.finalize(KEY, status="error", error="ValueError")
    row = ledger.get(KEY)
    assert row is not None and (row.verification_probe, row.status, row.error) == (
        "p",
        "error",
        "ValueError",
    )
    assert [r.side_effect_id for r in ledger.pending("run-1")] == [KEY]  # resume sees it


def test_finalize_of_an_unknown_key_writes_nothing(ledger: SideEffectLedger) -> None:
    assert ledger.finalize("nope", status="completed") is False
    assert ledger.get("nope") is None


# --------------------------------------------------------------------- the default kernel
def test_the_pre_row_is_written_last_at_pre_tool_use(tmp_path: Path) -> None:
    """After every hook that can deny or ask for approval -- the credential broker too."""
    kernel = build_default_kernel(
        audit_log=AuditLog(db_path=tmp_path / "audit.db"),
        side_effect_ledger=SideEffectLedger(tmp_path / "s.db"),
    )
    names = kernel.hook_names(HookPoint.PRE_TOOL_USE)
    assert names[-1] == "pre_tool_use_ledger"
    assert names.index("destructive_approval") < names.index("pre_tool_use_ledger")


async def test_a_later_allowing_hook_cannot_hide_the_row(tmp_path: Path) -> None:
    """``fire`` returns only the last hook's decision; the key rides on the context."""
    from iris_harness.kernel.governance.hooks.tool_payload import side_effect_id_of
    from iris_harness.kernel.governance.hooks.types import HookDecision
    from iris_harness.kernel.governance.kernel import GovernanceKernel

    class _Later:
        name = "later_allow"
        hook_point = HookPoint.PRE_TOOL_USE
        priority = 150

        async def __call__(self, ctx: HookContext) -> HookDecision:
            return HookDecision(outcome="allow", reason="later")

    kernel = GovernanceKernel(audit_log=AuditLog(db_path=tmp_path / "audit.db"))
    kernel.register(PreToolUseLedgerHook(SideEffectLedger(tmp_path / "s.db")))
    kernel.register(_Later())
    kernel.init_lock()
    decision, final_ctx = await kernel.fire(HookPoint.PRE_TOOL_USE, _pre())
    assert side_effect_id_of(decision.audit_metadata) is None
    assert side_effect_id_of(final_ctx.metadata) == KEY


def test_the_row_is_settled_before_any_post_hook_can_withhold_the_result(
    tmp_path: Path,
) -> None:
    kernel = build_default_kernel(
        audit_log=AuditLog(db_path=tmp_path / "audit.db"),
        side_effect_ledger=SideEffectLedger(tmp_path / "s.db"),
    )
    names = list(kernel.hook_names(HookPoint.POST_TOOL_USE))
    ledger_at = names.index("post_tool_use_ledger")
    # The hooks that can deny at POST_TOOL_USE all come after the ledger.
    for later in ("mcp_client_egress",):
        assert names.index(later) > ledger_at, names


def test_a_default_kernel_opens_the_default_ledger(tmp_path: Path) -> None:
    from iris_harness.foundation.paths import governance_data_dir

    kernel = build_default_kernel(audit_log=AuditLog(db_path=tmp_path / "audit.db"))
    assert "post_tool_use_ledger" in kernel.hook_names(HookPoint.POST_TOOL_USE)
    assert (governance_data_dir() / "side_effects.db").exists()


def test_a_kernel_built_without_a_ledger_still_guards_high_risk_calls(tmp_path: Path) -> None:
    kernel = build_default_kernel(
        audit_log=AuditLog(db_path=tmp_path / "audit.db"), side_effect_ledger_enabled=False
    )
    assert "post_tool_use_ledger" not in kernel.hook_names(HookPoint.POST_TOOL_USE)
    assert kernel.hook_names(HookPoint.PRE_TOOL_USE)[-1] == "pre_tool_use_ledger"


@pytest.mark.parametrize("raw", [None, "1", "on", "maybe"])
def test_the_ledger_is_on_unless_opted_out(
    raw: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_DB_PATH", str(tmp_path / "s.db"))
    if raw is None:
        monkeypatch.delenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER", raising=False)
    else:
        monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER", raw)
    kernel = kernel_from_env()
    assert kernel is not None
    assert "post_tool_use_ledger" in kernel.hook_names(HookPoint.POST_TOOL_USE)
    assert (tmp_path / "s.db").exists()


@pytest.mark.parametrize("raw", ["0", "false", "off"])
async def test_opting_out_turns_high_risk_calls_off(
    raw: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER", raw)
    with caplog.at_level(logging.WARNING, logger="iris_harness.kernel.governance.wiring"):
        kernel = kernel_from_env()
    assert kernel is not None
    assert "post_tool_use_ledger" not in kernel.hook_names(HookPoint.POST_TOOL_USE)
    assert any("side-effect ledger is OFF" in r.getMessage() for r in caplog.records)
    decision = await PreToolUseLedgerHook(None)(_pre())
    assert decision.outcome == "deny"


def test_a_ledger_that_will_not_open_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_DB_PATH", str(blocker / "s.db"))
    kernel = kernel_from_env()
    assert kernel is not None
    assert "post_tool_use_ledger" not in kernel.hook_names(HookPoint.POST_TOOL_USE)
    assert kernel.hook_names(HookPoint.PRE_TOOL_USE)[-1] == "pre_tool_use_ledger"
