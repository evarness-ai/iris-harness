"""The side-effect ledger's write-ahead row for high-risk calls (issue #73), in the kernel.

``PreToolUseLedgerHook`` writes a ``pending`` row before a destructive tool or a pinned
write runs, or denies the call when it cannot; ``PostToolUseLedgerHook`` settles that row
(``SideEffectLedger.finalize``). That class is always recorded (the ledger is created on
first use); ``IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL=1`` records every non-read call too, and
``IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER=0`` turns high-risk calls off. The runner's side is
``tests/unit/iris_harness/agent/test_agent/test_pre_execution_record.py``.
"""

from __future__ import annotations

import logging
import threading
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
from iris_harness.kernel.governance.side_effects import (
    DeferredSideEffectLedger,
    SideEffectLedger,
)
from iris_harness.kernel.governance.side_effects.probes import NO_PROBE, run_probe
from iris_harness.kernel.governance.wiring import parse_side_effect_ledger_settings

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


def test_a_default_kernel_guards_high_risk_calls_without_touching_the_ledger(
    tmp_path: Path,
) -> None:
    from iris_harness.foundation.paths import governance_data_dir

    kernel = build_default_kernel(audit_log=AuditLog(db_path=tmp_path / "audit.db"))
    assert "post_tool_use_ledger" in kernel.hook_names(HookPoint.POST_TOOL_USE)
    assert kernel.hook_names(HookPoint.PRE_TOOL_USE)[-1] == "pre_tool_use_ledger"
    # Built, not used: the default ledger opens when a high-risk call first needs it.
    assert not (governance_data_dir() / "side_effects.db").exists()


async def test_the_default_ledger_opens_on_the_first_high_risk_call_and_records_it(
    tmp_path: Path,
) -> None:
    db = tmp_path / "s.db"
    ledger = DeferredSideEffectLedger(db)
    assert not db.exists()
    decision = await PreToolUseLedgerHook(ledger)(_pre())
    assert decision.outcome == "allow" and db.exists()
    row = ledger.get(decision.audit_metadata[SIDE_EFFECT_ID])
    assert row is not None and row.status == "pending"


def test_a_second_thread_never_sees_a_deferred_ledger_before_its_schema_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first open holds every other caller until the schema is complete.

    Thread A is parked inside schema creation; thread B then records. B must wait for A, not
    run against a database with no table (which would deny a legitimate high-risk call).
    """
    ledger = DeferredSideEffectLedger(tmp_path / "s.db")
    real_init = SideEffectLedger._init_schema
    in_schema = threading.Event()
    release = threading.Event()
    parked: list[bool] = []

    def slow_init(self: SideEffectLedger) -> None:
        if not parked:  # only the first opener parks
            parked.append(True)
            in_schema.set()
            assert release.wait(timeout=10)
        real_init(self)

    monkeypatch.setattr(SideEffectLedger, "_init_schema", slow_init)
    errors: list[BaseException] = []

    def record(key: str) -> None:
        try:
            ledger.record(
                run_id="r", step_id=1, tool="wipe", verification_probe=NO_PROBE, side_effect_id=key
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    a = threading.Thread(target=record, args=("a",))
    b = threading.Thread(target=record, args=("b",))
    a.start()
    assert in_schema.wait(timeout=10)  # A is inside the schema creation
    b.start()
    b.join(timeout=1.0)  # unfixed: B finishes (failing) here; fixed: B is still waiting on A
    release.set()
    a.join(timeout=10)
    b.join(timeout=10)

    assert errors == []
    assert ledger.get("a") is not None and ledger.get("b") is not None


async def test_a_deferred_ledger_that_will_not_open_denies_the_high_risk_call(
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    ledger = DeferredSideEffectLedger(blocker / "s.db")
    decision = await PreToolUseLedgerHook(ledger)(_pre())
    assert decision.outcome == "deny"


@pytest.mark.parametrize("effect", ["write", "send"])
async def test_the_default_post_hook_leaves_a_plain_write_alone(
    effect: str, tmp_path: Path
) -> None:
    db = tmp_path / "s.db"
    hook = PostToolUseLedgerHook(DeferredSideEffectLedger(db), high_risk_only=True)
    ctx = _post(effect=effect)
    ctx.metadata["tool_confirm"] = "once"
    decision = await hook(ctx)
    assert decision.outcome == "allow"
    assert not db.exists()


async def test_the_default_post_hook_settles_a_high_risk_row(tmp_path: Path) -> None:
    ledger = DeferredSideEffectLedger(tmp_path / "s.db")
    pre = await PreToolUseLedgerHook(ledger)(_pre())
    hook = PostToolUseLedgerHook(ledger, high_risk_only=True)
    await hook(_post())
    row = ledger.get(pre.audit_metadata[SIDE_EFFECT_ID])
    assert row is not None and row.status == "completed"


@pytest.mark.parametrize("raw", [None, "maybe"])
def test_an_unset_flag_is_the_high_risk_only_default(
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
    assert not (tmp_path / "s.db").exists()  # nothing touched until a high-risk call


def test_an_unset_flag_kernel_records_nothing_for_a_plain_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the kernel's own hooks, as ``kernel_from_env`` wires them."""
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.delenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER", raising=False)
    kernel = kernel_from_env()
    assert kernel is not None
    ctx = _post(effect="write")
    ctx.metadata["tool_confirm"] = "once"
    kernel.fire_sync(HookPoint.POST_TOOL_USE, ctx)
    assert not (tmp_path / "s.db").exists()
    # ... and a destructive call through the same kernel is recorded, in the same file.
    kernel.fire_sync(HookPoint.POST_TOOL_USE, _post())
    assert (tmp_path / "s.db").exists()


def test_an_unrecognised_flag_value_warns_and_applies_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER", "maybe")
    with caplog.at_level(logging.WARNING, logger="iris_harness.kernel.governance.wiring"):
        kernel = kernel_from_env()
    assert kernel is not None
    (warning,) = [r.getMessage() for r in caplog.records if "not recognised" in r.getMessage()]
    assert "'maybe'" in warning and "1/true/yes/on" in warning and "high-risk" in warning
    assert not (tmp_path / "s.db").exists()  # the default applied, not "on"


def test_an_unset_flag_does_not_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.delenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER", raising=False)
    with caplog.at_level(logging.WARNING, logger="iris_harness.kernel.governance.wiring"):
        kernel_from_env()
    assert not any("SIDE_EFFECT_LEDGER" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("raw", ["1", "on", "true", "yes", "TRUE"])
def test_an_explicit_ledger_on_is_the_same_as_unset(
    raw: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit true no longer widens scope: high-risk only, nothing opened at build."""
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.delenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL", raising=False)
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER", raw)
    kernel = kernel_from_env()
    assert kernel is not None
    assert "post_tool_use_ledger" in kernel.hook_names(HookPoint.POST_TOOL_USE)
    ctx = _post(effect="write")
    ctx.metadata["tool_confirm"] = "once"
    kernel.fire_sync(HookPoint.POST_TOOL_USE, ctx)
    assert not (tmp_path / "s.db").exists()


@pytest.mark.parametrize("raw", ["1", "on", "true", "yes"])
def test_the_all_scope_records_every_non_read_call(
    raw: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.delenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER", raising=False)
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL", raw)
    kernel = kernel_from_env()
    assert kernel is not None
    assert "post_tool_use_ledger" in kernel.hook_names(HookPoint.POST_TOOL_USE)
    assert (tmp_path / "s.db").exists()  # opened at build, as the flag always did


@pytest.mark.parametrize("raw", [None, "0", "false", "no", "off", ""])
def test_the_all_scope_off_or_unset_is_high_risk_only(
    raw: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_DB_PATH", str(tmp_path / "s.db"))
    if raw is None:
        monkeypatch.delenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL", raising=False)
    else:
        monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL", raw)
    with caplog.at_level(logging.WARNING, logger="iris_harness.kernel.governance.wiring"):
        assert kernel_from_env() is not None
    assert not (tmp_path / "s.db").exists()
    assert not any("SIDE_EFFECT_LEDGER" in r.getMessage() for r in caplog.records)


def test_an_unrecognised_all_value_warns_and_applies_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL", "maybe")
    with caplog.at_level(logging.WARNING, logger="iris_harness.kernel.governance.wiring"):
        assert kernel_from_env() is not None
    (warning,) = [r.getMessage() for r in caplog.records if "not recognised" in r.getMessage()]
    assert "_LEDGER_ALL='maybe'" in warning and "1/true/yes/on" in warning
    assert "0/false/no/off" in warning
    assert not (tmp_path / "s.db").exists()  # the default (off) applied


@pytest.mark.parametrize("ledger", ["0", "false", "off"])
async def test_all_while_the_ledger_is_off_warns_once_and_keeps_the_denial(
    ledger: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER", ledger)
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL", "1")
    with caplog.at_level(logging.WARNING, logger="iris_harness.kernel.governance.wiring"):
        kernel = kernel_from_env()
    assert kernel is not None
    messages = [r.getMessage() for r in caplog.records]
    assert len([m for m in messages if "has no effect" in m]) == 1
    assert any("side-effect ledger is OFF" in m for m in messages)
    assert "post_tool_use_ledger" not in kernel.hook_names(HookPoint.POST_TOOL_USE)
    assert not (tmp_path / "s.db").exists()
    assert (await PreToolUseLedgerHook(None)(_pre())).outcome == "deny"


@pytest.mark.parametrize(
    ("ledger", "scope", "enabled", "record_all", "problems"),
    [
        (None, None, True, False, 0),
        ("", "", True, False, 0),
        ("true", None, True, False, 0),
        ("0", None, False, False, 0),
        ("No", None, False, False, 0),
        ("maybe", None, True, False, 1),
        (None, "yes", True, True, 0),
        ("1", "on", True, True, 0),
        (None, "maybe", True, False, 1),
        ("off", "1", False, False, 1),
        ("maybe", "maybe", True, False, 2),
    ],
)
def test_the_two_settings_parse_as_booleans(
    ledger: str | None, scope: str | None, enabled: bool, record_all: bool, problems: int
) -> None:
    got = parse_side_effect_ledger_settings(ledger, scope)
    assert (got.enabled, got.record_all, len(got.problems)) == (enabled, record_all, problems)


@pytest.mark.parametrize(
    ("ledger", "scope", "notices"),
    [
        (None, None, 0),
        ("", None, 0),
        ("true", None, 1),
        ("1", "", 1),
        ("On", "no", 1),
        ("true", "1", 0),
        ("1", "yes", 0),
        (None, "1", 0),
        ("0", None, 0),
        ("false", "1", 0),
        ("maybe", None, 0),
    ],
)
def test_an_explicit_ledger_on_without_the_scope_is_a_notice_never_a_default(
    ledger: str | None, scope: str | None, notices: int
) -> None:
    """Only an EXPLICIT truthy value says what it no longer records; unset stays silent."""
    got = parse_side_effect_ledger_settings(ledger, scope)
    assert len(got.notices) == notices
    for text in got.notices:
        assert "IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL=1" in text


def test_the_notice_is_logged_once_as_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.delenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL", raising=False)
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER", "true")
    with caplog.at_level(logging.WARNING, logger="iris_harness.kernel.governance.wiring"):
        kernel_from_env()
    hits = [r for r in caplog.records if "covers high-risk calls only" in r.getMessage()]
    assert len(hits) == 1
    assert hits[0].levelno == logging.WARNING


def test_a_kernel_built_without_a_ledger_still_guards_high_risk_calls(tmp_path: Path) -> None:
    kernel = build_default_kernel(
        audit_log=AuditLog(db_path=tmp_path / "audit.db"), side_effect_ledger_enabled=False
    )
    assert "post_tool_use_ledger" not in kernel.hook_names(HookPoint.POST_TOOL_USE)
    assert kernel.hook_names(HookPoint.PRE_TOOL_USE)[-1] == "pre_tool_use_ledger"


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
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL", "1")
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_DB_PATH", str(blocker / "s.db"))
    kernel = kernel_from_env()
    assert kernel is not None
    assert "post_tool_use_ledger" not in kernel.hook_names(HookPoint.POST_TOOL_USE)
    assert kernel.hook_names(HookPoint.PRE_TOOL_USE)[-1] == "pre_tool_use_ledger"
