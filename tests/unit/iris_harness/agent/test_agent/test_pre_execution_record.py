"""A high-risk call's durable record before it runs (issue #73), at the runner.

A destructive tool or a pinned write (``approved_per_call``) -- and a capability method by
its declared effect and confirm -- gets a ``pending`` side-effect ledger row *before* it
runs, keyed by a call id minted before ``PRE_TOOL_USE``; ``POST_TOOL_USE`` settles that same
row: ``completed`` when it returned, ``error`` (the exception class, never its message) when
it raised. If the row cannot be written, or the kernel never confirmed one, nothing runs.
Reads and every other write keep the post-only row.

The kernels here carry only the ledger hooks (``register_side_effect_ledger``) and, where
a test needs one, a stub: the approval queue's own checks are tested with the loop
(``test_destructive_approval_loop``) and ToolService (``test_approved_call_executor``).
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, Protocol

import pytest

from iris_harness.agent.agentic_core import ToolSpec
from iris_harness.agent.tool_runner import (
    NO_PRE_RECORD,
    CapabilityCall,
    GovernedToolRunner,
    ToolCall,
    ToolUnavailable,
)
from iris_harness.foundation.capabilities import CapabilityDenied, CapabilitySpec, MethodSpec
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.side_effects import (
    DeferredSideEffectLedger,
    SideEffectLedger,
    SideEffectRow,
)
from iris_harness.kernel.governance.side_effects.probes import NO_PROBE
from iris_harness.kernel.governance.wiring import register_side_effect_ledger

SECRET_TEXT = "owner@example.com wants it gone"


class _Hook:
    """A stub hook: records what it saw, answers ``outcome``."""

    def __init__(self, name: str, point: HookPoint, priority: int, outcome: str = "allow") -> None:
        self.name = name
        self.hook_point = point
        self.priority = priority
        self.outcome = outcome
        self.seen: list[HookContext] = []

    async def __call__(self, ctx: HookContext) -> HookDecision:
        self.seen.append(ctx)
        return HookDecision(outcome=self.outcome, reason=f"{self.name} says {self.outcome}")  # type: ignore[arg-type]


class _BrokenLedger(SideEffectLedger):
    """A ledger whose writes fail, with a message that must never surface."""

    def record(self, **kwargs: Any) -> str:
        raise sqlite3.OperationalError(f"disk I/O error at /secret/{SECRET_TEXT}")


def _kernel(
    tmp_path: Path,
    ledger: SideEffectLedger | None,
    *hooks: Any,
    ledger_hooks: bool = True,
    high_risk_only: bool = False,
) -> GovernanceKernel:
    kernel = GovernanceKernel(audit_log=AuditLog(db_path=tmp_path / "audit.db"))
    for hook in hooks:
        kernel.register(hook)
    if ledger_hooks:
        register_side_effect_ledger(kernel, ledger, high_risk_only=high_risk_only)
    kernel.init_lock()
    return kernel


def _all_rows(ledger: SideEffectLedger) -> list[SideEffectRow]:
    with sqlite3.connect(ledger.db_path) as conn:
        keys = [r[0] for r in conn.execute("SELECT side_effect_id FROM side_effect_ledger")]
    rows = [ledger.get(key) for key in keys]
    return [row for row in rows if row is not None]


# ---------------------------------------------------------------- GovernedToolRunner.execute
class _Tools:
    def __init__(self) -> None:
        self.ran: list[str] = []

    def spec(
        self,
        name: str,
        effect: str = "destructive",
        confirm: str = "approval",
        *,
        fail: bool = False,
        result: str = "done",
        verify: str | None = None,
    ) -> ToolSpec:
        def call(args: dict[str, Any]) -> str:
            self.ran.append(name)
            if fail:
                raise RuntimeError(SECRET_TEXT)
            return result

        return ToolSpec(name, name, call, effect=effect, confirm=confirm, verify=verify)


def _approved(run_id: str = "run-1") -> ToolCall:
    return ToolCall(run_id=run_id, step_id=2, approved_by="appr-1", caller="model:chat")


@pytest.mark.parametrize(
    ("effect", "confirm"), [("destructive", "approval"), ("write", "approval")]
)
def test_a_high_risk_call_is_pending_before_it_runs_and_completed_after(
    tmp_path: Path, effect: str, confirm: str
) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    seen_before: list[SideEffectRow | None] = []
    tools = _Tools()
    base = tools.spec("wipe", effect, confirm)

    def call(args: dict[str, Any]) -> str:
        seen_before.extend(_all_rows(ledger))
        return base.call(args)

    runner = GovernedToolRunner(kernel=_kernel(tmp_path, ledger), agent_type="chat")
    outcome = runner.execute(base._replace(call=call), {"id": 7}, _approved())

    assert outcome.status == "ran" and outcome.ok and tools.ran == ["wipe"]
    # The row was there, pending, while the tool ran -- keyed by the call id PRE minted.
    (before,) = seen_before
    assert before is not None and before.status == "pending"
    assert before.probe_metadata["pre_recorded"] is True
    (after,) = _all_rows(ledger)
    # POST settled the same row: the id minted before PRE reached POST.
    assert after.side_effect_id == before.side_effect_id
    assert after.side_effect_id.startswith("run-1:2:")
    assert (after.status, after.error) == ("completed", None)
    assert after.completed_at is not None
    # The row holds no arguments.
    assert "7" not in json.dumps(after.probe_metadata).replace(after.side_effect_id, "")


def test_a_high_risk_call_that_raises_is_settled_as_an_error_by_class_only(
    tmp_path: Path,
) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    tools = _Tools()
    runner = GovernedToolRunner(kernel=_kernel(tmp_path, ledger), agent_type="chat")
    outcome = runner.execute(tools.spec("wipe", fail=True), {}, _approved())

    assert outcome.status == "ran" and not outcome.ok
    (row,) = _all_rows(ledger)
    assert (row.status, row.error, row.completed_at) == ("error", "RuntimeError", None)
    assert SECRET_TEXT not in json.dumps(dataclasses.asdict(row))


def _unavailable_tool(name: str, *, cause: bool = True) -> ToolSpec:
    """A tool as the plugin fault boundary hands it on: its own code raised, and the
    boundary re-raised ``ToolUnavailable`` (message: text from the call) from the cause."""

    def call(args: dict[str, Any]) -> str:
        if not cause:
            raise ToolUnavailable(f"{name} is unavailable. {SECRET_TEXT}")
        try:
            raise ValueError(SECRET_TEXT)
        except ValueError as exc:
            raise ToolUnavailable(f"{name} is unavailable ({SECRET_TEXT}).") from exc

    return ToolSpec(name, name, call, effect="destructive", confirm="approval")


def test_a_high_risk_call_whose_tool_was_unavailable_is_settled_as_an_error(
    tmp_path: Path,
) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    runner = GovernedToolRunner(kernel=_kernel(tmp_path, ledger), agent_type="chat")
    outcome = runner.execute(_unavailable_tool("wipe"), {}, _approved())

    assert outcome.status == "ran" and not outcome.ok
    (row,) = _all_rows(ledger)
    # The class of what the tool's code raised (the cause), never the message; and not
    # ``completed`` with its probe attached: the call did not run.
    assert (row.status, row.error, row.completed_at) == ("error", "ValueError", None)
    assert SECRET_TEXT not in json.dumps(dataclasses.asdict(row))


def test_an_unavailable_tool_with_no_cause_is_settled_under_its_own_class(
    tmp_path: Path,
) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    runner = GovernedToolRunner(kernel=_kernel(tmp_path, ledger), agent_type="chat")
    outcome = runner.execute(_unavailable_tool("wipe", cause=False), {}, _approved())

    assert not outcome.ok
    (row,) = _all_rows(ledger)
    assert (row.status, row.error) == ("error", "ToolUnavailable")
    assert SECRET_TEXT not in json.dumps(dataclasses.asdict(row))


# ---------------------------------------------------- the default scope: high-risk only
def test_by_default_a_plain_write_leaves_no_row_and_never_opens_the_ledger(
    tmp_path: Path,
) -> None:
    db = tmp_path / "side_effects.db"
    ledger = DeferredSideEffectLedger(db)
    tools = _Tools()
    runner = GovernedToolRunner(
        kernel=_kernel(tmp_path, ledger, high_risk_only=True), agent_type="chat"
    )
    for effect, confirm in [("write", "once"), ("write", "never"), ("read", "never")]:
        outcome = runner.execute(
            tools.spec(f"w-{effect}-{confirm}", effect, confirm),
            {},
            ToolCall(run_id="run-1", step_id=2, caller="model:chat", asked_user=True),
        )
        assert outcome.status == "ran" and outcome.ok
    assert len(tools.ran) == 3
    assert not db.exists()  # no row, no commit, no file


@pytest.mark.parametrize(
    ("effect", "confirm"), [("destructive", "approval"), ("write", "approval")]
)
def test_by_default_a_high_risk_call_is_pre_recorded_and_settles(
    tmp_path: Path, effect: str, confirm: str
) -> None:
    db = tmp_path / "side_effects.db"
    ledger = DeferredSideEffectLedger(db)
    tools = _Tools()
    seen_before: list[SideEffectRow] = []
    base = tools.spec("wipe", effect, confirm)

    def call(args: dict[str, Any]) -> str:
        seen_before.extend(_all_rows(ledger))
        return base.call(args)

    runner = GovernedToolRunner(
        kernel=_kernel(tmp_path, ledger, high_risk_only=True), agent_type="chat"
    )
    outcome = runner.execute(base._replace(call=call), {}, _approved())

    assert outcome.status == "ran" and outcome.ok
    (before,) = seen_before
    assert before.status == "pending"
    (after,) = _all_rows(ledger)
    assert (after.side_effect_id, after.status) == (before.side_effect_id, "completed")


def test_by_default_a_high_risk_call_that_raises_settles_as_an_error(tmp_path: Path) -> None:
    ledger = DeferredSideEffectLedger(tmp_path / "side_effects.db")
    tools = _Tools()
    runner = GovernedToolRunner(
        kernel=_kernel(tmp_path, ledger, high_risk_only=True), agent_type="chat"
    )
    runner.execute(tools.spec("wipe", fail=True), {}, _approved())
    (row,) = _all_rows(ledger)
    assert (row.status, row.error) == ("error", "RuntimeError")


def test_by_default_a_confirm_only_pinned_write_is_pre_recorded_and_settled(
    tmp_path: Path,
) -> None:
    """``effect=write, confirm=approval`` is high-risk by its confirm alone.

    Under the default scope the post hook tells it from a plain write by the ``tool_confirm``
    the runner stamps on the POST metadata; without it the row would stay ``pending``.
    """
    ledger = DeferredSideEffectLedger(tmp_path / "side_effects.db")
    tools = _Tools()
    seen: list[str] = []
    base = tools.spec("send", "write", "approval")

    def call(args: dict[str, Any]) -> str:
        seen.extend(r.status for r in _all_rows(ledger))
        return base.call(args)

    runner = GovernedToolRunner(
        kernel=_kernel(tmp_path, ledger, high_risk_only=True), agent_type="chat"
    )
    outcome = runner.execute(base._replace(call=call), {}, _approved())

    assert outcome.status == "ran" and outcome.ok
    assert seen == ["pending"]
    (row,) = _all_rows(ledger)
    assert (row.status, row.error) == ("completed", None)


def test_a_plain_write_with_the_flag_on_is_still_recorded_after_the_call(
    tmp_path: Path,
) -> None:
    """``IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER=1``: every non-read call, as before."""
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    tools = _Tools()
    runner = GovernedToolRunner(kernel=_kernel(tmp_path, ledger), agent_type="chat")
    runner.execute(
        tools.spec("note", "write", "once"),
        {},
        ToolCall(run_id="run-1", step_id=2, caller="model:chat", asked_user=True),
    )
    (row,) = _all_rows(ledger)
    assert (row.tool, row.status) == ("note", "pending")


def test_with_the_flag_off_a_destructive_call_is_denied_and_a_plain_write_runs(
    tmp_path: Path,
) -> None:
    """``IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER=0``: no ledger, high-risk calls denied (D3)."""
    tools = _Tools()
    runner = GovernedToolRunner(kernel=_kernel(tmp_path, None), agent_type="chat")
    held = runner.execute(tools.spec("wipe"), {}, _approved())
    plain = runner.execute(
        tools.spec("note", "write", "once"),
        {},
        ToolCall(run_id="run-1", step_id=2, caller="model:chat", asked_user=True),
    )
    assert held.status == "held" and plain.status == "ran"
    assert tools.ran == ["note"]


def test_a_high_risk_call_whose_row_cannot_be_written_never_runs(tmp_path: Path) -> None:
    tools = _Tools()
    broken = _BrokenLedger(tmp_path / "side_effects.db")
    runner = GovernedToolRunner(kernel=_kernel(tmp_path, broken), agent_type="chat")
    outcome = runner.execute(tools.spec("wipe"), {}, _approved())

    assert outcome.status == "held" and tools.ran == []
    assert outcome.decision is not None and outcome.decision.outcome == "deny"
    assert "OperationalError" in outcome.decision.reason
    assert SECRET_TEXT not in outcome.decision.reason


def test_a_kernel_with_no_ledger_runs_no_high_risk_call(tmp_path: Path) -> None:
    tools = _Tools()
    runner = GovernedToolRunner(kernel=_kernel(tmp_path, None), agent_type="chat")
    outcome = runner.execute(tools.spec("wipe"), {}, _approved())

    assert outcome.status == "held" and tools.ran == []
    assert outcome.decision is not None
    assert "no side-effect ledger is configured" in outcome.decision.reason


def test_a_kernel_without_the_ledger_hook_runs_no_high_risk_call(tmp_path: Path) -> None:
    """The runner's own check: allowed, but no row was confirmed -- it does not run."""
    tools = _Tools()
    kernel = _kernel(
        tmp_path, None, _Hook("allow_all", HookPoint.PRE_TOOL_USE, 50), ledger_hooks=False
    )
    runner = GovernedToolRunner(kernel=kernel, agent_type="chat")
    outcome = runner.execute(tools.spec("wipe"), {}, _approved())

    assert outcome.status == "held" and tools.ran == []
    assert outcome.decision is not None and outcome.decision.reason == NO_PRE_RECORD


def test_a_call_denied_before_the_ledger_hook_leaves_no_row(tmp_path: Path) -> None:
    """The ledger hook runs last: a call another hook refuses never reaches it."""
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    tools = _Tools()
    kernel = _kernel(tmp_path, ledger, _Hook("no", HookPoint.PRE_TOOL_USE, 50, "deny"))
    outcome = GovernedToolRunner(kernel=kernel, agent_type="chat").execute(
        tools.spec("wipe"), {}, _approved()
    )

    assert outcome.status == "held" and tools.ran == [] and _all_rows(ledger) == []


def test_an_unapproved_high_risk_call_is_refused_and_leaves_no_row(tmp_path: Path) -> None:
    """With no approval hook the runner refuses it (ADR-0118); the ledger skips it."""
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    tools = _Tools()
    runner = GovernedToolRunner(kernel=_kernel(tmp_path, ledger), agent_type="chat")
    outcome = runner.execute(tools.spec("wipe"), {}, ToolCall(run_id="run-1"))

    assert outcome.status == "refused" and tools.ran == [] and _all_rows(ledger) == []


def test_a_result_withheld_at_post_is_still_settled_completed(tmp_path: Path) -> None:
    """The call ran: a POST_TOOL_USE deny after the ledger (40) withholds the result but
    cannot un-run it."""
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    tools = _Tools()
    kernel = _kernel(tmp_path, ledger, _Hook("withhold", HookPoint.POST_TOOL_USE, 45, "deny"))
    outcome = GovernedToolRunner(kernel=kernel, agent_type="chat").execute(
        tools.spec("wipe"), {}, _approved()
    )

    assert outcome.status == "ran" and outcome.post is not None and outcome.post.withheld
    (row,) = _all_rows(ledger)
    assert row.status == "completed"


def test_a_mapped_tool_gets_its_probe_subject_when_it_returns(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    tools = _Tools()
    spec = tools.spec("git_commit", result=json.dumps({"commit_sha": "abc123"}))
    GovernedToolRunner(kernel=_kernel(tmp_path, ledger), agent_type="chat").execute(
        spec, {}, _approved()
    )

    (row,) = _all_rows(ledger)
    assert (row.status, row.verification_probe, row.probe_subject) == (
        "completed",
        "git_commit",
        "abc123",
    )
    assert row.probe_metadata["pre_recorded"] is True  # the pre-call keys survive


def test_a_declared_probe_is_set_once_the_call_returned_not_before(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    tools = _Tools()
    base = tools.spec("wipe", verify="mail_trashed")
    probes_before: list[str] = []

    def call(args: dict[str, Any]) -> str:
        probes_before.extend(r.verification_probe for r in _all_rows(ledger))
        return base.call(args)

    GovernedToolRunner(kernel=_kernel(tmp_path, ledger), agent_type="chat").execute(
        base._replace(call=call), {}, _approved()
    )

    # A crash in the call would leave this row: no probe, so ``resume`` asks the owner.
    assert probes_before == [NO_PROBE]
    (row,) = _all_rows(ledger)
    assert (row.verification_probe, row.probe_subject) == ("mail_trashed", row.side_effect_id)


@pytest.mark.parametrize(("effect", "confirm"), [("write", "once"), ("write", "never")])
def test_a_plain_write_keeps_its_post_only_row(tmp_path: Path, effect: str, confirm: str) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    tools = _Tools()
    runner = GovernedToolRunner(kernel=_kernel(tmp_path, ledger), agent_type="chat")
    outcome = runner.execute(
        tools.spec("note", effect, confirm), {}, ToolCall(run_id="run-1", asked_user=True)
    )

    assert outcome.status == "ran"
    (row,) = _all_rows(ledger)
    assert row.status == "pending" and "pre_recorded" not in row.probe_metadata


def test_a_plain_write_runs_with_no_ledger_at_all(tmp_path: Path) -> None:
    """Unchanged: only high-risk calls need the ledger to run."""
    tools = _Tools()
    runner = GovernedToolRunner(kernel=_kernel(tmp_path, None), agent_type="chat")
    outcome = runner.execute(tools.spec("note", "write", "never"), {}, ToolCall(run_id="r"))
    assert outcome.status == "ran" and tools.ran == ["note"]


def test_a_read_leaves_no_row(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    tools = _Tools()
    GovernedToolRunner(kernel=_kernel(tmp_path, ledger), agent_type="chat").execute(
        tools.spec("look", "read", "never"), {}, ToolCall(run_id="run-1")
    )
    assert _all_rows(ledger) == []


def test_the_ledger_hook_never_sees_a_resolved_secret_on_the_row(tmp_path: Path) -> None:
    """It runs after the credential broker, and writes nothing from the arguments."""
    from iris_harness.kernel.governance.plugins.credential_broker import CredentialBroker

    class _Vault:
        def get(self, handle: str) -> str | None:
            return "s3cret-value" if handle == "vault://token" else None

    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    tools = _Tools()
    kernel = _kernel(tmp_path, ledger, CredentialBroker(vault=_Vault()))
    GovernedToolRunner(kernel=kernel, agent_type="chat").execute(
        tools.spec("wipe"), {"key": "vault://token"}, _approved()
    )

    (row,) = _all_rows(ledger)
    assert "s3cret" not in json.dumps(dataclasses.asdict(row))


# ------------------------------------------------------------------------ capability calls
@dataclasses.dataclass(frozen=True)
class Receipt:
    note: str


class Shredder(Protocol):
    def shred(self, id: int) -> Receipt: ...
    async def ashred(self, id: int) -> Receipt: ...
    def shred_all(self, ids: list[int]) -> Iterator[Receipt]: ...
    def ashred_all(self, ids: list[int]) -> AsyncIterator[Receipt]: ...


# A MethodSpec declares ``read`` / ``write`` today; the runner follows whatever effect and
# confirm the call carries, so the high-risk calls are built from a plain write's.
_SPEC = CapabilitySpec(
    name="test.shredder",
    protocol=Shredder,
    methods={
        m: MethodSpec(effect="write", confirm="never", fields=("note",))
        for m in ("shred", "ashred", "shred_all", "ashred_all")
    },
)


def _cap(method: str, *, effect: str = "destructive", confirm: str = "approval") -> CapabilityCall:
    return CapabilityCall(
        caller="plugin:consumer",
        provider="provider",
        capability=_SPEC.name,
        method=method,
        effect=effect,
        confirm=confirm,
        fields=("note",),
        shape=_SPEC.shapes[method],
        value_type=_SPEC.value_types[method],
    )


class _Provider:
    def __init__(self, ledger: SideEffectLedger, *, fail: bool = False) -> None:
        self.ledger = ledger
        self.fail = fail
        self.calls: list[str] = []
        self.rows_while_running: list[SideEffectRow] = []

    def _run(self, name: str) -> None:
        self.calls.append(name)
        self.rows_while_running.extend(_all_rows(self.ledger))
        if self.fail:
            raise ValueError(SECRET_TEXT)

    def shred(self, id: int) -> Receipt:
        self._run("shred")
        return Receipt("shredded")

    async def ashred(self, id: int) -> Receipt:
        self._run("ashred")
        return Receipt("shredded")

    def shred_all(self, ids: list[int]) -> Iterator[Receipt]:
        self._run("shred_all")
        for i in ids:
            yield Receipt(f"shredded {i}")

    async def ashred_all(self, ids: list[int]) -> AsyncIterator[Receipt]:
        self._run("ashred_all")
        for i in ids:
            yield Receipt(f"shredded {i}")


def _runner(kernel: GovernanceKernel) -> GovernedToolRunner:
    return GovernedToolRunner(kernel=kernel, agent_type="plugin:consumer")


def _one_row(ledger: SideEffectLedger) -> SideEffectRow:
    (row,) = _all_rows(ledger)
    return row


def test_execute_call_records_before_and_settles_after(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    provider = _Provider(ledger)
    result = _runner(_kernel(tmp_path, ledger)).execute_call(
        _cap("shred"), provider.shred, {"id": 1}
    )

    assert result == Receipt("shredded")
    (before,) = provider.rows_while_running
    assert before.status == "pending" and before.probe_metadata["pre_recorded"] is True
    row = _one_row(ledger)
    assert (row.side_effect_id, row.status) == (before.side_effect_id, "completed")


def test_execute_call_that_raises_is_settled_as_an_error(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    provider = _Provider(ledger, fail=True)
    with pytest.raises(ValueError):
        _runner(_kernel(tmp_path, ledger)).execute_call(_cap("shred"), provider.shred, {"id": 1})

    row = _one_row(ledger)
    assert (row.status, row.error) == ("error", "ValueError")
    assert SECRET_TEXT not in json.dumps(dataclasses.asdict(row))


async def test_aexecute_call_records_before_and_settles_after(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    provider = _Provider(ledger)
    result = await _runner(_kernel(tmp_path, ledger)).aexecute_call(
        _cap("ashred"), provider.ashred, {"id": 1}
    )

    assert result == Receipt("shredded")
    assert [r.status for r in provider.rows_while_running] == ["pending"]
    assert _one_row(ledger).status == "completed"


async def test_aexecute_call_that_raises_is_settled_as_an_error(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    provider = _Provider(ledger, fail=True)
    with pytest.raises(ValueError):
        await _runner(_kernel(tmp_path, ledger)).aexecute_call(
            _cap("ashred"), provider.ashred, {"id": 1}
        )
    assert (_one_row(ledger).status, _one_row(ledger).error) == ("error", "ValueError")


def test_a_sync_stream_is_settled_once_at_its_end(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    provider = _Provider(ledger)
    stream = _runner(_kernel(tmp_path, ledger)).execute_call(
        _cap("shred_all"), provider.shred_all, {"ids": [1, 2]}
    )
    items = []
    for item in stream:
        items.append(item)
        assert _one_row(ledger).status == "pending"  # an item settles nothing

    assert items == [Receipt("shredded 1"), Receipt("shredded 2")]
    assert (_one_row(ledger).status, _one_row(ledger).error) == ("completed", None)


def test_a_sync_stream_stopped_part_way_is_settled_as_an_error(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    provider = _Provider(ledger)
    stream = _runner(_kernel(tmp_path, ledger)).execute_call(
        _cap("shred_all"), provider.shred_all, {"ids": [1, 2]}
    )
    next(stream)
    stream.close()  # the consumer stops early

    assert (_one_row(ledger).status, _one_row(ledger).error) == ("error", "GeneratorExit")


async def test_aexecute_stream_is_settled_once_at_its_end(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    provider = _Provider(ledger)
    stream = _runner(_kernel(tmp_path, ledger)).aexecute_stream(
        _cap("ashred_all"), provider.ashred_all, {"ids": [1, 2]}
    )
    items = [item async for item in stream]

    assert len(items) == 2 and provider.calls == ["ashred_all"]
    assert [r.status for r in provider.rows_while_running] == ["pending"]
    assert (_one_row(ledger).status, _one_row(ledger).error) == ("completed", None)


async def test_aexecute_stream_that_raises_is_settled_with_the_class(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    provider = _Provider(ledger, fail=True)
    stream = _runner(_kernel(tmp_path, ledger)).aexecute_stream(
        _cap("ashred_all"), provider.ashred_all, {"ids": [1]}
    )
    with pytest.raises(ValueError):
        async for _ in stream:
            pass

    row = _one_row(ledger)
    assert (row.status, row.error) == ("error", "ValueError")
    assert SECRET_TEXT not in json.dumps(dataclasses.asdict(row))


@pytest.mark.parametrize("ledger_kind", ["none", "broken", "no_hook"])
def test_a_high_risk_capability_call_with_no_record_never_runs(
    tmp_path: Path, ledger_kind: str
) -> None:
    ledger: SideEffectLedger | None = {
        "none": None,
        "broken": _BrokenLedger(tmp_path / "side_effects.db"),
        "no_hook": None,
    }[ledger_kind]
    kernel = _kernel(
        tmp_path,
        ledger,
        _Hook("allow_all", HookPoint.PRE_TOOL_USE, 50),
        ledger_hooks=ledger_kind != "no_hook",
    )
    provider = _Provider(SideEffectLedger(tmp_path / "other.db"))
    with pytest.raises(CapabilityDenied) as denied:
        _runner(kernel).execute_call(_cap("shred"), provider.shred, {"id": 1})

    assert provider.calls == []
    assert SECRET_TEXT not in denied.value.reason
    if ledger_kind == "no_hook":
        assert NO_PRE_RECORD in denied.value.reason


async def test_an_async_high_risk_capability_call_with_no_ledger_never_runs(
    tmp_path: Path,
) -> None:
    provider = _Provider(SideEffectLedger(tmp_path / "other.db"))
    runner = _runner(_kernel(tmp_path, None))
    with pytest.raises(CapabilityDenied):
        await runner.aexecute_call(_cap("ashred"), provider.ashred, {"id": 1})
    with pytest.raises(CapabilityDenied):
        async for _ in runner.aexecute_stream(
            _cap("ashred_all"), provider.ashred_all, {"ids": [1]}
        ):
            pass
    assert provider.calls == []


@pytest.mark.parametrize(
    ("effect", "confirm"), [("destructive", "approval"), ("write", "approval")]
)
def test_by_default_a_pinned_capability_call_is_pre_recorded_and_settled(
    tmp_path: Path, effect: str, confirm: str
) -> None:
    ledger = DeferredSideEffectLedger(tmp_path / "side_effects.db")
    provider = _Provider(ledger)
    _runner(_kernel(tmp_path, ledger, high_risk_only=True)).execute_call(
        _cap("shred", effect=effect, confirm=confirm), provider.shred, {"id": 1}
    )

    (before,) = provider.rows_while_running
    assert before.status == "pending"
    row = _one_row(ledger)
    assert (row.side_effect_id, row.status) == (before.side_effect_id, "completed")


def test_a_plain_capability_write_keeps_its_post_only_row(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    provider = _Provider(ledger)
    _runner(_kernel(tmp_path, ledger)).execute_call(
        _cap("shred", effect="write", confirm="never"), provider.shred, {"id": 1}
    )

    assert provider.rows_while_running == []  # nothing before it ran
    row = _one_row(ledger)
    assert row.status == "pending" and "pre_recorded" not in row.probe_metadata


def test_a_plain_capability_write_runs_with_no_ledger(tmp_path: Path) -> None:
    provider = _Provider(SideEffectLedger(tmp_path / "other.db"))
    _runner(_kernel(tmp_path, None)).execute_call(
        _cap("shred", effect="write", confirm="never"), provider.shred, {"id": 1}
    )
    assert provider.calls == ["shred"]


# ----------------------------------------- a later hook, and a result that cannot be handled
def test_a_later_allowing_hook_does_not_orphan_the_row(tmp_path: Path) -> None:
    """``kernel.fire`` hands back only the last hook's decision: the row's key rides on the
    context, so a hook after the ledger hook cannot make the runner refuse a recorded call."""
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    tools = _Tools()
    kernel = _kernel(tmp_path, ledger, _Hook("later", HookPoint.PRE_TOOL_USE, 150))
    outcome = GovernedToolRunner(kernel=kernel, agent_type="chat").execute(
        tools.spec("wipe"), {}, _approved()
    )

    assert outcome.status == "ran" and tools.ran == ["wipe"]
    assert _one_row(ledger).status == "completed"


def test_a_later_denying_hook_settles_the_row_it_could_not_prevent(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    tools = _Tools()
    kernel = _kernel(tmp_path, ledger, _Hook("late_no", HookPoint.PRE_TOOL_USE, 150, "deny"))
    outcome = GovernedToolRunner(kernel=kernel, agent_type="chat").execute(
        tools.spec("wipe"), {}, _approved()
    )

    assert outcome.status == "held" and tools.ran == []
    row = _one_row(ledger)
    assert (row.status, row.error) == ("error", "NotRun")


def test_a_capability_refused_after_the_ledger_hook_settles_its_row(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    provider = _Provider(ledger)
    kernel = _kernel(tmp_path, ledger, _Hook("late_no", HookPoint.PRE_TOOL_USE, 150, "deny"))
    with pytest.raises(CapabilityDenied):
        _runner(kernel).execute_call(_cap("shred"), provider.shred, {"id": 1})

    assert provider.calls == []
    row = _one_row(ledger)
    assert (row.status, row.error) == ("error", "NotRun")


async def test_an_async_capability_refused_after_the_ledger_hook_settles_its_row(
    tmp_path: Path,
) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    provider = _Provider(ledger)
    kernel = _kernel(tmp_path, ledger, _Hook("late_no", HookPoint.PRE_TOOL_USE, 150, "deny"))
    with pytest.raises(CapabilityDenied):
        await _runner(kernel).aexecute_call(_cap("ashred"), provider.ashred, {"id": 1})

    assert provider.calls == []
    assert (_one_row(ledger).status, _one_row(ledger).error) == ("error", "NotRun")


def test_a_wrong_result_type_settles_the_row_instead_of_leaving_it_pending(
    tmp_path: Path,
) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")
    provider = _Provider(ledger)

    def wrong(id: int) -> str:
        provider.calls.append("shred")
        return "not a Receipt"

    with pytest.raises(CapabilityDenied):
        _runner(_kernel(tmp_path, ledger)).execute_call(_cap("shred"), wrong, {"id": 1})

    assert provider.calls == ["shred"]  # it ran: the row says an error, not "pending"
    row = _one_row(ledger)
    assert (row.status, row.error) == ("error", "ResultMismatch")


async def test_an_async_wrong_result_type_settles_the_row(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "side_effects.db")

    async def wrong(id: int) -> str:
        return "not a Receipt"

    with pytest.raises(CapabilityDenied):
        await _runner(_kernel(tmp_path, ledger)).aexecute_call(_cap("ashred"), wrong, {"id": 1})

    row = _one_row(ledger)
    assert (row.status, row.error) == ("error", "ResultMismatch")
