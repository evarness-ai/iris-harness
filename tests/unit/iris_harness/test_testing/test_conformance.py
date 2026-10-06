"""``iris_harness.testing.conformance``: the suite a plugin's own CI runs (#77).

A conforming plugin passes against a real governed harness; each rule's verdict is
pinned against the ledger rows that would break it, since governance itself never lets a
real plugin produce them.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Protocol

import pytest

from iris_harness.foundation import capabilities as catalogue
from iris_harness.foundation.capabilities import CapabilitySpec, MethodSpec
from iris_harness.kernel.governance.audit.log import AuditRow
from iris_harness.runtime.tool_service import ToolInfo, ToolResult
from iris_harness.sdk import PluginAPI
from iris_harness.testing import (
    ConformanceError,
    Violation,
    assert_conformant,
    check_conformance,
    conformance,
    plugin,
)

MANIFEST: dict[str, Any] = {
    "name": "board",
    "provides": ["tool"],
    "tools": {
        "list_notes": {"effect": "read"},
        "pin_note": {"effect": "write", "confirm": "once"},
        "remove_note": {"effect": "destructive"},
    },
}


@dataclass
class Board:
    notes: dict[str, bool]


def _board_plugin(board: Board, *, broken: str | None = None) -> Any:
    mounts: list[None] = []

    def list_notes(args: dict[str, Any]) -> str:
        return ", ".join(sorted(board.notes))

    def pin_note(args: dict[str, Any]) -> str:
        board.notes[args["note_id"]] = True
        return f"Pinned {args['note_id']}"

    def remove_note(args: dict[str, Any]) -> str:
        if broken == "remove raises":
            raise RuntimeError("the board is locked")
        board.notes.pop(args["note_id"])
        return f"Removed {args['note_id']}"

    def setup(api: PluginAPI) -> None:
        mounts.append(None)
        if broken == "setup raises" or (broken == "second setup raises" and len(mounts) > 1):
            raise RuntimeError("cannot start")
        api.register_tool("list_notes", "List the notes.", list_notes)
        api.register_tool("pin_note", 'Pin a note. Args: {"note_id": str}.', pin_note)
        api.register_tool("remove_note", 'Remove a note. Args: {"note_id": str}.', remove_note)

    return plugin(setup, manifest=MANIFEST)


EXAMPLES = {"list_notes": {}, "pin_note": {"note_id": "n1"}, "remove_note": {"note_id": "n2"}}


def _fresh() -> Board:
    return Board(notes={"n1": False, "n2": False, "n3": False})


# ------------------------------------------------------------------- against a harness
def test_a_conforming_plugin_has_no_violations() -> None:
    board = _fresh()
    assert check_conformance(_board_plugin(board), tools=EXAMPLES) == []
    # The approving run removed n2 and pinned n1; the rejecting run changed nothing more.
    assert "n2" not in board.notes and board.notes["n1"] is True


def test_assert_conformant_passes_quietly() -> None:
    assert_conformant(_board_plugin(_fresh()), tools=EXAMPLES)


def test_a_declared_tool_without_an_example_is_reported_not_passed() -> None:
    examples = {k: v for k, v in EXAMPLES.items() if k != "remove_note"}
    violations = check_conformance(_board_plugin(_fresh()), tools=examples)
    assert violations == [
        Violation("coverage", "remove_note", "declared but no example call was given")
    ]


def test_an_example_for_an_undeclared_tool_is_reported() -> None:
    violations = check_conformance(_board_plugin(_fresh()), tools={**EXAMPLES, "nope": {}})
    assert [(v.check, v.subject) for v in violations] == [("coverage", "nope")]


def test_a_plugin_that_does_not_mount_is_reported() -> None:
    violations = check_conformance(_board_plugin(_fresh(), broken="setup raises"), tools=EXAMPLES)
    assert [(v.check, v.subject) for v in violations] == [("mount", "board")]


def test_an_example_call_that_fails_is_reported() -> None:
    violations = check_conformance(_board_plugin(_fresh(), broken="remove raises"), tools=EXAMPLES)
    assert any(v.check == "example" and v.subject == "remove_note" for v in violations)


def test_assert_conformant_lists_every_violation() -> None:
    with pytest.raises(ConformanceError) as raised:
        assert_conformant(_board_plugin(_fresh()), tools={})
    assert len(raised.value.violations) == 3
    assert "[coverage] pin_note" in str(raised.value)


def test_a_capability_this_sdk_does_not_publish_is_reported() -> None:
    violations = check_conformance(
        _board_plugin(_fresh()), tools=EXAMPLES, capabilities={"no.such": {"go": {}}}
    )
    assert [(v.check, v.subject) for v in violations] == [("coverage", "no.such")]


# ------------------------------------------------- the verdicts, on rows governance forbids
def _row(
    hook: str,
    tool: str,
    *,
    caller: str = "plugin:board",
    digest: str | None = "d1",
    plugin: str = "tool_policy",
    decision: str = "allow",
) -> AuditRow:
    payload: dict[str, Any] = {"tool_name": tool, "caller": caller}
    if digest is not None:
        payload["args_digest"] = digest
    return AuditRow(
        id=1,
        ts=datetime.now(UTC).isoformat(),
        run_id="r1",
        step_id=None,
        agent_type="plugin:board",
        hook_point=hook,
        plugin=plugin,
        decision=decision,
        classification=None,
        tier=None,
        cost_usd=None,
        severity="info",
        reason="",
        payload_json=json.dumps(payload),
    )


def test_a_call_with_no_pre_row_is_an_audit_violation() -> None:
    rows = [_row("post_tool_use", "t")]
    assert [v.check for v in conformance._audited("t", rows, "plugin:board", ran=True)] == ["audit"]


def test_a_call_that_ran_with_no_post_row_is_an_audit_violation() -> None:
    rows = [_row("pre_tool_use", "t")]
    assert [v.check for v in conformance._audited("t", rows, "plugin:board", ran=True)] == ["audit"]
    assert conformance._audited("t", rows, "plugin:board", ran=False) == []


def test_a_row_naming_another_caller_is_a_caller_violation() -> None:
    rows = [_row("pre_tool_use", "t"), _row("post_tool_use", "t", caller="plugin:system")]
    [violation] = conformance._audited("t", rows, "plugin:board", ran=True)
    assert violation.check == "caller" and "plugin:system" in violation.detail


# ------------------------------------------------------------------------- capabilities
@dataclass(frozen=True)
class Note:
    text: str


class Notes(Protocol):
    def get(self, q: str) -> Note: ...
    async def aget(self, q: str) -> Note: ...
    def stream(self, q: str) -> Iterator[Note]: ...
    def astream(self, q: str) -> AsyncIterator[Note]: ...


NOTES = CapabilitySpec(
    name="test.notes",
    protocol=Notes,
    methods={m: MethodSpec(fields=("text",)) for m in ("get", "aget", "stream", "astream")},
)


class NotesProvider:
    def get(self, q: str) -> Note:
        return Note(q)

    async def aget(self, q: str) -> Note:
        return Note(q)

    def stream(self, q: str) -> Iterator[Note]:
        yield Note(q)

    async def astream(self, q: str) -> AsyncIterator[Note]:
        yield Note(q)


def _notes_plugin() -> Any:
    return plugin(
        lambda api: api.provide("test.notes", NotesProvider()),
        manifest={"name": "notes", "capabilities": {"provides": ["test.notes"]}},
    )


NOTE_EXAMPLES = {"test.notes": {m: {"q": "hi"} for m in ("get", "aget", "stream", "astream")}}


def test_a_provided_capability_is_called_as_a_consumer_would(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(catalogue, "CAPABILITIES", MappingProxyType({"test.notes": NOTES}))
    assert check_conformance(_notes_plugin(), capabilities=NOTE_EXAMPLES) == []


def test_a_provided_capability_method_without_an_example_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(catalogue, "CAPABILITIES", MappingProxyType({"test.notes": NOTES}))
    examples = {"test.notes": {"get": {"q": "hi"}}}
    violations = check_conformance(_notes_plugin(), capabilities=examples)
    assert sorted(v.subject for v in violations) == [
        "capability:test.notes.aget",
        "capability:test.notes.astream",
        "capability:test.notes.stream",
    ]


def test_the_capability_calls_are_checked_against_the_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not vacuous: with the ledger's rows hidden, every method is an audit violation."""
    monkeypatch.setattr(catalogue, "CAPABILITIES", MappingProxyType({"test.notes": NOTES}))
    monkeypatch.setattr(conformance, "_tool_rows", lambda *a, **k: [])
    violations = check_conformance(_notes_plugin(), capabilities=NOTE_EXAMPLES)
    assert sorted((v.check, v.subject) for v in violations) == [
        ("audit", f"capability:test.notes.{m}")
        for m in ("aget", "astream", "get", "stream")
        for _ in ("no pre row", "no post row")
    ]


def test_the_tool_calls_are_checked_against_the_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conformance, "_tool_rows", lambda *a, **k: [])
    violations = check_conformance(_board_plugin(_fresh()), tools={"list_notes": {}})
    assert [(v.check, v.subject) for v in violations if v.subject == "list_notes"] == [
        ("audit", "list_notes"),
        ("audit", "list_notes"),
    ]


# ----------------------------------------- the approval verdicts, one scripted call each
HOOKS = ("tool_policy", "output_classifier", "post_tool_use_ledger")


def _pre(digest: str | None = "d1") -> list[AuditRow]:
    return [_row("pre_tool_use", "t", digest=digest)]


def _post(times: int = 1) -> list[AuditRow]:
    """What ``times`` executions leave: each hook writes one post row per execution."""
    return [_row("post_tool_use", "t", digest=None, plugin=hook) for hook in HOOKS] * times


class _Calls:
    def __init__(self, result: ToolResult) -> None:
        self._result = result

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        return self._result


class _Service:
    def __init__(self, info: ToolInfo | None, result: ToolResult) -> None:
        self._info, self._result = info, result

    def describe(self, name: str) -> list[ToolInfo]:
        return [self._info] if self._info is not None else []

    def for_caller(self, caller: str) -> _Calls:
        return _Calls(self._result)


class _Registry:
    def plugins(self) -> list[Any]:
        return []


class _Runtime:
    tool_service: Any = None
    plugin_registry = _Registry()


class _Harness:
    """Just what ``_tool_call`` reads from a harness; its answers are scripted."""

    def __init__(self) -> None:
        self._runtime = _Runtime()
        self.answered: list[tuple[str, bool]] = []
        self._mark = 0

    def _audit_high_water(self) -> int:
        self._mark += 1
        return self._mark

    def respond_to_approval(self, approval_id: str, *, approve: bool) -> str:
        self.answered.append((approval_id, approve))
        return ""


HELD = ToolResult(ok=False, text="Queued", held=True, approval_id="a1")
RAN = ToolResult(ok=True, text="done")


def _verdict(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: ToolResult,
    first: list[AuditRow],
    after: list[AuditRow] | None = None,
    status: str | None = "ran",
    approve: bool = True,
    effect: str = "destructive",
    confirm: str = "never",
) -> list[Violation]:
    h = _Harness()
    info = ToolInfo(name="t", description="", effect=effect, confirm=confirm)
    h._runtime.tool_service = _Service(info, result)
    monkeypatch.setattr(conformance, "_tool_service", lambda _h: h._runtime.tool_service)
    rows = deque([first, after or []])
    monkeypatch.setattr(conformance, "_tool_rows", lambda *a, **k: rows.popleft())
    monkeypatch.setattr(conformance, "_approval_status", lambda *a, **k: status)
    out = conformance._tool_call(h, "plugin:board", "t", {}, approve=approve)  # type: ignore[arg-type]
    if result.held and result.approval_id is not None and (effect != "read"):
        assert h.answered == [("a1", approve)]
    return out


def test_a_held_call_approved_and_run_once_with_its_arguments_has_no_violations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _verdict(monkeypatch, result=HELD, first=_pre(), after=_pre() + _post()) == []


def test_a_destructive_tool_that_ran_without_approval_is_a_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    violations = _verdict(monkeypatch, result=RAN, first=_pre() + _post())
    assert [(v.check, v.detail) for v in violations] == [
        ("approval", "ran from code without the owner's approval")
    ]


def test_a_write_that_confirms_and_ran_without_approval_is_a_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    violations = _verdict(
        monkeypatch, result=RAN, first=_pre() + _post(), effect="write", confirm="once"
    )
    assert [v.check for v in violations] == ["approval"]


def test_a_tool_held_though_its_declaration_says_it_runs_is_a_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    violations = _verdict(monkeypatch, result=HELD, first=_pre(), effect="read")
    assert [(v.check, v.detail.startswith("held, but its declaration")) for v in violations] == [
        ("approval", True)
    ]


def test_a_hold_with_no_queued_approval_is_a_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    held = ToolResult(ok=False, text="refused", held=True, approval_id=None)
    violations = _verdict(monkeypatch, result=held, first=_pre())
    assert [
        (v.check, v.detail.startswith("held without a queued approval")) for v in violations
    ] == [("approval", True)]


def test_a_call_that_ran_after_the_owner_rejected_it_is_a_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran = _verdict(
        monkeypatch,
        result=HELD,
        first=_pre(),
        after=_pre() + _post(),
        approve=False,
        status="ran",
    )
    assert [(v.check, v.detail) for v in ran] == [("approval", "ran after the owner rejected it")]
    # A post row alone, whatever the queue recorded, is a run too.
    posted = _verdict(
        monkeypatch, result=HELD, first=_pre(), after=_post(), approve=False, status="rejected"
    )
    assert [v.check for v in posted] == ["approval"]


def test_a_rejected_call_that_never_ran_has_no_violations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _verdict(monkeypatch, result=HELD, first=_pre(), approve=False, status="rejected") == []


def test_a_rejection_with_no_outcome_row_is_an_audit_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    violations = _verdict(monkeypatch, result=HELD, first=_pre(), approve=False, status=None)
    assert [v.check for v in violations] == ["audit"]


def test_an_approved_call_that_ran_more_than_once_is_a_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both runs carry the run id the call opened, so distinct run ids see one."""
    violations = _verdict(monkeypatch, result=HELD, first=_pre(), after=_pre() + _post(times=2))
    assert [(v.check, v.detail) for v in violations] == [
        ("approval", "approved, it ran 2 time(s), not once")
    ]


def test_an_approved_call_that_never_ran_is_a_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    violations = _verdict(monkeypatch, result=HELD, first=_pre(), after=_pre(), status="denied")
    assert sorted(v.detail for v in violations) == [
        "approved, but it did not run (denied)",
        "approved, it ran 0 time(s), not once",
        "ran with no post_tool_use row",
    ]


def test_an_approved_call_that_ran_with_other_arguments_is_a_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    violations = _verdict(monkeypatch, result=HELD, first=_pre("d1"), after=_pre("d2") + _post())
    assert [(v.check, v.detail) for v in violations] == [
        ("approval", "approved, it ran with arguments other than those queued")
    ]


@pytest.mark.parametrize("which", ["queued", "run", "both"])
def test_arguments_that_cannot_be_compared_are_a_violation_not_a_pass(
    monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    first = _pre(None if which in ("queued", "both") else "d1")
    after = _pre(None if which in ("run", "both") else "d1") + _post()
    violations = _verdict(monkeypatch, result=HELD, first=first, after=after)
    assert [v.check for v in violations] == ["approval"]
    assert "args_digest" in violations[0].detail


def test_an_approval_with_no_outcome_row_is_an_audit_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    violations = _verdict(
        monkeypatch, result=HELD, first=_pre(), after=_pre() + _post(), status=None
    )
    assert [v.check for v in violations] == ["audit"]


# ------------------------------------------- the same verdicts, with governance itself broken
def _broken(monkeypatch: pytest.MonkeyPatch, how: str) -> None:
    from iris_harness.kernel.governance.approvals.queue import ApprovalQueue
    from iris_harness.kernel.governance.approvals.store import ApprovalItem
    from iris_harness.kernel.governance.hooks.types import HookDecision
    from iris_harness.kernel.governance.plugins import destructive_approval as da
    from iris_harness.runtime.tool_service import ToolService

    if how == "skips approval":
        # Governance no longer reads the tool's declaration as needing approval.
        from iris_harness.agent import tool_runner

        monkeypatch.setattr(da, "approved_per_call", lambda metadata: False)
        monkeypatch.setattr(tool_runner, "pinned_by_declaration", lambda effect, confirm: False)
    elif how == "approval hook allows":
        # Only the hook is broken: the runner's own fail-closed guard (ADR-0118) refuses.

        async def allow(self: Any, ctx: Any) -> Any:
            return HookDecision(outcome="allow", reason="broken: no approval")

        monkeypatch.setattr(da.DestructiveApprovalHook, "__call__", allow)
    elif how == "runs twice":
        run = ToolService._run_approved

        def twice(self: Any, row: Any) -> Any:
            run(self, row)
            return run(self, row)

        monkeypatch.setattr(ToolService, "_run_approved", twice)
        monkeypatch.setattr(ApprovalQueue, "claim_execution", lambda self, aid: self.get(aid))
    elif how == "mutates the pinned arguments":
        of = ApprovalItem.of.__func__  # type: ignore[attr-defined]
        monkeypatch.setattr(
            ApprovalItem,
            "of",
            classmethod(lambda cls, tool, args: of(cls, tool, {**args, "note_id": "n3"})),
        )
    elif how == "runs after rejection":
        settle = ToolService.settle_unrun_call

        def run_then_settle(self: Any, row: Any, status: Any) -> Any:
            self._run_approved(row)
            return settle(self, row, status)

        monkeypatch.setattr(ToolService, "settle_unrun_call", run_then_settle)
        monkeypatch.setattr(
            da.DestructiveApprovalHook,
            "_verify",
            lambda self, approval_id, item, *, caller: HookDecision(
                outcome="allow", reason="broken: any answer will do"
            ),
        )


@pytest.mark.parametrize(
    ("how", "expected"),
    [
        ("skips approval", "ran from code without the owner's approval"),
        ("approval hook allows", "held without a queued approval"),
        ("runs twice", "approved, it ran 2 time(s), not once"),
        (
            "mutates the pinned arguments",
            "approved, it ran with arguments other than those queued",
        ),
        ("runs after rejection", "ran after the owner rejected it"),
    ],
)
def test_the_suite_fails_a_harness_whose_governance_is_broken(
    monkeypatch: pytest.MonkeyPatch, how: str, expected: str
) -> None:
    _broken(monkeypatch, how)
    violations = check_conformance(_board_plugin(_fresh()), tools=EXAMPLES)
    found = [v for v in violations if v.check == "approval" and v.subject == "remove_note"]
    assert any(v.detail.startswith(expected) for v in found), violations


def test_a_plugin_that_mounts_once_and_not_twice_is_reported() -> None:
    violations = check_conformance(
        _board_plugin(_fresh(), broken="second setup raises"), tools=EXAMPLES
    )
    assert [(v.check, v.subject) for v in violations] == [("mount", "board")]
    assert "cannot start" in violations[0].detail


async def test_the_suite_runs_inside_a_running_event_loop() -> None:
    assert check_conformance(_board_plugin(_fresh()), tools=EXAMPLES) == []


# ------------------------------------------------------- a capability write that confirms
class Pins(Protocol):
    def pin(self, q: str) -> Note: ...


PINS = CapabilitySpec(
    name="test.pins",
    protocol=Pins,
    methods={"pin": MethodSpec(effect="write", fields=("text",))},
)
PIN_EXAMPLES = {"test.pins": {"pin": {"q": "hi"}}}


def _pins_plugin(ran: list[str]) -> Any:
    class Provider:
        def pin(self, q: str) -> Note:
            ran.append(q)
            return Note(q)

    return plugin(
        lambda api: api.provide("test.pins", Provider()),
        manifest={"name": "pins", "capabilities": {"provides": ["test.pins"]}},
    )


def test_a_capability_write_that_confirms_is_held_and_never_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(catalogue, "CAPABILITIES", MappingProxyType({"test.pins": PINS}))
    ran: list[str] = []
    assert check_conformance(_pins_plugin(ran), capabilities=PIN_EXAMPLES) == []
    assert ran == []


def test_a_held_capability_call_with_a_post_row_is_reported_as_having_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(catalogue, "CAPABILITIES", MappingProxyType({"test.pins": PINS}))
    real = conformance._tool_rows

    def with_a_post_row(h: Any, subject: str, *, after: int) -> list[AuditRow]:
        rows = real(h, subject, after=after)
        caller = f"plugin:{conformance.CONSUMER}"
        return [*rows, _row("post_tool_use", subject, caller=caller, digest=None)]

    monkeypatch.setattr(conformance, "_tool_rows", with_a_post_row)
    violations = check_conformance(_pins_plugin([]), capabilities=PIN_EXAMPLES)
    assert [(v.check, v.subject) for v in violations] == [("approval", "capability:test.pins.pin")]
    assert "it ran" in violations[0].detail


def test_a_held_capability_call_with_no_recorded_hold_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(catalogue, "CAPABILITIES", MappingProxyType({"test.pins": PINS}))
    real = conformance._tool_rows

    def allowed_rows(h: Any, subject: str, *, after: int) -> list[AuditRow]:
        return [
            _row(r.hook_point, subject, caller=f"plugin:{conformance.CONSUMER}", digest=None)
            for r in real(h, subject, after=after)
        ]

    monkeypatch.setattr(conformance, "_tool_rows", allowed_rows)
    violations = check_conformance(_pins_plugin([]), capabilities=PIN_EXAMPLES)
    assert [(v.check, v.subject) for v in violations] == [("approval", "capability:test.pins.pin")]
    assert "no pre_tool_use row records the hold" in violations[0].detail
