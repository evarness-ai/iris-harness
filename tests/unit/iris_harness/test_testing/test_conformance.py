"""``iris_harness.testing.conformance``: the suite a plugin's own CI runs (#77).

A conforming plugin passes against a real governed harness; each rule's verdict is
pinned against the ledger rows that would break it, since governance itself never lets a
real plugin produce them.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Protocol

import pytest

from iris_harness.foundation import capabilities as catalogue
from iris_harness.foundation.capabilities import CapabilitySpec, MethodSpec
from iris_harness.kernel.governance.audit.log import AuditRow
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
        if broken == "setup raises":
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
def _row(hook: str, tool: str, *, caller: str = "plugin:board", digest: str = "d1") -> AuditRow:
    payload = {"tool_name": tool, "caller": caller, "args_digest": digest}
    return AuditRow(
        id=1,
        ts=datetime.now(UTC).isoformat(),
        run_id="r1",
        step_id=None,
        agent_type="plugin:board",
        hook_point=hook,
        plugin="tool_policy",
        decision="allow",
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
