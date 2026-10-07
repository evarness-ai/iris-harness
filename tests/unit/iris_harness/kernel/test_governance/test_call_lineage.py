"""Where a call sits: its parent call, its attempt, its turn (issue #134, stage 2).

Every audit row of a governed call names the governed call it ran inside
(``parent_call_id``), which attempt it is (``attempt``; 2 on the approved re-execution of a
held call, whose ``replay_of`` names the held attempt) and the turn it was written in
(``turn_id``). The kernel writes them from the harness's own record of the call
(``kernel/governance/call_context.py``), never from a payload, an argument or a caller. A
nested call does NOT take its parent's ``run_id`` (D4: no existing field changes).
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.agentic_core import ToolSpec
from iris_harness.agent.tool_runner import GovernedToolRunner, ToolCall
from iris_harness.foundation.ids import is_ulid
from iris_harness.foundation.observability.session_log import turn_scope
from iris_harness.kernel.governance import GovernanceKernel, HookPoint
from iris_harness.kernel.governance.audit.log import AuditLog
from iris_harness.kernel.governance.call_context import (
    call_scope,
    current_call_id,
    is_run_resumed,
    lineage_of,
    mark_run_resumed,
    register_call,
)
from iris_harness.kernel.governance.caller_policy import register_caller_policy
from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision
from iris_harness.kernel.governance.plugins.caller_policy import CallerPolicyHook
from iris_harness.kernel.governance.plugins.capability_redaction import CapabilityRedactionHook
from iris_harness.kernel.governance.plugins.tool_policy import ToolPolicyHook

from .test_capability_calls import Msg, _isolation, _registry  # noqa: F401  (fixture)

FORGED = {
    "parent_call_id": "FORGED-PARENT",
    "turn_id": "FORGED-TURN",
    "attempt": 9,
    "replay_of": "FORGED-REPLAY",
    "resumed_from_run": "FORGED-RUN",
}


class _ForgingHook:
    """A hook that writes the identity keys into its own audit metadata, as a plugin could."""

    name = "forger"
    hook_point = HookPoint.PRE_TOOL_USE
    priority = 5

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="forge", audit_metadata=dict(FORGED))


class _Rig:
    def __init__(self, tmp_path: Path, *extra: Any) -> None:
        self.audit = AuditLog(db_path=tmp_path / "audit.db")
        self.kernel = GovernanceKernel(audit_log=self.audit)
        for hook in (CallerPolicyHook(), ToolPolicyHook(), CapabilityRedactionHook(), *extra):
            self.kernel.register(hook)
        self.kernel.init_lock()
        self.runner = GovernedToolRunner(kernel=self.kernel, agent_type="t")

    def tool(self, name: str, body: Any) -> ToolSpec:
        return ToolSpec(name, f"{name} tool", body, effect="read", confirm="never")

    def run(self, tool: ToolSpec, args: dict[str, Any] | None = None, **call: Any) -> Any:
        return self.runner.execute(tool, dict(args or {}), ToolCall(**call))

    def rows(self, tool_name: str) -> list[dict[str, Any]]:
        return [
            json.loads(r.payload_json)
            for r in self.audit.query()
            if r.hook_point in ("pre_tool_use", "post_tool_use")
            and json.loads(r.payload_json).get("tool_name") == tool_name
        ]

    def call_ids(self, tool_name: str) -> set[str]:
        return {p["call_id"] for p in self.rows(tool_name)}


@pytest.fixture(autouse=True)
def _no_caller_policy() -> Iterator[None]:
    register_caller_policy(None)
    yield
    register_caller_policy(None)


# ------------------------------------------------------------------- the parent edge
def test_a_nested_tool_call_names_the_call_it_ran_inside(tmp_path: Path) -> None:
    rig = _Rig(tmp_path)
    inner = rig.tool("inner", lambda args: "inner done")
    outer = rig.tool("outer", lambda args: rig.run(inner, run_id="child-run").text)

    rig.run(outer, run_id="parent-run")

    (outer_id,) = rig.call_ids("outer")
    (inner_id,) = rig.call_ids("inner")
    assert is_ulid(outer_id) and is_ulid(inner_id) and inner_id != outer_id
    assert all("parent_call_id" not in p for p in rig.rows("outer"))  # a top-level call
    assert all(p["parent_call_id"] == outer_id for p in rig.rows("inner"))  # every row of it
    assert all(p["attempt"] == 1 for p in rig.rows("outer") + rig.rows("inner"))
    # D4: the child keeps its own run; nothing about run_id moved.
    runs = {r.run_id for r in rig.audit.query() if r.hook_point == "pre_tool_use"}
    assert {"parent-run", "child-run"} <= runs


def test_a_nested_call_with_no_run_still_mints_its_own_run(tmp_path: Path) -> None:
    rig = _Rig(tmp_path)
    inner = rig.tool("inner", lambda args: "x")
    outer = rig.tool("outer", lambda args: rig.run(inner).text)

    rig.run(outer, run_id="parent-run")

    inner_runs = {
        r.run_id
        for r in rig.audit.query()
        if json.loads(r.payload_json).get("tool_name") == "inner"
    }
    assert len(inner_runs) == 1 and "parent-run" not in inner_runs


def test_a_capability_call_made_from_a_tool_names_that_tool_call(tmp_path: Path) -> None:
    rig = _Rig(tmp_path)
    registry, _ = _registry(tmp_path, rig.kernel)
    inbox = registry.resolve_capability("fin", "test.inbox")
    outer = rig.tool("outer", lambda args: str(len(inbox.search("a"))))

    rig.run(outer)

    (outer_id,) = rig.call_ids("outer")
    capability_rows = rig.rows("capability:test.inbox.search")
    assert capability_rows
    assert {p["parent_call_id"] for p in capability_rows} == {outer_id}
    assert len({p["call_id"] for p in capability_rows}) == 1


def test_the_identity_keys_in_arguments_or_a_hook_are_never_read(tmp_path: Path) -> None:
    rig = _Rig(tmp_path, _ForgingHook())
    inner = rig.tool("inner", lambda args: "x")
    outer = rig.tool("outer", lambda args: rig.run(inner).text)

    rig.run(outer, dict(FORGED))

    (outer_id,) = rig.call_ids("outer")
    for p in rig.rows("outer") + rig.rows("inner"):
        for key, forged in FORGED.items():
            assert p.get(key) != forged, (key, p)
    assert "resumed_from_run" not in json.dumps(rig.rows("outer"))
    assert {p["parent_call_id"] for p in rig.rows("inner")} == {outer_id}
    assert all("turn_id" not in p for p in rig.rows("outer"))  # no turn scope: none invented


# --------------------------------------------------------------------------- attempt
def test_an_approved_reexecution_is_attempt_two_and_replays_the_held_attempt(
    tmp_path: Path,
) -> None:
    rig = _Rig(tmp_path)
    tool = rig.tool("shred", lambda args: "done")

    rig.run(tool, caller="plugin:p", held_call_id="HELD-ID-1")
    rig.run(tool, caller="plugin:p")

    by_id: dict[str, list[dict[str, Any]]] = {}
    for p in rig.rows("shred"):
        by_id.setdefault(p["call_id"], []).append(p)
    replay = [rows for rows in by_id.values() if rows[0].get("held_call_id") == "HELD-ID-1"]
    fresh = [rows for rows in by_id.values() if "held_call_id" not in rows[0]]
    assert len(replay) == 1 and len(fresh) == 1
    assert all(p["attempt"] == 2 and p["replay_of"] == "HELD-ID-1" for p in replay[0])
    assert all(p["attempt"] == 1 and "replay_of" not in p for p in fresh[0])


# ------------------------------------------------------------------------------ turn
def test_a_row_names_the_turn_it_was_written_in_and_none_outside_one(tmp_path: Path) -> None:
    rig = _Rig(tmp_path)
    tool = rig.tool("t", lambda args: "x")

    with turn_scope() as first:
        rig.run(tool)
    with turn_scope() as second:
        rig.run(tool)
    rig.run(tool)  # outside any turn (a heartbeat, the HTTP MCP route)

    by_call: dict[str, set[str | None]] = {}
    for p in rig.rows("t"):
        by_call.setdefault(p["call_id"], set()).add(p.get("turn_id"))
    ordered = [next(iter(v)) for _, v in sorted(by_call.items())]
    assert all(len(v) == 1 for v in by_call.values())
    assert ordered == [first, second, None] and first != second


# ------------------------------------------------------------------ the resumed run
def test_the_rows_of_a_resumed_run_say_so_and_earlier_rows_do_not(tmp_path: Path) -> None:
    rig = _Rig(tmp_path)
    tool = rig.tool("t", lambda args: "x")
    run_id = "run-resumed-1"

    rig.run(tool, run_id=run_id)  # written before the run was re-entered
    assert not is_run_resumed(run_id)
    mark_run_resumed(run_id)
    rig.run(tool, run_id=run_id)
    rig.run(tool, run_id="run-other")

    by_run: dict[str, list[dict[str, Any]]] = {}
    for r in rig.audit.query():
        if r.hook_point in ("pre_tool_use", "post_tool_use"):
            by_run.setdefault(r.run_id, []).append(json.loads(r.payload_json))
    marked = [p for p in by_run[run_id] if p.get("resumed_from_run") == run_id]
    unmarked = [p for p in by_run[run_id] if "resumed_from_run" not in p]
    assert marked and unmarked and len(marked) + len(unmarked) == len(by_run[run_id])
    assert all("resumed_from_run" not in p for p in by_run["run-other"])


# ----------------------------------------------------- propagation and reset on error
def test_a_nested_call_sees_its_parent_across_asyncio_and_to_thread(tmp_path: Path) -> None:
    rig = _Rig(tmp_path)
    inner = rig.tool("inner", lambda args: "x")

    async def via_to_thread() -> None:
        await asyncio.to_thread(rig.run, inner)  # copies the context

    outer = rig.tool("outer", lambda args: asyncio.run(via_to_thread()) or "ok")
    rig.run(outer)

    (outer_id,) = rig.call_ids("outer")
    assert {p["parent_call_id"] for p in rig.rows("inner")} == {outer_id}


def test_a_bare_thread_or_run_in_executor_has_no_parent_and_that_is_stated(
    tmp_path: Path,
) -> None:
    """Neither ``threading.Thread`` nor ``loop.run_in_executor`` copies the context, so a
    governed call started there has no parent. The limit is documented in ``call_context``
    (the egress scope has the same one); it is not silently papered over."""
    rig = _Rig(tmp_path)
    inner = rig.tool("inner", lambda args: "x")

    def in_thread(args: dict[str, Any]) -> str:
        seen: list[str | None] = []
        worker = threading.Thread(target=lambda: seen.append(rig.run(inner) and current_call_id()))
        worker.start()
        worker.join()

        async def in_executor() -> None:
            await asyncio.get_running_loop().run_in_executor(None, rig.run, inner)

        asyncio.run(in_executor())
        return "ok"

    rig.run(rig.tool("outer", in_thread))

    assert rig.rows("inner")
    assert all("parent_call_id" not in p for p in rig.rows("inner"))


def test_a_streaming_capability_call_is_current_while_its_body_runs(tmp_path: Path) -> None:
    rig = _Rig(tmp_path)
    registry, provider = _registry(tmp_path, rig.kernel)
    inner = rig.tool("inner", lambda args: "x")
    seen: list[str | None] = []

    def stream(query: str) -> Iterator[Any]:
        seen.append(current_call_id())
        rig.run(inner)  # a governed call started inside the stream's body
        yield Msg(1, query, "body")
        seen.append(current_call_id())  # a later step of the same body

    async def astream(query: str) -> Any:
        seen.append(current_call_id())
        await asyncio.to_thread(rig.run, inner)
        yield Msg(2, query, "body")
        seen.append(current_call_id())

    provider.stream = stream  # type: ignore[method-assign]
    provider.astream = astream  # type: ignore[method-assign]
    inbox = registry.resolve_capability("fin", "test.inbox")

    assert [m.subject for m in inbox.stream("a")] == ["a"]

    async def drain() -> list[Any]:
        return [m async for m in inbox.astream("b")]

    assert [m.subject for m in asyncio.run(drain())] == ["b"]
    sync_id, async_id = (
        next(iter(rig.call_ids(f"capability:test.inbox.{m}"))) for m in ("stream", "astream")
    )
    assert seen == [sync_id, sync_id, async_id, async_id]
    assert current_call_id() is None  # nothing leaked out of either stream
    assert {p["parent_call_id"] for p in rig.rows("inner")} == {sync_id, async_id}


def test_the_current_call_is_reset_when_the_call_raises(tmp_path: Path) -> None:
    rig = _Rig(tmp_path)
    registry, provider = _registry(tmp_path, rig.kernel)

    def boom(args: dict[str, Any]) -> str:
        raise RuntimeError("tool broke")

    rig.run(rig.tool("boom", boom))  # the runner turns a raise into a failed outcome
    assert current_call_id() is None

    def explode(query: str) -> list[Any]:
        raise RuntimeError("provider broke")

    provider.search = explode  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        registry.resolve_capability("fin", "test.inbox").search("a")
    assert current_call_id() is None

    with pytest.raises(ValueError), call_scope("OUTER"):
        with call_scope("INNER"):
            assert current_call_id() == "INNER"
            raise ValueError
    assert current_call_id() is None


def test_an_abandoned_stream_leaves_no_current_call(tmp_path: Path) -> None:
    rig = _Rig(tmp_path)
    registry, provider = _registry(tmp_path, rig.kernel)

    async def astream(query: str) -> Any:
        yield Msg(1, "one", "b")
        yield Msg(2, "two", "b")

    provider.astream = astream  # type: ignore[method-assign]
    inbox = registry.resolve_capability("fin", "test.inbox")

    async def first_only() -> None:
        agen = inbox.astream("q")
        await agen.__anext__()
        await agen.aclose()  # the consumer stops early

    asyncio.run(first_only())
    assert current_call_id() is None


# ------------------------------------------------------------------ the registry itself
def test_a_call_nobody_registered_has_no_lineage_and_the_registry_is_bounded() -> None:
    assert lineage_of(None) is None and lineage_of("nobody") is None
    with call_scope("P"):
        lineage = register_call("C1", held_call_id="H")
    assert (lineage.parent_call_id, lineage.attempt, lineage.replay_of) == ("P", 2, "H")
    assert lineage_of("C1") == lineage
    for n in range(9000):  # past the bound: the oldest go, the newest stay
        register_call(f"bulk-{n}")
    assert lineage_of("C1") is None and lineage_of("bulk-8999") is not None
