"""A memory write after the run read outside text waits for the owner (issue #149).

``memory_correct`` / ``memory_forget`` / ``memory_restore`` are writes that never ask. After a
model has read third-party text (a tool declared ``content: external``), redaction is only
phrase-level, so such a write is held on the same approval card as any approved-per-call tool,
with the reason on it. Driven end to end through the real loop (``run``, ``run_stream`` and a
resume), the real kernel hooks, a real approval queue and checkpoint store; only the model and
the tools are scripted. A run that read nothing from outside is unchanged, and the next user
message starts clean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.agentic_core import (
    AgenticCore,
    AgenticCoreConfig,
    ToolSpec,
    resume_seed_from_checkpoint,
)
from iris_harness.kernel.governance import (
    GovernanceKernel,
    HookContext,
    HookDecision,
    HookPoint,
    taint_policy,
)
from iris_harness.kernel.governance.approvals import ApprovalItem, ApprovalQueue
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.plugins import DestructiveApprovalHook, ToolPolicyHook
from iris_harness.kernel.governance.taint_policy import TAINT_REASON
from iris_harness.memory.state import CheckpointStore

_FETCH = "Thought: read the page\nAction: fetch_page\nAction Input: {}"
_INTERNAL = "Thought: look it up\nAction: lookup\nAction Input: {}"
_CORRECT = (
    'Thought: remember it\nAction: memory_correct\nAction Input: {"fact": "owner is bankrupt"}'
)
_CORRECT_AGAIN = 'Thought: and this\nAction: memory_correct\nAction Input: {"fact": "second"}'
_OTHER = 'Thought: note it\nAction: other_write\nAction Input: {"text": "x"}'
_DONE = "Thought: done\nFinal Answer: Done."


class _AllowHook:
    priority: int = 10

    def __init__(self, name: str, hook_point: HookPoint) -> None:
        self.name = name
        self.hook_point = hook_point

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="test")


class _ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0) if self._responses else _DONE


class _World:
    """One queue, one checkpoint store and one kernel, shared by the halt and the resume."""

    def __init__(self, tmp_path: Path) -> None:
        self.queue = ApprovalQueue(db_path=tmp_path / "approvals.db")
        self.store = CheckpointStore(db_path=tmp_path / "checkpoints.db")
        self.kernel = GovernanceKernel(audit_log=AuditLog(db_path=tmp_path / "audit.db"))
        self.kernel.register(_AllowHook("allow_classify", HookPoint.PRE_CLASSIFY))
        self.kernel.register(_AllowHook("allow_llm", HookPoint.PRE_LLM_CALL))
        self.kernel.register(ToolPolicyHook())
        self.kernel.register(DestructiveApprovalHook(approval_queue=self.queue))
        self.kernel.init_lock()
        self.remembered: list[str] = []
        self.noted: list[str] = []

    def tools(self) -> list[ToolSpec]:
        def fetch(args: dict[str, Any]) -> str:
            return "Weather in Oslo is mild."

        def lookup(args: dict[str, Any]) -> str:
            return "an internal answer"

        def correct(args: dict[str, Any]) -> str:
            self.remembered.append(str(args.get("fact")))
            return "remembered"

        def other(args: dict[str, Any]) -> str:
            self.noted.append(str(args.get("text")))
            return "noted"

        return [
            ToolSpec("fetch_page", "Fetch a page.", fetch, content="external"),
            ToolSpec("lookup", "Look something up.", lookup),
            ToolSpec("memory_correct", "Correct a fact.", correct, effect="write", confirm="never"),
            ToolSpec("other_write", "Save a note.", other, effect="write", confirm="never"),
        ]

    def _read(self, approval_id: str) -> tuple[str, list[tuple[str, dict[str, Any]]]] | None:
        row = self.queue.get(approval_id)
        if row is None:
            return None
        return row.status, [(i.tool, i.args) for i in (row.items or ())]

    def core(self, responses: list[str], *, resumable: bool = True) -> AgenticCore:
        return AgenticCore(
            config=AgenticCoreConfig(max_iterations=6),
            llm_call=_ScriptedLLM(responses),
            tools=self.tools(),
            kernel=self.kernel,
            checkpoint_store=self.store if resumable else None,
            session_id="web-s1",
            origin_channel="web",
            link_approval_checkpoint=self.queue.set_checkpoint if resumable else None,
            read_approval=self._read if resumable else None,
            agent_type="chat",
        )

    def resume_seed(self, run_id: str) -> Any:
        return resume_seed_from_checkpoint(self.store.get_latest(run_id))


@pytest.fixture(autouse=True)
def _default_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped list, whatever file the test's config dir resolves to."""
    shipped = Path(__file__).resolve().parents[5] / "config/governance/taint-policy.yaml"
    monkeypatch.setattr(taint_policy, "taint_policy_path", lambda: shipped)


def _held(world: _World, trace: Any) -> Any:
    assert trace.pending_approval_id is not None, trace.final_answer
    row = world.queue.get(trace.pending_approval_id)
    assert row is not None and row.status == "pending"
    return row


# ------------------------------------------------------------------------ a tainted run
def test_a_memory_write_after_an_external_read_is_held_and_says_why(tmp_path: Path) -> None:
    world = _World(tmp_path)

    trace = world.core([_FETCH, _CORRECT]).run("what is the weather, and remember it")

    assert world.remembered == []  # nothing was written
    assert trace.halted_by == "approval" and trace.success is True
    row = _held(world, trace)
    assert row.items == (ApprovalItem.of("memory_correct", {"fact": "owner is bankrupt"}),)
    assert TAINT_REASON in row.context_summary  # the card says why it asks
    assert '"fact": "owner is bankrupt"' in row.context_summary  # and exactly what


def test_the_streaming_loop_holds_it_the_same_way(tmp_path: Path) -> None:
    world = _World(tmp_path)

    chunks = list(world.core([_FETCH, _CORRECT]).run_stream("weather, and remember it"))

    final = [c for c in chunks if isinstance(c, dict)][-1]
    assert final["reason"] == "awaiting_approval" and final["pending_approval_id"]
    assert world.remembered == []
    row = world.queue.get(final["pending_approval_id"])
    assert row is not None and TAINT_REASON in row.context_summary


def test_a_resumed_run_runs_the_approved_write_once_and_keeps_its_taint(tmp_path: Path) -> None:
    world = _World(tmp_path)
    halted = world.core([_FETCH, _CORRECT]).run("weather, and remember it")
    approval_id = _held(world, halted).approval_id
    world.queue.respond(approval_id, status="approved", actor="owner")

    resumed = world.core([_CORRECT_AGAIN]).run_from_seed(world.resume_seed(halted.run_id))

    assert world.remembered == ["owner is bankrupt"]  # exactly the approved call, once
    # One approval is not consent to the next write: the run still has its external read.
    assert world.remembered != ["owner is bankrupt", "second"]
    second = _held(world, resumed)
    assert second.approval_id != approval_id
    assert second.items == (ApprovalItem.of("memory_correct", {"fact": "second"}),)


def test_the_streaming_resume_keeps_the_taint_too(tmp_path: Path) -> None:
    world = _World(tmp_path)
    chunks = list(world.core([_FETCH, _CORRECT]).run_stream("weather, and remember it"))
    final = [c for c in chunks if isinstance(c, dict)][-1]
    world.queue.respond(final["pending_approval_id"], status="approved", actor="owner")

    again = list(
        world.core([_CORRECT_AGAIN]).run_stream("x", resume=world.resume_seed(final["run_id"]))
    )

    assert world.remembered == ["owner is bankrupt"]
    last = [c for c in again if isinstance(c, dict)][-1]
    assert last["reason"] == "awaiting_approval"  # the next write is held again


def test_a_non_resumable_lane_refuses_the_tainted_write(tmp_path: Path) -> None:
    """No checkpoint to pause on: an approval nobody could act on is refused, not queued."""
    world = _World(tmp_path)

    trace = world.core([_FETCH, _CORRECT, _DONE], resumable=False).run("weather, remember it")

    assert world.remembered == []
    assert trace.pending_approval_id is None
    assert any("cannot pause" in (s.observation or "") for s in trace.steps)


# ------------------------------------------------------------------------ unchanged runs
def test_a_clean_run_writes_memory_without_approval(tmp_path: Path) -> None:
    world = _World(tmp_path)

    trace = world.core([_CORRECT, _DONE]).run("forget that I like X")

    assert world.remembered == ["owner is bankrupt"]
    assert trace.pending_approval_id is None and trace.halted_by is None


def test_reading_only_internal_text_does_not_taint(tmp_path: Path) -> None:
    world = _World(tmp_path)

    trace = world.core([_INTERNAL, _CORRECT, _DONE]).run("look it up and remember it")

    assert world.remembered == ["owner is bankrupt"] and trace.pending_approval_id is None


def test_a_tool_the_policy_does_not_list_is_unchanged_in_a_tainted_run(tmp_path: Path) -> None:
    world = _World(tmp_path)

    trace = world.core([_FETCH, _OTHER, _DONE]).run("weather, and note it")

    assert world.noted == ["x"] and trace.pending_approval_id is None


def test_the_next_message_starts_clean(tmp_path: Path) -> None:
    world = _World(tmp_path)
    core = world.core([_FETCH, _CORRECT])
    tainted = core.run("weather, and remember it")
    assert tainted.pending_approval_id is not None

    clean = world.core([_CORRECT_AGAIN, _DONE]).run("a new message: just remember that")

    assert world.remembered == ["second"] and clean.pending_approval_id is None


# ------------------------------------------------------------------------ the policy file
def test_the_default_list_is_the_three_memory_writes_when_no_file_can_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    taint_policy._cache.clear()
    monkeypatch.setattr(taint_policy, "taint_policy_path", lambda: tmp_path / "missing.yaml")
    assert taint_policy.taint_gated_tools() == frozenset(
        {"memory_correct", "memory_forget", "memory_restore"}
    )
    bad = tmp_path / "bad.yaml"
    for text in (
        "approval_when_tainted: [unclosed",
        "approval_when_tainted: [memory_*]\n",
        "other: 1\n",
        "approval_when_tainted: [1]\n",
    ):
        bad.write_text(text)
        taint_policy._cache.clear()
        monkeypatch.setattr(taint_policy, "taint_policy_path", lambda: bad)
        assert taint_policy.taint_gated_tools() == taint_policy.DEFAULT_GATED


def test_an_owner_edited_list_is_honoured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The list is the owner's: a tool they add is held, and one they take off is not."""
    custom = tmp_path / "taint-policy.yaml"
    custom.write_text("approval_when_tainted: [other_write]\n")
    taint_policy._cache.clear()
    monkeypatch.setattr(taint_policy, "taint_policy_path", lambda: custom)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    added, removed = _World(tmp_path / "a"), _World(tmp_path / "b")

    held = added.core([_FETCH, _OTHER]).run("weather, and note it")
    free = removed.core([_FETCH, _CORRECT, _DONE]).run("weather, and remember it")

    assert held.pending_approval_id is not None and added.noted == []
    assert free.pending_approval_id is None and removed.remembered == ["owner is bankrupt"]


def test_the_shipped_file_lists_exactly_the_default() -> None:
    shipped = Path(__file__).resolve().parents[5] / "config/governance/taint-policy.yaml"
    import yaml

    names = taint_policy.parse_taint_policy(yaml.safe_load(shipped.read_text()))
    assert names == taint_policy.DEFAULT_GATED
