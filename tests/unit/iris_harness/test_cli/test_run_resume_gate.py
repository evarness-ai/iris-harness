"""``iris run resume`` — the approval gate, and the resume it never did.

Two long-standing gaps meet here.

`ApprovalGate.await_approval` shipped in Phase 3 and had **no caller for its whole
life**, because the flow it was written for did not exist: an evaluator halt ends the
turn rather than blocking, so nothing was ever *waiting* on an answer. A resume is the
one place something is.

And `iris run resume` never resumed. It verified side effects against the ledger and
printed "re-run the original command with the same run_id" — while the halt message
that advertised it told the user to run it to continue. Before ADR-0106 Tier B there was
no way to re-enter a halted loop; now there is, so this uses it.

The safety property is the one worth pinning: a run the evaluator halted **for
approval** must not be resumed until that approval is answered. Stepping past it would
walk straight through the human the evaluator stopped the run to consult.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris_harness.kernel.governance.approvals.store import ApprovalStore
from iris_harness.main import _await_gating_approval


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ApprovalStore:
    """Point `ApprovalQueue()` / `ApprovalStore()` at a tmp database."""
    from iris_harness.kernel.governance.approvals import store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_APPROVALS_DB_PATH", tmp_path / "approvals.db")
    return ApprovalStore(db_path=tmp_path / "approvals.db")


def _gating(store: ApprovalStore, **kwargs: Any) -> str:
    return store.enqueue("run-1", None, "goal_drift", "thought drifted", **kwargs)


# ── nothing in the way ────────────────────────────────────────────────────────


def test_a_run_with_no_approval_may_continue(store: ApprovalStore) -> None:
    assert _await_gating_approval("run-1", wait=False) is True


def test_an_already_answered_approval_does_not_gate(store: ApprovalStore) -> None:
    approval_id = _gating(store)
    store.respond(approval_id, status="approved", actor="web:owner")

    assert _await_gating_approval("run-1", wait=False) is True


def test_another_runs_approval_does_not_gate_this_one(store: ApprovalStore) -> None:
    store.enqueue("run-2", None, "cost_budget", "over budget")

    assert _await_gating_approval("run-1", wait=False) is True


# ── the gate ──────────────────────────────────────────────────────────────────


def test_a_pending_approval_refuses_the_resume(store: ApprovalStore) -> None:
    """The property this exists for. Without it, resuming would be a way around the
    queue — and the queue is the only thing putting a human in the path."""
    _gating(store)

    assert _await_gating_approval("run-1", wait=False) is False


def test_the_refusal_names_the_approval_to_answer(
    store: ApprovalStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same lesson as the halt message: an id the user cannot see is no use."""
    approval_id = _gating(store)

    _await_gating_approval("run-1", wait=False)

    assert approval_id in capsys.readouterr().out


# ── --wait, which is what the gate is for ─────────────────────────────────────


def _answer_on_first_poll(
    monkeypatch: pytest.MonkeyPatch, store: ApprovalStore, approval_id: str, *, status: str
) -> dict[str, int]:
    """Answer the approval from another 'channel' the first time the gate polls.

    `await_approval` reads once to pre-check, then reads again inside its loop; flipping
    the status on that second read exercises the loop without waiting out a poll
    interval. The counter is returned so a test can assert the loop was actually entered
    rather than short-circuited.
    """
    import iris_harness.kernel.governance.approvals.gate as gate_mod

    calls = {"n": 0}
    original = gate_mod.ApprovalStore.get

    def patched(self: Any, aid: str) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            store.respond(aid, status=status, actor="web:owner")
        return original(self, aid)

    monkeypatch.setattr(gate_mod.ApprovalStore, "get", patched)
    return calls


def test_waiting_continues_once_it_is_approved(
    store: ApprovalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Answered on another channel while the terminal blocks — the flow the gate was
    written for and never got."""
    approval_id = _gating(store)
    calls = _answer_on_first_poll(monkeypatch, store, approval_id, status="approved")

    assert _await_gating_approval("run-1", wait=True) is True
    assert calls["n"] >= 2  # it really went through the polling loop


def test_waiting_stops_when_it_is_rejected(
    store: ApprovalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    approval_id = _gating(store)
    _answer_on_first_poll(monkeypatch, store, approval_id, status="rejected")

    assert _await_gating_approval("run-1", wait=True) is False


def test_waiting_stops_when_it_times_out(store: ApprovalStore) -> None:
    """`await_approval` raises `ApprovalTimedOutError` past the row's deadline; the run
    stays halted rather than proceeding unapproved."""
    _gating(store, timeout_minutes=-1)

    assert _await_gating_approval("run-1", wait=True) is False


def test_a_rejected_run_is_not_resumed(
    store: ApprovalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is the only thing between "rejected" and the loop running anyway."""
    approval_id = _gating(store)
    _answer_on_first_poll(monkeypatch, store, approval_id, status="rejected")

    assert _await_gating_approval("run-1", wait=True) is False
    assert store.get(approval_id).status == "rejected"  # type: ignore[union-attr]
