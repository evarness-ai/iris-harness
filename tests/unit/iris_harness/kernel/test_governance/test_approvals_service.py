"""``respond_to_approval`` — one capability, four surfaces.

Answering an approval used to exist three times: the CLI wrote to ``ApprovalStore``
directly (and so left no audit row), the Telegram handler did the same separately, and
the web could not do it at all. None of the three could continue the run it had just
approved. This is the single implementation they now share, and what these tests are
mostly about is the **degradations** — every way a resume can fail has to leave the
human's decision intact, because that is the durable, auditable thing they did.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.kernel.governance.approvals.queue import ApprovalQueue
from iris_harness.kernel.governance.approvals.service import (
    ApprovalOutcome,
    ResumedRun,
    parse_checkpoint_id,
    pending_approvals,
    respond_to_approval,
)
from iris_harness.kernel.governance.approvals.store import (
    ApprovalAlreadyAnsweredError,
    ApprovalNotFoundError,
    ApprovalStore,
)


class _Resumer:
    """A runtime stand-in; records the resume point it was asked for."""

    def __init__(self, *, answer: str = "Going with Stardog.", boom: bool = False) -> None:
        self._answer = answer
        self._boom = boom
        self.calls: list[tuple[str, int]] = []
        self.channels: list[str] = []

    def resume_halted_run(
        self, *, run_id: str, step_id: int, channel: str = "console"
    ) -> ResumedRun:
        self.calls.append((run_id, step_id))
        self.channels.append(channel)
        if self._boom:
            raise RuntimeError("the model is down")
        return ResumedRun(answer=self._answer, session_id="web-6d670ccd")


def _queue(tmp_path: Path) -> ApprovalQueue:
    return ApprovalQueue(store=ApprovalStore(db_path=tmp_path / "approvals.db"))


def _queued(tmp_path: Path, *, linked: bool = True) -> tuple[ApprovalQueue, str]:
    q = _queue(tmp_path)
    approval_id = q.enqueue("run-1", None, "goal_drift", "drifted", channel="web")
    if linked:
        q.set_checkpoint(approval_id, "run-1:1")
    return q, approval_id


# ── the checkpoint link ───────────────────────────────────────────────────────


def test_the_link_is_parsed_as_run_and_step() -> None:
    assert parse_checkpoint_id("b7c928e2-5a09-41ee-b771-d0322391c3b3:4") == (
        "b7c928e2-5a09-41ee-b771-d0322391c3b3",
        4,
    )


@pytest.mark.parametrize("raw", [None, "", "no-colon", "run-1:", "run-1:abc", ":3"])
def test_an_unusable_link_is_none_not_an_exception(raw: str | None) -> None:
    assert parse_checkpoint_id(raw) is None


# ── approving ─────────────────────────────────────────────────────────────────


def test_approving_records_the_decision_and_continues_the_run(tmp_path: Path) -> None:
    q, approval_id = _queued(tmp_path)
    resumer = _Resumer()

    outcome = respond_to_approval(
        approval_id, status="approved", actor="web:owner", queue=q, resumer=resumer
    )

    assert outcome.row.status == "approved"
    assert outcome.row.response_actor == "web:owner"
    assert outcome.resumed is True
    assert outcome.detail == "Going with Stardog."
    assert resumer.calls == [("run-1", 1)]
    # The row's channel travels with the resume, so a second halt reaches the same
    # surface as the first instead of defaulting back to the console.
    assert resumer.channels == ["web"]


def test_rejecting_records_the_decision_and_resumes_nothing(tmp_path: Path) -> None:
    q, approval_id = _queued(tmp_path)
    resumer = _Resumer()

    outcome = respond_to_approval(
        approval_id, status="rejected", actor="cli:owner", queue=q, resumer=resumer
    )

    assert outcome.row.status == "rejected"
    assert outcome.resumed is False
    assert resumer.calls == []
    assert "stays halted" in outcome.detail


def test_answering_leaves_an_audit_row(tmp_path: Path) -> None:
    """The CLI path wrote straight to the store and so recorded nothing. A governance
    decision that leaves no trace is the one kind that most needs one."""

    class _Audit:
        def __init__(self) -> None:
            self.rows: list[dict[str, object]] = []

        def record(self, **payload: object) -> None:
            self.rows.append(payload)

    audit = _Audit()
    q = ApprovalQueue(store=ApprovalStore(db_path=tmp_path / "approvals.db"), audit_log=audit)
    approval_id = q.enqueue("run-1", None, "goal_drift", "drifted", channel="web")

    respond_to_approval(approval_id, status="approved", actor="web:owner", queue=q)

    decisions = [r for r in audit.rows if r.get("decision") == "approved"]
    assert len(decisions) == 1


# ── every way the resume can fail keeps the decision ──────────────────────────


def test_an_unlinked_approval_is_answered_but_not_resumed(tmp_path: Path) -> None:
    q, approval_id = _queued(tmp_path, linked=False)
    resumer = _Resumer()

    outcome = respond_to_approval(
        approval_id, status="approved", actor="web:owner", queue=q, resumer=resumer
    )

    assert outcome.row.status == "approved"
    assert outcome.resumed is False
    assert resumer.calls == []
    assert "nothing to continue" in outcome.detail


def test_no_runtime_available_is_answered_but_not_resumed(tmp_path: Path) -> None:
    """A CLI invocation in a process with no runtime built, for instance."""
    q, approval_id = _queued(tmp_path)

    outcome = respond_to_approval(approval_id, status="approved", actor="cli:owner", queue=q)

    assert outcome.row.status == "approved"
    assert outcome.resumed is False
    assert "no runtime is available" in outcome.detail


def test_a_resume_that_raises_does_not_lose_the_approval(tmp_path: Path) -> None:
    q, approval_id = _queued(tmp_path)

    outcome = respond_to_approval(
        approval_id,
        status="approved",
        actor="web:owner",
        queue=q,
        resumer=_Resumer(boom=True),
    )

    assert outcome.row.status == "approved"  # recorded first, and durably
    assert outcome.resumed is False
    assert "could not be continued" in outcome.detail
    assert "the model is down" in outcome.detail


# ── the errors every surface already renders ──────────────────────────────────


def test_an_unknown_approval_raises(tmp_path: Path) -> None:
    with pytest.raises(ApprovalNotFoundError):
        respond_to_approval("nope", status="approved", actor="a", queue=_queue(tmp_path))


def test_answering_twice_raises(tmp_path: Path) -> None:
    q, approval_id = _queued(tmp_path)
    respond_to_approval(approval_id, status="approved", actor="a", queue=q)

    with pytest.raises(ApprovalAlreadyAnsweredError):
        respond_to_approval(approval_id, status="rejected", actor="b", queue=q)


# ── the read side ─────────────────────────────────────────────────────────────


def test_pending_approvals_lists_only_what_is_waiting(tmp_path: Path) -> None:
    q, approval_id = _queued(tmp_path)
    other = q.enqueue("run-2", None, "cost_budget", "over budget", channel="cli")

    assert {r.approval_id for r in pending_approvals(queue=q)} == {approval_id, other}
    respond_to_approval(approval_id, status="approved", actor="a", queue=q)
    assert {r.approval_id for r in pending_approvals(queue=q)} == {other}


def test_the_outcome_serialises_for_an_api_response(tmp_path: Path) -> None:
    q, approval_id = _queued(tmp_path)
    outcome = respond_to_approval(
        approval_id, status="approved", actor="web:owner", queue=q, resumer=_Resumer()
    )

    payload = outcome.as_dict()
    assert payload["approval_id"] == approval_id
    assert payload["status"] == "approved"
    assert payload["resumed"] is True
    assert payload["checkpoint_id"] == "run-1:1"
    assert isinstance(outcome, ApprovalOutcome)


# ── ADR-0118: a destructive-tool approval resumes on a rejection too ──────────


def _pinned(tmp_path: Path) -> tuple[ApprovalQueue, str]:
    from iris_harness.kernel.governance.approvals.store import ApprovalItem

    q = _queue(tmp_path)
    approval_id = q.enqueue(
        "run-1",
        None,
        "trash_email wants to delete or overwrite your data",
        "- trash_email {...}",
        channel="web",
        items=(ApprovalItem.of("trash_email", {"ids": ["m1"]}),),
    )
    q.set_checkpoint(approval_id, "run-1:1")
    return q, approval_id


def test_rejecting_a_destructive_approval_resumes_so_the_owner_hears_back(tmp_path: Path) -> None:
    """The loop settles the row itself — it reads 'rejected' and executes nothing — and
    the model closes the conversation, instead of the chat going silent."""
    q, approval_id = _pinned(tmp_path)
    resumer = _Resumer(answer="I did not delete it.")

    outcome = respond_to_approval(
        approval_id, status="rejected", actor="web:owner", queue=q, resumer=resumer
    )

    assert outcome.row.status == "rejected"
    assert outcome.resumed is True
    assert resumer.calls == [("run-1", 1)]
    assert outcome.detail == "I did not delete it."


def test_a_rejected_destructive_approval_with_no_runtime_says_nothing_changed(
    tmp_path: Path,
) -> None:
    q, approval_id = _pinned(tmp_path)
    outcome = respond_to_approval(approval_id, status="rejected", actor="cli:owner", queue=q)
    assert outcome.resumed is False
    assert outcome.detail.startswith("Rejected, nothing was changed.")
