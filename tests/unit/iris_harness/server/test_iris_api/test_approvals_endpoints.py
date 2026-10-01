"""The HTTP surface for the approval queue.

Before these endpoints the web had no way to answer an approval at all, so a halt
raised in the browser was unanswerable from the browser. They are deliberately thin:
the decision, its audit row and the resume live in
``governance.approvals.service``, which the CLI and the Telegram handler call too. A
core capability is never isolated in one channel.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.kernel.governance.approvals.queue import ApprovalQueue
from iris_harness.kernel.governance.approvals.service import ResumedRun
from iris_harness.kernel.governance.approvals.store import ApprovalStore
from iris_harness.server.iris_api.main import create_app


class _Runtime(SimpleNamespace):
    """A runtime that can resume and (decision 1) run a code caller's approved call."""

    def __init__(self, tmp_path: Path, *, boom: bool = False, tool_service: Any = None) -> None:
        super().__init__(data_dir=tmp_path, tool_service=tool_service)
        self.boom = boom
        self.calls: list[tuple[str, int]] = []
        self.channels: list[str] = []

    def resume_halted_run(
        self, *, run_id: str, step_id: int, channel: str = "console"
    ) -> ResumedRun:
        self.calls.append((run_id, step_id))
        self.channels.append(channel)
        if self.boom:
            raise RuntimeError("the model is down")
        return ResumedRun(answer="Going with Stardog.", session_id="web-6d670ccd")


@pytest.fixture()
def queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ApprovalQueue:
    """Point the service's default queue at a tmp database."""
    q = ApprovalQueue(store=ApprovalStore(db_path=tmp_path / "approvals.db"))
    from iris_harness.kernel.governance.approvals import service

    monkeypatch.setattr(service, "_queue", lambda given: given or q)
    return q


def _client(
    tmp_path: Path, *, boom: bool = False, tool_service: Any = None
) -> tuple[TestClient, _Runtime]:
    runtime = _Runtime(tmp_path, boom=boom, tool_service=tool_service)
    client = TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    )
    return client, runtime


def _queued(queue: ApprovalQueue, *, linked: bool = True, channel: str = "web") -> str:
    approval_id = queue.enqueue("run-1", None, "goal_drift", "drifted", channel=channel)
    if linked:
        queue.set_checkpoint(approval_id, "run-1:1")
    return approval_id


# ── listing ───────────────────────────────────────────────────────────────────


def test_pending_approvals_are_listed(tmp_path: Path, queue: ApprovalQueue) -> None:
    approval_id = _queued(queue)
    client, _ = _client(tmp_path)

    body = client.get("/governance/approvals").json()

    assert body["count"] == 1
    row = body["approvals"][0]
    assert row["approval_id"] == approval_id
    assert row["signal"] == "goal_drift"
    assert row["channel"] == "web"
    assert row["checkpoint_id"] == "run-1:1"
    assert row["resumable"] is True


def test_an_evaluator_approval_is_listed_as_such(tmp_path: Path, queue: ApprovalQueue) -> None:
    _queued(queue)
    client, _ = _client(tmp_path)
    row = client.get("/governance/approvals").json()["approvals"][0]
    assert (row["kind"], row["card"], row["items"]) == ("evaluator", None, [])


def test_a_destructive_approval_lists_its_card_and_exact_calls(
    tmp_path: Path, queue: ApprovalQueue
) -> None:
    """ADR-0118 step 4: what the phone card renders, and the raw call behind it."""
    from iris_harness.kernel.governance.approvals.store import ApprovalCard, ApprovalItem

    queue.enqueue(
        "run-1",
        None,
        "Trash 2 emails",
        "Trash 2 emails\n- Deals — Store X",
        channel="web",
        items=(ApprovalItem.of("trash_email", {"ids": ["m1", "m2"]}),),
        card=ApprovalCard(
            title="Trash 2 emails",
            lines=("Deals — Store X", "Sale — Shop Y"),
            undo_tool="restore_email",
            undo_window_days=30,
            asked="clean up the promos",
        ),
    )
    client, _ = _client(tmp_path)
    row = client.get("/governance/approvals").json()["approvals"][0]

    assert row["kind"] == "destructive"
    assert row["card"] == {
        "title": "Trash 2 emails",
        "lines": ["Deals — Store X", "Sale — Shop Y"],
        "undo_tool": "restore_email",
        "undo_window_days": 30,
        "asked": "clean up the promos",
        "effect": "destructive",
    }
    assert row["items"] == [{"tool": "trash_email", "args": {"ids": ["m1", "m2"]}}]


def test_an_unlinked_approval_is_listed_as_not_resumable(
    tmp_path: Path, queue: ApprovalQueue
) -> None:
    """So the UI can say what the button will do rather than promise a resume it
    cannot deliver."""
    _queued(queue, linked=False)
    client, _ = _client(tmp_path)

    assert client.get("/governance/approvals").json()["approvals"][0]["resumable"] is False


def test_an_empty_queue_lists_nothing(tmp_path: Path, queue: ApprovalQueue) -> None:
    client, _ = _client(tmp_path)
    body = client.get("/governance/approvals").json()
    assert body == {"count": 0, "approvals": []}


# ── responding ────────────────────────────────────────────────────────────────


def test_approving_records_the_decision_and_resumes_the_run(
    tmp_path: Path, queue: ApprovalQueue
) -> None:
    approval_id = _queued(queue)
    client, runtime = _client(tmp_path)

    body = client.post(
        f"/governance/approvals/{approval_id}/respond",
        json={"status": "approved", "actor": "web:owner"},
    ).json()

    assert body["status"] == "approved"
    assert body["resumed"] is True
    assert body["detail"] == "Going with Stardog."
    assert runtime.calls == [("run-1", 1)]
    assert queue.get(approval_id).status == "approved"  # type: ignore[union-attr]


def test_rejecting_records_the_decision_and_resumes_nothing(
    tmp_path: Path, queue: ApprovalQueue
) -> None:
    approval_id = _queued(queue)
    client, runtime = _client(tmp_path)

    body = client.post(
        f"/governance/approvals/{approval_id}/respond", json={"status": "rejected"}
    ).json()

    assert body["status"] == "rejected"
    assert body["resumed"] is False
    assert runtime.calls == []


def test_the_actor_defaults_to_web(tmp_path: Path, queue: ApprovalQueue) -> None:
    approval_id = _queued(queue)
    client, _ = _client(tmp_path)

    body = client.post(
        f"/governance/approvals/{approval_id}/respond", json={"status": "approved"}
    ).json()

    assert body["response_actor"] == "web"


def test_a_failed_resume_still_reports_the_recorded_approval(
    tmp_path: Path, queue: ApprovalQueue
) -> None:
    approval_id = _queued(queue)
    client, _ = _client(tmp_path, boom=True)

    response = client.post(
        f"/governance/approvals/{approval_id}/respond", json={"status": "approved"}
    )

    assert response.status_code == 200  # the decision succeeded; the resume did not
    body = response.json()
    assert body["status"] == "approved"
    assert body["resumed"] is False
    assert "could not be continued" in body["detail"]


# ── the errors ────────────────────────────────────────────────────────────────


def test_an_unknown_approval_is_a_404(tmp_path: Path, queue: ApprovalQueue) -> None:
    client, _ = _client(tmp_path)
    response = client.post(
        "/governance/approvals/does-not-exist/respond", json={"status": "approved"}
    )
    assert response.status_code == 404


def test_answering_twice_is_a_409(tmp_path: Path, queue: ApprovalQueue) -> None:
    approval_id = _queued(queue)
    client, _ = _client(tmp_path)
    client.post(f"/governance/approvals/{approval_id}/respond", json={"status": "approved"})

    response = client.post(
        f"/governance/approvals/{approval_id}/respond", json={"status": "rejected"}
    )
    assert response.status_code == 409


def test_an_invalid_status_is_a_400(tmp_path: Path, queue: ApprovalQueue) -> None:
    approval_id = _queued(queue)
    client, _ = _client(tmp_path)

    response = client.post(f"/governance/approvals/{approval_id}/respond", json={"status": "maybe"})
    assert response.status_code == 400


# ── an overdue row must not read as live ──────────────────────────────────────


def test_an_overdue_approval_is_flagged(tmp_path: Path, queue: ApprovalQueue) -> None:
    """`approval_timeout_tick` sweeps every 60s, so a row can be past its deadline and
    not yet swept. Before the sweep existed that window was forever, and the Action
    Center showed a dead request as though it were still being considered."""
    queue.enqueue("run-1", None, "goal_drift", "drifted", channel="web", timeout_minutes=-1)
    client, _ = _client(tmp_path)

    row = client.get("/governance/approvals").json()["approvals"][0]

    assert row["overdue"] is True


def test_a_live_approval_is_not_flagged(tmp_path: Path, queue: ApprovalQueue) -> None:
    queue.enqueue("run-1", None, "goal_drift", "drifted", channel="web", timeout_minutes=60)
    client, _ = _client(tmp_path)

    assert client.get("/governance/approvals").json()["approvals"][0]["overdue"] is False


def test_the_listing_carries_the_session(tmp_path: Path, queue: ApprovalQueue) -> None:
    queue.enqueue("run-1", None, "goal_drift", "drifted", channel="web", session_id="web-6d670ccd")
    client, _ = _client(tmp_path)

    assert client.get("/governance/approvals").json()["approvals"][0]["session_id"] == (
        "web-6d670ccd"
    )


# ── the service secret may answer approvals, and nothing else (owner, 2026-09-21) ──


def test_the_service_secret_may_answer_an_approval_with_writes_off(
    tmp_path: Path, queue: ApprovalQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The channel gateway answers Telegram taps through this route with the shared
    secret, on a deployment that leaves IRIS_WEBUI_ALLOW_WRITES unset (the VM)."""
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "0")
    approval_id = _queued(queue)
    client, runtime = _client(tmp_path)

    response = client.post(
        f"/governance/approvals/{approval_id}/respond",
        json={"status": "approved", "actor": "telegram:42"},
    )

    assert response.status_code == 200
    assert queue.get(approval_id).response_actor == "telegram:42"  # type: ignore[union-attr]
    assert runtime.calls == [("run-1", 1)]


def test_the_service_secret_still_may_not_make_other_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "0")
    client, _ = _client(tmp_path)
    response = client.post("/tasks", json={"title": "not allowed"})
    assert response.status_code == 403
    assert "control writes are disabled" in response.json()["detail"]
