"""A lapsed approval says so.

`expire_stale` existed, was tested, and **had no production caller** — not the runtime,
not a heartbeat, not a read path. So an approval never actually timed out: the row sat
`pending` past its deadline, `list_pending` kept returning it, and `policy_on_timeout`
had never once been applied to anything. A lapsed approval did not go quiet so much as
go stale — the Action Center showed a dead request as though it were still live, and
the run it halted waited on a decision nobody could still make.

`approval_timeout_tick` now sweeps every 60s, and each lapse is announced twice: on the
channel that was asked, and in the conversation the halted run belongs to.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.kernel.governance.approvals.queue import ApprovalQueue
from iris_harness.kernel.governance.approvals.service import lapse_notice, sweep_expired
from iris_harness.kernel.governance.approvals.store import ApprovalRow, ApprovalStore


class _Router:
    def __init__(self, *, boom: bool = False) -> None:
        self.announced: list[ApprovalRow] = []
        self._boom = boom

    def notify_timeout(self, approval: ApprovalRow) -> None:
        self.announced.append(approval)
        if self._boom:
            raise RuntimeError("telegram is down")


class _Notifier:
    def __init__(self, *, boom: bool = False) -> None:
        self.notices: list[tuple[str | None, str, str]] = []
        self._boom = boom

    def deliver_lapse_notice(self, *, session_id: str | None, channel: str, text: str) -> None:
        self.notices.append((session_id, channel, text))
        if self._boom:
            raise RuntimeError("no channel registered")


def _queue(tmp_path: Path) -> ApprovalQueue:
    return ApprovalQueue(store=ApprovalStore(db_path=tmp_path / "approvals.db"))


def _overdue(q: ApprovalQueue, **kwargs: object) -> str:
    return q.enqueue(
        "run-1",
        None,
        "goal_drift",
        "thought drifted from original task",
        channel="web",
        timeout_minutes=-1,
        session_id="web-6d670ccd",
        **kwargs,  # type: ignore[arg-type]
    )


# ── the sweep ─────────────────────────────────────────────────────────────────


def test_an_overdue_approval_is_timed_out_and_announced(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    approval_id = _overdue(q)
    router, notifier = _Router(), _Notifier()

    expired = sweep_expired(queue=q, router=router, notifier=notifier)

    assert [r.approval_id for r in expired] == [approval_id]
    assert q.get(approval_id).status == "timed_out"  # type: ignore[union-attr]
    assert [r.approval_id for r in router.announced] == [approval_id]
    assert len(notifier.notices) == 1


def test_the_notice_reaches_the_conversation_that_was_left_waiting(tmp_path: Path) -> None:
    """The chat that was told "paused for approval" is where the silence shows."""
    q = _queue(tmp_path)
    _overdue(q)
    notifier = _Notifier()

    sweep_expired(queue=q, router=_Router(), notifier=notifier)

    session_id, channel, text = notifier.notices[0]
    assert session_id == "web-6d670ccd"
    assert channel == "web"
    assert "run-1" in text
    assert "still stopped" in text


def test_the_channel_that_was_asked_is_the_one_told(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    _overdue(q)
    router = _Router()

    sweep_expired(queue=q, router=router, notifier=_Notifier())

    assert router.announced[0].channel == "web"


def test_a_future_approval_is_left_alone(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    q.enqueue("run-1", None, "goal_drift", "drifted", timeout_minutes=60)
    router, notifier = _Router(), _Notifier()

    assert sweep_expired(queue=q, router=router, notifier=notifier) == []
    assert router.announced == []
    assert notifier.notices == []


def test_an_answered_approval_is_never_swept(tmp_path: Path) -> None:
    """Overdue but already decided: the window closing after the fact is not news."""
    q = _queue(tmp_path)
    approval_id = _overdue(q)
    q.respond(approval_id, status="approved", actor="web:owner")
    notifier = _Notifier()

    assert sweep_expired(queue=q, router=_Router(), notifier=notifier) == []
    assert notifier.notices == []


def test_each_lapse_is_announced_exactly_once(tmp_path: Path) -> None:
    """The sweep runs on a 60s heartbeat *and* is safe to call from anywhere. The only
    thing making that true is that `expire_stale` transitions rows out of `pending`."""
    q = _queue(tmp_path)
    _overdue(q)
    router, notifier = _Router(), _Notifier()

    sweep_expired(queue=q, router=router, notifier=notifier)
    sweep_expired(queue=q, router=router, notifier=notifier)
    sweep_expired(queue=q, router=router, notifier=notifier)

    assert len(router.announced) == 1
    assert len(notifier.notices) == 1


def test_several_lapses_are_each_announced(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    _overdue(q)
    q.enqueue("run-2", None, "cost_budget", "over budget", timeout_minutes=-1)
    router, notifier = _Router(), _Notifier()

    sweep_expired(queue=q, router=router, notifier=notifier)

    assert len(router.announced) == 2
    assert len(notifier.notices) == 2


# ── an approval with no conversation behind it ─────────────────────────────────


def test_a_sessionless_approval_still_gets_the_channel_notice(tmp_path: Path) -> None:
    """Rows written before `session_id` existed, and halts raised outside any chat."""
    q = _queue(tmp_path)
    q.enqueue("run-1", None, "goal_drift", "drifted", channel="cli", timeout_minutes=-1)
    router, notifier = _Router(), _Notifier()

    sweep_expired(queue=q, router=router, notifier=notifier)

    assert len(router.announced) == 1
    assert notifier.notices[0][0] is None  # no session to post into
    assert notifier.notices[0][1] == "cli"


# ── a failing delivery must not strand the rest ───────────────────────────────


def test_a_broken_channel_does_not_stop_the_in_chat_notice(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    _overdue(q)
    notifier = _Notifier()

    sweep_expired(queue=q, router=_Router(boom=True), notifier=notifier)

    assert len(notifier.notices) == 1


def test_a_broken_notifier_does_not_stop_the_sweep(tmp_path: Path) -> None:
    """The status transition is the durable part; delivery is best-effort by contract."""
    q = _queue(tmp_path)
    approval_id = _overdue(q)

    expired = sweep_expired(queue=q, router=_Router(), notifier=_Notifier(boom=True))

    assert len(expired) == 1
    assert q.get(approval_id).status == "timed_out"  # type: ignore[union-attr]


def test_one_failure_does_not_hide_the_next_lapse(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    _overdue(q)
    q.enqueue("run-2", None, "cost_budget", "over budget", timeout_minutes=-1)
    notifier = _Notifier(boom=True)

    sweep_expired(queue=q, router=_Router(boom=True), notifier=notifier)

    assert len(notifier.notices) == 2


def test_the_sweep_works_with_no_delivery_wired_at_all(tmp_path: Path) -> None:
    """A CLI process with no runtime: the rows must still expire."""
    q = _queue(tmp_path)
    approval_id = _overdue(q)

    assert len(sweep_expired(queue=q)) == 1
    assert q.get(approval_id).status == "timed_out"  # type: ignore[union-attr]


# ── the wording ───────────────────────────────────────────────────────────────


def test_the_notice_says_what_happened_why_and_what_now(tmp_path: Path) -> None:
    """ADR-0107's rule, which the halt message already follows."""
    q = _queue(tmp_path)
    approval_id = _overdue(q)
    row = q.get(approval_id)
    assert row is not None

    text = lapse_notice(row)

    assert "expired" in text
    assert "run-1" in text  # the only handle the user has on it afterwards
    assert "thought drifted from original task" in text  # why it stopped
    assert "fail_closed" in text
    assert "how you'd like to proceed" in text


# ── the sweep actually runs ────────────────────────────────────────────────────
#
# The defect being fixed was not a wrong sweep, it was a sweep nobody called. So the
# wiring is pinned structurally, the way `test_no_bypass.py` pins the governed loops:
# a handler that exists but is never registered, or a definition that is declared but
# disabled, both reproduce the original bug exactly.


def test_the_sweep_is_declared_as_a_heartbeat() -> None:
    from pathlib import Path as _Path

    from iris_harness.services.heartbeat.config import load_heartbeats

    definitions = {d.name: d for d in load_heartbeats(_Path("config/heartbeats.yaml"))}

    assert "approval_timeout_tick" in definitions
    tick = definitions["approval_timeout_tick"]
    assert tick.enabled is True
    assert tick.schedule == "interval:60"
    assert tick.handler == "approval_timeout_tick"


def test_the_runtime_registers_a_handler_for_it() -> None:
    """`expire_stale` was complete and tested with no caller for its whole life; a
    handler registered under no name would be the same bug again."""
    import inspect

    from iris_harness.runtime import bootstrap

    source = inspect.getsource(bootstrap.IrisRuntime._register_default_heartbeats)
    assert '"approval_timeout_tick"' in source


def test_the_runtime_notifier_satisfies_the_protocol() -> None:
    from iris_harness.kernel.governance.approvals.service import LapseNotifier
    from iris_harness.runtime.activity_notices import ActivityNotices

    assert issubclass(ActivityNotices, LapseNotifier)


def test_the_runtime_sweep_hands_each_lapse_to_its_notifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The protocol check above says the notifier *could* deliver; this says the
    runtime's heartbeat actually hands it one. Since the notify leg left `IrisRuntime`
    (OSS plan M5.7 track C, slice 8) the heartbeat passes a collaborator rather than
    `self`, so a heartbeat that passed nothing would still expire the row and pass
    every sweep-level test in this file while the conversation heard nothing."""
    from iris_harness.runtime import build_runtime
    from iris_harness.services.heartbeat.config import load_heartbeats

    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    (tmp_path / "config").mkdir()
    (tmp_path / "data").mkdir()
    runtime = build_runtime(
        config_dir=tmp_path / "config",
        data_dir=tmp_path / "data",
        use_background_scheduler=False,
    )
    q = _queue(tmp_path)
    _overdue(q)
    runtime.confirmations._approvals_queue_cache = q
    runtime.confirmations._approvals_router_cache = _Router()
    definition = {d.name: d for d in load_heartbeats(Path("config/heartbeats.yaml"))}[
        "approval_timeout_tick"
    ]

    run = runtime.confirmations.approval_timeout_heartbeat(definition)

    assert run.output == "approval_timeout_tick expired=1"
    notices = runtime.sessions.conversations["web-6d670ccd"]
    assert len(notices) == 1 and "expired" in notices[0].content
