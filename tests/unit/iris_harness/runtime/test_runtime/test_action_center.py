"""Tests for the harness Action Center composer (ADR-0073/0074)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.runtime.action_center import collect_pending_actions, render_pending_actions
from iris_harness.services.health.models import CheckKind, HealthCheck, HealthSnapshot, HealthState
from iris_harness.services.tasks import TaskAction, TaskStore


@pytest.fixture(autouse=True)
def _own_approvals_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test reads its own (empty) approval queue.

    ``collect_pending_actions`` lists pending approvals from the default queue, whose
    database path is fixed when the store module is imported (under the session's
    ``IRIS_HOME``). A test elsewhere that halts a turn on an approval -- the destructive
    approval runtime test does -- leaves a pending row there, and in the same process
    these counts then included it: pass or fail depended on test order.
    """
    from iris_harness.kernel.governance.approvals import store as approvals_store

    monkeypatch.setattr(
        approvals_store, "DEFAULT_APPROVALS_DB_PATH", tmp_path / "governance" / "approvals.db"
    )


@pytest.fixture
def store(tmp_path: Path) -> TaskStore:
    s = TaskStore(db_path=tmp_path / "tasks.db")
    s.ensure_schema()
    return s


def _snap(*checks: HealthCheck) -> HealthSnapshot:
    return HealthSnapshot(checks=checks, sampled_at="2026-06-22T00:00:00Z")


def test_collect_unions_tasks_and_health(store: TaskStore) -> None:
    store.create(
        title="Wingtip card needs a password",
        source_kind="finance-statements",
        action=TaskAction(kind="copy_command", label="Set", command="iris finance secret ..."),
    )
    snap = _snap(
        HealthCheck(
            target="gmail",
            kind=CheckKind.CREDENTIAL,
            state=HealthState.RED,
            detail="token expired",
            action="iris auth gmail login",
        )
    )
    actions = collect_pending_actions(store, snap)
    assert sorted(a.origin for a in actions) == ["health", "task"]


def test_collect_without_snapshot_is_tasks_only(store: TaskStore) -> None:
    store.create(
        title="x",
        source_kind="finance-statements",
        action=TaskAction(kind="re_extract", label="Re-extract", target_id="s1", safe=True),
    )
    actions = collect_pending_actions(store, None)
    assert [a.origin for a in actions] == ["task"]


def test_collect_excludes_plain_todos(store: TaskStore) -> None:
    store.create(title="buy milk")  # no action
    assert collect_pending_actions(store, None) == []


def test_collect_stamps_feedback_ref(store: TaskStore, tmp_path: Path) -> None:
    from iris_harness.services.learning.suppression import SurfaceFeedbackStore, decode_ref

    fb = SurfaceFeedbackStore(db_path=tmp_path / "learning.db")
    fb.ensure_schema()
    store.create(
        title="Northwind card needs a password",
        source_kind="finance-statements",
        dedup_key="stmt:abc:password",
        action=TaskAction(kind="copy_command", label="Set", command="iris finance ..."),
    )
    snap = _snap(
        HealthCheck(
            target="gmail",
            kind=CheckKind.CREDENTIAL,
            state=HealthState.RED,
            detail="token expired",
            action="iris auth gmail login",
        )
    )
    by_origin = {a.origin: a for a in collect_pending_actions(store, snap, feedback_store=fb)}

    # Health keys on (kind, target) so feedback generalizes across probe instances.
    assert decode_ref(by_origin["health"].feedback_ref) == (
        "system",
        "health_alert",
        {"kind": "credential", "target": "gmail"},
    )
    # Task-backed keys on its dedup_key.
    assert decode_ref(by_origin["task"].feedback_ref) == (
        "finance-statements",
        "pending_action",
        {"key": "stmt:abc:password"},
    )


def test_not_useful_hides_health_item(store: TaskStore, tmp_path: Path) -> None:
    from iris_harness.services.learning.suppression import NOT_USEFUL, SurfaceFeedbackStore

    fb = SurfaceFeedbackStore(db_path=tmp_path / "learning.db")
    fb.ensure_schema()
    snap = _snap(
        HealthCheck(
            target="ollama",
            kind=CheckKind.SERVICE,
            state=HealthState.RED,
            detail="down",
            action="brew services start ollama",
        )
    )
    assert len(collect_pending_actions(store, snap, feedback_store=fb)) == 1
    fb.record("system", "health_alert", {"kind": "service", "target": "ollama"}, NOT_USEFUL)
    assert collect_pending_actions(store, snap, feedback_store=fb) == []


def test_not_useful_hides_task_item(store: TaskStore, tmp_path: Path) -> None:
    from iris_harness.services.learning.suppression import NOT_USEFUL, SurfaceFeedbackStore

    fb = SurfaceFeedbackStore(db_path=tmp_path / "learning.db")
    fb.ensure_schema()
    store.create(
        title="Northwind card needs a password",
        source_kind="finance-statements",
        dedup_key="stmt:abc:password",
        action=TaskAction(kind="copy_command", label="Set", command="iris finance ..."),
    )
    assert len(collect_pending_actions(store, None, feedback_store=fb)) == 1
    fb.record("finance-statements", "pending_action", {"key": "stmt:abc:password"}, NOT_USEFUL)
    assert collect_pending_actions(store, None, feedback_store=fb) == []


def test_render_empty() -> None:
    assert "nothing needs your attention" in render_pending_actions([]).lower()


def test_render_shows_command_and_count(store: TaskStore) -> None:
    store.create(
        title="Wingtip card needs a password",
        source_kind="finance-statements",
        action=TaskAction(
            kind="copy_command",
            label="Set",
            command="poetry run iris finance secret set-password wingtip",
        ),
    )
    text = render_pending_actions(collect_pending_actions(store, None))
    assert "1 pending action" in text
    assert "poetry run iris finance secret set-password wingtip" in text


# ── pending approvals are the third source (governance) ───────────────────────
#
# A run halted waiting on a human is the plainest "needs your attention" there is, and
# it used to appear on no such list — `iris approvals list` knew about it and nothing
# else did. Going through the Action Center is what makes it visible on every channel
# at once: `GET /actions`, the ReAct `pending_actions` tool, and the CLI all read this.


@pytest.fixture
def approvals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A tmp-path approval queue wired in as the Action Center's source."""
    from iris_harness.kernel.governance.approvals import service
    from iris_harness.kernel.governance.approvals.queue import ApprovalQueue
    from iris_harness.kernel.governance.approvals.store import ApprovalStore

    q = ApprovalQueue(store=ApprovalStore(db_path=tmp_path / "approvals.db"))
    monkeypatch.setattr(service, "_queue", lambda given: given or q)
    return q


def test_a_pending_approval_reaches_the_action_center(store: TaskStore, approvals: object) -> None:
    approval_id = approvals.enqueue(  # type: ignore[attr-defined]
        "run-1", None, "goal_drift", "thought drifted from original task", channel="web"
    )

    actions = collect_pending_actions(store, None)

    approval_items = [a for a in actions if a.origin == "approval"]
    assert len(approval_items) == 1
    item = approval_items[0]
    assert item.id == f"approval:{approval_id}"
    assert item.source_kind == "governance-approval"
    assert "goal_drift" in item.title
    assert "drifted" in item.description
    assert approval_id in (item.action.command or "")


def test_an_answered_approval_disappears(store: TaskStore, approvals: object) -> None:
    """Read-time like the Health source: the item lives exactly as long as its row is
    pending, and nothing is ever copied into the task store."""
    approval_id = approvals.enqueue("run-1", None, "goal_drift", "drifted")  # type: ignore[attr-defined]
    assert any(a.origin == "approval" for a in collect_pending_actions(store, None))

    approvals.respond(approval_id, status="approved", actor="cli:owner")  # type: ignore[attr-defined]

    assert not any(a.origin == "approval" for a in collect_pending_actions(store, None))


def test_a_blocked_run_is_listed_before_what_the_harness_merely_noticed(
    store: TaskStore, approvals: object
) -> None:
    store.create(
        title="Wingtip card needs a password",
        source_kind="finance-statements",
        action=TaskAction(kind="copy_command", label="Set", command="iris finance secret ..."),
    )
    approvals.enqueue("run-1", None, "goal_drift", "drifted")  # type: ignore[attr-defined]

    actions = collect_pending_actions(store, None)

    assert actions[0].origin == "approval"


def test_an_approval_carries_no_suppression_affordance(store: TaskStore, approvals: object) -> None:
    """Everything else on this list is something the harness noticed; an approval is a
    run *waiting on the user*. "Don't show me these again" would strand runs silently."""
    approvals.enqueue("run-1", None, "goal_drift", "drifted")  # type: ignore[attr-defined]

    item = next(a for a in collect_pending_actions(store, None) if a.origin == "approval")

    assert item.feedback_ref is None


def test_the_approval_action_is_display_only(store: TaskStore, approvals: object) -> None:
    """Answering has two possible answers and is a privileged write, which does not fit
    the single-CTA `/actions/{id}/invoke` lifecycle — and these ids are not in the task
    store, so an invoke was never going to find them. The surfaces that can answer use
    the approvals endpoint; everyone else gets the command."""
    approvals.enqueue("run-1", None, "goal_drift", "drifted")  # type: ignore[attr-defined]

    item = next(a for a in collect_pending_actions(store, None) if a.origin == "approval")

    assert item.action.kind == "copy_command"
    assert item.action.safe is False


def test_a_broken_approval_source_does_not_break_the_read(
    store: TaskStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Health and task blockers have nothing to do with approvals; a governance surface
    that erred would otherwise take the whole Action Center down with it."""
    from iris_harness.kernel.governance.approvals import service

    def _boom(**_kwargs: object) -> list[object]:
        raise RuntimeError("approvals.db is unreadable")

    monkeypatch.setattr(service, "pending_approvals", _boom)
    store.create(
        title="Wingtip card needs a password",
        source_kind="finance-statements",
        action=TaskAction(kind="copy_command", label="Set", command="iris finance secret ..."),
    )

    actions = collect_pending_actions(store, None)

    assert len(actions) == 1
    assert actions[0].origin == "task"


def test_an_approval_renders_in_the_channel_agnostic_text(
    store: TaskStore, approvals: object
) -> None:
    """`render_pending_actions` is what chat and the CLI show, so the approval has to
    read as something a person can act on there, not only in a UI."""
    approvals.enqueue("run-1", None, "goal_drift", "drifted")  # type: ignore[attr-defined]

    text = render_pending_actions(collect_pending_actions(store, None))

    assert "goal_drift" in text
    assert "iris approvals approve" in text
