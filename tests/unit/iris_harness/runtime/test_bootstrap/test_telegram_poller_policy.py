"""Tests for Telegram poller ownership between runtime and channel gateway."""

from __future__ import annotations

from iris_harness.runtime.bootstrap import _runtime_telegram_poller_enabled


def test_runtime_telegram_poller_disabled_when_gateway_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("IRIS_CHANNEL_GATEWAY_TELEGRAM_ENABLED", raising=False)
    monkeypatch.delenv("IRIS_RUNTIME_TELEGRAM_POLLER_ENABLED", raising=False)

    assert _runtime_telegram_poller_enabled() is False


def test_runtime_telegram_poller_enabled_when_gateway_explicitly_disabled(monkeypatch) -> None:
    monkeypatch.setenv("IRIS_CHANNEL_GATEWAY_TELEGRAM_ENABLED", "0")
    monkeypatch.delenv("IRIS_RUNTIME_TELEGRAM_POLLER_ENABLED", raising=False)

    assert _runtime_telegram_poller_enabled() is True


def test_runtime_telegram_poller_explicit_flag_overrides_gateway_default(monkeypatch) -> None:
    monkeypatch.delenv("IRIS_CHANNEL_GATEWAY_TELEGRAM_ENABLED", raising=False)
    monkeypatch.setenv("IRIS_RUNTIME_TELEGRAM_POLLER_ENABLED", "1")

    assert _runtime_telegram_poller_enabled() is True


def test_runtime_telegram_poller_explicit_false_wins(monkeypatch) -> None:
    monkeypatch.setenv("IRIS_CHANNEL_GATEWAY_TELEGRAM_ENABLED", "0")
    monkeypatch.setenv("IRIS_RUNTIME_TELEGRAM_POLLER_ENABLED", "false")

    assert _runtime_telegram_poller_enabled() is False


def test_the_runtime_poller_answers_approvals_in_process(tmp_path) -> None:
    """With the gateway off, the runtime's own poller answers a tapped approval and the
    runtime resumes the run right here (approvals follow-up, 2026-09-21)."""
    from iris_harness.kernel.governance.approvals import ApprovalQueue
    from iris_harness.kernel.governance.approvals.service import ResumedRun
    from iris_harness.runtime.channel_wiring import _telegram_approval_commands

    class _Runtime:
        def __init__(self) -> None:
            self.resumed: list[tuple[str, int]] = []
            self.data_dir = tmp_path
            self.tool_service = None  # no code caller's approval here

        def resume_halted_run(
            self, *, run_id: str, step_id: int, channel: str = "console"
        ) -> ResumedRun:
            self.resumed.append((run_id, step_id))
            return ResumedRun(answer="Continued.")

    queue = ApprovalQueue()  # the shared default store, as the runtime's is
    approval_id = queue.enqueue("run-tg", None, "step_cap", "Too many steps", channel="telegram")
    queue.set_checkpoint(approval_id, "run-tg:2")
    runtime = _Runtime()

    handler = _telegram_approval_commands(runtime)  # type: ignore[arg-type]
    reply = handler(f"/approve {approval_id}", "42")

    assert reply is not None and reply.text.startswith("Approved: step_cap.")
    assert runtime.resumed == [("run-tg", 2)]
    assert queue.get(approval_id).response_actor == "telegram:42"  # type: ignore[union-attr]
    assert handler("hello", "42") is None  # not a command: it goes to chat


def test_the_runtime_poller_answers_reminder_done_and_snooze_in_process(
    tmp_path, monkeypatch
) -> None:
    """Chained after approvals: a reminder button, a typed command and a reply to the
    reminder's message all write the one store in process (PR 3b)."""
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    from iris_harness.runtime.channel_wiring import _telegram_approval_commands
    from iris_harness.services.notifications.store import ReminderStore

    monkeypatch.setenv("IRIS_TZ", "America/Chicago")
    monkeypatch.setattr(
        "iris_harness.services.channels.approval_commands.load_allowed_users",
        lambda path=None: frozenset(),
    )
    monkeypatch.setattr(
        "iris_harness.services.channels.reminder_commands.load_allowed_users",
        lambda path=None: frozenset(),
    )
    store = ReminderStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    due = datetime.now(UTC) - timedelta(minutes=1)
    tapped = store.create(target_kind="task", target_id="t1", remind_at=due, note="Call")
    replied = store.create(target_kind="task", target_id="t2", remind_at=due, note="Water")
    store.mark_sent(
        replied.id,
        delivered_channels=["telegram"],
        message_refs=[{"channel": "telegram", "chat_id": "555", "message_id": "77"}],
    )

    handler = _telegram_approval_commands(SimpleNamespace(data_dir=tmp_path, tool_service=None))  # type: ignore[arg-type]

    reply = handler(f"/done {tapped.id}", "42")
    assert reply is not None and reply.text == "✅ Marked done."
    assert store.get(tapped.id).status == "done"  # type: ignore[union-attr]

    answer = handler.on_reply("snooze 1h", "42", "555", "77")  # type: ignore[attr-defined]
    assert answer is not None and answer.text.startswith("⏰ Snoozed — I'll remind you")
    row = store.get(replied.id)
    assert row is not None and row.status == "pending" and row.remind_at > datetime.now(UTC)
    # Not a reminder's message, or not a Done / Snooze: chat, unchanged.
    assert handler.on_reply("snooze 1h", "42", "555", "78") is None  # type: ignore[attr-defined]
    assert handler.on_reply("thanks!", "42", "555", "77") is None  # type: ignore[attr-defined]
    assert handler("hello", "42") is None
