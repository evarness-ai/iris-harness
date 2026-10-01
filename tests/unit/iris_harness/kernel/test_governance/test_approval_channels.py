"""Tests for approval channels (story 12.gov-4.8)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from iris_harness.kernel.governance.approvals import (
    ApprovalQueue,
    ApprovalRow,
    ApprovalStore,
)
from iris_harness.kernel.governance.approvals.channels.cli_channel import CLIChannel
from iris_harness.kernel.governance.approvals.router import ChannelRouter, _StderrFallbackChannel
from iris_harness.services.channels.approval_delivery import TelegramApprovalChannel
from iris_harness.services.channels.models import DeliveryReceipt, DeliveryStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(approval_id: str = "a" * 36, run_id: str = "run-1") -> ApprovalRow:
    return ApprovalRow(
        approval_id=approval_id,
        run_id=run_id,
        checkpoint_id=None,
        signal="step_cap",
        context_summary="Too many steps reached",
        requested_at="2026-01-01T00:00:00+00:00",
        channel="cli",
        status="pending",
        responded_at=None,
        response_actor=None,
        timeout_at="2026-01-01T00:10:00+00:00",
        policy_on_timeout="fail_closed",
    )


def _make_queue(tmp_path: Path) -> ApprovalQueue:
    return ApprovalQueue(store=ApprovalStore(db_path=tmp_path / "approvals.db"))


# ---------------------------------------------------------------------------
# CLIChannel
# ---------------------------------------------------------------------------


def test_cli_channel_interactive_approve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    queue = _make_queue(tmp_path)
    approval_id = queue.enqueue("run-1", None, "step_cap", "Too many steps")
    row = queue.get(approval_id)
    assert row is not None

    channel = CLIChannel(queue, force_interactive=True)
    monkeypatch.setattr("builtins.input", lambda _: "y")
    channel.notify(row)

    updated = queue.get(approval_id)
    assert updated is not None
    assert updated.status == "approved"


def test_cli_channel_interactive_reject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    queue = _make_queue(tmp_path)
    approval_id = queue.enqueue("run-1", None, "step_cap", "Too many steps")
    row = queue.get(approval_id)
    assert row is not None

    channel = CLIChannel(queue, force_interactive=True)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    channel.notify(row)

    updated = queue.get(approval_id)
    assert updated is not None
    assert updated.status == "rejected"


def test_cli_channel_interactive_default_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = _make_queue(tmp_path)
    approval_id = queue.enqueue("run-1", None, "step_cap", "Too many steps")
    row = queue.get(approval_id)
    assert row is not None

    channel = CLIChannel(queue, force_interactive=True)
    monkeypatch.setattr("builtins.input", lambda _: "")  # empty = default yes
    channel.notify(row)

    updated = queue.get(approval_id)
    assert updated is not None
    assert updated.status == "approved"


def test_cli_channel_non_interactive_no_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = _make_queue(tmp_path)
    approval_id = queue.enqueue("run-1", None, "step_cap", "Too many steps")
    row = queue.get(approval_id)
    assert row is not None

    input_called = False

    def _should_not_be_called(prompt: str) -> str:
        nonlocal input_called
        input_called = True
        return "y"

    monkeypatch.setattr("builtins.input", _should_not_be_called)
    channel = CLIChannel(queue, force_interactive=False)
    channel.notify(row)

    assert not input_called
    # status should remain pending since no interaction happened
    updated = queue.get(approval_id)
    assert updated is not None
    assert updated.status == "pending"


def test_cli_channel_eof_no_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    queue = _make_queue(tmp_path)
    approval_id = queue.enqueue("run-1", None, "step_cap", "Too many steps")
    row = queue.get(approval_id)
    assert row is not None

    def _raise_eof(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)
    channel = CLIChannel(queue, force_interactive=True)
    channel.notify(row)  # must not raise

    updated = queue.get(approval_id)
    assert updated is not None
    assert updated.status == "pending"  # no answer given


# ---------------------------------------------------------------------------
# TelegramApprovalChannel
# ---------------------------------------------------------------------------


def test_telegram_channel_sends_message(tmp_path: Path) -> None:
    row = _make_row(approval_id="12345678-1234-1234-1234-123456789012")

    mock_connector = MagicMock()
    mock_connector.send.return_value = DeliveryReceipt(
        channel="telegram", status=DeliveryStatus.SENT
    )

    channel = TelegramApprovalChannel(connector=mock_connector, chat_id="@testchat")
    channel.notify(row)

    assert mock_connector.send.call_count == 1
    sent_msg = mock_connector.send.call_args[0][0]
    # One tap answers: the commands ride on the buttons, not in text to copy.
    [[approve, reject]] = sent_msg.metadata["inline_keyboard"]
    assert approve["callback_data"] == f"/approve {row.approval_id}"
    assert reject["callback_data"] == f"/reject {row.approval_id}"
    assert "Tap a button below" in sent_msg.body


def test_telegram_sends_a_destructive_approval_in_plain_words(tmp_path: Path) -> None:
    """ADR-0118 step 4: the card, not Signal/Context, and a subject's Markdown specials
    escaped so Telegram neither mangles nor rejects the message."""
    from dataclasses import replace

    from iris_harness.kernel.governance.approvals import ApprovalCard, ApprovalItem

    row = replace(
        _make_row(approval_id="12345678-1234-1234-1234-123456789012"),
        signal="Trash 2 emails",
        # A destructive row always pins its calls; that is what makes it destructive.
        items=(ApprovalItem.of("trash_email", {"ids": ["m1", "m2"]}),),
        card=ApprovalCard(
            title="Trash 2 emails",
            lines=("*50% off* _today_ — Store X", "Your [weekly] deals — Shop Y"),
            undo_tool="restore_email",
            undo_window_days=30,
            asked="clean up the promos",
        ),
    )
    mock_connector = MagicMock()
    mock_connector.send.return_value = DeliveryReceipt(
        channel="telegram", status=DeliveryStatus.SENT
    )

    TelegramApprovalChannel(connector=mock_connector, chat_id="@testchat").notify(row)

    body = mock_connector.send.call_args[0][0].body
    assert body.startswith("*Approve?* Trash 2 emails")
    assert "• \\*50% off\\* \\_today\\_ — Store X" in body
    assert "• Your \\[weekly] deals — Shop Y" in body
    assert "Reversible for 30 days (undo: restore\\_email)." in body
    assert 'You asked: "clean up the promos"' in body
    assert "Signal:" not in body
    [[action, reject]] = mock_connector.send.call_args[0][0].metadata["inline_keyboard"]
    assert (action["text"], action["callback_data"]) == (
        "Trash 2 emails",
        f"/ask {row.approval_id}",
    )
    assert reject["callback_data"] == f"/reject {row.approval_id}"


def test_telegram_channel_degraded_no_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    row = _make_row()
    channel = TelegramApprovalChannel()  # no connector, no token
    channel.notify(row)  # must not raise


# ---------------------------------------------------------------------------
# ChannelRouter
# ---------------------------------------------------------------------------


def test_channel_router_interactive_picks_cli(tmp_path: Path) -> None:
    queue = _make_queue(tmp_path)
    router = ChannelRouter(queue=queue, force_interactive=True)
    row = _make_row()
    selected = router.select(row)
    assert isinstance(selected, CLIChannel)


def test_channel_router_non_interactive_with_telegram_picks_telegram(tmp_path: Path) -> None:
    queue = _make_queue(tmp_path)
    mock_connector = MagicMock()
    # The transport is injected now: the kernel keeps the protocol and the routing,
    # the channels layer owns the Telegram delivery (M6.2, decision 6).
    router = ChannelRouter(
        queue=queue,
        force_interactive=False,
        remote=TelegramApprovalChannel(connector=mock_connector, chat_id="@testchat"),
    )
    row = _make_row()
    selected = router.select(row)
    assert isinstance(selected, TelegramApprovalChannel)


def test_channel_router_fallback_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    queue = _make_queue(tmp_path)
    router = ChannelRouter(queue=queue, force_interactive=False)
    row = _make_row()
    selected = router.select(row)
    assert isinstance(selected, _StderrFallbackChannel)
