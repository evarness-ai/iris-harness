"""The notification_reminder_tick heartbeat: accept-then-fire, and a run that reports
an undelivered reminder as FAILED rather than SUCCESS (loop-proof D14)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from iris_harness.runtime.handlers.ticks import build_notification_reminder_tick_handler
from iris_harness.services.channels import ChannelGateway, ChannelMessage, DeliveryStatus
from iris_harness.services.channels.models import DeliveryReceipt
from iris_harness.services.heartbeat import HeartbeatDefinition, HeartbeatStatus
from iris_harness.services.notifications.store import ReminderStore

DEFINITION = HeartbeatDefinition(
    name="notification_reminder_tick", handler="notification_reminder_tick", schedule="interval:60"
)


class _Channel:
    def __init__(self, name: str, status: DeliveryStatus) -> None:
        self.name = name
        self.status = status
        self.sent: list[ChannelMessage] = []

    def send(self, message: ChannelMessage) -> DeliveryReceipt:
        self.sent.append(message)
        return DeliveryReceipt(channel=self.name, status=self.status, error="down")


def _runtime(tmp_path: Path, telegram: DeliveryStatus) -> tuple[SimpleNamespace, _Channel]:
    config = tmp_path / "config"
    config.mkdir()
    (config / "notifications.yaml").write_text("reminder_channels: [telegram, web_push]\n")
    gateway = ChannelGateway()
    tg = _Channel("telegram", telegram)
    gateway.register(tg)
    gateway.register(_Channel("console", DeliveryStatus.SENT))
    runtime = SimpleNamespace(
        data_dir=tmp_path, config_dir=config, channels=gateway, default_channel="console"
    )
    return runtime, tg


def _due(tmp_path: Path) -> str:
    store = ReminderStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    at = datetime.now(UTC) - timedelta(seconds=5)
    return store.create(target_kind="task", target_id="t", remind_at=at, note="stretch").id


@pytest.fixture(autouse=True)
def _chicago(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_TZ", "America/Chicago")


def test_a_delivered_reminder_is_a_successful_run(tmp_path: Path) -> None:
    runtime, tg = _runtime(tmp_path, DeliveryStatus.SENT)
    rid = _due(tmp_path)

    run = build_notification_reminder_tick_handler(runtime)(DEFINITION)  # type: ignore[arg-type]

    assert run.status is HeartbeatStatus.SUCCESS
    assert "sent=1" in run.output
    assert tg.sent and tg.sent[0].body.startswith("⏰ <b>stretch</b>")
    row = ReminderStore(db_path=tmp_path / "tasks.db").get(rid)
    assert row is not None and row.status == "sent"


def test_an_undelivered_reminder_fails_the_run(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path, DeliveryStatus.FAILED)
    rid = _due(tmp_path)

    run = build_notification_reminder_tick_handler(runtime)(DEFINITION)  # type: ignore[arg-type]

    assert run.status is HeartbeatStatus.FAILED
    assert "retrying=1" in run.error
    row = ReminderStore(db_path=tmp_path / "tasks.db").get(rid)
    assert row is not None and row.status == "pending" and row.attempts == 1


def test_nothing_due_is_a_quiet_success(tmp_path: Path) -> None:
    runtime, tg = _runtime(tmp_path, DeliveryStatus.SENT)
    run = build_notification_reminder_tick_handler(runtime)(DEFINITION)  # type: ignore[arg-type]
    assert run.status is HeartbeatStatus.SUCCESS and tg.sent == []
