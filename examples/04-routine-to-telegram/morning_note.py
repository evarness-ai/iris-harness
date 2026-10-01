"""A routine delivered to a channel: a morning note, sent to Telegram on a schedule.

``api.register_heartbeat`` registers a job the harness's scheduler runs -- on its own
``schedule`` (cron, or ``interval:<seconds>``), when the owner fires it by hand, or when
a routine the owner set up in chat ("every weekday at 7, send me the morning note on
Telegram") names it. The job builds the note and hands it to the channel gateway,
``api.services.channels.broadcast``; which channel it goes to is the routine's choice
(``definition.params["channel"]``), else the owner's default channel.

The plugin never talks to Telegram itself: the Telegram channel (the ``telegram_channel``
plugin, configured with the owner's bot) does, like every other channel.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.channels import ChannelMessage, DeliveryStatus
from iris_harness.sdk.time import local_today
from iris_harness.sdk.types import HeartbeatDefinition, HeartbeatRun, HeartbeatStatus

NAME = "morning_note"
NOTE_FILE = Path(__file__).with_name("note.yaml")


def compose(lines: list[str], today: str) -> str:
    return "\n".join([f"Good morning. It is {today}.", *(f"- {line}" for line in lines)])


def setup(api: PluginAPI) -> None:
    raw = yaml.safe_load(NOTE_FILE.read_text(encoding="utf-8")) or {}
    schedule = str(raw.get("schedule", "0 7 * * 1-5"))
    lines = [str(line) for line in raw.get("lines") or []]
    services = api.services

    def send_note(definition: HeartbeatDefinition) -> HeartbeatRun:
        started = datetime.now(UTC)
        channel = definition.params.get("channel") or (
            services.default_channel() if services.default_channel else None
        )
        # An empty recipient: each channel sends to its own configured owner.
        message = ChannelMessage(recipient="", body=compose(lines, local_today().isoformat()))
        receipts = services.channels.broadcast(
            message, channels=[str(channel)] if channel else None
        )
        sent = [r.channel for r in receipts if r.status is DeliveryStatus.SENT]
        failed = [
            f"{r.channel}: {r.error}" for r in receipts if r.status is not DeliveryStatus.SENT
        ]
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS if sent and not failed else HeartbeatStatus.FAILED,
            started_at=started,
            finished_at=datetime.now(UTC),
            output=f"sent to {', '.join(sent) or 'nobody'}",
            error="; ".join(failed),
            result={"sent": sent},
        )

    api.register_heartbeat(
        NAME,
        send_note,
        schedule=schedule,
        description="The morning note, sent to the owner's channel.",
    )
