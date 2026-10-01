"""Wiring the channel gateway, and picking the default channel.

The default can only be resolved AFTER plugins mount -- chat surfaces are plugins,
so the set of valid channels is not known until then. That ordering rule is the
reason these four live together rather than inline in `build_runtime`, where the
sequence would read as if any of it could move (OSS plan M4.5, release gate 1).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from iris_harness.foundation.env import env_flag as _env_flag
from iris_harness.runtime.facade import IrisRuntime
from iris_harness.services.channels import (
    ChannelGateway,
    ChannelMessage,
    DeliveryStatus,
)
from iris_harness.services.channels.approval_commands import (
    CommandHandler,
    LocalApprovalBackend,
    handle_approval_command,
)
from iris_harness.services.channels.connectors.console import ConsoleConnector
from iris_harness.services.channels.reminder_commands import (
    ChainedCommands,
    LocalReminderBackend,
    ReminderCommands,
)
from iris_harness.services.heartbeat import (
    HeartbeatStatus,
)

logger = logging.getLogger(__name__)


def _runtime_telegram_poller_enabled() -> bool:
    """Return whether the legacy runtime Telegram poller should start.

    The channel gateway owns Telegram inbound by default. Running both the
    gateway poller and runtime poller with the same bot token causes Telegram
    Bot API HTTP 409 conflicts because only one getUpdates long-poll can be
    active per bot.
    """
    explicit_runtime = os.getenv("IRIS_RUNTIME_TELEGRAM_POLLER_ENABLED")
    if explicit_runtime is not None:
        return _env_flag("IRIS_RUNTIME_TELEGRAM_POLLER_ENABLED", default=False)
    return not _env_flag("IRIS_CHANNEL_GATEWAY_TELEGRAM_ENABLED", default=True)


def _telegram_approval_commands(runtime: IrisRuntime) -> CommandHandler:
    """Commands for the runtime's own Telegram poller, answered in process: approvals
    first (the runtime is the resumer, so approving continues the run right here), then
    a reminder's Done / Snooze — a tap, a typed command, or (``on_reply``) a reply to
    the reminder's message — written straight to the one reminder store (PR 3b)."""
    from iris_harness.foundation.eventbus import get_default_bus
    from iris_harness.kernel.governance.approvals import ApprovalQueue
    from iris_harness.kernel.governance.audit import AuditLog

    backend = LocalApprovalBackend(
        queue=ApprovalQueue(audit_log=AuditLog()),
        resumer=runtime,
        # A code caller's approved call runs here, through the runtime's governed tools.
        executor=runtime.tool_service,
    )
    reminders = LocalReminderBackend(data_dir=runtime.data_dir, bus=get_default_bus())
    return ChainedCommands(
        lambda text, user_id: handle_approval_command(text, user_id, backend),
        ReminderCommands(reminders),
    )


def _channel_dispatch(
    channels: ChannelGateway,
    channel: str,
    msg: ChannelMessage,
) -> tuple[HeartbeatStatus, str | None]:
    """Send to one channel or broadcast to all when channel == 'all'."""
    if channel == "all":
        receipts = channels.broadcast(msg)
        failed = [r for r in receipts if r.status is not DeliveryStatus.SENT]
        if failed:
            return HeartbeatStatus.FAILED, "; ".join(r.error or r.channel for r in failed)
        return HeartbeatStatus.SUCCESS, None
    receipt = channels.send(channel, msg)
    status = (
        HeartbeatStatus.SUCCESS if receipt.status is DeliveryStatus.SENT else HeartbeatStatus.FAILED
    )
    return status, receipt.error


def _load_channels(cfg: Path) -> tuple[ChannelGateway, str]:
    """Build the gateway with the CORE sinks, and read the operator's default name.

    Only ``type: console`` rows register here. A chat *surface* — Telegram, the web
    UI — is a plugin (OSS plan M4.5), so its connector arrives during
    ``_mount_plugins``, which is why this returns the **desired** default name
    rather than a validated one. ``_resolve_default_channel`` validates it once
    every surface has had its chance to register.

    Console stays core for one reason: it is the guaranteed fallback. A profile
    that mounts no channel plugin at all must still have somewhere for a proactive
    notice to go.
    """
    from iris_harness.services.channels.channel_config import load_channel_config

    gateway = ChannelGateway()
    config = load_channel_config(cfg)
    for row in config.of_type("console"):
        gateway.register(ConsoleConnector(name=row.name))
    if not gateway.channels():
        # Either the file is absent, or it declares no console sink. Register one
        # anyway — see the fallback argument above.
        gateway.register(ConsoleConnector(name="console"))
    return gateway, config.default


def _resolve_default_channel(runtime: IrisRuntime, desired: str) -> None:
    """Pin ``runtime.default_channel`` once every channel plugin has registered.

    Called right after ``_mount_plugins``. Before M4.5 this ran inside
    ``_load_channels``, which worked only because the core built the Telegram
    connector itself; with surfaces as plugins, validating that early would demote
    ``default: telegram`` to console on every boot.
    """
    registered = runtime.channels.channels()
    if desired in registered:
        runtime.default_channel = desired
    else:
        fallback = registered[0] if registered else "console"
        logger.warning("default channel %r not registered, falling back to %r", desired, fallback)
        runtime.default_channel = fallback
    logger.info("channels registered: %s  |  default: %s", registered, runtime.default_channel)
