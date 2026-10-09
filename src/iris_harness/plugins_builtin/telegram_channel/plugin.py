"""``setup(api)`` for the Telegram channel plugin (OSS plan M4.5).

One registration kind. The connector construction — reading ``channels.yaml``,
resolving ``${TELEGRAM_BOT_TOKEN}``, skipping the row when the token is absent —
is the code that used to be a branch in ``bootstrap._load_channels``; it is here
now, unchanged, so the core no longer names a surface.

The approvals router is untouched by this. It builds its own ``TelegramConnector``
from the environment when the gateway has none (see
``governance/approvals/channels/telegram_channel.py``), so an approval still
reaches a phone under a profile that does not mount this plugin.
"""

from __future__ import annotations

import logging

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.channels import TelegramConnector, telegram_rows

logger = logging.getLogger(__name__)


def setup(api: PluginAPI) -> None:
    for row in telegram_rows(api.services.config_dir):
        if not row.bot_token:
            # channels.yaml documents this skip as silent: Telegram is opt-in, and a row whose
            # token is unset is a surface the operator has not set up, not a fault.
            logger.info("channel %r skipped — TELEGRAM_BOT_TOKEN not set", row.name)
            continue
        api.register_channel(
            TelegramConnector(
                bot_token=row.bot_token,
                name=row.name,
                default_chat_id=row.default_chat_id or None,
            )
        )
        logger.debug("telegram_channel: registered %r", row.name)
