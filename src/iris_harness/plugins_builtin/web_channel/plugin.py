"""``setup(api)`` for the web channel plugin (OSS plan M4.5).

One registration kind. The plugin mounts only when the harness exposes the
in-chat delivery path — without it there is nothing to deliver *into*, and a
connector that silently swallowed every message would be worse than an absent
one (a brief addressed to ``web`` would report SENT and vanish).
"""

from __future__ import annotations

import logging

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.channels import load_channel_config

from .connector import WebConnector

logger = logging.getLogger(__name__)


def setup(api: PluginAPI) -> None:
    deliver = api.services.deliver_in_chat
    if deliver is None:
        logger.warning("web_channel not mounted: the harness exposes no in-chat delivery path")
        return
    # Honour an explicit `type: web` row's name if the operator declared one, so a
    # second web surface can be addressed separately; otherwise the name is `web`.
    rows = load_channel_config(api.services.config_dir).of_type("web")
    names = [r.name for r in rows] or ["web"]
    for name in names:
        api.register_channel(WebConnector(deliver, name=name))
        logger.debug("web_channel: registered %r", name)
