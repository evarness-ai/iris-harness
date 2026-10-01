"""Approval delivery channels the kernel itself can build.

Telegram delivery is not here: it reaches an outside service, so it lives in the
channels layer above and registers through :func:`register_remote_channel`
(OSS plan M6, decision 6).
"""

from iris_harness.kernel.governance.approvals.channels.base import (
    ApprovalChannel,
    register_remote_channel,
    remote_channel,
)
from iris_harness.kernel.governance.approvals.channels.cli_channel import CLIChannel
from iris_harness.kernel.governance.approvals.channels.web_channel import WebChannel

__all__ = [
    "ApprovalChannel",
    "CLIChannel",
    "WebChannel",
    "register_remote_channel",
    "remote_channel",
]
