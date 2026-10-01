"""ApprovalChannel protocol — best-effort delivery; the queue is the source of truth.

The kernel owns *which* channel an approval goes to and *what* it says; it does not own
any transport. A channel that reaches an outside service -- Telegram today -- is built
above this layer and registered here, because the kernel cannot import the channels
package without importing upward (OSS plan M6, decision 6).

CLI, web and the stderr fallback need no transport: they write to a terminal, to a row
the browser polls, or to stderr. They stay in the kernel.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Protocol, runtime_checkable

from iris_harness.foundation.process_state import track_globals
from iris_harness.kernel.governance.approvals.store import ApprovalRow

logger = logging.getLogger(__name__)


@runtime_checkable
class ApprovalChannel(Protocol):
    name: str

    def notify(self, approval: ApprovalRow) -> None:
        """Deliver the approval request to the user. Best-effort — must not raise."""
        ...

    def is_configured(self) -> bool:
        """Whether this channel can actually deliver right now.

        A channel that needs credentials answers False without them, and the router
        skips it rather than dropping the notice into the void.
        """
        ...

    def notify_timeout(self, approval: ApprovalRow) -> None:
        """Tell the user the request lapsed unanswered. Best-effort — must not raise.

        A separate method rather than a reused ``notify``, because the request message
        is wrong twice over once the window has closed: it asks for a decision that can
        no longer be made, and — on Telegram — it offers ``/approve <id>`` commands that
        would now be refused. ``ChannelRouter`` calls this defensively, so a channel
        written before it still delivers requests.
        """
        ...


_lock = Lock()
_remote: ApprovalChannel | None = None


def register_remote_channel(channel: ApprovalChannel | None) -> None:
    """Install the channel that reaches the user off-box (``None`` clears it).

    Called by the composition root once the channel layer is up. With none registered,
    delivery falls back to CLI / web / stderr — which is the shape of an install that
    has no remote surface configured, and was already the behaviour when no bot token
    was set.
    """
    global _remote
    with _lock:
        _remote = channel


def remote_channel() -> ApprovalChannel | None:
    """The registered off-box channel, if any."""
    with _lock:
        return _remote


__all__ = ["ApprovalChannel", "register_remote_channel", "remote_channel"]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_remote")
