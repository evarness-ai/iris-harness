"""WebChannel — delivery for an approval raised by a web turn.

The base protocol already says the important thing: a channel is *best-effort
delivery; the queue is the source of truth*. For the web that is the whole design.
The UI is a polling REST client, so an approval becomes visible the moment its row
exists — through ``GET /governance/approvals`` and, because a pending approval is
something that needs the user's attention, through the Action Center's ``GET
/actions`` alongside health items and task blockers. There is nothing to push.

So ``notify`` records rather than delivers, and that is not a stub — it is what
pull-based delivery looks like. The channel earns its place by existing: before it,
``ChannelRouter.select`` had no web branch, so a web-raised approval fell through
``sys.stdin.isatty()`` (false under the API server) to Telegram, or to a line on the
server's stderr. The user watching the browser was the one person guaranteed not to
be told.
"""

from __future__ import annotations

import logging

from iris_harness.kernel.governance.approvals.store import ApprovalRow

logger = logging.getLogger(__name__)


class WebChannel:
    """Make a web-raised approval discoverable to the surfaces the web UI reads."""

    name = "web"

    def is_configured(self) -> bool:
        """Always. The queue is the transport, and it is always there."""
        return True

    def notify(self, approval: ApprovalRow) -> None:
        logger.info(
            "approval %s awaiting a web response (run=%s signal=%s checkpoint=%s)",
            approval.approval_id,
            approval.run_id,
            approval.signal,
            approval.checkpoint_id or "unlinked",
        )

    def notify_timeout(self, approval: ApprovalRow) -> None:
        """Pull-based here too: the row's status is the notice.

        The web UI stops listing it as waiting because ``GET /governance/approvals``
        and the Action Center both read pending rows only. What tells the *user* is
        the in-chat notice the sweep posts into the session the run belongs to —
        delivered by the runtime, because that is what owns conversations.
        """
        logger.info(
            "approval %s lapsed unanswered (run=%s signal=%s)",
            approval.approval_id,
            approval.run_id,
            approval.signal,
        )
