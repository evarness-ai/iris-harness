"""ChannelRouter — selects and dispatches the right approval channel."""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

from iris_harness.kernel.governance.approvals.channels.base import ApprovalChannel, remote_channel
from iris_harness.kernel.governance.approvals.channels.cli_channel import CLIChannel
from iris_harness.kernel.governance.approvals.channels.web_channel import WebChannel
from iris_harness.kernel.governance.approvals.store import ApprovalRow

if TYPE_CHECKING:
    from iris_harness.kernel.governance.approvals.queue import ApprovalQueue

logger = logging.getLogger(__name__)


class _StderrFallbackChannel:
    """Emit a one-line stderr notice when neither CLI nor Telegram is available."""

    name = "stderr"

    def is_configured(self) -> bool:
        """Always: stderr is the one destination that cannot be unconfigured."""
        return True

    def notify(self, approval: ApprovalRow) -> None:
        print(  # noqa: T201 — this channel IS stderr: the fallback when nothing else is set
            f"[iris/approval-required] run={approval.run_id} id={approval.approval_id} "
            f"  signal={approval.signal}: {approval.context_summary}\n"
            f"  -> iris approvals approve {approval.approval_id}",
            file=sys.stderr,
        )

    def notify_timeout(self, approval: ApprovalRow) -> None:
        print(  # noqa: T201 — this channel IS stderr: the fallback when nothing else is set
            f"[iris/approval-timed-out] run={approval.run_id} id={approval.approval_id}"
            f"  signal={approval.signal}: no answer before {approval.timeout_at[:19]};"
            f" {approval.lapse_consequence()}.",
            file=sys.stderr,
        )


class ChannelRouter:
    """Select the best delivery channel per approval and call notify()."""

    def __init__(
        self,
        *,
        queue: ApprovalQueue,
        force_interactive: bool | None = None,
        remote: ApprovalChannel | None = None,
    ) -> None:
        self._queue = queue
        self._force_interactive = force_interactive
        self._cli = CLIChannel(queue, force_interactive=force_interactive)
        # The off-box channel (Telegram today) is built above the kernel and registered
        # by the composition root; an explicit one wins, for tests (M6.2, decision 6).
        self._remote_override = remote
        self._web = WebChannel()
        self._fallback = _StderrFallbackChannel()

    @property
    def _remote(self) -> ApprovalChannel | None:
        """Resolved per call: registration happens after the kernel is built."""
        return self._remote_override if self._remote_override is not None else remote_channel()

    def _is_interactive(self) -> bool:
        if self._force_interactive is not None:
            return self._force_interactive
        return sys.stdin.isatty() and os.environ.get("IRIS_INTERACTIVE_SESSION", "1") != "0"

    def select(self, approval: ApprovalRow) -> ApprovalChannel:
        """Deliver where the request came from; guess only when it did not say.

        The ``channel`` column has been on every row since Phase 3 and was, until the
        evaluator started stamping it, always the ``"cli"`` default — so this method
        could only ever guess, and under the API server it guessed wrong twice over:
        ``sys.stdin.isatty()`` is false there, so a web turn's approval went to
        Telegram if a bot happened to be configured and to the server's stderr
        otherwise. Honouring the origin first is the fix; the old chain stays as the
        fallback for rows that genuinely carry no origin.
        """
        origin = (approval.channel or "").strip().lower()
        if origin == "web":
            return self._web
        remote = self._remote
        if remote is not None and origin == remote.name and remote.is_configured():
            return remote

        if self._is_interactive():
            return self._cli
        if remote is not None and remote.is_configured():
            return remote
        return self._fallback

    def copy_to(self, approval: ApprovalRow) -> ApprovalChannel | None:
        """Where a second copy goes, besides the channel ``select`` chose.

        The web channel is pull-based: it reaches the owner only while the app is open.
        A phone that backgrounded the app mid-turn never sees it, and the halt message
        offers Telegram as a way to answer — so the owner looked there, and there was
        nothing (2026-09-21). A web-raised approval therefore also goes off-box when a
        bot is configured. Whichever copy is answered first settles it; the other is
        refused as already answered. Other origins are unchanged: Telegram is already
        off-box, and an interactive CLI prompt is in front of the person asked.
        """
        if (approval.channel or "").strip().lower() != "web":
            return None
        remote = self._remote
        if remote is None or not remote.is_configured():
            return None
        return remote

    def notify(self, approval: ApprovalRow) -> None:
        channel = self.select(approval)
        logger.info(
            "routing approval %s to channel=%s (run=%s)",
            approval.approval_id,
            channel.name,
            approval.run_id,
        )
        channel.notify(approval)
        copy = self.copy_to(approval)
        if copy is not None:
            logger.info("approval %s: a copy goes to channel=%s", approval.approval_id, copy.name)
            try:
                copy.notify(approval)
            except Exception:  # the copy is best-effort; the queue holds it
                logger.warning(
                    "channel=%s failed to deliver a copy of approval %s",
                    copy.name,
                    approval.approval_id,
                    exc_info=True,
                )
        self._announce(approval, channel.name)

    def _announce(self, approval: ApprovalRow, channel_name: str) -> None:
        """Say on the bus that something is waiting, for anyone who nudges.

        Best-effort and after delivery: the approval is already queued and
        routed, so a subscriber that raises must not take the routing with it.
        The bus is in ``foundation``, below this layer — nothing here learns
        what a push or a Telegram message is.
        """
        from iris_harness.foundation.eventbus import get_default_bus

        from .events import APPROVAL_REQUESTED, ApprovalRequestedPayload

        try:
            get_default_bus().emit_sync(
                APPROVAL_REQUESTED,
                ApprovalRequestedPayload(
                    approval_id=approval.approval_id,
                    run_id=approval.run_id,
                    signal=approval.signal,
                    context_summary=approval.context_summary,
                    timeout_at=approval.timeout_at,
                    channel=channel_name,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - a nudge must never break an approval
            logger.warning("approval %s: announce failed (%s)", approval.approval_id, exc)

    def notify_timeout(self, approval: ApprovalRow) -> None:
        """Tell the surface that asked that the window closed unanswered.

        Same channels as the request (its copy too), because where the owner was asked
        is where to tell them. Called defensively: ``notify_timeout`` is newer than the
        protocol, and a channel without one should still deliver requests rather than
        break the sweep.
        """
        for channel in (self.select(approval), self.copy_to(approval)):
            if channel is not None:
                self._notify_timeout_on(channel, approval)

    def _notify_timeout_on(self, channel: ApprovalChannel, approval: ApprovalRow) -> None:
        notify_timeout = getattr(channel, "notify_timeout", None)
        if notify_timeout is None:
            logger.info(
                "channel=%s cannot announce timeouts; approval %s lapsed silently there",
                channel.name,
                approval.approval_id,
            )
            return
        logger.info(
            "routing approval %s timeout to channel=%s (run=%s)",
            approval.approval_id,
            channel.name,
            approval.run_id,
        )
        try:
            notify_timeout(approval)
        except Exception:  # delivery is best-effort by contract
            logger.warning(
                "channel=%s failed to announce the timeout of approval %s",
                channel.name,
                approval.approval_id,
                exc_info=True,
            )
