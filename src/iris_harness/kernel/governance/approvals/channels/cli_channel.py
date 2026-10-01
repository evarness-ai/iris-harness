"""CLIChannel — synchronous Y/n prompt on an interactive TTY."""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

from iris_harness.kernel.governance.approvals.store import ApprovalAlreadyAnsweredError, ApprovalRow

if TYPE_CHECKING:
    from iris_harness.kernel.governance.approvals.queue import ApprovalQueue

logger = logging.getLogger(__name__)


class CLIChannel:
    """Blocks on input() when the session is interactive; no-op otherwise."""

    name = "cli"

    def __init__(
        self,
        queue: ApprovalQueue,
        *,
        force_interactive: bool | None = None,
    ) -> None:
        self._queue = queue
        self._force_interactive = force_interactive

    def _is_interactive(self) -> bool:
        if self._force_interactive is not None:
            return self._force_interactive
        return sys.stdin.isatty() and os.environ.get("IRIS_INTERACTIVE_SESSION", "1") != "0"

    def is_configured(self) -> bool:
        """Always: this channel writes to the terminal the process already has."""
        return True

    def notify(self, approval: ApprovalRow) -> None:
        if approval.is_deferred_call:
            # A code caller's call (plugin-capabilities decision 1). This prompt would run
            # inside that call's own PreToolUse, where a "yes" can execute nothing: the
            # answer has to reach ``respond_to_approval``, which runs the call once. So
            # say where to answer it instead of asking.
            print(  # noqa: T201 — TTY channel
                f"\n[approval-required] {approval.context_summary}\n"
                f"  -> iris approvals approve {approval.approval_id}",
                file=sys.stderr,
            )
            return
        if not self._is_interactive():
            logger.info(
                "non-interactive: approval %s queued for manual review via `iris approvals list`",
                approval.approval_id,
            )
            return

        prompt = f"\n[approval-required] {approval.context_summary}\n  Approve? [Y/n] "
        try:
            answer = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            logger.warning("approval %s not answered (EOF/interrupt)", approval.approval_id)
            return

        status = "approved" if answer in ("", "y", "yes") else "rejected"
        actor = f"cli:{os.environ.get('USER', 'user')}"
        try:
            self._queue.respond(approval.approval_id, status=status, actor=actor)
            print(f"[approval] {status} ({approval.approval_id[:8]}…)")  # noqa: T201 — TTY channel
        except ApprovalAlreadyAnsweredError:
            logger.warning("approval %s already answered", approval.approval_id)

    def notify_timeout(self, approval: ApprovalRow) -> None:
        """Say it lapsed on stderr, whether or not anyone was at the prompt.

        Not gated on interactivity the way the request is: a prompt needs someone
        there to answer it, but a notice is worth leaving in the log either way.
        """
        print(  # noqa: T201 — the TTY channel answers on the terminal it prompted
            f"[iris/approval-timed-out] run={approval.run_id} "
            f"id={approval.approval_id} signal={approval.signal} — no answer before "
            f"{approval.timeout_at[:19]}; {approval.lapse_consequence()}.",
            file=sys.stderr,
        )
