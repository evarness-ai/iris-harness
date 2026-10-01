"""ApprovalGate — async polling helper for run-resume flows (design §11.2)."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from iris_harness.kernel.governance.approvals.store import (
    ApprovalNotFoundError,
    ApprovalRow,
    ApprovalStore,
)

logger = logging.getLogger(__name__)


class ApprovalRejectedError(Exception):
    """await_approval() raised because the approval was rejected."""

    def __init__(self, approval_id: str, run_id: str) -> None:
        self.approval_id = approval_id
        self.run_id = run_id
        super().__init__(f"approval {approval_id!r} was rejected (run={run_id})")


class ApprovalTimedOutError(Exception):
    """await_approval() raised because the approval timed out."""

    def __init__(self, approval_id: str, run_id: str) -> None:
        self.approval_id = approval_id
        self.run_id = run_id
        super().__init__(f"approval {approval_id!r} timed out (run={run_id})")


class ApprovalGate:
    """Poll ApprovalStore until the approval is answered or times out.

    ``await_approval`` is the only public method; it is called by the
    ``iris run resume`` / ``iris-code resume`` flow after the run was
    paused on a ``require_approval`` decision.
    """

    def __init__(self, store: ApprovalStore) -> None:
        self._store = store

    async def await_approval(
        self,
        approval_id: str,
        *,
        poll_interval: float = 1.0,
        timeout: float | None = None,
    ) -> ApprovalRow:
        """Poll until the approval is answered.

        Raises:
            ApprovalNotFoundError: if no row with approval_id exists.
            ApprovalRejectedError: if the user rejected the request.
            ApprovalTimedOutError: if timeout_at has passed or the caller's
                timeout elapses first.
        """
        row = self._store.get(approval_id)
        if row is None:
            raise ApprovalNotFoundError(approval_id)

        # Parse row's timeout; normalise to UTC-aware
        row_deadline = _parse_iso(row.timeout_at)
        if timeout is not None:
            caller_deadline = datetime.now(UTC) + timedelta(seconds=timeout)
            effective_deadline = min(row_deadline, caller_deadline)
        else:
            effective_deadline = row_deadline

        while True:
            row = self._store.get(approval_id)
            if row is None:
                raise ApprovalNotFoundError(approval_id)

            if row.status == "approved":
                return row
            if row.status == "rejected":
                raise ApprovalRejectedError(approval_id, row.run_id)
            if row.status == "timed_out":
                raise ApprovalTimedOutError(approval_id, row.run_id)

            if datetime.now(UTC) >= effective_deadline:
                raise ApprovalTimedOutError(approval_id, row.run_id)

            logger.debug(
                "approval %s still pending; next poll in %.1fs", approval_id, poll_interval
            )
            await asyncio.sleep(poll_interval)


def _parse_iso(s: str) -> datetime:
    """Parse ISO-8601; add UTC if tz-naive."""
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
