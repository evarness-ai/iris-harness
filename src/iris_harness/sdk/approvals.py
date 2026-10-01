"""Asking the owner to approve something a plugin's own flow will do (R17).

The model's calls reach the approval queue through the kernel (a destructive tool, a
pinned write); ``api.tools`` queues a code caller's call the same way. This is for the
other case: a flow the owner drives step by step -- email setup's "may IRIS change your
mailbox?" -- that must leave the same governed record before it acts. Enqueue a row
with a card (``ApprovalQueue.enqueue``: the Action Center, ``iris approvals`` and the
Governance screen show it like any other), then answer it only on the owner's explicit
word through :func:`respond_to_approval`, the one function every surface answers with,
so the answer leaves its audit row wherever it was given.

Build the queue as ``ApprovalQueue(store=ApprovalStore(), audit_log=AuditLog(...))``
(``iris_harness.sdk.audit``) so every state change is on the ledger. Never answer an
approval from code the owner did not direct: the queue exists to put them between a
decision and the action.
"""

from __future__ import annotations

from iris_harness.kernel.governance.approvals import (
    ApprovalAlreadyAnsweredError,
    ApprovalCard,
    ApprovalNotFoundError,
    ApprovalQueue,
    ApprovalRow,
    ApprovalStore,
)
from iris_harness.kernel.governance.approvals.service import respond_to_approval

__all__ = [
    "ApprovalAlreadyAnsweredError",
    "ApprovalCard",
    "ApprovalNotFoundError",
    "ApprovalQueue",
    "ApprovalRow",
    "ApprovalStore",
    "respond_to_approval",
]
