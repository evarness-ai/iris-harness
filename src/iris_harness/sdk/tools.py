"""Calling a tool from code, through governance.

``api.tools`` (a ``BoundTools``, bound to your plugin by the harness) runs any registered
tool through the same governed path the model's calls take: ``PRE_TOOL_USE``, the approval
rules, ``POST_TOOL_USE`` and an audit row naming your plugin as the caller. Never import
another plugin's tool function: that call would skip all of it.

``call(name, args)`` returns a ``ToolResult``: ``ok`` and the output, or why not (``held``
when governance stopped it). ``describe()`` lists the registered tools and what each
declares (``ToolInfo``: effect and confirm), the contract you call against.

A code caller cannot answer an approval: a destructive tool, a pinned write or a
``confirm: once`` write is queued for the owner and comes back held with its
``approval_id``, never silently written. It runs once, when the owner approves it, as
your plugin; ``api.on_approved_call(handler)`` hears the outcome
(``ApprovalCallCompletedPayload`` on ``APPROVAL_CALL_COMPLETED``).
"""

from __future__ import annotations

from iris_harness.kernel.governance.approvals.events import (
    APPROVAL_CALL_COMPLETED,
    ApprovalCallCompletedPayload,
    ApprovedCallStatus,
)
from iris_harness.runtime.tool_service import BoundTools, ToolInfo, ToolResult

__all__ = [
    "APPROVAL_CALL_COMPLETED",
    "ApprovalCallCompletedPayload",
    "ApprovedCallStatus",
    "BoundTools",
    "ToolInfo",
    "ToolResult",
]
