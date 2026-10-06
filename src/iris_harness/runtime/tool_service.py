"""Tools for code: a plugin, a workflow or the core calling a tool through governance.

The model reaches a plugin's tools through the governed ReAct loop. Code used to reach
them only by importing the function, which skipped ``PRE_TOOL_USE``, approvals and the
audit row. ``ToolService`` runs a registered tool through the same ``GovernedToolRunner``
the loop uses (docs/architecture/plugin-capabilities.md, rollout step 2), so a code call
cannot be governed differently from a model call.

Callers never name themselves. A plugin gets ``api.tools``, bound to ``plugin:<name>`` by
the harness; the core asks ``services.tools.for_caller("core:<workflow>")``. The caller
lands on the ``PRE_TOOL_USE`` context and the audit row, and the permission contract
(step 3) allows or denies by it — which only means something if it cannot be spoofed.
So ``for_caller`` is the core's: a plugin's ``services.tools`` is a :class:`ToolCatalogue`
(``describe`` only), and ``api.tools`` is its one bound entry.

A code call made during a chat turn is governed as that turn: the harness stamps the turn's
label on it (``kernel/governance/turn_label.py``), and what its result earns lifts the turn's
label for the model calls after it. The caller cannot set either.

A code caller has no chat turn in which to ask the owner, so the effect rules are:
``read`` and ``write`` with ``confirm: never`` run; a destructive tool, a pinned write
and a ``confirm: once`` write are queued for the owner's approval and come back held,
with the approval's id (plugin-capabilities decision 1). Nothing writes silently.

This is also the **executor** for those approvals (``ApprovedCallExecutor``): when the
owner approves, ``respond_to_approval`` hands the row here, and the
pinned call runs through the same governed runner — the caller's permission re-checked
at execution, the approval verified, ``POST_TOOL_USE``, the audit — attributed to the
caller that asked. Ran, failed, denied, rejected or expired, the caller hears once, on
``approval.call_completed``, and the conversation it came from gets a notice.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from iris_harness.agent.tool_runner import GovernedToolRunner, ToolCall, governance_block_message
from iris_harness.foundation.observability.session_log import current_session_id
from iris_harness.kernel.governance.approvals.events import (
    APPROVAL_CALL_COMPLETED,
    ApprovalCallCompletedPayload,
    ApprovedCallStatus,
)
from iris_harness.kernel.governance.approvals.service import ApprovedCallResult
from iris_harness.kernel.governance.display_mask import mask_text
from iris_harness.kernel.governance.external_content import wrap_scanned
from iris_harness.kernel.governance.turn_label import current_turn_label

if TYPE_CHECKING:
    from iris_harness.agent.agentic_core import ToolSpec
    from iris_harness.foundation.eventbus import EventBus
    from iris_harness.kernel.governance import GovernanceKernel
    from iris_harness.kernel.governance.approvals.store import ApprovalRow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolResult:
    """What a code call came to.

    ``ok``: the tool ran and did not raise; ``text`` is its output. Otherwise ``text`` says
    why not — the tool raised, governance denied or held it, or no such tool is
    registered — and ``held`` is true when governance, not the tool, stopped it.
    """

    ok: bool
    text: str
    held: bool = False
    # When held for the owner's approval: the queued approval's id. The call runs once
    # they approve it, and the caller hears on ``approval.call_completed``.
    approval_id: str | None = None
    # Whether the tool declares ``content: external``: ``text`` is then text a third party
    # wrote. A code caller gets it redacted but not wrapped (it may show it to the owner);
    # hand it to a model through :meth:`for_model`. ``source`` and ``tool`` say where it came
    # from (the owning plugin or ``skill:<name>``, and the tool's name).
    external: bool = False
    source: str = ""
    tool: str = ""

    def for_model(self) -> str:
        """``text`` as it may go into a model prompt: inside the untrusted-content envelope
        when the tool is external, unchanged when it is not.

        ``text`` stays the owner-facing form. Instruction-like spans are redacted again, an
        envelope already on the text is replaced rather than nested, and the envelope names
        this result's own ``source`` and ``tool``.
        """
        if not self.external:
            return self.text
        return wrap_scanned(self.text, source=self.source or "plugin", tool=self.tool or "tool")


@dataclass(frozen=True)
class ToolInfo:
    """A registered tool and what its manifest declares — the contract a caller reads."""

    name: str
    description: str
    effect: str
    confirm: str


class ToolService:
    """Registered tools, run through the governed runner on a caller's behalf."""

    def __init__(
        self,
        *,
        tools: Callable[[], list[ToolSpec]],
        kernel: Callable[[], GovernanceKernel | None],
        events: Callable[[], EventBus | None] | None = None,
        deliver_in_chat: Callable[[str, str], None] | None = None,
    ) -> None:
        # All late-bound: plugins register tools after this is built, and the kernel is
        # the runtime's, read at call time.
        self._tools = tools
        self._kernel = kernel
        self._events = events
        self._deliver_in_chat = deliver_in_chat

    def for_caller(self, caller: str) -> BoundTools:
        """The tools, bound to ``caller`` (``plugin:<name>`` or ``core:<workflow>``)."""
        return BoundTools(self, caller)

    def describe(self, name: str | None = None) -> list[ToolInfo]:
        """Every registered tool's declaration, or just ``name``'s (empty if unknown)."""
        return [
            ToolInfo(tool.name, tool.description, tool.effect, tool.confirm)
            for tool in self._tools()
            if name is None or tool.name == name
        ]

    def _call(self, caller: str, name: str, args: dict[str, Any]) -> ToolResult:
        tool = next((t for t in self._tools() if t.name == name), None)
        if tool is None:
            return ToolResult(ok=False, text=f"Error: unknown tool {name!r}.")
        # Deferred: a call that needs the owner's approval is queued, and this service
        # runs it on approval.
        return self._run(caller, tool, args, deferred=True)

    def call_for_client(self, caller: str, tool: ToolSpec, args: dict[str, Any]) -> ToolResult:
        """Run ``tool`` for a client outside IRIS (``mcp:<client>``), through governance.

        The core's, like ``for_caller``: the harness names the caller, never the client.
        ``tool`` is one the core chose to serve -- a registered tool, or a skill tool it
        wrapped with a declared effect -- so it is passed in rather than looked up.

        Not deferred: a client outside IRIS cannot answer an approval, and a call queued
        for one would run later, out of its sight, its result never reaching it. So a
        destructive tool, a pinned write or a ``confirm: once`` write is refused by
        ``PRE_TOOL_USE`` (the approval hook's "this run cannot pause"; the tool policy's
        confirm-once rule) -- audited, and nothing runs. With no kernel bound nothing runs
        at all: an outside client is never served ungoverned.
        """
        if self._kernel() is None:
            return ToolResult(
                ok=False,
                text=f"Refused: {tool.name!r} was not run -- no governance kernel is bound, "
                "and a call from outside IRIS is never run ungoverned.",
                held=True,
            )
        return self._run(caller, tool, args, deferred=False)

    def _run(
        self, caller: str, tool: ToolSpec, args: dict[str, Any], *, deferred: bool
    ) -> ToolResult:
        runner = GovernedToolRunner(
            kernel=self._kernel(),
            agent_type=caller,
            session_id=current_session_id(),
            # A code caller is not a run that can pause and be resumed by an answer.
            resumable=False,
        )
        # The turn's label: the call is governed as the turn it runs in (None outside
        # one), and POST_TOOL_USE lifts the turn's label with what the result earned. Both
        # stamped here, never by the caller.
        call = ToolCall(caller=caller, deferred=deferred, classification=current_turn_label())
        outcome = runner.execute(tool, dict(args), call)
        if outcome.status == "held":
            assert outcome.decision is not None
            approval_id = (
                str(outcome.decision.approval_request_id)
                if outcome.decision.approval_request_id is not None
                else None
            )
            # Surface-neutral: the owner answers wherever they are (web, phone, CLI).
            text = (
                f"Queued for approval {approval_id}; it runs once the owner approves it."
                if approval_id is not None
                else governance_block_message(outcome.decision)
            )
            return ToolResult(ok=False, text=text, held=True, approval_id=approval_id)
        if outcome.status == "refused":
            return ToolResult(ok=False, text=outcome.text, held=True)
        # The runner already ran POST_TOOL_USE: ``text`` is the result as governance left
        # it (redacted, or the block message when it was withheld).
        withheld = outcome.post is not None and outcome.post.withheld
        return ToolResult(
            ok=outcome.ok,
            text=outcome.text,
            held=withheld,
            external=tool.content == "external",
            source=tool.plugin or "core",
            tool=tool.name,
        )

    # -- the executor for a code caller's approved call (decision 1) -------------------
    def execute_approved_call(self, row: ApprovalRow) -> ApprovedCallResult:
        """Run the call an approved, claimed row pins, as the caller that asked for it.

        Satisfies ``governance.approvals.service.ApprovedCallExecutor``. The approval hook
        verifies the row against the queue (approved, this exact call, this caller) and
        claims it, once; the caller policy re-checks the caller's permission as it stands
        now, not as it stood when the owner was asked.
        """
        status, text = self._run_approved(row)
        return self._settle(row, status, text)

    def settle_unrun_call(self, row: ApprovalRow, status: ApprovedCallStatus) -> ApprovedCallResult:
        """Tell the caller its call will never run: rejected, or its approval lapsed."""
        if status == "rejected":
            text = "The owner rejected it; nothing was run."
        else:
            text = "The approval expired unanswered; nothing was run."
        return self._settle(row, status, text)

    def _run_approved(self, row: ApprovalRow) -> tuple[ApprovedCallStatus, str]:
        items = row.items or ()
        if row.caller is None or len(items) != 1:
            return "denied", "Not a code caller's single pinned call; nothing was run."
        item = items[0]
        tool = next((t for t in self._tools() if t.name == item.tool), None)
        if tool is None:
            return "denied", f"Tool {item.tool!r} is no longer registered; nothing was run."
        runner = GovernedToolRunner(
            kernel=self._kernel(),
            agent_type=row.caller,
            origin_channel=row.channel,
            session_id=row.session_id,
            resumable=False,
        )
        # The approved call belongs to the run the original call opened (the approval row
        # carries it), so its PRE and POST rows, and any side-effect row, sit beside the
        # rows that queued it. No turn label is stamped: this runs on the owner's answer,
        # outside the turn that queued it (POST_TOOL_USE still labels the call itself).
        call = ToolCall(
            run_id=row.run_id,
            caller=row.caller,
            approved_by=row.approval_id,
            deferred=True,
            # A new call with its own id; the held attempt's is the row's (#134).
            held_call_id=row.call_id,
        )
        outcome = runner.execute(tool, item.args, call)
        if outcome.status == "held":
            assert outcome.decision is not None
            return "denied", governance_block_message(outcome.decision)
        if outcome.status == "refused":
            return "denied", outcome.text
        # POST_TOOL_USE ran inside ``execute``; ``text`` is what governance let through.
        return ("ran" if outcome.ok else "failed"), outcome.text

    def _settle(
        self, row: ApprovalRow, status: ApprovedCallStatus, text: str
    ) -> ApprovedCallResult:
        """One result, one event, one notice — whatever became of the call."""
        tool = row.items[0].tool if row.items else ""
        summary = mask_text(text)[:500]
        result = ApprovedCallResult(status=status, summary=summary)
        bus = self._events() if self._events is not None else None
        if bus is not None:
            bus.emit_sync(
                APPROVAL_CALL_COMPLETED,
                ApprovalCallCompletedPayload(
                    approval_id=row.approval_id,
                    caller=row.caller or "",
                    tool=tool,
                    status=status,
                    summary=summary,
                ),
            )
        if row.session_id and self._deliver_in_chat is not None:
            try:
                self._deliver_in_chat(row.session_id, result.notice(tool))
            except Exception:  # the decision and the call are already recorded
                logger.warning(
                    "could not post the notice for approval %s", row.approval_id, exc_info=True
                )
        return result


class ToolCatalogue:
    """The registered tools as a plugin's ``services.tools`` shows them: what exists, no calls.

    A plugin calls tools only through ``api.tools``, bound to it by the harness. The tool
    service itself can bind any caller (``for_caller``), so it is never handed to a plugin;
    this view keeps ``describe`` and nothing else.
    """

    def __init__(self, service: ToolService) -> None:
        self._service = service

    def describe(self, name: str | None = None) -> list[ToolInfo]:
        """The registered tools and their declarations (see ``ToolService.describe``)."""
        return self._service.describe(name)


class BoundTools:
    """The tools as one caller sees them. The caller is fixed; there is no way to set it."""

    def __init__(self, service: ToolService, caller: str) -> None:
        self._service = service
        self._caller = caller

    @property
    def caller(self) -> str:
        return self._caller

    def call(self, name: str, args: dict[str, Any] | None = None) -> ToolResult:
        """Run tool ``name`` with ``args`` through governance, as this caller."""
        return self._service._call(self._caller, name, dict(args or {}))

    def describe(self, name: str | None = None) -> list[ToolInfo]:
        """The registered tools and their declarations (see ``ToolService.describe``)."""
        return self._service.describe(name)


__all__ = ["BoundTools", "ToolCatalogue", "ToolInfo", "ToolResult", "ToolService"]
