"""DestructiveApprovalHook — every destructive tool call waits for the owner (ADR-0118).

A tool declared ``effect: destructive`` removes or overwrites the owner's data. At
``PreToolUse`` this hook turns such a call into an approval row that pins the exact call
(tool + canonical arguments), tells the owner through the channel router (which also
fires ``approval.requested``, so the phone gets a banner), and returns
``require_approval`` carrying the row's id. That id is what makes the loop **halt** and
checkpoint rather than read the decision as advice — the confirm-once decisions carry
none.

When an approved run resumes, the loop executes the pinned call with
``metadata["approved_by"]`` set. This hook checks that claim against the queue rather
than trusting the loop: the row must exist, be ``approved``, and pin exactly this tool
with exactly these arguments. Anything else is denied — a changed argument, a second
item the owner never saw, or a forged id.

Priority 50: last at PreToolUse, so every other policy's deny (blocked tools, persona
surface, sandboxes, egress) wins before a row is ever created. With no queue there is
nothing a human could answer, so the call is denied — fail-closed.

A ``write`` declared ``approval: pinned`` (gate ``"approval"``, ADR-0118 amendment:
``send_email``) takes this same path. It is not data loss, so its card and reasons say
it acts on the owner's behalf rather than that it deletes anything.

A **code caller** (``services.tools`` — plugin-capabilities decision 1) has no run to
pause. The harness stamps ``deferred_executor`` on its calls, and only on its calls: the
harness itself runs the pinned call once the owner approves. Such a call is queued like
any other, with the caller on the row, and a ``confirm: once`` write from code is queued
the same way (the runner stamps ``per_call_approval``). When the executor runs it, the
claim is checked here too: the row must name this caller and have been claimed for
execution: the first execution claims the row (one conditional write), and any later
one is denied, so an approved call runs at most once.
"""

from __future__ import annotations

import logging
import uuid as _uuid
from typing import TYPE_CHECKING, Any

from iris_harness.foundation.observability.session_log import current_turn_id
from iris_harness.kernel.governance.approvals.store import ApprovalCard, ApprovalItem
from iris_harness.kernel.governance.hooks.tool_payload import CALL_ID, HELD_CALL_ID
from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint

if TYPE_CHECKING:
    from iris_harness.kernel.governance.approvals import ApprovalQueue
    from iris_harness.kernel.governance.approvals.router import ChannelRouter

logger = logging.getLogger(__name__)

SIGNAL = "destructive_tool"

#: How long a destructive approval stays answerable. The owner often answers from the
#: phone, away from the conversation, so it gets an hour (owner's decision 2026-09-21)
#: rather than the queue's 10-minute default, which evaluator halts keep.
DESTRUCTIVE_TIMEOUT_MINUTES = 60


def pinned_by_declaration(effect: object, confirm: object) -> bool:
    """Whether a tool's own declaration makes every call of it wait for a pinned approval:
    ``effect: destructive``, or a write whose gate is ``approval`` (``approval: pinned``).

    The high-risk class: these calls also get a durable side-effect ledger row before they
    run (``pre_tool_use_ledger``). A code caller's ``confirm: once`` write is approved per
    call too (``approved_per_call``), but by who calls it, not by what it is: it stays a
    plain write everywhere else.
    """
    return effect == "destructive" or confirm == "approval"


def approved_per_call(metadata: dict[str, Any]) -> bool:
    """Whether the call in this PreToolUse context waits for a pinned approval: a
    destructive tool, a write whose declared gate is ``approval``, or a call the runner
    says is approved per call (a code caller's ``confirm: once`` write)."""
    return (
        pinned_by_declaration(metadata.get("tool_effect"), metadata.get("tool_confirm"))
        or metadata.get("per_call_approval") is True
    )


def card_title(tool: str, raw: Any) -> str:
    """The one-line headline for a call: the plugin's ``describe`` title, else a fallback
    worded by the tool's effect (a write acts for the owner; only a destructive tool
    deletes or overwrites). The queue card and the chat halt message both use this, so
    the two cannot word the same approval differently."""
    d = raw if isinstance(raw, dict) else {}
    title = str(d.get("title") or "").strip()
    if title:
        return title
    if d.get("effect") == "write":
        return f"{tool} wants to act on your behalf"
    return f"{tool} wants to delete or overwrite your data"


def build_card(item: ApprovalItem, raw: Any) -> ApprovalCard:
    """The owner-facing card for one call, from what the loop passed (ADR-0118 step 4).

    ``raw`` is the loop's ``approval_card`` metadata: the plugin's ``describe`` title and
    lines (absent when the tool has none, or it failed), the declared undo tool and
    window, and the owner's request. Missing pieces fall back to the raw call, so a card
    always says what will run.
    """
    d = raw if isinstance(raw, dict) else {}
    effect = "write" if d.get("effect") == "write" else "destructive"
    title = card_title(item.tool, raw)
    lines = tuple(str(line) for line in (d.get("lines") or ()) if str(line).strip())
    lines = lines or (item.render(),)
    # Why a tool that does not always ask is asking now (the run read outside text, #149).
    reason = str(d.get("reason") or "").strip()
    if reason:
        lines = (*lines, reason)
    days = d.get("undo_window_days")
    return ApprovalCard(
        title=title,
        lines=lines,
        undo_tool=str(d["undo_tool"]) if d.get("undo_tool") else None,
        undo_window_days=int(days) if isinstance(days, int) and days > 0 else None,
        asked=str(d["asked"]).strip() if d.get("asked") else None,
        effect=effect,
    )


def summarise(card: ApprovalCard, items: tuple[ApprovalItem, ...]) -> str:
    """The card as plain text, for every surface that reads ``context_summary``: the
    CLI, Telegram, the lapse notice. Plain words first; the exact call last."""
    parts = [card.title, *(f"- {line}" for line in card.lines), card.undo_sentence()]
    if card.asked:
        parts.append(f'You asked: "{card.asked}"')
    parts.append("Exact call: " + "; ".join(item.render() for item in items))
    return "\n".join(parts)


class DestructiveApprovalHook:
    """Queue an itemised approval for a destructive call, or verify an approved one."""

    name: str = "destructive_approval"
    hook_point: HookPoint = HookPoint.PRE_TOOL_USE
    priority: int = 50

    def __init__(
        self,
        *,
        approval_queue: ApprovalQueue | None,
        channel_router: ChannelRouter | None = None,
        timeout_minutes: int = DESTRUCTIVE_TIMEOUT_MINUTES,
    ) -> None:
        self._queue = approval_queue
        self._router = channel_router
        self._timeout_minutes = timeout_minutes

    async def __call__(self, ctx: HookContext) -> HookDecision:
        if not approved_per_call(ctx.metadata):
            return HookDecision(
                outcome="allow", reason="destructive_approval: not approved per call"
            )

        tool_name = str(ctx.payload.get("tool_name") or "")
        raw_args = ctx.payload.get("args")
        args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
        item = ApprovalItem.of(tool_name, args)

        if self._queue is None:
            return HookDecision(
                outcome="deny",
                reason=(
                    f"destructive_approval: {tool_name!r} needs the owner's approval and no "
                    "approval queue is configured; nothing was changed"
                ),
                severity="warn",
                audit_metadata={"tool_name": tool_name},
            )

        approved_by = ctx.metadata.get("approved_by")
        if approved_by:
            return self._verify(str(approved_by), item, caller=_metadata_str(ctx, "caller"))

        deferred = ctx.metadata.get("deferred_executor") is True
        if ctx.metadata.get("resumable") is not True and not deferred:
            # An approval this run could never resume from is a promise nobody keeps:
            # the owner would approve and nothing would happen. Refuse instead.
            return HookDecision(
                outcome="deny",
                reason=(
                    f"destructive_approval: {tool_name!r} needs the owner's approval, and "
                    "this run cannot pause for one; nothing was changed"
                ),
                severity="warn",
                audit_metadata={"tool_name": tool_name},
            )
        return self._request(ctx, item)

    def _verify(self, approval_id: str, item: ApprovalItem, *, caller: str | None) -> HookDecision:
        assert self._queue is not None
        row = self._queue.get(approval_id)
        audit = {"tool_name": item.tool, "approval_id": approval_id, "caller": caller}
        if row is not None and row.call_id:
            # From the queue, not the caller: which attempt was held.
            audit[HELD_CALL_ID] = row.call_id
        if row is None or row.status != "approved":
            status = "missing" if row is None else row.status
            return HookDecision(
                outcome="deny",
                reason=f"destructive_approval: approval {approval_id} is {status}",
                severity="warn",
                audit_metadata=audit,
            )
        if item not in (row.items or ()):
            return HookDecision(
                outcome="deny",
                reason=(
                    f"destructive_approval: {item.tool!r} with these arguments is not what "
                    f"approval {approval_id} pinned"
                ),
                severity="warn",
                audit_metadata=audit,
            )
        if row.caller is not None:
            # A code caller's approval runs once, as the caller that asked for it: the
            # executor claims it before running, and no other caller may use it.
            if caller != row.caller:
                return HookDecision(
                    outcome="deny",
                    reason=(
                        f"destructive_approval: approval {approval_id} was given to "
                        f"{row.caller}, not {caller}"
                    ),
                    severity="warn",
                    audit_metadata=audit,
                )
            # The claim is the single use: one conditional write, which only the first
            # call to reach here wins. A retried, replayed or concurrent execution of
            # the same approval is denied, so the call runs at most once.
            if self._queue.claim_execution(approval_id) is None:
                return HookDecision(
                    outcome="deny",
                    reason=f"destructive_approval: approval {approval_id} has already run",
                    severity="warn",
                    audit_metadata=audit,
                )
        return HookDecision(
            outcome="allow",
            reason=f"destructive_approval: pinned by approved {approval_id}",
            audit_metadata=audit,
        )

    def _request(self, ctx: HookContext, item: ApprovalItem) -> HookDecision:
        assert self._queue is not None
        items = (item,)
        card = build_card(item, ctx.metadata.get("approval_card"))
        approval_id = self._queue.enqueue(
            ctx.run_id,
            None,  # checkpoint_id — the loop writes the checkpoint after this and links it
            # The signal is what the phone banner shows: the card's title, in plain words.
            card.title,
            summarise(card, items),
            channel=_metadata_str(ctx, "origin_channel") or "cli",
            session_id=_metadata_str(ctx, "session_id"),
            items=items,
            card=card,
            timeout_minutes=self._timeout_minutes,
            # Only a deferred call records its caller: that is what makes the harness,
            # not a resumed run, the one that executes it on approval.
            caller=(
                _metadata_str(ctx, "caller")
                if ctx.metadata.get("deferred_executor") is True
                else None
            ),
            # The id of THIS (held) attempt: the approved re-execution is a new call, and
            # records this one as its ``held_call_id`` (#134).
            call_id=_metadata_str(ctx, CALL_ID),
            # Where it was raised (#134 stage 3): the loop step and the chat turn.
            step_id=ctx.step_id,
            turn_id=current_turn_id(),
        )
        row = self._queue.get(approval_id)
        if row is not None and self._router is not None:
            try:
                self._router.notify(row)
            except Exception:  # the row is the source of truth; a nudge is not
                logger.warning("could not notify approval %s", approval_id, exc_info=True)
        return HookDecision(
            outcome="require_approval",
            reason=(
                f"destructive_approval: {item.tool!r} "
                + (
                    "acts on the owner's behalf"
                    if card.effect == "write"
                    else "deletes or overwrites the owner's data"
                )
                + " and waits for their approval"
            ),
            severity="warn",
            approval_request_id=_uuid.UUID(approval_id),
            audit_metadata={"tool_name": item.tool, "approval_id": approval_id, "signal": SIGNAL},
        )


def _metadata_str(ctx: HookContext, key: str) -> str | None:
    raw = ctx.metadata.get(key)
    value = str(raw).strip() if raw else ""
    return value or None
