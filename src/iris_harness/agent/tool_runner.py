"""The one governed path a tool call takes, whoever makes it.

Every tool call — the model's, from the ReAct loop, and (next) code's, through
``services.tools`` — passes the same governance: the per-call approval rule for a
destructive tool or a pinned write, the plugin's argument check before a card reaches
the owner, the kernel's ``PRE_TOOL_USE`` hooks, fail-closed when a per-call tool is
allowed with no approval, the timeline events, the call itself, and ``POST_TOOL_USE``
over the result, whose verdict is enforced (``deny`` withholds the result, ``transform``
hands on the rewritten one). Every payload is built by ``kernel/governance/hooks/
tool_payload.py``, the one definition of the keys the tool hooks read. It used to live inside ``AgenticCore._execute_tool``, where only the
loop could reach it; a second caller would have had to re-implement it and could drift
(docs/architecture/plugin-capabilities.md, rollout step 1).

What stays in the loop is what only a loop has: resolving the name the model wrote
(the reserve pool, near-miss aliases) and turning an approval into a halt of the run.

A capability method call (plugin-capabilities §2, §4) takes the same kernel path as a
pseudo-tool named ``capability:<name>.<method>``: ``execute_call`` / ``aexecute_call`` /
``aexecute_stream`` fire ``PRE_TOOL_USE`` before the provider runs and ``POST_TOOL_USE``
over what it returned. Like a plain tool's ``post``, the capability path *uses* the
final context: the redacted field map is written back into a copy of the typed result the
consumer receives, and a ``POST_TOOL_USE`` deny withholds the result. Streams are governed
at the call, then per yielded item, then once more when exhausted.

Every governed call is audited by keyed digest (``kernel/governance/audit/digest.py``):
``PRE_TOOL_USE`` carries ``args_digest`` of the arguments as the caller wrote them,
``POST_TOOL_USE`` ``result_digest`` of the result it judges, both with ``digest_alg``. With a
kernel bound and no audit key, no call runs: ``execute`` refuses before ``PRE_TOOL_USE`` and
the tool is never invoked; a capability call raises ``CapabilityDenied`` the same way.

A ``POST_TOOL_USE`` that raises the label lifts the turn's label too
(``kernel/governance/turn_label.py``): never lowers it, and a no-op outside a turn.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from iris_harness.foundation.capabilities import CapabilityDenied, capability_tool_name
from iris_harness.foundation.capability_fields import (
    ResultMismatch,
    extract_fields,
    rebuild,
)
from iris_harness.kernel.governance import (
    DataClassification,
    GovernanceKernel,
    HookContext,
    HookDecision,
    HookPoint,
)
from iris_harness.kernel.governance.audit.digest import (
    AuditDigester,
    AuditKeyUnavailable,
    audit_digester,
)
from iris_harness.kernel.governance.hooks.tool_payload import (
    TOOL_SENDS_TO,
    ToolContent,
    args_of,
    post_tool_payload,
    pre_tool_payload,
    result_of,
    tool_post_metadata,
)
from iris_harness.kernel.governance.plugins.output_classifier import more_restrictive
from iris_harness.kernel.governance.turn_label import lift_turn_label

if TYPE_CHECKING:
    from iris_harness.agent.agentic_core import ToolSpec

logger = logging.getLogger(__name__)

# The refusal for a destructive tool when no approval queue can take its card.
_DESTRUCTIVE_REFUSED = (
    "Refused: {name!r} deletes or overwrites the user's data, and each such call needs "
    "the owner's itemised approval, which this harness cannot take here (no approval "
    "queue). Nothing was changed. Tell the user plainly that you did not do it."
)
# The same refusal for a write approved per call (``approval: pinned``), which is not
# data loss and must not be described as one.
_PINNED_WRITE_REFUSED = (
    "Refused: each {name!r} call needs the owner's approval, which this harness cannot "
    "take here (no approval queue). It was not run: nothing was sent or changed. Tell the "
    "user plainly that you did not do it."
)


def approved_per_call(tool: ToolSpec) -> bool:
    """Whether every call of ``tool`` waits for the owner's approval on a pinned card
    (ADR-0118): a destructive tool, or a write declared ``approval: pinned`` (the
    gate ``"approval"``). One predicate, so the two share one path end to end."""
    return tool.effect == "destructive" or tool.confirm == "approval"


def approved_per_call_for(tool: ToolSpec, call: ToolCall) -> bool:
    """``approved_per_call`` for this caller. A deferred (code) caller has no chat turn in
    which to ask, so its ``confirm: once`` write waits for a pinned approval too
    (plugin-capabilities decision 1); the loop's confirm-once rule is unchanged."""
    return approved_per_call(tool) or (call.deferred and tool.confirm == "once")


def refused(tool: ToolSpec) -> str:
    template = _DESTRUCTIVE_REFUSED if tool.effect == "destructive" else _PINNED_WRITE_REFUSED
    return template.format(name=tool.name)


def governance_block_message(decision: HookDecision) -> str:
    action = "blocked" if decision.outcome == "deny" else "needs approval"
    return f"Request {action} by governance: {decision.reason}"


class ToolUnavailable(RuntimeError):
    """A tool's own code failed and a fault boundary caught it (a plugin's tool raised).

    Its message is what the caller is told instead of a result. Raised, not returned: a
    tool that failed must not read as one that ran, so the call's ``ok`` is false and
    an approved call settles as ``failed``, not ``ran``.
    """


@dataclass(frozen=True)
class ToolCall:
    """Who is calling, and where the call belongs, for one governed tool call."""

    run_id: str | None = None
    classification: DataClassification | None = None
    step_id: int | None = None
    # The run's own evidence that it asked the user after proposing this write.
    asked_user: bool = False
    # A resumed run executing a pinned call claims the approval it was given.
    approved_by: str | None = None
    # The owner's request, shown on an approval card (None when there is none).
    query: str | None = None
    # Who is calling: ``model:<agent>`` for the loop, ``plugin:<name>`` for a plugin's
    # bound ``api.tools``, ``core:<workflow>`` for the core. Stamped by the harness, never
    # chosen by the caller; the permission contract reads it (plugin-capabilities §4).
    caller: str | None = None
    # A code call the harness can run later, once the owner approves (decision 1). Set
    # only by ``ToolService``, never by a caller: it is what lets the approval hook queue
    # a call no run will resume.
    deferred: bool = False


@dataclass(frozen=True)
class PostOutcome:
    """What ``POST_TOOL_USE`` made of a tool's result.

    ``text`` is what the caller hands on -- to the model, or to a code caller -- and never
    the raw result when a hook rewrote it (``transform``: the rewritten text) or refused it
    (``deny`` / ``require_approval``: the governance block message; ``withheld``).
    ``classification`` is the run's label after the result, never lower than before it.
    ``decision`` is the kernel's final decision (None when no kernel is bound).
    """

    classification: DataClassification | None
    text: str
    decision: HookDecision | None = None
    withheld: bool = False


@dataclass(frozen=True)
class ToolOutcome:
    """What a governed call came to.

    ``ran``: the tool ran and ``POST_TOOL_USE`` judged its result; ``text`` is what the
    caller hands on (the output, a hook's rewrite of it, ``Tool error: ...``, or the block
    message when governance withheld it) and ``ok`` says whether it is a usable result.
    ``held``: governance denied it or wants an approval; ``decision`` says which (an
    approval with an id exists in the queue). ``refused``: the runner turned it back
    itself (no approval queue, invalid arguments, allowed without approval); ``text`` is
    what to tell the caller.

    ``classification`` is the run's label after the call: raised by ``POST_TOOL_USE`` when
    the result is more sensitive than the question was, otherwise the caller's own.
    """

    status: Literal["ran", "held", "refused"]
    text: str = ""
    ok: bool = False
    decision: HookDecision | None = None
    approval_card: dict[str, Any] | None = None
    classification: DataClassification | None = None
    post: PostOutcome | None = None


@dataclass(frozen=True)
class CapabilityCall:
    """One capability method call, as governance sees it.

    ``caller`` is bound by the harness (``plugin:<consumer>`` / ``core:<module>``);
    ``effect`` and ``confirm`` come from the method's ``MethodSpec``, as a tool's come from
    its manifest; ``fields`` are the declared text paths of the result; ``shape`` how the
    result comes back (``MethodShape``).
    """

    caller: str
    provider: str
    capability: str
    method: str
    effect: str
    confirm: str
    fields: tuple[str, ...]
    shape: str
    # The declared type the result (a stream's item) is walked as -- never its own shape.
    value_type: Any
    # Whether the result is text a third party wrote (``MethodSpec.content``).
    content: ToolContent = "internal"
    # The turn's label (``kernel/governance/turn_label.py``), stamped by the harness when
    # it builds the call; None outside a turn. Never the caller's to set.
    classification: DataClassification | None = None

    @property
    def tool_name(self) -> str:
        return capability_tool_name(self.capability, self.method)


_NO_KERNEL = "no governance kernel is bound, so capability calls fail closed"
_IN_LOOP = (
    "a sync capability method cannot be governed inside a running event loop (the kernel "
    "is async); call it from sync code, or declare the method async"
)


class GovernedToolRunner:
    """The governance around one tool call. Holds no state between calls."""

    def __init__(
        self,
        *,
        kernel: GovernanceKernel | None,
        agent_type: str,
        origin_channel: str | None = None,
        session_id: str | None = None,
        resumable: bool = False,
    ) -> None:
        self._kernel = kernel
        self._agent_type = agent_type
        self._origin_channel = origin_channel
        self._session_id = session_id
        self._resumable = resumable

    def execute(
        self,
        tool: ToolSpec,
        args: dict[str, Any],
        call: ToolCall,
        *,
        effects: list[str] | None = None,
    ) -> ToolOutcome:
        """Govern and run ``tool`` with ``args``: ``PRE_TOOL_USE``, the call, and
        ``POST_TOOL_USE`` over its result, whose verdict the outcome already carries (a
        withheld or rewritten ``text``, the raised ``classification``). The caller cannot
        skip the ``POST`` step or forget to honour it: every caller gets the same outcome.

        One run id covers both steps: the caller's, or one minted here for a caller that
        has no run (a code caller), so the ``PRE`` and ``POST`` rows of one call correlate.
        """
        name = tool.name
        if self._kernel is not None:
            # No audit key, no call: the refusal comes before PRE_TOOL_USE, the plugin's
            # argument check and the tool, so nothing runs that the ledger cannot record.
            try:
                audit_digester()
            except AuditKeyUnavailable as exc:
                logger.warning("tool %r refused: %s", name, exc)
                return ToolOutcome(
                    status="refused", text=str(exc), classification=call.classification
                )
        if call.run_id is None:
            call = replace(call, run_id=str(uuid.uuid4()))
        # A destructive tool, or a write declared ``approval: pinned``: every call waits
        # for the owner's approval on a pinned card, through one path (ADR-0118).
        per_call = approved_per_call_for(tool, call)
        if per_call and self._kernel is None:
            # No kernel, so no approval queue and nobody to ask (ADR-0118).
            logger.info("react: refused %r (approved per call; no governance kernel)", name)
            return ToolOutcome(
                status="refused", text=refused(tool), classification=call.classification
            )

        # Whether this call could be queued for the owner at all: a run that can pause and
        # resume, or a caller the harness runs once approved. When it cannot (a lane with
        # no checkpoint, a client outside IRIS) the approval hook refuses it at
        # PRE_TOOL_USE -- audited -- so there is no card to check arguments for, and a
        # refusal here would leave no row in the ledger.
        can_queue = self._resumable or call.deferred
        if per_call and not call.approved_by and can_queue:
            problem = invalid_destructive_args(tool, args)
            if problem is not None:
                logger.info("react: %r arguments refused before approval: %s", name, problem)
                return ToolOutcome(
                    status="refused",
                    text=f"Error: {problem} Nothing was sent for approval and nothing changed.",
                    classification=call.classification,
                )

        # A destructive call about to ask for approval carries its card (ADR-0118 step
        # 4); a pinned call being executed on an approved resume does not need one.
        approval_card = (
            approval_card_for(tool, args, query=call.query)
            if per_call and not call.approved_by and can_queue
            else None
        )
        decision, tool_args = self.pre(
            tool, args, call, approval_card=approval_card, per_call=per_call
        )
        if decision is not None and decision.outcome in ("deny", "require_approval"):
            return ToolOutcome(
                status="held",
                decision=decision,
                approval_card=approval_card,
                classification=call.classification,
            )
        if per_call and not call.approved_by:
            # Governance allowed a call nobody approved: the approval hook is not
            # registered. Fail closed rather than run it (ADR-0118).
            logger.warning("react: %r (approved per call) allowed without an approval", name)
            return ToolOutcome(
                status="refused", text=refused(tool), classification=call.classification
            )

        # Emit session-log tool events so tool calls show up in the trace graph and the
        # Reasoning tab. Best-effort: logging is a no-op outside a session_scope and must
        # never break execution.
        from iris_harness.foundation.observability.session_log import log_timeline_event

        tool_call_id = uuid.uuid4().hex[:12]
        log_timeline_event(
            "tool.invoke.start",
            phase="tool.invoke.start",
            # The arguments as the caller wrote them, never ``tool_args``: a PRE_TOOL_USE
            # transform may have resolved a ``vault://`` handle into a secret there
            # (credential broker), and a secret must never reach a log.
            payload={"tool": name, "tool_call_id": tool_call_id, "arguments": args},
        )
        # Recorded before the call, not after it succeeds: a write that raised half-way
        # may still have changed something.
        if effects is not None and tool.effect != "read":
            effects.append(tool.effect)
        ok = True
        try:
            result = str(tool.call(tool_args))
        except ToolUnavailable as exc:
            result = str(exc)
            ok = False
        except Exception as exc:  # noqa: BLE001
            result = f"Tool error: {exc}"
            ok = False
        post = self.post(tool, result, call, tool_call_id=tool_call_id)
        log_timeline_event(
            "tool.invoke.end",
            phase="tool.invoke.end",
            payload={
                "tool": name,
                "tool_call_id": tool_call_id,
                "ok": ok,
                "withheld": post.withheld,
                # What the caller was handed: a withheld result is not logged either.
                "result_preview": post.text[:500],
            },
        )
        return ToolOutcome(
            status="ran",
            text=post.text,
            ok=ok and not post.withheld,
            classification=post.classification,
            post=post,
        )

    def pre(
        self,
        tool: ToolSpec,
        args: dict[str, Any],
        call: ToolCall,
        *,
        approval_card: dict[str, Any] | None = None,
        per_call: bool | None = None,
    ) -> tuple[HookDecision | None, dict[str, Any]]:
        """Run ``PRE_TOOL_USE`` before invoking a registered tool.

        ``call.asked_user`` is the run's own evidence that it asked the user after
        proposing this write. The tool policy's confirm-once rule reads it: a write tool
        that asks first may proceed, one that has not is turned back with the instruction
        to ask (multi-step loop plan, decision 8).
        """
        if self._kernel is None:
            return None, args
        name = tool.name
        if per_call is None:
            per_call = approved_per_call_for(tool, call)
        tool_ctx = HookContext(
            hook_point=HookPoint.PRE_TOOL_USE,
            run_id=call.run_id or str(uuid.uuid4()),
            agent_type=self._agent_type,
            step_id=call.step_id,
            route=f"tool/{name}",
            classification=call.classification,
            # ``args`` as the caller wrote them: this context is built before any hook's
            # transform, so a secret the credential broker resolves is never digested.
            payload=pre_tool_payload(name, args, **audit_digester().args_fields(args)),
            metadata={
                "caller": call.caller or f"model:{self._agent_type}",
                "asked_user": call.asked_user,
                # ADR-0110: the tool's own declaration (from its manifest, or the core's
                # ToolSpec). The tool policy's confirm-once rule reads it.
                "tool_effect": tool.effect,
                "tool_confirm": tool.confirm,
                # ADR-0125: where the arguments go, when the tool declares a destination
                # the owner-PII guards treat on its own (a web search provider).
                TOOL_SENDS_TO: tool.sends_to,
                # ADR-0118: where a destructive call's approval is delivered, which
                # conversation it belongs to, and — when a resumed run executes a pinned
                # call — the approval it claims. The hook checks that claim itself.
                "origin_channel": self._origin_channel,
                "session_id": self._session_id,
                "approved_by": call.approved_by,
                # ADR-0118 step 4: how the approval reads to the owner, built from the
                # tool's describe() and its declared undo. The hook freezes it on the row.
                "approval_card": approval_card,
                # Whether this run can pause and be resumed by an answer. A core built
                # without a checkpoint store, the link and the reader (a plugin's own
                # degrade-path loop) cannot, so an approval there could never be acted on.
                "resumable": self._resumable,
                # Decision 1: the harness runs this call itself once approved (a code
                # caller), and whether it waits for a pinned approval. Both stamped
                # here, from the ToolCall the harness built, never from the caller.
                "deferred_executor": call.deferred,
                "per_call_approval": per_call,
            },
        )
        decision, final_ctx = self._kernel.fire_sync(HookPoint.PRE_TOOL_USE, tool_ctx)
        transformed_args = args_of(final_ctx.payload)
        return decision, transformed_args if transformed_args is not None else args

    def post(
        self,
        tool: ToolSpec,
        observation: str,
        call: ToolCall,
        *,
        tool_call_id: str | None = None,
    ) -> PostOutcome:
        """Run ``POST_TOOL_USE`` over a tool result; what the caller may hand on.

        Design §6.5. The label was derived once from the user's question and cached for
        the whole run, but a tool is where personal data actually enters a turn — so the
        result is where it has to be re-derived, or the egress gate keeps deciding
        against a question that no longer describes what is in the prompt. The label is
        never lowered.

        The verdict is enforced, not advisory: ``deny`` / ``require_approval`` withhold the
        result (the caller gets the governance block message), ``transform`` hands on the
        rewritten result (the injection guard's redaction). A final result that is not
        text cannot be handed on as the tool's output, so it is withheld too, naming the
        hooks that rewrote it.

        The tool's declaration rides on the metadata (``tool_payload.tool_post_metadata``):
        the side-effect ledger reads ``tool_effect`` / ``tool_verify`` / ``tool_call_id``,
        the injection guard ``tool_content``.
        """
        if self._kernel is None:
            return PostOutcome(classification=call.classification, text=observation)
        name = tool.name
        tool_ctx = HookContext(
            hook_point=HookPoint.POST_TOOL_USE,
            run_id=call.run_id or str(uuid.uuid4()),
            agent_type=self._agent_type,
            step_id=call.step_id,
            route=f"tool/{name}",
            classification=call.classification,
            payload=post_tool_payload(
                name, observation, **audit_digester().result_fields(observation)
            ),
            metadata={
                "caller": call.caller or f"model:{self._agent_type}",
                "session_id": self._session_id,
                **tool_post_metadata(
                    effect=tool.effect,
                    content=tool.content,
                    verify=tool.verify,
                    tool_call_id=tool_call_id,
                ),
            },
        )
        decision, final_ctx = self._kernel.fire_sync(HookPoint.POST_TOOL_USE, tool_ctx)
        classification = more_restrictive(call.classification, final_ctx.classification)
        # What the result earned is the turn's too (a floor: never lowered).
        lift_turn_label(classification)
        if decision.outcome in ("deny", "require_approval"):
            return PostOutcome(
                classification=classification,
                text=governance_block_message(decision),
                decision=decision,
                withheld=True,
            )
        final = result_of(final_ctx.payload)
        if not isinstance(final, str):
            hooks = ", ".join(final_ctx.metadata.get("transformed_by", ())) or "unknown"
            refusal = HookDecision(
                outcome="deny",
                reason=(
                    f"POST_TOOL_USE left a result for {name!r} that is not text "
                    f"(transformed by: {hooks}), so it is withheld"
                ),
                severity="error",
            )
            return PostOutcome(
                classification=classification,
                text=governance_block_message(refusal),
                decision=refusal,
                withheld=True,
            )
        return PostOutcome(classification=classification, text=final, decision=decision)

    # ---------------------------------------------------------------- capability calls
    def execute_call(
        self, call: CapabilityCall, provider_call: Callable[..., Any], args: dict[str, Any]
    ) -> Any:
        """Govern and run a sync capability method: its value, or a stream.

        ``PRE_TOOL_USE`` runs now; ``POST_TOOL_USE`` over the value, or for a stream per item
        and at the end. What the consumer gets is always the redacted copy the kernel's
        final context describes.
        """
        run_id = str(uuid.uuid4())
        args = self.capability_pre(call, args, run_id)
        result = provider_call(**args)
        if call.shape == "stream":
            return self._governed_stream(call, iter(result), run_id)
        return self.capability_post(call, result, run_id)

    async def aexecute_call(
        self, call: CapabilityCall, provider_call: Callable[..., Any], args: dict[str, Any]
    ) -> Any:
        """Govern and run an ``async def`` capability method, through ``kernel.fire``."""
        run_id = str(uuid.uuid4())
        args = await self.capability_apre(call, args, run_id)
        result = await provider_call(**args)
        return await self.capability_apost(call, result, run_id)

    async def aexecute_stream(
        self, call: CapabilityCall, provider_call: Callable[..., Any], args: dict[str, Any]
    ) -> AsyncIterator[Any]:
        """Govern an async stream: ``PRE`` before it starts, ``POST`` per item and at the end."""
        run_id = str(uuid.uuid4())
        args = await self.capability_apre(call, args, run_id)
        stream = provider_call(**args)
        digester = _capability_digester(call)
        digests: list[str] = []
        finished = False
        try:
            async for item in stream:
                digests.append(digester.digest(item))
                yield await self.capability_apost(call, item, run_id, stream_item=len(digests) - 1)
            finished = True
        finally:
            # Also when the consumer stops early or the provider raises: the stream's end is
            # audited either way, marked partial when it did not run to the end.
            try:
                await self.capability_apost(
                    call, None, run_id, stream_end=digests, partial=not finished
                )
            except CapabilityDenied:
                if finished:
                    raise
                logger.info("capability stream %s: partial end denied", call.tool_name)

    def _governed_stream(
        self, call: CapabilityCall, stream: Iterator[Any], run_id: str
    ) -> Iterator[Any]:
        digester = _capability_digester(call)
        digests: list[str] = []
        finished = False
        try:
            for item in stream:
                digests.append(digester.digest(item))
                yield self.capability_post(call, item, run_id, stream_item=len(digests) - 1)
            finished = True
        finally:
            try:
                self.capability_post(call, None, run_id, stream_end=digests, partial=not finished)
            except CapabilityDenied:
                if finished:
                    raise
                logger.info("capability stream %s: partial end denied", call.tool_name)

    def _capability_ctx(
        self, point: HookPoint, call: CapabilityCall, run_id: str, payload: dict[str, Any]
    ) -> HookContext:
        """The context a capability call carries on the tool hooks.

        ``payload`` is built by the tool-payload builders (``tool_payload``), so the
        caller policy, the tool policy, the output classifier and every other tool hook
        judge a capability call exactly as they judge a tool. The capability's own keys
        are metadata for the audit row; the row holds no argument or result text, only
        their digests (``kernel._AUDITED_PAYLOAD_KEYS``). One call is one ``run_id``,
        which is also its ``tool_call_id``: a stream's items are one call.
        """
        return HookContext(
            hook_point=point,
            run_id=run_id,
            agent_type=self._agent_type,
            route=f"tool/{call.tool_name}",
            # The turn's label, stamped by the harness when it built the call.
            classification=call.classification,
            payload={
                **payload,
                "caller": call.caller,
                "capability": call.capability,
                "method": call.method,
                "capability_provider": call.provider,
            },
            metadata={
                "caller": call.caller,
                "asked_user": False,
                "tool_confirm": call.confirm,
                "origin_channel": self._origin_channel,
                "session_id": self._session_id,
                "approved_by": None,
                "approval_card": None,
                # A code caller is not a run that can pause and be resumed by an answer.
                "resumable": False,
                **tool_post_metadata(
                    effect=call.effect, content=call.content, verify=None, tool_call_id=run_id
                ),
            },
        )

    def _pre_ctx(self, call: CapabilityCall, args: dict[str, Any], run_id: str) -> HookContext:
        if self._kernel is None:
            raise CapabilityDenied(_NO_KERNEL)
        # No audit key, no call: raised before PRE_TOOL_USE, so the provider never runs.
        digester = _capability_digester(call)
        return self._capability_ctx(
            HookPoint.PRE_TOOL_USE,
            call,
            run_id,
            pre_tool_payload(call.tool_name, args, **digester.args_fields(args)),
        )

    @staticmethod
    def _allowed_args(
        decision: HookDecision, final_ctx: HookContext, args: dict[str, Any]
    ) -> dict[str, Any]:
        if decision.outcome in ("deny", "require_approval"):
            raise CapabilityDenied(governance_block_message(decision), outcome=decision.outcome)
        new_args = args_of(final_ctx.payload)
        return new_args if new_args is not None else args

    def capability_pre(
        self, call: CapabilityCall, args: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        """``PRE_TOOL_USE`` for a capability call; the arguments to run with, or a denial."""
        ctx = self._pre_ctx(call, args, run_id)
        _refuse_inside_a_loop()
        assert self._kernel is not None
        decision, final_ctx = self._kernel.fire_sync(HookPoint.PRE_TOOL_USE, ctx)
        return self._allowed_args(decision, final_ctx, args)

    async def capability_apre(
        self, call: CapabilityCall, args: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        """The async twin of :meth:`capability_pre`, through ``kernel.fire``."""
        ctx = self._pre_ctx(call, args, run_id)
        assert self._kernel is not None
        decision, final_ctx = await self._kernel.fire(HookPoint.PRE_TOOL_USE, ctx)
        return self._allowed_args(decision, final_ctx, args)

    def _post_ctx(
        self,
        call: CapabilityCall,
        value: Any,
        run_id: str,
        stream_item: int | None,
        stream_end: list[str] | None,
        partial: bool,
    ) -> tuple[HookContext, dict[str, str]]:
        if self._kernel is None:
            raise CapabilityDenied(_NO_KERNEL)
        digester = _capability_digester(call)
        if stream_end is not None:
            # The end of a stream: nothing more to redact, one row for the whole of it.
            payload = post_tool_payload(
                call.tool_name,
                "",
                fields={},
                **digester.result_fields(stream_end),
                stream_end=True,
                stream_partial=partial,
            )
            return self._capability_ctx(HookPoint.POST_TOOL_USE, call, run_id, payload), {}
        try:
            fields = extract_fields(value, call.value_type, call.fields)
        except ResultMismatch as exc:
            raise CapabilityDenied(
                f"{call.tool_name}: provider {call.provider!r} returned a value that is not its "
                f"declared type, so it cannot be redacted ({exc})"
            ) from exc
        payload = post_tool_payload(
            call.tool_name,
            "\n".join(fields.values()),
            fields=dict(fields),
            **digester.result_fields(value),
        )
        if stream_item is not None:
            payload["stream_item"] = stream_item
        return self._capability_ctx(HookPoint.POST_TOOL_USE, call, run_id, payload), fields

    @staticmethod
    def _redacted(
        call: CapabilityCall,
        decision: HookDecision,
        final_ctx: HookContext,
        value: Any,
        fields: dict[str, str],
        *,
        end: bool,
    ) -> Any:
        """The consumer's copy: the final context's field map written back, or a denial.

        The field map is the redacted value; ``result`` is its joined text, for hooks that
        read one string. A hook that rewrote ``result`` without rewriting the map would have
        its redaction silently dropped, so a final ``result`` that disagrees with the join of
        the final map is refused, naming the hooks that transformed the payload.
        """
        if decision.outcome in ("deny", "require_approval"):
            raise CapabilityDenied(governance_block_message(decision), outcome=decision.outcome)
        if end:
            return None
        final_fields = final_ctx.payload.get("fields")
        if (
            not isinstance(final_fields, dict)
            or set(final_fields) != set(fields)
            or not all(isinstance(text, str) for text in final_fields.values())
            or final_ctx.payload.get("result") != "\n".join(final_fields[k] for k in fields)
        ):
            hooks = ", ".join(final_ctx.metadata.get("transformed_by", ())) or "unknown"
            raise CapabilityDenied(
                f"{call.tool_name}: POST_TOOL_USE left payload['result'] and payload['fields'] "
                f"disagreeing (transformed by: {hooks}); a result-rewriting hook must rewrite "
                "the field map, so the result is withheld"
            )
        try:
            return rebuild(value, call.value_type, final_fields)
        except ResultMismatch as exc:
            raise CapabilityDenied(
                f"{call.tool_name}: the consumer's copy could not be built ({exc})"
            ) from exc

    def capability_post(
        self,
        call: CapabilityCall,
        value: Any,
        run_id: str,
        *,
        stream_item: int | None = None,
        stream_end: list[str] | None = None,
        partial: bool = False,
    ) -> Any:
        """``POST_TOOL_USE`` over a capability result; the redacted copy, or a denial."""
        ctx, fields = self._post_ctx(call, value, run_id, stream_item, stream_end, partial)
        _refuse_inside_a_loop()
        assert self._kernel is not None
        decision, final_ctx = self._kernel.fire_sync(HookPoint.POST_TOOL_USE, ctx)
        lift_turn_label(more_restrictive(call.classification, final_ctx.classification))
        return self._redacted(call, decision, final_ctx, value, fields, end=stream_end is not None)

    async def capability_apost(
        self,
        call: CapabilityCall,
        value: Any,
        run_id: str,
        *,
        stream_item: int | None = None,
        stream_end: list[str] | None = None,
        partial: bool = False,
    ) -> Any:
        """The async twin of :meth:`capability_post`, through ``kernel.fire``."""
        ctx, fields = self._post_ctx(call, value, run_id, stream_item, stream_end, partial)
        assert self._kernel is not None
        decision, final_ctx = await self._kernel.fire(HookPoint.POST_TOOL_USE, ctx)
        lift_turn_label(more_restrictive(call.classification, final_ctx.classification))
        return self._redacted(call, decision, final_ctx, value, fields, end=stream_end is not None)


def _capability_digester(call: CapabilityCall) -> AuditDigester:
    """The audit digester, or the call is denied: no capability call runs unaudited."""
    try:
        return audit_digester()
    except AuditKeyUnavailable as exc:
        logger.warning("capability call %s refused: %s", call.tool_name, exc)
        raise CapabilityDenied(str(exc)) from exc


def _refuse_inside_a_loop() -> None:
    """``fire_sync`` cannot run inside an event loop; refuse clearly, and fail closed."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise CapabilityDenied(_IN_LOOP)


def invalid_destructive_args(tool: ToolSpec, args: dict[str, Any]) -> str | None:
    """Why this call, approved per call, must not reach the owner, or None (ADR-0118).

    The plugin's ``validate`` checks the arguments against its own data, so an approval
    card only ever names things that exist. It is plugin code: one that raises refuses the
    call (fail-closed), because a card built on arguments nobody could check is exactly
    what this guards against.
    """
    if tool.validate is None:
        return None
    try:
        problem = tool.validate(dict(args))
    except Exception:  # plugin code; refuse rather than guess
        logger.warning("validate() failed for %r; the call is refused", tool.name, exc_info=True)
        return f"{tool.name} could not check these arguments."
    text = str(problem).strip() if problem else ""
    return text or None


def approval_card_for(tool: ToolSpec, args: dict[str, Any], *, query: str | None) -> dict[str, Any]:
    """What the owner will see for this call: the plugin's ``describe`` (title and one line
    per item), the declared undo, and their own request (ADR-0118 step 4).

    ``describe`` is plugin code, so it is guarded: one that raises or returns the wrong
    shape costs the nice wording, not the approval — the hook falls back to the raw call,
    which always says exactly what will run.
    """
    title: str | None = None
    lines: list[str] = []
    if tool.describe is not None:
        try:
            described = tool.describe(dict(args))
            title = str(described.title).strip() or None
            lines = [str(line) for line in described.lines]
        except Exception:  # plugin code; fall back to the raw call
            logger.warning(
                "describe() failed for %r; the card shows the raw call",
                tool.name,
                exc_info=True,
            )
    return {
        "title": title,
        "lines": lines,
        "undo_tool": tool.undo,
        "undo_window_days": tool.undo_window_days,
        "asked": query,
        # What the card warns of: data loss, or a write acting for the owner.
        "effect": tool.effect,
    }


__all__ = [
    "CapabilityCall",
    "GovernedToolRunner",
    "PostOutcome",
    "ToolCall",
    "ToolOutcome",
    "ToolUnavailable",
    "approval_card_for",
    "approved_per_call",
    "approved_per_call_for",
    "governance_block_message",
    "invalid_destructive_args",
    "refused",
]
