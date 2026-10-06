"""PostToolUseLedgerHook — records side effects at PostToolUse for safe resume.

Story 12.gov-4.10 / design §10.3.

Every tool call whose declared effect is not ``read`` gets a row in the side-effect
ledger, so ``iris run resume`` can ask whether the effect already landed before the run
goes on. Opt-in is the tool's own declaration, never a list of names: the runner stamps it
on the ``POST_TOOL_USE`` context (``kernel/governance/hooks/tool_payload.py``).

Read from the context:

    payload  tool_name     : str  -- the tool (``tool_payload.tool_name_of``)
             result        : Any  -- its output; a mapped tool's id is read from it
    metadata tool_effect   : str  -- ``read`` calls are never recorded
             tool_verify   : str | None -- the declared probe, if any
             call_id       : str | None -- the call's ULID (#134); makes the row's key unique
    ctx      run_id, step_id

The row's key is ``<run_id>:<step_id>:<call_id>`` -- one row per call. A key that
already holds a row (not this call's pre-record) is reported as a warning, never confirmed
as recorded; a capability stream's items are one call and are settled once, at the end.

Which probe verifies it, in order:

1. ``TOOL_PROBE_MAP`` -- the coding agent's tool names (git commit/push, PR creation,
   file writes), each with its probe and the result field that names the effect (the
   commit SHA, the PR URL). The probe is handed that id as its subject; when the result
   does not carry it, the row falls back to no probe.
2. The tool's declared ``verify:`` probe, handed the row's key as its subject.
3. No probe: ``run_probe`` answers ``ambiguous``, so resume asks the owner to approve
   before the call is retried.

A high-risk call (``pre_tool_use_ledger``) already has its row, written before it ran: this
hook then *settles* that row instead of inserting -- ``completed`` when the tool returned,
``error`` when it raised (``tool_error`` metadata: the exception class name, never the
message) -- and sets the probe and its subject when the result named the effect
(``SideEffectLedger.finalize``). A capability stream is settled once, at its end
(``stream_end``): ``completed``, or ``error`` when it stopped part-way (the class of what
stopped it). This hook runs before every ``POST_TOOL_USE`` hook that can withhold a result,
so a call whose result is then denied is still settled: it ran. The row stays ``pending``
only if the process never got here.

By default every non-read call is recorded (``IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL=1``).
With ``high_risk_only`` (the default when that flag is unset) only the high-risk class is: a
plain write is not recorded and the ledger is not touched.

The hook always returns ``allow`` — it is an observer, not a gatekeeper. The call has run by
now, so a failed write is a warning, never a refusal.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from iris_harness.kernel.governance.hooks.tool_payload import (
    TOOL_EFFECT,
    TOOL_ERROR,
    TOOL_VERIFY,
    call_id_of,
    result_of,
    tool_name_of,
)
from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.plugins.destructive_approval import pinned_by_declaration
from iris_harness.kernel.governance.side_effects.probes import NO_PROBE
from iris_harness.kernel.governance.side_effects.store import SideEffectKeyExists

if TYPE_CHECKING:
    from iris_harness.kernel.governance.side_effects import SideEffectLedger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static tool → (probe_name, id_field, metadata_extractor) mapping, for the coding agent's
# tool names. metadata_extractor receives the full ctx.payload dict and returns the dict
# stored in probe_metadata (passed back to the probe at resume time).
# ---------------------------------------------------------------------------

_MetaExtractor = Any  # Callable[[dict[str, Any]], dict[str, Any]]


def _git_commit_meta(payload: dict[str, Any]) -> dict[str, Any]:
    return {"repo_path": str(payload.get("cwd", "."))}


def _git_push_meta(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "repo_path": str(payload.get("cwd", ".")),
        "commit_sha": str(payload.get("commit_sha", "")),
    }


def _github_pr_meta(payload: dict[str, Any]) -> dict[str, Any]:
    return {}


def _write_file_meta(payload: dict[str, Any]) -> dict[str, Any]:
    return {}


# Maps tool name → (probe_name, side_effect_id_field, meta_extractor)
# side_effect_id_field: key in ctx.payload["result"] that carries the unique identifier
# for the side effect (e.g. commit SHA, PR URL) -- the probe's subject.
TOOL_PROBE_MAP: dict[str, tuple[str, str, _MetaExtractor]] = {
    # git tool names as used by the coding agent's shell runner
    "git_commit": ("git_commit", "commit_sha", _git_commit_meta),
    "git_push": ("git_push", "branch_remote", _git_push_meta),
    "github_create_pr": ("github_create_pr", "pr_url", _github_pr_meta),
    "write_file": ("write_file", "file_path_sha", _write_file_meta),
    # aliases used by some tool callers
    "create_pull_request": ("github_create_pr", "pr_url", _github_pr_meta),
}


#: ``probe_metadata`` flag of a row written before its call ran (``pre_tool_use_ledger``).
PRE_RECORDED = "pre_recorded"

#: The ``error`` of a stream that ended part-way when no exception class was handed on.
PARTIAL_STREAM = "PartialStream"


def side_effect_key(run_id: str, step_id: int, call_id: str | None) -> str:
    """The ledger key of one call: unique per run, step and call."""
    return f"{run_id}:{step_id}:{call_id or '-'}"


def _mapped_subject(id_field: str, result: Any) -> str | None:
    """The side effect's id in a mapped tool's result, or None when it is not there.

    The result is a dict, or -- through the governed runner, which hands on text -- a JSON
    object as text.
    """
    if isinstance(result, str) and result.lstrip().startswith("{"):
        try:
            result = json.loads(result)
        except ValueError:
            return None
    if isinstance(result, dict):
        value = result.get(id_field)
        if value:
            return str(value)
    return None


class PostToolUseLedgerHook:
    """``PostToolUse`` hook that records every non-read call's side effect for safe resume."""

    name: str = "post_tool_use_ledger"
    hook_point: HookPoint = HookPoint.POST_TOOL_USE
    priority: int = 40

    def __init__(self, ledger: SideEffectLedger, *, high_risk_only: bool = False) -> None:
        self._ledger = ledger
        # ``True``: only the high-risk class (``pinned_by_declaration``) is recorded -- the
        # rows ``PreToolUseLedgerHook`` wrote before the call, settled here. A plain write
        # leaves no row and does not touch the ledger. The default records every non-read.
        self._high_risk_only = high_risk_only

    async def __call__(self, ctx: HookContext) -> HookDecision:
        tool = tool_name_of(ctx.payload)
        effect = ctx.metadata.get(TOOL_EFFECT)
        if not tool or not isinstance(effect, str) or effect == "read":
            # Undeclared (no effect stamped) is not a write anyone declared: a producer
            # that runs non-read tools stamps the declaration (``tool_post_metadata``).
            return HookDecision(outcome="allow", reason="post_tool_use_ledger: not a side effect")

        if self._high_risk_only and not pinned_by_declaration(
            effect, ctx.metadata.get("tool_confirm")
        ):
            return HookDecision(
                outcome="allow", reason="post_tool_use_ledger: not recorded (high-risk only)"
            )

        step_id: int = ctx.step_id or 0
        key = side_effect_key(ctx.run_id, step_id, call_id_of(ctx.metadata))
        probe_name, subject, meta = self._probe_for(tool, ctx)
        if self._written_before(key):
            return self._settle(key, tool, probe_name, subject, meta, ctx)
        try:
            self._ledger.record(
                side_effect_id=key,
                run_id=ctx.run_id,
                step_id=step_id,
                tool=tool,
                verification_probe=probe_name,
                probe_metadata={**meta, "subject": subject or key, "effect": effect},
                exclusive=True,
            )
        except SideEffectKeyExists:
            if self._is_own_stream_row(key, tool, ctx):
                return HookDecision(
                    outcome="allow", reason="post_tool_use_ledger: stream already recorded"
                )
            # The call has run; the ledger already holds a row under its key that is not a
            # pre-record of it. Say so, never confirm a record that was not written.
            logger.warning(
                "post_tool_use_ledger: the ledger key for run=%s tool=%s is already taken; "
                "this call's side effect was not recorded",
                ctx.run_id,
                tool,
            )
            return HookDecision(
                outcome="allow",
                reason=f"post_tool_use_ledger: could not record {tool}: its key is already taken",
                severity="warn",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "post_tool_use_ledger: failed to record side effect for run=%s tool=%s: %s",
                ctx.run_id,
                tool,
                exc,
            )
            return HookDecision(
                outcome="allow",
                reason=f"post_tool_use_ledger: could not record {tool}",
                severity="warn",
            )

        return HookDecision(
            outcome="allow",
            reason=f"post_tool_use_ledger: recorded {tool} side effect",
            audit_metadata={"side_effect_id": key, "probe": probe_name or "none"},
        )

    def _is_own_stream_row(self, key: str, tool: str, ctx: HookContext) -> bool:
        """Whether the taken ``key`` is this call's own row: a later item of its stream.

        A stream is one call under one key; its first item (or its end) inserts the row and
        the rest find it there, which is expected and not a collision.
        """
        if ctx.payload.get("stream_item") is None and ctx.payload.get("stream_end") is not True:
            return False
        try:
            row = self._ledger.get(key)
        except Exception:  # noqa: BLE001
            return False
        return row is not None and row.tool == tool

    def _written_before(self, key: str) -> bool:
        """Whether ``key`` is a row ``PreToolUseLedgerHook`` wrote and nothing has settled yet.

        A pre-row that is no longer ``pending`` was settled by a call already; settling it
        again would rewrite that call's outcome, so it answers no and the insert below
        reports the taken key. A ledger that cannot be read answers no: the insert is then
        tried, and a key already there is reported (``record(..., exclusive=True)``).
        """
        try:
            row = self._ledger.get(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "post_tool_use_ledger: could not read the ledger row %s (%s)",
                key,
                type(exc).__name__,
            )
            return False
        return (
            row is not None
            and row.probe_metadata.get(PRE_RECORDED) is True
            and row.status == "pending"
        )

    def _settle(
        self,
        key: str,
        tool: str,
        probe_name: str,
        subject: str | None,
        meta: dict[str, Any],
        ctx: HookContext,
    ) -> HookDecision:
        """Settle the row ``PreToolUseLedgerHook`` wrote before the call ran."""
        if ctx.payload.get("stream_item") is not None:
            # An item of a stream: the stream is one call, settled at its end.
            return HookDecision(outcome="allow", reason="post_tool_use_ledger: stream item")
        error = ctx.metadata.get(TOOL_ERROR)
        if not (isinstance(error, str) and error) and ctx.payload.get("stream_partial") is True:
            # A stream that stopped part-way, with no exception the runner could name.
            error = PARTIAL_STREAM
        status = "error" if isinstance(error, str) and error else "completed"
        update: dict[str, Any] = {}
        if status == "completed" and probe_name != NO_PROBE:
            # The row was written with no probe; now the call has returned, the declared
            # probe (subject: the row's key) or the mapped one (subject: the effect's id).
            update = {
                "verification_probe": probe_name,
                "probe_metadata": {**meta, "subject": subject or key},
            }
        try:
            self._ledger.finalize(
                key,
                status=status,
                error=error if status == "error" else None,
                **update,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "post_tool_use_ledger: could not settle %s for run=%s tool=%s (%s)",
                key,
                ctx.run_id,
                tool,
                type(exc).__name__,
            )
            return HookDecision(
                outcome="allow",
                reason=f"post_tool_use_ledger: could not settle {tool}",
                severity="warn",
            )
        return HookDecision(
            outcome="allow",
            reason=f"post_tool_use_ledger: settled {tool} as {status}",
            audit_metadata={"side_effect_id": key, "status": status},
        )

    @staticmethod
    def _probe_for(tool: str, ctx: HookContext) -> tuple[str, str | None, dict[str, Any]]:
        """``(probe, subject, probe_metadata)`` for this call (module docstring, 1-3)."""
        mapped = TOOL_PROBE_MAP.get(tool)
        if mapped is not None:
            probe_name, id_field, meta_extractor = mapped
            subject = _mapped_subject(id_field, result_of(ctx.payload))
            if subject is not None:
                return probe_name, subject, meta_extractor(ctx.payload)
            # The probe needs the effect's own id; without it, it could only guess.
            return NO_PROBE, None, {}
        declared = ctx.metadata.get(TOOL_VERIFY)
        if isinstance(declared, str) and declared:
            return declared, None, {}
        return NO_PROBE, None, {}
