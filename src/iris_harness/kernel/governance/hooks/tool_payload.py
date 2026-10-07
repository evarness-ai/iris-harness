"""The tool-hook payload contract: what ``PRE_TOOL_USE`` and ``POST_TOOL_USE`` carry.

Every producer of a tool hook context builds its payload here, and every tool hook reads
it through the accessors below, so a key cannot be written under one name and read under
another. That drift happened: three hooks read ``tool`` / ``tool_arguments`` while every
producer sent ``tool_name`` / ``args``, and each silently did nothing on every real call.
``tests/unit/iris_harness/kernel/test_governance/test_tool_payload_contract.py`` runs the
tool hooks over payloads built here and fails when one no longer acts.

Payload (``HookContext.payload``):

- ``PRE_TOOL_USE``: ``tool_name`` (str) and ``args`` (dict). A hook may rewrite ``args``
  in a ``transform``; the caller runs the tool with the final ``args``.
- ``POST_TOOL_USE``: ``tool_name`` (str) and ``result`` (the tool's output). A hook may
  rewrite ``result`` in a ``transform``; the caller hands on the final ``result``.

The pre-execution record (``PRE_TOOL_USE``): a high-risk call -- destructive, or a write
approved per call -- is written to the side-effect ledger before it runs, by the last
``PRE_TOOL_USE`` hook (``PreToolUseLedgerHook``). The hook confirms it by putting
``side_effect_id`` (the row's key) in its decision's ``audit_metadata``; the runner runs the
call only when the final decision carries it, so a call without a durable record never runs.
(Not a payload transform: the hook runs after the credential broker and never reads the
arguments, which by then may hold a resolved secret.)

Audit fingerprints, beside the contract keys (the only trace of the call's text an audit
row keeps, ``kernel._AUDITED_PAYLOAD_KEYS``): ``args_digest`` on ``PRE_TOOL_USE`` -- of the
arguments as the caller wrote them, never a hook's rewrite (the credential broker puts
secrets there) -- ``result_digest`` on ``POST_TOOL_USE``, and ``digest_alg`` on both. Each is
an HMAC under the install's audit key (``kernel/governance/audit/digest.py``); a governed
call that cannot get one does not run.

Declarations (``HookContext.metadata``), stamped by the runner from the tool's own
declaration, never chosen by the caller:

- ``tool_effect``: ``read`` | ``write`` | ``destructive`` (ADR-0110 / ADR-0118).
- ``tool_content``: ``internal`` (the owner's or IRIS's own data) | ``external`` (text a
  third party wrote: web pages, email, retrieved documents) -- what the always-on
  external-content floor marks and scans, and what the opt-in retrieved-content injection
  guard (model) also scans.
- ``tool_verify``: the side-effect probe that can tell whether a write landed, or None.
- ``call_id`` (alias ``tool_call_id``, same value): this call attempt's ULID, minted once by the
  runner and stamped by the kernel on every audit row of the call (``kernel._audit``).
- ``tool_error`` (``POST_TOOL_USE``): the exception class name when the tool raised, else
  None. Never the message: it can carry the call's content.
- ``tool_sends_to`` (``PRE_TOOL_USE``): where the arguments go, when the tool declares it:
  ``search_engine`` for a tool that hands its arguments to a web search provider (the
  owner-PII web-search column applies to exactly these calls, ADR-0125), or
  ``external_service`` for a tool that hands them to any other service outside the machine
  (the owner-PII egress column applies; the hosts are the plugin's ``egress`` declaration,
  issue #103). By declaration rather than by a list of tool names.
"""

from __future__ import annotations

from typing import Any, Literal

#: Payload keys.
TOOL_NAME = "tool_name"
ARGS = "args"
RESULT = "result"

#: Audit fingerprint keys (keyed digests; ``AuditDigester.args_fields`` / ``result_fields``).
ARGS_DIGEST = "args_digest"
RESULT_DIGEST = "result_digest"
DIGEST_ALG = "digest_alg"

#: Audit identity key (both tool hooks): who owns the tool called -- the plugin that
#: registered it, ``skill:<name>``, ``mcp:<server>`` for a bridged server's tool, or
#: ``system`` for a tool the core provides. Not ``plugin``: an audit row's ``plugin`` column
#: is the governance check that wrote the row.
TOOL_PLUGIN = "tool_plugin"

#: Metadata keys (the tool's declaration, stamped by the runner).
TOOL_EFFECT = "tool_effect"
TOOL_CONTENT = "tool_content"
TOOL_VERIFY = "tool_verify"
#: The call's id (a ULID, #134), minted once per call attempt by the runner -- never taken
#: from a caller. ``TOOL_CALL_ID`` is the old name, stamped beside it with the same value
#: for readers that predate ``CALL_ID`` (the session log's trace builder, old logs).
CALL_ID = "call_id"
TOOL_CALL_ID = "tool_call_id"
#: On the approved attempt of a held call: the id of the attempt that was held, read by
#: the harness from the approval row, never from a caller.
HELD_CALL_ID = "held_call_id"
#: On an egress request's rows (``PRE_EGRESS`` / ``POST_EGRESS``): the id of the governed
#: call the request was made inside (the calling tool's ``CALL_ID``). The governed HTTP
#: client reads it from the harness's call scope, never from the plugin; the request's own
#: ``CALL_ID`` is a fresh ULID the client mints (#103, #134).
PARENT_CALL_ID = "parent_call_id"
#: Identity of a call beyond its own id (#134 stage 2). Each is written by the kernel from
#: the harness's own state (``kernel/governance/call_context.py``, the turn scope), never
#: from a payload, an argument or a caller: the turn the row was written in, which attempt
#: of the call it is (2 for the approved re-execution of a held call), the held attempt it
#: replays, and the run a halted run was re-entered as.
TURN_ID = "turn_id"
ATTEMPT = "attempt"
REPLAY_OF = "replay_of"
RESUMED_FROM_RUN = "resumed_from_run"
TOOL_SENDS_TO = "tool_sends_to"
TOOL_ERROR = "tool_error"

#: ``audit_metadata`` key of the final ``PRE_TOOL_USE`` decision: the ledger row written
#: before the call ran.
SIDE_EFFECT_ID = "side_effect_id"

ToolContent = Literal["internal", "external"]
TOOL_CONTENTS: tuple[ToolContent, ...] = ("internal", "external")

#: Where a tool's arguments go, when that is a governed destination of its own.
ToolSendsTo = Literal["search_engine", "external_service"]
SEARCH_ENGINE: ToolSendsTo = "search_engine"
EXTERNAL_SERVICE: ToolSendsTo = "external_service"


def pre_tool_payload(tool_name: str, args: dict[str, Any], /, **extra: Any) -> dict[str, Any]:
    """The ``PRE_TOOL_USE`` payload: the tool's name and the arguments it will run with.

    ``extra`` carries producer-specific keys beside the contract (an MCP call's server and
    tool, a capability call's digests); it may not redefine a contract key.
    """
    _no_contract_keys(extra)
    return {TOOL_NAME: tool_name, ARGS: args, **extra}


def post_tool_payload(tool_name: str, result: Any, /, **extra: Any) -> dict[str, Any]:
    """The ``POST_TOOL_USE`` payload: the tool's name and what it returned."""
    _no_contract_keys(extra)
    return {TOOL_NAME: tool_name, RESULT: result, **extra}


def tool_post_metadata(
    *,
    effect: str | None,
    content: ToolContent,
    verify: str | None,
    tool_call_id: str | None,
    error: str | None = None,
    held_call_id: str | None = None,
) -> dict[str, Any]:
    """The declaration a ``POST_TOOL_USE`` hook reads, as metadata.

    ``effect`` is None only for a tool that declares none (an external MCP server's tool):
    nothing is assumed about it. ``error`` is the exception class name when the tool raised.
    """
    return {
        TOOL_EFFECT: effect,
        TOOL_CONTENT: content,
        TOOL_VERIFY: verify,
        CALL_ID: tool_call_id,
        TOOL_CALL_ID: tool_call_id,
        HELD_CALL_ID: held_call_id,
        TOOL_ERROR: error,
    }


def call_id_of(metadata: dict[str, Any]) -> str | None:
    """The call's id from a context's metadata: ``call_id``, else its old name
    ``tool_call_id`` (a hand-built context that predates #134 stamps only that)."""
    value = metadata.get(CALL_ID) or metadata.get(TOOL_CALL_ID)
    return value if isinstance(value, str) and value else None


def tool_name_of(payload: dict[str, Any]) -> str:
    """The tool's name, or ``""`` when the payload names none."""
    name = payload.get(TOOL_NAME)
    return name if isinstance(name, str) else ""


def args_of(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The call's arguments, or None when the payload carries none (or not a dict)."""
    args = payload.get(ARGS)
    return args if isinstance(args, dict) else None


def result_of(payload: dict[str, Any]) -> Any:
    """What the tool returned (None when the payload carries no result)."""
    return payload.get(RESULT)


def side_effect_id_of(audit_metadata: dict[str, Any]) -> str | None:
    """The ledger key of the row written before the call ran, or None when there is none.

    Read from the final ``PRE_TOOL_USE`` decision's ``audit_metadata``.
    """
    key = audit_metadata.get(SIDE_EFFECT_ID)
    return key if isinstance(key, str) and key else None


def sends_to_search_engine(metadata: dict[str, Any]) -> bool:
    """Whether the tool declared that its arguments go to a web search provider."""
    return metadata.get(TOOL_SENDS_TO) == SEARCH_ENGINE


def sends_to_external_service(metadata: dict[str, Any]) -> bool:
    """Whether the tool declared that its arguments go to a service outside the machine."""
    return metadata.get(TOOL_SENDS_TO) == EXTERNAL_SERVICE


def is_external(metadata: dict[str, Any]) -> bool:
    """Whether the tool declared its output third-party content."""
    return metadata.get(TOOL_CONTENT) == "external"


def _no_contract_keys(extra: dict[str, Any]) -> None:
    clash = {TOOL_NAME, ARGS, RESULT} & set(extra)
    if clash:
        raise ValueError(f"tool payload extras may not redefine contract keys: {sorted(clash)}")


__all__ = [
    "ARGS",
    "ARGS_DIGEST",
    "ATTEMPT",
    "CALL_ID",
    "DIGEST_ALG",
    "HELD_CALL_ID",
    "PARENT_CALL_ID",
    "REPLAY_OF",
    "RESULT_DIGEST",
    "RESUMED_FROM_RUN",
    "RESULT",
    "EXTERNAL_SERVICE",
    "SEARCH_ENGINE",
    "SIDE_EFFECT_ID",
    "TOOL_CALL_ID",
    "TURN_ID",
    "TOOL_CONTENT",
    "TOOL_CONTENTS",
    "TOOL_EFFECT",
    "TOOL_ERROR",
    "TOOL_NAME",
    "TOOL_SENDS_TO",
    "TOOL_VERIFY",
    "ToolContent",
    "ToolSendsTo",
    "args_of",
    "call_id_of",
    "is_external",
    "post_tool_payload",
    "pre_tool_payload",
    "result_of",
    "side_effect_id_of",
    "sends_to_external_service",
    "sends_to_search_engine",
    "tool_name_of",
    "tool_post_metadata",
]
