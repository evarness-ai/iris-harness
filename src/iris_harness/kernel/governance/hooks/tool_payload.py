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
  third party wrote: web pages, email, retrieved documents) -- what the retrieved-content
  injection guard scans.
- ``tool_verify``: the side-effect probe that can tell whether a write landed, or None.
- ``tool_call_id``: this call's id, unique within its run step.
- ``tool_sends_to`` (``PRE_TOOL_USE``): where the arguments go, when the tool declares it:
  ``search_engine`` for a tool that hands its arguments to a web search provider. The
  owner-PII web-search column applies to exactly these calls (ADR-0125), by declaration
  rather than by a list of tool names.
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

#: Metadata keys (the tool's declaration, stamped by the runner).
TOOL_EFFECT = "tool_effect"
TOOL_CONTENT = "tool_content"
TOOL_VERIFY = "tool_verify"
TOOL_CALL_ID = "tool_call_id"
TOOL_SENDS_TO = "tool_sends_to"

ToolContent = Literal["internal", "external"]
TOOL_CONTENTS: tuple[ToolContent, ...] = ("internal", "external")

#: Where a tool's arguments go, when that is a governed destination of its own.
ToolSendsTo = Literal["search_engine"]
SEARCH_ENGINE: ToolSendsTo = "search_engine"


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
) -> dict[str, Any]:
    """The declaration a ``POST_TOOL_USE`` hook reads, as metadata.

    ``effect`` is None only for a tool that declares none (an external MCP server's tool):
    nothing is assumed about it.
    """
    return {
        TOOL_EFFECT: effect,
        TOOL_CONTENT: content,
        TOOL_VERIFY: verify,
        TOOL_CALL_ID: tool_call_id,
    }


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


def sends_to_search_engine(metadata: dict[str, Any]) -> bool:
    """Whether the tool declared that its arguments go to a web search provider."""
    return metadata.get(TOOL_SENDS_TO) == SEARCH_ENGINE


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
    "DIGEST_ALG",
    "RESULT_DIGEST",
    "RESULT",
    "SEARCH_ENGINE",
    "TOOL_CALL_ID",
    "TOOL_CONTENT",
    "TOOL_CONTENTS",
    "TOOL_EFFECT",
    "TOOL_NAME",
    "TOOL_SENDS_TO",
    "TOOL_VERIFY",
    "ToolContent",
    "ToolSendsTo",
    "args_of",
    "is_external",
    "post_tool_payload",
    "pre_tool_payload",
    "result_of",
    "sends_to_search_engine",
    "tool_name_of",
    "tool_post_metadata",
]
