"""Stateless helpers the general handler uses — no closure, no state, no I/O.

Gate-1 extraction (OSS plan M5.7), slice 4a and the first cut into
``_make_general_handler``. That function is a 1,106-line closure factory returning
``(handler, stream_handler)``, of which only 54 lines are glue; the rest is 22 nested
definitions. These seven read **nothing** from the enclosing scope, which is what makes
them a move rather than the signature change the remaining clusters need — every other
cluster closes over ``tier_router``, ``skill_registry``, ``repo_root``, ``wiki``,
``semantic_index`` or ``runtime_holder``, and extracting those means turning closure
variables into parameters.

They are a mixed bag on purpose: tool-binding schemas, tool-result shaping, the
recovery texts and the stream-directly predicate. Grouped by *having no dependencies*
rather than by subject, because that is the property that made this slice safe. Slices
4b-4d may re-home individual helpers next to the cluster that uses them if that reads
better once those modules exist; one module now beats seven near-empty ones.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from iris_harness.agent.agent_executor import AgentTask, HandlerResult

logger = logging.getLogger(__name__)


GENERAL_TOOL_RESULT_LIMIT = 12_000


GENERAL_STREAM_TOOLISH_RE = re.compile(
    r"\b("
    r"search|look\s*up|latest|current|today|tonight|news|weather|"
    r"run|execute|code|script|calculate|compute|scrape|fetch|download|"
    r"file|pdf|csv|excel|xlsx|github|repo|repository|web|url|https?://|"
    r"memory|wiki|tool"
    r")\b",
    re.IGNORECASE,
)


def local_skill_input_schema(tool_class: type[Any]) -> dict[str, Any]:
    try:
        schema = tool_class().get_input_schema().model_json_schema()
        if isinstance(schema, dict):
            return schema
    except Exception:
        logger.debug("local skill schema inference failed via tool instance", exc_info=True)

    try:
        from pydantic import BaseModel

        model_field = getattr(tool_class, "model_fields", {}).get("args_schema")
        args_schema = getattr(model_field, "default", None)
        if isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
            schema = args_schema.model_json_schema()
            if isinstance(schema, dict):
                return schema
    except Exception:
        logger.debug("local skill schema inference failed via args_schema", exc_info=True)
    return {"type": "object", "properties": {}, "additionalProperties": True}


def direct_recovery_hint(
    result: tuple[str, dict[str, object]] | None,
) -> tuple[tuple[str, dict[str, object]] | None, str | None, str | None]:
    """Return direct result or a recovery hint when direct execution failed."""
    if result is None:
        return None, None, None
    text, metadata = result
    if not metadata.get("skill_error"):
        return result, None, None
    capability = str(metadata.get("skill_tool") or "capability")
    hint = (
        f"Direct capability `{capability}` failed before completing. "
        "Continue by using other available tools, then provide a concrete answer "
        "or ask a short clarification question if data is still missing."
    )
    return None, hint, capability


def structured_recovery_clarification(query: str, capabilities: tuple[str, ...]) -> str:
    """Return deterministic next-step options when fallback stays inconclusive."""
    cap_text = ", ".join(f"`{name}`" for name in capabilities) if capabilities else "`capability`"
    return (
        f"I couldn't complete that via {cap_text}, and the fallback tool pass was inconclusive.\n\n"
        "Reply with one option:\n"
        "1. `retry now` to run another live lookup with tools.\n"
        "2. `list routines` to show approved routines and IDs.\n"
        "3. `run <routine-id>` to execute a specific routine now.\n"
        "4. Share an exact source/URL or exact skill name to target."
    )


def plugin_tool_binding(tool: Any) -> dict[str, Any]:
    """A generic binding for a plugin tool.

    A ``ToolSpec`` carries a name, a description and a callable — no JSON
    schema — so the binding offers the one argument every plugin tool in this
    repo reads (``args.get("query") or args.get("input")``). It is the honest
    floor: richer per-tool schemas are a v2 registration-surface question.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The request to pass to this tool, in full.",
                    }
                },
                "required": ["query"],
            },
        },
    }


def truncate_tool_result(value: object) -> str:
    text = str(value)
    if len(text) <= GENERAL_TOOL_RESULT_LIMIT:
        return text
    return text[:GENERAL_TOOL_RESULT_LIMIT] + "\n...[truncated]"


def enrich_tool_result(tool_name: str, result_payload: dict[str, object]) -> str:
    """Wrap bare/empty tool results with context hints so local models know what to do next."""
    if result_payload.get("ok"):
        raw = str(result_payload.get("result", ""))
        if raw.strip():
            return raw
        return (
            f"The {tool_name} tool returned an empty result. "
            "Try rephrasing the query with different keywords, or use a broader search term."
        )
    err = str(result_payload.get("error", "unknown error"))
    return (
        f"The {tool_name} tool failed: {err}. " "Try a different approach or rephrase your query."
    )


def should_stream_directly(task: AgentTask) -> bool:
    """Return True when a general turn is low-risk to stream without tool binding."""
    if task.session_id == "__warmup__":
        return True
    return GENERAL_STREAM_TOOLISH_RE.search(task.query) is None


# Shared by BOTH handler builders (react and general), which is why it is here rather
# than in either -- it left bootstrap.py with them for release gate 1.
def _normalize_handler_result(raw: HandlerResult) -> tuple[str, dict[str, object]]:
    if isinstance(raw, tuple):
        output, metadata = raw
        return str(output), dict(metadata)
    return str(raw), {}
