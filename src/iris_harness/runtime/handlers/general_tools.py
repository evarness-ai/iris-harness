"""Binding and executing the general handler's tools.

Gate-1 extraction (OSS plan M5.7), slice 4c. Three functions: the plugin tools the loaded
profile contributes, the LangChain bindings the model is offered, and the dispatcher that
runs whichever tool it picked.

Moved as a **factory**, the shape 4b established. The measurement said this cluster closes
over five names — ``repo_root``, ``wiki``, ``semantic_index``, ``runtime_holder`` and 4b's
``local_skills`` bundle — so threading them through as parameters would have meant three
signature changes and a behaviour-preserving refactor. ``make_general_tools`` keeps the
closure and relocates it: the bodies below are byte-identical to their originals inside
``_make_general_handler``, and the change at the call site is one line.

That ``local_skills`` arrives as an argument rather than a closure variable is the visible
payoff of doing 4b first — the dependency between the two clusters is now in a signature
instead of implied by both living in the same 1,100-line function.

A plugin's tool (``research`` included) runs through ``GovernedToolRunner``, the path the
ReAct loop's calls take, never by calling the tool directly (R14: every call audited).
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from iris_harness.agent.agent_executor import AgentTask
from iris_harness.agent.provenance import ProvenanceLedger
from iris_harness.agent.tool_runner import GovernedToolRunner, ToolCall, governance_block_message
from iris_harness.foundation.observability.session_log import log_timeline_event
from iris_harness.kernel.governance.turn_label import current_turn_label
from iris_harness.llm.client import LLMToolCall
from iris_harness.llm.errors import friendly_llm_error as _friendly_llm_error
from iris_harness.memory.graph_context import memory_graph_description, memory_graph_tool
from iris_harness.memory.knowledge.wiki_engine import WikiEngine
from iris_harness.memory.semantic_index import SemanticIndex
from iris_harness.runtime.handlers.general_support import (
    plugin_tool_binding,
    truncate_tool_result,
)
from iris_harness.runtime.handlers.local_skills import LocalSkills
from iris_harness.tools.propose_skill_from_sandbox import propose_skill_from_sandbox

if TYPE_CHECKING:
    from iris_harness.agent.agentic_core import ToolSpec

logger = logging.getLogger(__name__)

# Per-request retrieval-provenance ledger (Phase 5 grounding, P1). Set by the general
# handler around the tool loop; recorded into by the dispatcher below; read back into
# AgentResult.metadata. ContextVar → race-safe across executor threads.
#
# It lives here rather than in bootstrap because the dispatcher that writes it moved here
# while the handler that brackets it stayed: one of the two had to import the other, and
# bootstrap already imports this module.
PROVENANCE_LEDGER: ContextVar[ProvenanceLedger | None] = ContextVar(
    "iris_provenance_ledger", default=None
)


@dataclass(frozen=True)
class GeneralTools:
    """What the tool-calling loops need: the bindings to offer, and the dispatcher.

    ``execute(task, tool_call, run_id=...)``: one run id per turn's tool loop, so every
    governed call the turn makes is recorded under the same run.
    """

    bindings: Any
    execute: Any


def make_general_tools(
    *,
    repo_root: Path | None,
    wiki: WikiEngine | None,
    semantic_index: SemanticIndex | None,
    runtime_holder: list[Any] | None,
    local_skills: LocalSkills,
) -> GeneralTools:
    """Bind the tool set to one profile's stores.

    The bodies are unchanged from their nested originals; this is the closure they
    already had, given a name and a module.
    """
    _local_skills = local_skills

    def _plugin_tools() -> dict[str, Any]:
        """Tools contributed by mounted plugins, by name.

        This legacy native-tool-calling lane used to hard-code its own ``research``
        call straight into the engine, which silently skipped the tool's egress
        guards — a personally-framed money question on this lane reached a search
        provider, and the user's name was never stripped. Reading the registry
        instead means the lane calls the SAME guarded tool the ReAct loop calls,
        and picks up every other plugin tool for free (OSS plan M4.7).
        """
        if not runtime_holder or runtime_holder[0] is None:
            return {}
        return {tool.name: tool for tool in runtime_holder[0].plugin_registry.tools()}

    def _memory_store() -> Any:
        if not runtime_holder or runtime_holder[0] is None:
            return None
        return getattr(runtime_holder[0], "memory_store", None)

    def _tool_bindings(query: str = "") -> tuple[dict[str, Any], ...]:
        bindings: list[dict[str, Any]] = []
        # Plugin tools first — `research` is one of them now (OSS plan M4.7), so a
        # profile that mounts no web-access plugin simply never offers it, instead
        # of advertising a tool whose dispatch would fail.
        bindings.extend(plugin_tool_binding(tool) for tool in _plugin_tools().values())
        bindings += [
            {
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "description": (
                        "Search IRIS's long-term episodic memory — patterns and "
                        "routines previously observed about how the user works. "
                        "Use when the user asks about their habits, preferences, "
                        "or recurring behavior, or when prior context would help "
                        "answer a question. Returns up to 5 short pattern strings."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Concise search query describing the kind of "
                                    "pattern or context to recall."
                                ),
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
        ]
        # The memory graph (memris PR 6): the same tool body the ReAct loop calls, its
        # description generated from the ontology.
        store = _memory_store()
        if store is not None:
            bindings.append(
                {
                    "type": "function",
                    "function": {
                        "name": "memory_graph",
                        "description": memory_graph_description(store),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "entity": {"type": "string"},
                                "relation": {"type": "string"},
                                "direction": {"type": "string", "enum": ["out", "in", "both"]},
                                "hops": {"type": "integer"},
                                "as_of": {"type": "string"},
                            },
                            "required": ["entity"],
                        },
                    },
                }
            )
        if wiki is not None:
            bindings.append(
                {
                    "type": "function",
                    "function": {
                        "name": "wiki_search",
                        "description": (
                            "Search IRIS's compiled long-term knowledge wiki. "
                            "Use for durable knowledge, design decisions, entities, "
                            "concepts, source notes, routine details, or project history. "
                            "Returns the best matching wiki page and related context."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": (
                                        "Concise query describing the durable knowledge, "
                                        "entity, decision, or routine to retrieve."
                                    ),
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            )
        if repo_root is not None:
            bindings.append(
                {
                    "type": "function",
                    "function": {
                        "name": "propose_skill_from_sandbox",
                        "description": (
                            "Save a previously-run sandbox script as a draft skill "
                            "proposal in the user's quarantine queue. Only call this "
                            "when the user explicitly asks to save, remember, or "
                            "promote a script as a reusable skill. Drafts are reviewed "
                            "via the /queue command before any code lands; this tool "
                            "does NOT auto-create tools or modify the codebase. Pass "
                            "the verbatim script you ran (do not paraphrase)."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "script": {
                                    "type": "string",
                                    "description": (
                                        "Verbatim sandbox script content to save as the draft."
                                    ),
                                },
                                "intent": {
                                    "type": "string",
                                    "description": (
                                        "Short description of what the script does "
                                        "(e.g. 'fetch top github repos by stars')."
                                    ),
                                },
                                "narrative": {
                                    "type": "string",
                                    "description": (
                                        "Why this is useful and when to use it; becomes "
                                        "the draft's narrative for later promotion review."
                                    ),
                                },
                                "slug": {
                                    "type": "string",
                                    "description": (
                                        "Optional kebab-case slug; auto-derived from "
                                        "intent when omitted."
                                    ),
                                },
                            },
                            "required": ["script", "intent", "narrative"],
                        },
                    },
                }
            )
        for name, (_tool_class, description, parameters) in _local_skills.relevant_tools(
            query
        ).items():
            bindings.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )
        return tuple(bindings)

    def _run_plugin_tool(
        task: AgentTask, tool_call: LLMToolCall, tool: ToolSpec, run_id: str
    ) -> dict[str, object]:
        """Run a plugin's tool through the governed runner, as the ReAct loop does.

        This lane used to call ``tool.call`` itself, so research and every other plugin
        tool it offered ran with no ``PRE_TOOL_USE`` / ``POST_TOOL_USE``, no approval, no
        audit row and no refusal without an audit key (R14: every call audited). Now the
        call takes the one path every tool call takes (``agent/tool_runner.py``), with
        what the loop gives it: the model as the caller (``model:<agent>``, the runner's
        default), one run id for the whole turn's tool calls, the turn's label, and the
        conversation's session and channel. What comes back is what governance left: a
        held or refused call is an error the model reads, a withheld result is the block
        message, a rewritten one is the rewrite.
        """
        args = dict(tool_call.arguments)
        if tool.name == "research":
            # Snippet mode (fetch_content=False) — this lane's quick lookup;
            # the ReAct loop uses the full research tool with crawl.
            query = str(args.get("query") or task.query).strip()
            args = {"query": query, "fetch_content": False, "max_results": 5}
        # A plugin tool is only ever found through the runtime (``_plugin_tools``).
        assert runtime_holder and runtime_holder[0] is not None
        runner = GovernedToolRunner(
            # Read per call, like the tool service's: the runtime's own kernel.
            kernel=runtime_holder[0].governance_kernel,
            agent_type=task.agent_type,
            origin_channel=task.origin_channel,
            session_id=task.session_id,
            # This lane has no checkpoint to pause on and resume from: an approval it
            # raised could never be acted on, so the approval hook must not queue one.
            resumable=False,
        )
        call = ToolCall(run_id=run_id, classification=current_turn_label(), query=task.query)
        try:
            outcome = runner.execute(tool, args, call)
        except Exception as exc:
            logger.exception("general tool call failed: %s", tool.name)
            return {"ok": False, "error": _friendly_llm_error(exc)}
        if outcome.status == "held":
            assert outcome.decision is not None
            return {"ok": False, "error": governance_block_message(outcome.decision)}
        if not outcome.ok:
            # Refused by the runner, the tool raised, or POST_TOOL_USE withheld it.
            return {"ok": False, "error": outcome.text}
        result = truncate_tool_result(outcome.text)
        # Phase 5 grounding (P1): record retrieved context for the grounding judge.
        ledger = PROVENANCE_LEDGER.get()
        if ledger is not None:
            ledger.record(tool.name, result)
        return {"ok": True, "result": result}

    def _execute_general_tool_call(
        task: AgentTask, tool_call: LLMToolCall, *, run_id: str
    ) -> dict[str, object]:
        plugin_tool = _plugin_tools().get(tool_call.name)
        if plugin_tool is not None:
            # The runner logs this call's timeline events itself.
            return _run_plugin_tool(task, tool_call, plugin_tool, run_id)
        arguments = dict(tool_call.arguments)
        log_timeline_event(
            "tool.invoke.start",
            phase="tool.invoke.start",
            payload={
                "tool": tool_call.name,
                "tool_call_id": tool_call.id,
                "arguments": arguments,
            },
        )
        try:
            if tool_call.name == "research":
                result_payload = {
                    "ok": False,
                    "error": "research unavailable — no web-access plugin is mounted",
                }
            elif tool_call.name == "memory_search":
                if semantic_index is None or not semantic_index.is_ready:
                    result_payload = {
                        "ok": False,
                        "error": "memory_search unavailable — semantic index not ready",
                    }
                else:
                    query = str(arguments.get("query") or task.query).strip()
                    hits = semantic_index.query_episodic(query, n=5)
                    if not hits:
                        result_payload = {
                            "ok": True,
                            "result": "No matching episodic patterns found.",
                        }
                    else:
                        result_payload = {
                            "ok": True,
                            "result": "\n".join(f"- {h}" for h in hits),
                        }
            elif tool_call.name == "memory_graph":
                result_payload = {
                    "ok": True,
                    "result": truncate_tool_result(memory_graph_tool(_memory_store(), arguments)),
                }
            elif tool_call.name == "wiki_search":
                if wiki is None:
                    result_payload = {
                        "ok": False,
                        "error": "wiki_search unavailable — wiki not configured",
                    }
                else:
                    query = str(arguments.get("query") or task.query).strip()
                    result = wiki.query(query)
                    if not result.strip():
                        result_payload = {
                            "ok": True,
                            "result": "No matching wiki pages found.",
                        }
                    else:
                        result_payload = {"ok": True, "result": truncate_tool_result(result)}
            elif tool_call.name == "propose_skill_from_sandbox" and repo_root is not None:
                payload = propose_skill_from_sandbox(
                    repo_root,
                    script=str(arguments.get("script") or ""),
                    intent=str(arguments.get("intent") or ""),
                    narrative=str(arguments.get("narrative") or ""),
                    slug=(str(arguments["slug"]) if arguments.get("slug") else None),
                )
                if payload.get("ok"):
                    result_payload = {"ok": True, "result": truncate_tool_result(payload)}
                else:
                    result_payload = {
                        "ok": False,
                        "error": str(payload.get("error", "proposal failed")),
                    }
            else:
                local_skill = _local_skills.tools().get(tool_call.name)
                if local_skill is not None:
                    tool_class = local_skill[0]
                    tool = tool_class()
                    result = tool.invoke(arguments)
                    result_payload = {"ok": True, "result": truncate_tool_result(result)}
                else:
                    result_payload = {"ok": False, "error": f"unknown tool: {tool_call.name}"}
            # Phase 5 grounding (P1): record retrieved context for the grounding judge.
            _ledger = PROVENANCE_LEDGER.get()
            if _ledger is not None and result_payload.get("ok"):
                _ledger.record(tool_call.name, str(result_payload.get("result") or ""))
            log_timeline_event(
                "tool.invoke.end",
                phase="tool.invoke.end",
                payload={
                    "tool": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "ok": bool(result_payload.get("ok")),
                    "result_preview": str(
                        result_payload.get("result") or result_payload.get("error") or ""
                    ),
                },
            )
            return result_payload
        except Exception as exc:
            logger.exception("general tool call failed: %s", tool_call.name)
            result_payload = {"ok": False, "error": _friendly_llm_error(exc)}
            log_timeline_event(
                "tool.invoke.end",
                phase="tool.invoke.end",
                payload={
                    "tool": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "ok": False,
                    "error": result_payload["error"],
                },
            )
            return result_payload

    return GeneralTools(bindings=_tool_bindings, execute=_execute_general_tool_call)
