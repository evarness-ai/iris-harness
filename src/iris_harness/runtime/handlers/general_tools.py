"""Binding and executing the general handler's tools.

The legacy native-tool-calling lane (``IRIS_AGENTIC_CORE_ENABLED`` off or ``shadow``). Three
functions: the tool set the model is offered, the argument schemas for it, and the dispatcher
that runs whichever tool it picked.

There is ONE tool set and ONE way to run a tool. The set is built from the loop's own builders
(plugin tools from the registry, the lane's four built-ins from ``builtin_react_tools``, skill
tools from ``_skills_to_react_tools``), so a tool carries the declaration the loop gives it
(``effect``, ``confirm``, ``content``), and every call goes through ``GovernedToolRunner``
(R14: every call audited): ``PRE_TOOL_USE`` / ``POST_TOOL_USE``, the external-content floor, the
audit rows, the approval gate. This lane used to keep a second dispatch table that ran the
built-ins and every skill tool directly, with no hooks, no floor and no audit row, so an
injection in a fetched page reached the next prompt, the session log and the memory stores raw
(issue #155, #134 D7). The table is gone: a tool the lane offers cannot run any other way.

It is a **factory** (OSS plan M5.7, slice 4c): the closure over ``repo_root``, ``wiki``,
``semantic_index``, ``runtime_holder`` and the ``local_skills`` bundle is relocated here.
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
from iris_harness.kernel.governance.turn_label import current_turn_label
from iris_harness.llm.client import LLMToolCall
from iris_harness.llm.errors import friendly_llm_error as _friendly_llm_error
from iris_harness.memory.knowledge.wiki_engine import WikiEngine
from iris_harness.memory.semantic_index import SemanticIndex
from iris_harness.runtime.handlers.general_support import (
    plugin_tool_binding,
    truncate_tool_result,
)
from iris_harness.runtime.handlers.local_skills import LocalSkills

if TYPE_CHECKING:
    from iris_harness.agent.agentic_core import ToolSpec
    from iris_harness.tools.skills import SkillRegistry

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


# What the lane offers beyond the plugins' tools: the same four built-ins it always offered,
# now the loop's own ToolSpecs (``builtin_react_tools``), so there is one implementation, one
# declaration (effect, confirm, content) and one governed path for each. Nothing else of the
# loop's built-in pool is offered here (the lane's capability set does not grow).
_LANE_BUILTINS: tuple[str, ...] = (
    "memory_search",
    "memory_graph",
    "wiki_search",
    "propose_skill_from_sandbox",
)

# The argument schemas the model is offered for those built-ins. A ``ToolSpec`` carries no
# schema, so the lane keeps these (the loop describes its tools in prose instead).
_BUILTIN_PARAMETERS: dict[str, dict[str, Any]] = {
    "memory_search": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Concise search query describing what to recall.",
            },
            "scope": {
                "type": "string",
                "enum": ["facts", "patterns", "behaviors", "sessions", "all"],
                "description": "What to search; 'all' by default.",
            },
            "n": {"type": "integer", "description": "How many results per scope (default 5)."},
        },
        "required": ["query"],
    },
    "memory_graph": {
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
    "wiki_search": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Concise query describing the durable knowledge, entity, decision, or "
                    "routine to retrieve."
                ),
            }
        },
        "required": ["query"],
    },
    "propose_skill_from_sandbox": {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": "Verbatim sandbox script content to save as the draft.",
            },
            "intent": {
                "type": "string",
                "description": "Short description of what the script does.",
            },
            "narrative": {
                "type": "string",
                "description": "Why this is useful and when to use it.",
            },
            "slug": {
                "type": "string",
                "description": "Optional kebab-case slug; auto-derived from intent when omitted.",
            },
        },
        "required": ["script", "intent", "narrative"],
    },
}


def make_general_tools(
    *,
    repo_root: Path | None,
    wiki: WikiEngine | None,
    semantic_index: SemanticIndex | None,
    runtime_holder: list[Any] | None,
    local_skills: LocalSkills,
    skill_registry: SkillRegistry | None = None,
) -> GeneralTools:
    """Bind the lane's tool set to one profile's stores.

    There is one set of tools and one way to run them. The set is built from the loop's own
    builders (plugin tools from the registry, the four built-ins from ``builtin_react_tools``,
    skill tools from ``_skills_to_react_tools``), so each tool carries the declaration the
    loop gives it (effect, confirm, ``content``), and every call runs through
    :class:`GovernedToolRunner`: ``PRE_TOOL_USE`` / ``POST_TOOL_USE``, the external-content
    floor, the audit rows, the approval gate. The lane has no dispatch table of its own, so a
    tool it offers cannot run any other way (issue #155).
    """
    _local_skills = local_skills

    def _plugin_tools() -> dict[str, Any]:
        """Tools contributed by mounted plugins, by name (the registry's own, as the loop's)."""
        if not runtime_holder or runtime_holder[0] is None:
            return {}
        return {tool.name: tool for tool in runtime_holder[0].plugin_registry.tools()}

    def _memory_store() -> Any:
        if not runtime_holder or runtime_holder[0] is None:
            return None
        return getattr(runtime_holder[0], "memory_store", None)

    def _builtin_specs() -> dict[str, ToolSpec]:
        from iris_harness.runtime.react_tools import builtin_react_tools

        store = _memory_store()
        built = {
            spec.name: spec
            for spec in builtin_react_tools(
                semantic_index=semantic_index,
                wiki=wiki,
                repo_root=repo_root,
                memory_store=store,
            )
        }
        offered = {"memory_search"}
        if store is not None:
            offered.add("memory_graph")
        if wiki is not None:
            offered.add("wiki_search")
        if repo_root is not None:
            offered.add("propose_skill_from_sandbox")
        return {name: built[name] for name in _LANE_BUILTINS if name in offered and name in built}

    def _spec_pool(query: str | None) -> dict[str, ToolSpec]:
        """The lane's tools by name. ``query`` gates which SKILL tools are offered (the
        loop's relevance rule); ``None`` is every tool, which is what a call is checked
        against (a tool the model names is run if it exists, offered or not, as before)."""
        pool: dict[str, ToolSpec] = dict(_plugin_tools())
        for name, spec in _builtin_specs().items():
            pool.setdefault(name, spec)
        if skill_registry is not None:
            from iris_harness.runtime.handlers.react import _skills_to_react_tools

            for spec in _skills_to_react_tools(
                skill_registry, query=query or "", taken=frozenset(pool)
            ):
                pool.setdefault(spec.name, spec)
        return pool

    def _binding(spec: ToolSpec, local: dict[str, Any]) -> dict[str, Any]:
        """What the model is offered for ``spec``: a schema when this lane has one."""
        parameters = _BUILTIN_PARAMETERS.get(spec.name)
        if parameters is None and spec.name in local:
            parameters = local[spec.name][2]  # the skill tool's own input schema
        if parameters is None:
            return plugin_tool_binding(spec)  # a ToolSpec carries no schema: the generic one
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": parameters,
            },
        }

    def _tool_bindings(query: str = "") -> tuple[dict[str, Any], ...]:
        local = _local_skills.tools()
        return tuple(_binding(spec, local) for spec in _spec_pool(query).values())

    def _run_governed_tool(
        task: AgentTask, tool_call: LLMToolCall, tool: ToolSpec, run_id: str
    ) -> dict[str, object]:
        """Run one of the lane's tools through the governed runner, as the ReAct loop does.

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
        # The runtime's own kernel, read per call like the tool service's. None when governance
        # is switched off (the operator's explicit opt-out, said once at startup by
        # ``_warn_if_lane_serves_ungoverned``) and for a handler built with no runtime (a unit
        # test): the runner then runs the tool with no hooks and refuses only a call that needs
        # the owner's approval.
        runtime = runtime_holder[0] if runtime_holder else None
        runner = GovernedToolRunner(
            kernel=getattr(runtime, "governance_kernel", None),
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
        tool = _spec_pool(None).get(tool_call.name)
        if tool is None:
            return {"ok": False, "error": f"unknown tool: {tool_call.name}"}
        # The runner logs this call's timeline events itself.
        return _run_governed_tool(task, tool_call, tool, run_id)

    return GeneralTools(bindings=_tool_bindings, execute=_execute_general_tool_call)
