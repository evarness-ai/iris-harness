"""The general handler, the clarify handler, and the coding handoff.

Release gate 1 again: building a handler is not composition. These are the last three
builders that were still in `bootstrap.py`; `runtime/handlers/` already held the six
helpers they call.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from iris_harness.agent.agent_executor import (
    ActivityChunk,
    AgentTask,
    HandlerResult,
    StreamChunk,
    TraceChunk,
)
from iris_harness.agent.provenance import ProvenanceLedger
from iris_harness.foundation.observability.session_log import (
    agent_scope,
    log_timeline_event,
    session_scope,
)
from iris_harness.llm.errors import friendly_llm_error as _friendly_llm_error
from iris_harness.llm.tier_router import TierRouter
from iris_harness.memory.knowledge.wiki_engine import WikiEngine
from iris_harness.memory.semantic_index import SemanticIndex
from iris_harness.runtime.handlers.general_invoke import make_general_invoke
from iris_harness.runtime.handlers.general_support import (
    direct_recovery_hint,
    should_stream_directly,
    structured_recovery_clarification,
)
from iris_harness.runtime.handlers.general_tools import (
    PROVENANCE_LEDGER,
    make_general_tools,
)
from iris_harness.runtime.handlers.local_skills import make_local_skills
from iris_harness.runtime.nlu_parsing import _deterministic_time_date_reply
from iris_harness.tools.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


def _pressure_snapshot_payload(snapshot: Any) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    sampled_at = getattr(snapshot, "sampled_at", None)
    return {
        "ram_free_gb": getattr(snapshot, "ram_free_gb", None),
        "cpu_percent": getattr(snapshot, "cpu_percent", None),
        "cpu_speed_limit": getattr(snapshot, "cpu_speed_limit", None),
        "thermal_throttled": getattr(snapshot, "thermal_throttled", None),
        "sampled_at": sampled_at.isoformat() if sampled_at is not None else None,
    }


_UNRESOLVED_RECOVERY_RE = re.compile(
    r"\b("
    r"unable to find|"
    r"could not find|"
    r"no specific information|"
    r"don't have enough information|"
    r"do not have enough information|"
    r"i wasn't able to find"
    r")\b",
    re.IGNORECASE,
)


def _client_runtime_metadata(
    client: Any,
    *,
    intent: str,
    tier_router: TierRouter,
) -> dict[str, object]:
    cfg = getattr(client, "config", None)
    metadata: dict[str, object] = {"intent_tag": intent}
    if cfg is not None:
        metadata.update(
            {
                "provider": getattr(cfg, "provider", ""),
                "model": getattr(cfg, "model", ""),
                "max_tokens": getattr(cfg, "max_tokens", 0),
                "timeout_seconds": getattr(cfg, "timeout_seconds", 0),
            }
        )
        num_ctx = getattr(cfg, "num_ctx", None)
        if num_ctx:
            metadata["num_ctx"] = num_ctx
            metadata["context_window"] = num_ctx
        keep_alive = getattr(cfg, "keep_alive", None)
        if keep_alive is not None:
            metadata["keep_alive"] = keep_alive

    metadata.update(tier_router.trace_metadata_for_intent(intent))

    governor = getattr(tier_router, "governor", None)
    if governor is not None:
        try:
            metadata["governor_mode"] = str(governor.mode())
            metadata["governor_snapshot"] = _pressure_snapshot_payload(governor.snapshot())
        except Exception:  # noqa: BLE001, S110
            pass
    return metadata


_CODING_QUICK_PROMPT = (
    "You are a senior software engineer. Answer the user's coding request directly and concisely.\n"
    "- Write clean, idiomatic code with minimal comments.\n"
    "- If asked to write a function or class, provide the complete implementation.\n"
    "- Prefer standard-library solutions unless the user specifies otherwise.\n"
    "- Output only the code and a one-sentence explanation — no preamble, no filler."
)

_CODING_BUILD_PROMPT = (
    "You are IRIS Coding Agent — a senior software architect and engineer.\n"
    "The user wants you to BUILD something substantial (an agent, service, tool, or application).\n\n"
    "Structure your response as follows:\n"
    "## Overview\n"
    "One paragraph describing what you will build and the key design decisions.\n\n"
    "## Architecture\n"
    "Bullet-list the main components and how they interact.\n\n"
    "## Implementation\n"
    "Provide the complete, runnable code. Use multiple code blocks with file paths as headings.\n"
    "Write production-quality code — proper error handling, type hints, docstrings on public APIs.\n\n"
    "## How to Run\n"
    "Step-by-step instructions to install dependencies and run the implementation.\n\n"
    "## Next Steps\n"
    "What to extend or harden before using this in production."
)

_BUILD_PATTERN = re.compile(
    r"\b(build|create|make|develop|design|generate|produce)\b.{0,80}"
    r"\b(agent|tool|service|bot|system|app|application|pipeline|workflow|plugin|extension)\b",
    re.IGNORECASE,
)


def _make_general_handler(
    tier_router: TierRouter,
    *,
    repo_root: Path | None = None,
    skill_registry: SkillRegistry | None = None,
    semantic_index: SemanticIndex | None = None,
    wiki: WikiEngine | None = None,
    runtime_holder: list[Any] | None = None,
) -> tuple[Callable[[AgentTask], HandlerResult], Callable[[AgentTask], Iterator[StreamChunk]]]:
    """Build the sync + streaming handler pair for general/system intents.

    The general LLM receives bounded direct access to lightweight tools. This
    avoids needing a bespoke intent rule for every current-information or
    execution-shaped request while still keeping tool calls explicit.

    A ``preferred_model`` task param lets the REPL session override the tier
    model for the duration of a conversation.
    """
    _local_skills = make_local_skills(skill_registry, runtime_holder)
    _general_tools = make_general_tools(
        repo_root=repo_root,
        wiki=wiki,
        semantic_index=semantic_index,
        runtime_holder=runtime_holder,
        local_skills=_local_skills,
        skill_registry=skill_registry,
    )
    _general_invoke = make_general_invoke(tier_router=tier_router, general_tools=_general_tools)
    from iris_harness.llm.client import (
        CodingLLMClient,
    )

    _clients: dict[str, CodingLLMClient] = {}

    def handler(task: AgentTask) -> tuple[str, dict[str, object]]:
        try:
            with agent_scope("general"):
                deterministic = _deterministic_time_date_reply(task.query)
                if deterministic is not None:
                    return deterministic, {"deterministic_time_date": True}
                recovery_hints: list[str] = []
                failed_capabilities: list[str] = []

                direct_brief_result, brief_hint, brief_cap = direct_recovery_hint(
                    _local_skills.direct_brief_response(task)
                )
                if direct_brief_result is not None:
                    return direct_brief_result
                if brief_hint:
                    recovery_hints.append(brief_hint)
                if brief_cap:
                    failed_capabilities.append(brief_cap)

                direct_skill_result, skill_hint, skill_cap = direct_recovery_hint(
                    _local_skills.direct_skill_response(task)
                )
                if direct_skill_result is not None:
                    return direct_skill_result
                if skill_hint:
                    recovery_hints.append(skill_hint)
                if skill_cap:
                    failed_capabilities.append(skill_cap)

                client, system_prompt, user_prompt, mark, strategy = _general_invoke.build_prompts(
                    task,
                    recovery_hints=tuple(recovery_hints),
                )
                runtime_metadata = _client_runtime_metadata(
                    client,
                    intent=task.params.get("intent", "general"),
                    tier_router=tier_router,
                )
                log_timeline_event(
                    "agent.trace",
                    phase="llm.invoke.start",
                    payload={
                        "name": "llm.invoke.start",
                        "agent_type": "general",
                        "strategy": str(strategy),
                        **runtime_metadata,
                    },
                )
                provenance = ProvenanceLedger()
                token = PROVENANCE_LEDGER.set(provenance)
                try:
                    text = _general_invoke.invoke_with_tools(
                        task, client, system_prompt, user_prompt, strategy
                    )
                finally:
                    PROVENANCE_LEDGER.reset(token)
                fallback_clarification = False
                if recovery_hints and _UNRESOLVED_RECOVERY_RE.search(text):
                    text = structured_recovery_clarification(task.query, tuple(failed_capabilities))
                    fallback_clarification = True
            usage = client.get_token_usage_since(mark)
            metadata = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "fallback_clarification": fallback_clarification,
                **runtime_metadata,
            }
            # Phase 5 grounding (P1): surface retrieved context for the grounding judge.
            if not provenance.is_empty:
                metadata["retrieved_context"] = provenance.render()
                metadata["sources"] = list(provenance.sources())
            log_timeline_event(
                "agent.trace",
                phase="llm.invoke.end",
                payload={
                    "name": "llm.invoke.end",
                    "agent_type": "general",
                    "strategy": str(strategy),
                    "output_chars": len(text),
                    **metadata,
                },
            )
            return text, metadata
        except Exception as exc:
            logger.exception(
                "general LLM call failed for query=%r provider_profile=%r model=%r",
                task.query,
                task.params.get("provider_profile", ""),
                task.params.get("preferred_model", ""),
            )
            return _friendly_llm_error(exc), {}

    def stream_handler(task: AgentTask) -> Iterator[StreamChunk]:
        try:
            with agent_scope("general"):
                deterministic = _deterministic_time_date_reply(task.query)
                if deterministic is not None:
                    yield deterministic
                    yield {"deterministic_time_date": True}
                    return
                recovery_hints: list[str] = []
                failed_capabilities: list[str] = []

                direct_brief_result, brief_hint, brief_cap = direct_recovery_hint(
                    _local_skills.direct_brief_response(task)
                )
                if direct_brief_result is not None:
                    text, metadata = direct_brief_result
                    yield text
                    yield metadata
                    return
                if brief_hint:
                    recovery_hints.append(brief_hint)
                if brief_cap:
                    failed_capabilities.append(brief_cap)

                direct_skill_result, skill_hint, skill_cap = direct_recovery_hint(
                    _local_skills.direct_skill_response(task)
                )
                if direct_skill_result is not None:
                    text, metadata = direct_skill_result
                    yield text
                    yield metadata
                    return
                if skill_hint:
                    recovery_hints.append(skill_hint)
                if skill_cap:
                    failed_capabilities.append(skill_cap)

                client, system_prompt, user_prompt, mark, strategy = _general_invoke.build_prompts(
                    task,
                    recovery_hints=tuple(recovery_hints),
                )
                intent = task.params.get("intent", "general")
                metadata = _client_runtime_metadata(client, intent=intent, tier_router=tier_router)
                model = str(metadata.get("model") or "")
                num_ctx = metadata.get("num_ctx")
                if should_stream_directly(task):
                    status = f"model {model}" if model else "starting model stream"
                    if num_ctx:
                        status += f" · ctx {num_ctx}"
                    yield ActivityChunk(status)
                    yield TraceChunk(
                        json.dumps(
                            {
                                "event": "llm.start",
                                "agent_type": "general",
                                "intent": intent,
                                "streaming": True,
                                **metadata,
                            },
                            default=str,
                        )
                    )
                    with session_scope(task.session_id):
                        yield from client.invoke_stream(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                        )
                    yield TraceChunk(
                        json.dumps(
                            {
                                "event": "llm.end",
                                "agent_type": "general",
                                "intent": intent,
                                "streaming": True,
                            },
                            default=str,
                        )
                    )
                else:
                    yield ActivityChunk("running tool-capable model turn")
                    start_payload = {
                        "name": "llm.invoke.start",
                        "agent_type": "general",
                        "intent": intent,
                        "strategy": str(strategy),
                        **metadata,
                    }
                    yield TraceChunk(json.dumps(start_payload, default=str))
                    with session_scope(task.session_id):
                        text = _general_invoke.invoke_with_tools(
                            task, client, system_prompt, user_prompt, strategy
                        )
                    if recovery_hints and _UNRESOLVED_RECOVERY_RE.search(text):
                        text = structured_recovery_clarification(
                            task.query,
                            tuple(failed_capabilities),
                        )
                        metadata = {**metadata, "fallback_clarification": True}
                    yield text
                    yield TraceChunk(
                        json.dumps(
                            {
                                "name": "llm.invoke.end",
                                "agent_type": "general",
                                "intent": intent,
                                "strategy": str(strategy),
                                "output_chars": len(text),
                            },
                            default=str,
                        )
                    )
            usage = client.get_token_usage_since(mark)
            yield {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                **_client_runtime_metadata(
                    client,
                    intent=task.params.get("intent", "general"),
                    tier_router=tier_router,
                ),
            }
        except Exception as exc:
            logger.exception(
                "general LLM stream failed for query=%r provider_profile=%r model=%r",
                task.query,
                task.params.get("provider_profile", ""),
                task.params.get("preferred_model", ""),
            )
            yield _friendly_llm_error(exc)

    return handler, stream_handler


def _make_coding_handler(tier_router: TierRouter) -> Callable[[AgentTask], str]:
    """Coding agent handler — uses quick prompt for snippets, architect prompt for build requests."""
    from iris_harness.llm.client import CodingLLMClient

    _clients: dict[str, CodingLLMClient] = {}

    def handler(task: AgentTask) -> str:
        intent = task.params.get("intent", "coding")
        is_build_request = bool(_BUILD_PATTERN.search(task.query))
        system_prompt = _CODING_BUILD_PROMPT if is_build_request else _CODING_QUICK_PROMPT
        try:
            if intent not in _clients:
                cfg = tier_router.get_llm_config(intent)
                _clients[intent] = CodingLLMClient(cfg)  # type: ignore[arg-type]
            return _clients[intent].invoke(
                system_prompt=system_prompt,
                user_prompt=task.query,
            )
        except Exception:
            logger.exception("coding LLM call failed for query=%r", task.query)
            return (
                "The coding agent encountered an error. "
                "Check that a valid LLM provider is configured in your .env file."
            )

    return handler


def _make_clarify_handler(
    llm_call: Callable[[str], str] | None,
) -> tuple[Callable[[AgentTask], HandlerResult], Callable[[AgentTask], Iterator[StreamChunk]]]:
    """Ask ONE brief clarifying question when the router can't tell what the user
    wants (issue 0002) — better than guessing an agent or running a loop that
    stalls. Grounded in the recent conversation so the question is specific."""

    _DEFAULT = "I'm not sure what you'd like me to do — could you tell me a bit more?"

    def _question(task: AgentTask) -> str:
        recent = ""
        if task.memory_context and task.memory_context.recent_turns:
            recent = (
                "Conversation so far:\n" + "\n".join(task.memory_context.recent_turns[-6:]) + "\n\n"
            )
        prompt = (
            "You are IRIS. You could not tell what the user wants from their last "
            "message. Ask ONE short, friendly clarifying question to pin down their "
            "intent (offer the likely options if helpful). Reply with ONLY the "
            "question.\n\n"
            f"{recent}User: {task.query}"
        )
        if llm_call is None:
            return _DEFAULT
        try:
            text = (llm_call(prompt) or "").strip()
        except Exception:  # noqa: BLE001 — never 500 the turn
            return _DEFAULT
        return text or _DEFAULT

    def handler(task: AgentTask) -> HandlerResult:
        return _question(task), {"agent_type": "clarify", "clarify": True}

    def stream_handler(task: AgentTask) -> Iterator[StreamChunk]:
        yield _question(task)
        yield {"agent_type": "clarify", "clarify": True}

    return handler, stream_handler
