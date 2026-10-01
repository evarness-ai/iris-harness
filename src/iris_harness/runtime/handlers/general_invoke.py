"""Prompts, the per-profile LLM client, and the tool-calling loops.

Gate-1 extraction (OSS plan M5.7), slice 4d and the last cluster out of
``_make_general_handler`` before the two handlers themselves. Building the system and user
prompts, resolving (and memoising) the ``CodingLLMClient`` for an intent, and the two loops
— native tool-calling and the ReAct text fallback — plus the shim that picks between them.

Moved as a **factory**, as 4b and 4c were: the cluster closes over ``tier_router``, 4c's
``general_tools`` bundle, and ``_clients``. That last one is why the factory shape matters
here rather than merely being tidy. ``_clients`` is a **mutable memo of built clients, one
dict per factory call** — i.e. per runtime. Promoting it to a module-level dict would make it
process-global, so two runtimes in one process (which the test suite builds constantly) would
start sharing clients. Created inside ``make_general_invoke``, its lifetime is exactly what it
was.

The prompt template, tool-budget helper and ReAct parsing constants travel with the loops
because nothing else used them. Client/provider resolution went the other way, into
``runtime/client_config.py``, because six other bootstrap functions call it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any

from iris_harness.agent.agent_executor import AgentTask
from iris_harness.agent.dateparse import resolved_dates_line
from iris_harness.foundation.clock import local_now
from iris_harness.kernel.governance.disclosure import DISCLOSURE_STYLE_GUARDRAIL
from iris_harness.llm.budget import budget_for, estimate_tokens, trim_text
from iris_harness.llm.tier_router import TierRouter
from iris_harness.memory.identity import load_agent_name, load_soul_core
from iris_harness.runtime.client_config import (
    config_from_profile,
    effective_provider_profile,
    infer_provider,
    resolve_routing_intent,
)
from iris_harness.runtime.handlers.general_support import enrich_tool_result
from iris_harness.runtime.handlers.general_tools import GeneralTools

logger = logging.getLogger(__name__)


_GENERAL_SYSTEM_PROMPT_TEMPLATE = (
    "You are {agent_name}, a helpful personal AI assistant. Answer the user's question directly "
    "and concisely.\n"
    "Use the user profile and conversation history below to personalise your answers.\n"
    "If the answer is in the profile or history, use it confidently. Do not make up "
    "information.\n"
    "You may call tools when the answer needs live data, retrieval, calculations, "
    "code execution, or artifact generation. Use research for current public facts and "
    "anything needing depth or recency. Use code_exec for tasks that require running "
    "code, calculations, or producing a file. Do not claim that you lack live web "
    "access when a relevant tool is available.\n"
    "\n"
    "Tool selection guidance:\n"
    "- Use research for ALL web lookups — facts, headlines, weather, news, docs, and "
    "anything needing depth, recency, or multiple sources (e.g. 'tell me more about X', "
    "'what's the latest on Y', 'find out about Z'). It searches the web, crawls the top "
    "pages, and returns extracted page CONTENT with citations. Call it ONCE, then "
    "synthesize your answer from the returned content and cite the URLs — do NOT "
    "re-issue the same search; repeating a search will not yield more.\n"
    "- For a single quick fact, headline, or current value where a snippet is enough, "
    'still use research but pass "fetch_content": false (faster — snippets only, no '
    "page crawl).\n"
    "- Use code_exec (with requests/httpx) to fetch data from a specific known URL "
    "(e.g. an API endpoint, a documentation page) — more reliable than searching for it.\n"
    "- Only tell the user you could not find information after actually using research.\n"
    "\n"
    "When presenting lists:\n"
    "- Always return the COMPLETE list the user asked for. Never truncate with '...', "
    "'(other repositories)', '(and more)', or similar placeholders.\n"
    "- If the user asks for top 10, return all 10 items. If the tool result contains "
    "more data than requested, select the top N and list them fully.\n"
    "\n"
    "Current time: {time}\n"
    "Current date: {date}"
)


# Shared with the AgenticCore ReAct prompt builder
# (iris_harness.kernel.governance.disclosure.DISCLOSURE_STYLE_GUARDRAIL) so the legacy general handler
# and the ReAct path — which handles conversational "who are you?" turns — apply
# exactly the same concise/no-disclosure rules. The streaming-path safety screen
# is the deterministic backstop when a model ignores them.
_DISCLOSURE_STYLE_GUARDRAIL = DISCLOSURE_STYLE_GUARDRAIL


def _general_system_prompt(soul: str | None = None, message: str = "") -> str:
    """Return the general-handler system prompt with current time/date injected.

    Identity (soul.md) is the canonical source. The embedded template is a
    fallback that preserves behavior when ``~/.iris/identity/soul.md`` is
    missing — this keeps tests and fresh installs functional.
    """
    now = local_now()
    time_str = now.strftime("%H:%M %Z")
    date_str = now.strftime("%A, %B %d, %Y")
    # The user's day words, already resolved (small models misread "Friday").
    dates_line = resolved_dates_line(message, now=now)

    if soul is None:
        soul = load_soul_core()
    agent_name = load_agent_name()
    if soul:
        return (
            # Pin the current date/time at the HEAD of the prompt: local-tier
            # prompts are trimmed tail-first (llm.budget.trim_text), so a date
            # appended after a long SOUL is the first thing dropped under budget
            # pressure — the model then falls back to its training-era date
            # prior (exp-006 GAP-10: calendar "tomorrow" resolved to Apr 2025).
            f"Agent name: {agent_name}\n"
            f"Current time: {time_str}\nCurrent date: {date_str}\n{dates_line}\n"
            f"{soul}" + _DISCLOSURE_STYLE_GUARDRAIL
        )
    return (
        _GENERAL_SYSTEM_PROMPT_TEMPLATE.format(
            agent_name=agent_name,
            time=time_str,
            date=date_str,
        )
        + (f"\n{dates_line}" if dates_line else "")
        + _DISCLOSURE_STYLE_GUARDRAIL
    )


_GENERAL_TOOL_MAX_TURNS = 4


_GENERAL_TOOL_MAX_TURNS_CAP = 8


def _safe_int_env(name: str, default: int, *, min_value: int = 1, max_value: int = 128) -> int:
    """Read a bounded integer from env vars, returning ``default`` on invalid input."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


def _general_tool_budget(task: AgentTask) -> tuple[int, int]:
    """Return ``(base_budget, cap_budget)`` for the general tool loop.

    Budget is configurable and slightly adaptive: short continuation turns and
    requests that mention environment/auth details often need one extra tool
    hop (e.g., token-aware retries), so they receive a small base bump.
    """
    base = _safe_int_env("IRIS_GENERAL_TOOL_MAX_TURNS", _GENERAL_TOOL_MAX_TURNS)
    cap = _safe_int_env("IRIS_GENERAL_TOOL_MAX_TURNS_CAP", _GENERAL_TOOL_MAX_TURNS_CAP)
    if cap < base:
        cap = base

    lowered = task.query.lower()
    continuationish = bool(
        re.search(r"\b(yes|yep|yeah|do that|continue|go ahead|as discussed)\b", lowered)
    )
    env_or_auth = bool(re.search(r"\b(env|\.env|token|pat|api key|auth)\b", lowered))
    if continuationish or env_or_auth:
        base = min(cap, base + 1)
    return base, cap


_REACT_TOOL_SYSTEM_SUFFIX = """
---
You have access to tools. When you need live data, calculations, or web search,
use a tool by outputting EXACTLY this structure (no fences, no prose around it):

Thought: <why you need the tool>
Action: <tool_name>
Action Input: {{"key": "value"}}

After you receive the tool result (shown as "Observation:"), continue:

Thought: <what you learned>
Final Answer: <your answer to the user>

--- Example ---
User: What are the trending repos on GitHub today?
Thought: I need to search the web for current GitHub trending repositories.
Action: research
Action Input: {{"query": "GitHub trending repositories today", "fetch_content": false}}
Observation: 1. microsoft/TypeScript ★98k  2. vercel/next.js ★120k ...
Thought: I have the trending repos.
Final Answer: Here are today's top trending GitHub repositories: ...
--- End Example ---

Available tools:
{tool_list}
"""


_REACT_ACTION_RE = re.compile(r"Action:\s*(.+?)(?:\n|$)", re.IGNORECASE)


_REACT_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.+?\})", re.DOTALL | re.IGNORECASE)


_REACT_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.+?)$", re.DOTALL | re.IGNORECASE)


# Patterns that suggest a model "wanted" to call a tool but produced text instead
# of a proper native function call — used to detect failed native attempts.
_TOOL_INTENT_RE = re.compile(
    r"\b(research|code_exec|I(?:'ll| will| need to| should) (?:search|look up|use))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GeneralInvoke:
    """What the two handlers need from this cluster."""

    get_client: Any
    build_prompts: Any
    invoke_with_tools: Any


def make_general_invoke(*, tier_router: TierRouter, general_tools: GeneralTools) -> GeneralInvoke:
    """Bind the prompt/client/loop trio to one tier router and tool set.

    ``_clients`` is created here, per call, which is what keeps the client memo
    per-runtime rather than per-process. The bodies below are unchanged from their nested
    originals.
    """
    # Imported here, not at module scope, and deliberately so. `_install_fake_client` in
    # tests/unit/test_runtime/test_general_tool_loop.py patches
    # `iris_harness.llm.client.CodingLLMClient` — the attribute on the *source* module —
    # which only takes effect for a consumer that resolves the name per call. Hoisting
    # these to module level binds them at import time and silently un-patches ten tests;
    # bootstrap carried the same local import with the same noqa marker, and this is why.
    from iris_harness.llm.client import (
        CodingLLMClient,
        CodingLLMConfig,
        LLMMessage,
        LLMToolCall,
    )
    from iris_harness.llm.tier_router import (
        ToolStrategy,
        tool_strategy_for_model,
    )

    _clients: dict[str, CodingLLMClient] = {}
    _general_tools = general_tools

    def _get_client(intent: str, preferred_model: str, provider_profile: str) -> CodingLLMClient:
        provider_profile = effective_provider_profile(provider_profile, preferred_model)
        # ``intent`` is part of the cache key because the same profile can map
        # to different models per intent (e.g. coding → profile.coding_model).
        cache_key = (
            f"{intent}:{provider_profile}:{preferred_model}"
            if (provider_profile and preferred_model)
            else (
                f"{intent}:{provider_profile}"
                if provider_profile
                else (f"{intent}:{preferred_model}" if preferred_model else intent)
            )
        )
        if cache_key not in _clients:
            if provider_profile:
                cfg = config_from_profile(provider_profile, intent=intent)
                if preferred_model:
                    # Keep provider's auth/connection settings, override just the model.
                    cfg = cfg.model_copy(update={"model": preferred_model})
            elif preferred_model:
                provider, base_url, api_key_env = infer_provider(preferred_model)
                cfg = CodingLLMConfig(
                    provider=provider,
                    model=preferred_model,
                    base_url=base_url,
                    api_key_env=api_key_env,
                    temperature=0.5,
                    max_tokens=4096,
                    timeout_seconds=60,
                    tier_name="preferred_model",
                )
            else:
                cfg = CodingLLMConfig(**vars(tier_router.get_llm_config(intent)))
            _clients[cache_key] = CodingLLMClient(cfg)
        return _clients[cache_key]

    def _build_prompts(
        task: AgentTask,
        *,
        recovery_hints: tuple[str, ...] = (),
    ) -> tuple[CodingLLMClient, str, str, int, ToolStrategy]:
        """Resolve client + system/user prompts + usage-mark + tool strategy for a task."""
        intent = task.params.get("intent", "general")
        # Compound (multi-step) turns escalate to the stronger tier-2 model.
        is_multi_step = bool(task.params.get("is_multi_step"))
        routing_intent = resolve_routing_intent(intent, is_multi_step)
        preferred_model = task.params.get("preferred_model", "")
        provider_profile = effective_provider_profile(
            task.params.get("provider_profile", ""),
            preferred_model,
        )

        ctx = task.memory_context
        profile = ""
        user_md = ctx.user_profile if ctx else None
        if user_md:
            profile = f"\n\n{user_md}"
        elif ctx and ctx.user_facts:
            # Mark low-confidence recalled facts so the model doesn't treat them as
            # authoritative (recall quality filter; curated USER.md above is never marked).
            lines = "\n".join(
                f"- {f.key}: {f.value}{' (unconfirmed)' if f.uncertain else ''}"
                for f in ctx.user_facts
            )
            profile = f"\n\nUser profile:\n{lines}"
        linked = f"\n\n{ctx.linked.strip()}" if ctx and ctx.linked else ""
        active = ""
        active_md = ctx.active if ctx else None
        if active_md:
            active = f"\n\n{active_md}"
        episodic = ""
        episodic_digest = ctx.episodic_digest if ctx else None
        if episodic_digest:
            episodic = f"\n\n{episodic_digest}"
        elif ctx and ctx.episodic_patterns:
            bullets = "\n".join(f"- {p}" for p in ctx.episodic_patterns)
            episodic = f"\n\n## What I've noticed about you\n{bullets}"
        behavior_block = ""
        if ctx and ctx.behavior:
            behavior_name = ctx.behavior_name or "matched"
            behavior_block = f"\n\n## Behavior - {behavior_name}\n\n{ctx.behavior}"

        history = ""
        if ctx and ctx.recent_turns:
            history = "\n\nConversation history:\n" + "\n".join(ctx.recent_turns)

        client = _get_client(routing_intent, preferred_model, provider_profile)
        mark = client.get_usage_mark()

        # Determine model + provider for strategy detection
        if provider_profile:
            try:
                cfg_preview = config_from_profile(provider_profile, intent=intent)
                resolved_model = preferred_model or cfg_preview.model
                resolved_provider = cfg_preview.provider
            except Exception:  # noqa: BLE001
                resolved_model = preferred_model or ""
                resolved_provider = ""
        elif preferred_model:
            resolved_provider, _, _ = infer_provider(preferred_model)
            resolved_model = preferred_model
        else:
            tier = tier_router.get_tier(routing_intent)
            resolved_model = tier.model
            resolved_provider = tier.provider

        strategy = tool_strategy_for_model(resolved_model, resolved_provider)

        system_prompt = _general_system_prompt(ctx.soul if ctx else None, message=task.query)
        system_prompt += profile + linked + active + episodic + behavior_block + history
        user_prompt = task.query
        if recovery_hints:
            user_prompt += "\n\n[Execution context]\n" + "\n".join(
                f"- {hint}" for hint in recovery_hints
            )

        # Cap the assembled prompt for local Ollama tiers so identity blobs and
        # history can never blow past the model's KV-cache budget. Cloud models
        # advertise huge context windows, so leave them untouched (num_ctx None).
        tier = tier_router.get_tier(intent)
        num_ctx = getattr(tier, "num_ctx", None)
        if num_ctx:
            budget = budget_for(num_ctx, fraction=0.75)
            user_tokens = estimate_tokens(user_prompt)
            sys_budget = max(0, budget - user_tokens)
            system_prompt = trim_text(system_prompt, max_chars=sys_budget * 4)

        return client, system_prompt, user_prompt, mark, strategy

    def _invoke_with_tools_react(
        task: AgentTask,
        client: CodingLLMClient,
        system_prompt: str,
        user_prompt: str,
        *,
        run_id: str,
    ) -> str:
        """ReAct text-based tool loop for local models without native function calling.

        Injects the tool list and few-shot examples into the system prompt, then
        parses Action / Action Input from text responses. Each observation is fed
        back as a plain-text continuation so even small models can follow the loop.
        """
        bindings = () if task.session_id == "__warmup__" else _general_tools.bindings(task.query)
        tool_list = "\n".join(
            f"- {b['function']['name']}: {b['function']['description']}" for b in bindings
        )
        augmented_system = system_prompt + _REACT_TOOL_SYSTEM_SUFFIX.format(tool_list=tool_list)

        messages = [
            LLMMessage(role="system", content=augmented_system),
            LLMMessage(role="user", content=user_prompt),
        ]

        budget, budget_cap = _general_tool_budget(task)
        turn = 0
        while turn < budget:
            turn += 1
            response = client.invoke_turn(messages=messages)
            raw = response.content or ""

            # Check for Final Answer first
            if m := _REACT_FINAL_ANSWER_RE.search(raw):
                return m.group(1).strip()

            # Try to parse Action / Action Input
            action_match = _REACT_ACTION_RE.search(raw)
            input_match = _REACT_ACTION_INPUT_RE.search(raw)

            if not action_match:
                # No tool call pattern found — treat as final response
                return raw.strip()

            tool_name = action_match.group(1).strip()
            tool_args: dict[str, Any] = {}
            if input_match:
                try:
                    tool_args = json.loads(input_match.group(1))
                except (json.JSONDecodeError, ValueError):
                    tool_args = {"query": input_match.group(1).strip()}

            # Execute the tool
            dummy_call = LLMToolCall(id=f"react_{turn}", name=tool_name, arguments=tool_args)
            result_payload = _general_tools.execute(task, dummy_call, run_id=run_id)
            observation = enrich_tool_result(tool_name, result_payload)
            if result_payload.get("ok") and budget < budget_cap:
                budget += 1

            # Append the assistant's ReAct turn + observation to the conversation
            messages.append(LLMMessage(role="assistant", content=raw))
            messages.append(LLMMessage(role="user", content=f"Observation: {observation}"))

        return "I tried to use the available tools but reached the iteration limit."

    def _invoke_with_tools_native(
        task: AgentTask,
        client: CodingLLMClient,
        system_prompt: str,
        user_prompt: str,
        *,
        run_id: str,
        with_fallback: bool = False,
    ) -> str:
        """Native function-calling loop.  If ``with_fallback`` is True and the
        first response contains no tool_calls but looks like a failed tool attempt,
        a corrective nudge is sent before continuing.
        """
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        tool_bindings = (
            () if task.session_id == "__warmup__" else _general_tools.bindings(task.query)
        )

        budget, budget_cap = _general_tool_budget(task)
        turn = 0
        while turn < budget:
            turn += 1
            response = client.invoke_turn(messages=messages, bound_tools=tool_bindings)

            if not response.tool_calls:
                # On the first native turn, check if the model "wanted" to call a
                # tool but failed to emit a function call (text like "I will search…").
                if with_fallback and turn == 1 and _TOOL_INTENT_RE.search(response.content or ""):
                    logger.debug(
                        "native tool call failed (model likely lacks function-calling support); "
                        "nudging with format correction"
                    )
                    messages.append(LLMMessage(role="assistant", content=response.content or ""))
                    messages.append(
                        LLMMessage(
                            role="user",
                            content=(
                                "Please call the tool using the proper JSON function-call format "
                                "rather than describing it in prose."
                            ),
                        )
                    )
                    continue
                return response.content

            messages.append(
                LLMMessage(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )
            for tool_call in response.tool_calls:
                result_payload = _general_tools.execute(task, tool_call, run_id=run_id)
                enriched = enrich_tool_result(tool_call.name, result_payload)
                if result_payload.get("ok") and budget < budget_cap:
                    budget += 1
                messages.append(
                    LLMMessage(
                        role="tool",
                        name=tool_call.name,
                        tool_call_id=tool_call.id,
                        content=enriched,
                    )
                )

        return "I tried to use the available tools, but reached the tool-call limit."

    def _invoke_with_tools(
        task: AgentTask,
        client: CodingLLMClient,
        system_prompt: str,
        user_prompt: str,
        strategy: ToolStrategy,
    ) -> str:
        # One run for the turn's tool loop, as the ReAct loop has one per run: every
        # governed tool call it makes is recorded under this id.
        run_id = str(uuid.uuid4())
        if strategy == ToolStrategy.REACT:
            return _invoke_with_tools_react(task, client, system_prompt, user_prompt, run_id=run_id)
        return _invoke_with_tools_native(
            task,
            client,
            system_prompt,
            user_prompt,
            run_id=run_id,
            with_fallback=(strategy == ToolStrategy.NATIVE_WITH_FALLBACK),
        )

    return GeneralInvoke(
        get_client=_get_client,
        build_prompts=_build_prompts,
        invoke_with_tools=_invoke_with_tools,
    )
