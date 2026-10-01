"""OpenAI-compatible LLM client shared by the harness.

``CodingLLMClient`` is the one governed chat client every tier uses; construct it
through ``tier_router.get_llm_config(intent)``. The name is historical (it was born
in the coding agent) and kept to avoid churn. The coding agent's repo-config loader
(``load_coding_llm_config``) lives in ``iris_harness.llm.client``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from iris_harness.foundation.observability.logging_setup import log_egress
from iris_harness.foundation.observability.session_log import llm_call_scope
from iris_harness.kernel.governance import (
    GovernanceKernel,
    HookContext,
    HookDecision,
    HookPoint,
    kernel_from_env,
)
from iris_harness.kernel.governance.hooks.types import LLMTier
from iris_harness.kernel.governance.turn_label import apply_turn_floor
from iris_harness.llm.arbiter import (
    CircuitBreakerOpenError,
    OllamaCircuitBreaker,
    get_ollama_breaker,
)
from iris_harness.llm.egress import egress_headers
from iris_harness.llm.fake import FAKE_PROVIDER
from iris_harness.llm.locality import provider_locality

logger = logging.getLogger(__name__)

_RATE_LIMIT_WAIT_SECONDS_RE = re.compile(r"wait\s+(\d+)\s+seconds", re.IGNORECASE)

# Providers that talk to a LOCAL model server — the only ones the circuit breaker
# applies to. Cloud providers (anthropic/openrouter/github) bypass it entirely.
_LOCAL_PROVIDERS = frozenset({"ollama", "lmstudio"})

# Providers whose reply a JSON schema constrains (Ollama's ``format``). The scripted fake
# plays that decoder (llm/fake.py), so a structured caller -- a judge, a classifier --
# runs on it exactly as it does on Ollama.
_JSON_SCHEMA_PROVIDERS = frozenset({"ollama", FAKE_PROVIDER})


def supports_json_schema(provider: str) -> bool:
    """Whether a call on ``provider`` can ask for a schema-constrained JSON reply
    (:meth:`CodingLLMClient.invoke_json`)."""
    return provider in _JSON_SCHEMA_PROVIDERS


# Per-call usage events a client keeps for span accounting. One turn makes a handful of
# calls; this only has to outlast the longest span between a mark and its read.
_USAGE_EVENTS_MAX = 4096


def _is_connection_error(exc: BaseException) -> bool:
    """True if ``exc`` (or its cause chain) is a connectivity/timeout failure.

    Used to decide what trips the breaker: only 'server is down/unreachable' faults,
    NOT model/validation/rate-limit errors. Walks ``__cause__``/``__context__`` because
    LangChain wraps the underlying httpx error, and falls back to type-name matching so
    we don't hard-depend on every provider SDK's exception classes.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (ConnectionError, TimeoutError, httpx.TransportError)):
            return True
        name = type(cur).__name__
        if any(
            marker in name
            for marker in ("ConnectError", "ConnectionError", "Timeout", "NetworkError")
        ):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


# Some models (e.g. Llama variants via GitHub Models) emit tool calls as XML text
# instead of using native function-calling.  Format:
#   <function=tool_name><parameter=key>value</parameter>...</function>
# optionally wrapped in <tool_call>...</tool_call>.
_XML_FUNC_CALL_RE = re.compile(
    r"(?:<tool_call>\s*)?<function=(\w+)>(.*?)</function>(?:\s*</tool_call>)?",
    re.DOTALL | re.IGNORECASE,
)
_XML_PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)


class CodingLLMInvocationError(ValueError):
    """User-facing error raised when the configured chat backend cannot complete a turn."""


class GovernanceBlockedError(CodingLLMInvocationError):
    """Governance refused the call before anything was sent; ``decision`` says who and why."""

    def __init__(self, decision: HookDecision) -> None:
        super().__init__(f"Request blocked by governance: {decision.reason}")
        self.decision = decision


class LLMUnreachable(RuntimeError):
    """The model server could not be reached: refused, timed out, or its breaker open.

    Raised by :meth:`CodingLLMClient.invoke_json` so a batch caller (a classifier
    working through a queue) can stop at once and try later, instead of waiting out a
    timeout per item. Nothing retries another endpoint.
    """


class LLMBadReply(RuntimeError):
    """The server answered, but twice with no JSON object (:meth:`CodingLLMClient.invoke_json`)."""


@dataclass(frozen=True)
class JsonReply:
    """One structured reply: the JSON object, the time every attempt took, the model."""

    data: dict[str, Any]
    latency_ms: int
    model: str


def _json_object(content: str) -> dict[str, Any] | None:
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class _SupportsInvoke(Protocol):
    """Protocol for a chat model implementation compatible with this bootstrap wrapper."""

    def invoke(self, messages: Sequence[dict[str, Any]]) -> object: ...


class _SupportsBindTools(Protocol):
    """Protocol for chat models that can bind OpenAI-style tool definitions."""

    def bind_tools(self, tools: Sequence[dict[str, Any]]) -> _SupportsInvoke: ...


class CodingLLMConfig(BaseModel):
    """Provider-neutral LLM configuration for the coding bootstrap."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    provider: str = Field(default="ollama", min_length=1)
    model: str = Field(default="qwen3-coder:30b", min_length=1)
    # Routing-tier key from llm_tiers.yaml (e.g. "tier2", "router") when the
    # config came from TierRouter.get_llm_config. Lets the client report the
    # *configured* governance tier instead of inferring from provider alone.
    tier_name: str | None = Field(default=None)
    # Governance's tier label for this config, stamped by the tier router from where the
    # tier's provider runs (llm/locality.py). Unset on a config that did not come from the
    # router; the client then reads the provider's declared locality itself.
    governance_tier: LLMTier | None = Field(default=None)
    base_url: str = Field(default="http://localhost:11434/v1", min_length=1)
    api_key_env: str | None = Field(default=None)
    temperature: float = Field(default=0.3, ge=0, le=2)
    max_tokens: int = Field(default=8192, ge=1)
    timeout_seconds: int = Field(default=120, ge=1)
    auth_mode: str = Field(default="static", min_length=1)
    default_headers: tuple[tuple[str, str], ...] = Field(default_factory=tuple)
    # Ollama-native tuning. Honored only when provider == "ollama" and the
    # native ChatOllama path is taken in _default_model_factory.
    num_ctx: int | None = Field(default=None, ge=128)
    keep_alive: str | int | None = Field(default=None)
    num_thread: int | None = Field(default=None, ge=1)
    # A reasoning model's thinking: False off, True on, None the model's default.
    # Reaches Ollama as ChatOllama(reasoning=...); other providers ignore it.
    think: bool | None = None

    def headers_map(self) -> dict[str, str]:
        """Return ``default_headers`` as a plain mutable mapping."""
        return dict(self.default_headers)


class LLMMessage(BaseModel):
    """Typed chat message payload used by the coding bootstrap."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: str = Field(..., min_length=1)
    content: str = Field(default="")
    name: str | None = Field(default=None)
    tool_call_id: str | None = Field(default=None)
    tool_calls: tuple[LLMToolCall, ...] = Field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        """Render the chat message to an OpenAI-compatible payload."""
        payload: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [tool_call.to_payload() for tool_call in self.tool_calls]
        return payload


class LLMToolCall(BaseModel):
    """One tool call requested by the chat model."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Render the tool call to an OpenAI-compatible assistant message entry."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, sort_keys=True, separators=(",", ":")),
            },
        }


class LLMTokenUsage(BaseModel):
    """Per-invocation token consumption captured from the provider response."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class LLMInvocationResponse(BaseModel):
    """Normalized model turn output for multi-turn coding runs."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    content: str = Field(default="")
    tool_calls: tuple[LLMToolCall, ...] = Field(default_factory=tuple)
    token_usage: LLMTokenUsage | None = Field(default=None)


PROVIDER_DEFAULTS: dict[str, CodingLLMConfig] = {
    "github": CodingLLMConfig(
        provider="github",
        model="gpt-4o",
        base_url="https://models.inference.ai.azure.com",
        api_key_env="GITHUB_TOKEN",
    ),
    "openrouter": CodingLLMConfig(
        provider="openrouter",
        model="~anthropic/claude-sonnet-latest",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    ),
    "anthropic": CodingLLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-5",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "ollama": CodingLLMConfig(
        provider="ollama",
        model="qwen3-coder:30b",
        base_url="http://localhost:11434/v1",
        api_key_env=None,
        max_tokens=4096,
        timeout_seconds=60,
    ),
    "lmstudio": CodingLLMConfig(
        provider="lmstudio",
        model="qwen2.5-coder-14b-instruct",
        base_url="http://localhost:1234/v1",
        api_key_env="LM_STUDIO_API_KEY",
        max_tokens=4096,
        timeout_seconds=120,
    ),
    "copilot": CodingLLMConfig(
        provider="copilot",
        model="gpt-4o",
        base_url="https://api.githubcopilot.com",
        api_key_env=None,
        auth_mode="copilot",
        default_headers=(
            ("Editor-Version", "iris-coding/0.1"),
            ("Copilot-Integration-Id", "vscode-chat"),
            ("Editor-Plugin-Version", "iris-coding/0.1"),
        ),
        max_tokens=8192,
        timeout_seconds=120,
    ),
}

ChatModelFactory = Callable[..., _SupportsInvoke]


def _default_model_factory(**kwargs: Any) -> _SupportsInvoke:
    """Build a chat model. Branch on provider so Ollama-native options reach the daemon.

    For ``provider == "ollama"`` we use ``langchain_ollama.ChatOllama`` so that
    ``num_ctx``, ``keep_alive`` and ``num_thread`` actually reach the Ollama
    daemon. The OpenAI-compat endpoint at ``:11434/v1`` silently drops them.

    ``provider == "fake"`` is the scripted fake (``llm/fake.py``): only the transport is
    a script; the client around it (governance, audit, egress log, spans) is unchanged.
    """
    provider = kwargs.pop("provider", None)
    if provider == FAKE_PROVIDER:
        from iris_harness.llm.fake import chat_model_factory

        return chat_model_factory(**kwargs)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = str(kwargs.get("base_url") or "").removesuffix("/v1")
        # ChatOpenAI uses ``max_tokens``; ChatOllama uses ``num_predict``.
        chat_kwargs: dict[str, Any] = {
            "model": kwargs["model"],
            "base_url": base_url,
            "temperature": kwargs.get("temperature"),
            "num_predict": kwargs.get("max_tokens"),
        }
        # ``format``: a JSON schema the whole reply must fit (CodingLLMClient.invoke_json).
        for opt in ("num_ctx", "keep_alive", "num_thread", "format"):
            if kwargs.get(opt) is not None:
                chat_kwargs[opt] = kwargs[opt]
        if kwargs.get("think") is not None:
            # ChatOllama's name for Ollama's `think` flag.
            chat_kwargs["reasoning"] = kwargs["think"]
        timeout = kwargs.get("timeout")
        if timeout is not None:
            chat_kwargs["timeout"] = timeout
        return ChatOllama(**{k: v for k, v in chat_kwargs.items() if v is not None})  # type: ignore[return-value]

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(**kwargs)  # type: ignore[return-value]


def resolve_coding_llm_api_key(
    config: CodingLLMConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve the configured API key from the environment when required.

    If the env value is a ``vault://<handle>`` reference, it is resolved
    against the local credential vault. Plain values pass through
    unchanged (Phase 2 migration is opt-in per env var).
    """
    if config.api_key_env is None:
        return None

    source = os.environ if environ is None else environ
    api_key = source.get(config.api_key_env)
    if api_key is None:
        return None
    stripped = api_key.strip()
    if not stripped:
        return None

    from iris_harness.kernel.governance.vault import (
        resolve_secret_value,  # avoid import cycle
    )

    return resolve_secret_value(stripped)


class ApiCredentialProvider(Protocol):
    """Protocol for supplying API tokens and extra headers to the LLM client."""

    def get_token(self) -> str | None: ...

    def extra_headers(self) -> Mapping[str, str]: ...


class StaticEnvCredentialProvider:
    """Credential provider backed by a static environment variable lookup."""

    def __init__(
        self,
        config: CodingLLMConfig,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._config = config
        self._environ = environ

    def get_token(self) -> str | None:
        return resolve_coding_llm_api_key(self._config, environ=self._environ)

    def extra_headers(self) -> Mapping[str, str]:
        return {}


CredentialProviderFactory = Callable[..., ApiCredentialProvider]


_CREDENTIAL_PROVIDER_FACTORIES: dict[str, CredentialProviderFactory] = {}


def register_credential_provider_factory(
    auth_mode: str,
    factory: CredentialProviderFactory,
) -> None:
    """Register a credential provider factory for a given ``auth_mode``."""
    normalized = auth_mode.strip().lower()
    if not normalized:
        raise ValueError("auth_mode must be a non-empty string")
    _CREDENTIAL_PROVIDER_FACTORIES[normalized] = factory


def build_credential_provider(
    config: CodingLLMConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> ApiCredentialProvider:
    """Build the credential provider matching ``config.auth_mode``."""
    mode = (config.auth_mode or "static").strip().lower()
    if mode == "static":
        return StaticEnvCredentialProvider(config, environ=environ)
    factory = _CREDENTIAL_PROVIDER_FACTORIES.get(mode)
    if factory is None:
        # Trigger lazy import so callers don't need to pre-register known modes.
        if mode == "copilot":
            from iris_harness.llm import copilot_auth as _copilot_auth  # noqa: F401

            factory = _CREDENTIAL_PROVIDER_FACTORIES.get(mode)
    if factory is None:
        raise ValueError(f"Unknown auth_mode {mode!r}")
    return factory(config, environ=environ)


class CodingLLMClient:
    """Minimal OpenAI-compatible chat wrapper for coding bootstrap personas."""

    def __init__(
        self,
        config: CodingLLMConfig,
        *,
        environ: Mapping[str, str] | None = None,
        model_factory: ChatModelFactory | None = None,
        credential_provider: ApiCredentialProvider | None = None,
        governance_kernel: GovernanceKernel | None = None,
        governance_target_tier: LLMTier | None = None,
        governance_agent_type: str = "coding",
        governance_handled_upstream: bool = False,
        governance_stripped_public_content: bool = False,
        circuit_breaker: OllamaCircuitBreaker | None = None,
    ) -> None:
        """``governance_handled_upstream=True`` disables the client-internal
        PRE_CLASSIFY/PRE_LLM_CALL hooks. ONLY for callers that demonstrably
        fire those hooks themselves before every invocation (AgenticCore's
        ReAct loop — enforced by the AST tests in tests/security). Without
        this, the ReAct path fired governance twice per prompt: 2× audit
        rows and a duplicate classifier run per loop step.

        ``governance_stripped_public_content=True`` declares that every prompt this
        client sends is public content with the user's stored personal data stripped
        upstream. It is the one exemption from the turn's label floor
        (``kernel/governance/turn_label.apply_turn_floor``): personal is read as
        internal, ``secret`` never lowered. Today only the cloud search-synthesis client
        (``runtime/handlers/react.py``) makes it; do not set it for anything else.
        """
        self.config = config
        self._environ = environ
        self._model_factory = model_factory or _default_model_factory
        self._circuit_breaker = (
            circuit_breaker if circuit_breaker is not None else get_ollama_breaker()
        )
        self._credential_provider = credential_provider or build_credential_provider(
            config,
            environ=environ,
        )
        self._governance_kernel = (
            None if governance_handled_upstream else (governance_kernel or kernel_from_env())
        )
        # Where this call goes, never what its tier is called: the router's stamp, else
        # the provider's declared locality (local -> tier_1, cloud or undeclared -> tier_3).
        self._governance_target_tier: LLMTier = (
            governance_target_tier
            or config.governance_tier
            or ("tier_1" if provider_locality(config.provider) == "local" else "tier_3")
        )
        self._governance_agent_type = governance_agent_type
        self._governance_stripped_public_content = governance_stripped_public_content
        self._cumulative_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
        }
        # Per-call usage, for exact span accounting (get_usage_mark / get_token_usage_since).
        # Bounded: a client lives as long as the server, and a span only ever needs the
        # calls made since its mark. ``_usage_dropped`` keeps marks absolute.
        self._usage_events: deque[LLMTokenUsage] = deque(maxlen=_USAGE_EVENTS_MAX)
        self._usage_dropped = 0

    def model_kwargs(self) -> dict[str, Any]:
        """Return the kwargs passed to the chat-model factory.

        ``provider`` is included so ``_default_model_factory`` can route Ollama
        traffic through ``ChatOllama`` (native API) instead of the OpenAI-compat
        endpoint, which silently drops Ollama-only options like ``num_ctx``.
        """
        kwargs: dict[str, Any] = {
            "provider": self.config.provider,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout_seconds,
        }
        if self.config.provider == "ollama":
            # Ollama-only options. ChatOpenAI forwards unknown keywords into the request,
            # and the OpenAI client rejects them ("Completions.create() got an unexpected
            # keyword argument 'num_ctx'"), so a tier moved to lmstudio (or any
            # OpenAI-style provider) with its num_ctx kept failed every call; the router
            # then fell back to keywords without a word (FoundationModels bake-off).
            for opt in ("num_ctx", "keep_alive", "num_thread"):
                value = getattr(self.config, opt, None)
                if value is not None:
                    kwargs[opt] = value
            if self.config.think is not None:
                kwargs["think"] = self.config.think
            # ChatOllama doesn't take api_key/default_headers; the factory strips
            # ``provider`` and only forwards ChatOllama-compatible kwargs.
            return kwargs

        api_key = self._credential_provider.get_token()
        # ChatOpenAI requires api_key even for keyless local providers (lmstudio).
        kwargs["api_key"] = api_key if api_key is not None else "ollama"
        headers = dict(self.config.headers_map())
        headers.update(self._credential_provider.extra_headers())
        # The failover proxy's data-free mark (llm/egress.py): only an opted-in call.
        # The model is built from these kwargs on every call, so the mark is per call.
        headers.update(egress_headers())
        if headers:
            kwargs["default_headers"] = headers
        return kwargs

    def build_messages(
        self, *, system_prompt: str, user_prompt: str
    ) -> tuple[LLMMessage, LLMMessage]:
        """Build the two-message prompt shape used by the bootstrap agent."""
        return (
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        )

    def get_token_usage(self) -> dict[str, int]:
        """Return a copy of the cumulative token usage counters for this client instance."""
        return dict(self._cumulative_usage)

    def get_usage_mark(self) -> int:
        """Return the current provider-usage event index for exact span accounting."""
        return self._usage_dropped + len(self._usage_events)

    def _record_usage(self, usage: LLMTokenUsage) -> None:
        if len(self._usage_events) == self._usage_events.maxlen:
            self._usage_dropped += 1  # the oldest event falls off as this one goes in
        self._usage_events.append(usage)

    def get_token_usage_since(self, mark: int) -> dict[str, int]:
        """Return the exact provider-reported usage accumulated since ``mark``."""
        end = self._usage_dropped + len(self._usage_events)
        if mark < 0 or mark > end:
            raise ValueError("usage mark is out of range")
        if mark < self._usage_dropped:
            raise ValueError(f"usage mark is older than the last {_USAGE_EVENTS_MAX} calls")

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        for usage in islice(self._usage_events, mark - self._usage_dropped, None):
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens
            total_tokens += usage.total_tokens

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "call_count": end - mark,
        }

    def _breaker_active(self) -> bool:
        """The circuit breaker applies only to local providers; killable via env."""
        if os.getenv("IRIS_OLLAMA_BREAKER", "1").strip().lower() in {"0", "false", "no", "off"}:
            return False
        return self.config.provider in _LOCAL_PROVIDERS

    def invoke(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stop: Sequence[str] | None = None,
    ) -> str:
        """Invoke the configured chat model and normalize the response content to text.

        ``stop`` is an optional list of strings the provider should treat
        as completion-stop tokens. Threaded through to the underlying
        ``model.invoke(...)`` call. Used by the ReAct loop to prevent the
        LLM from hallucinating fake ``\\nUser:`` / ``\\nObservation:``
        continuations after a Final Answer.
        """
        response = self.invoke_turn(
            messages=self.build_messages(system_prompt=system_prompt, user_prompt=user_prompt),
            stop=stop,
        )
        return response.content

    def _log_llm_egress(self, method: str) -> None:
        """Emit one egress-audit line for an outbound LLM call (host only, no key)."""
        from urllib.parse import urlparse

        dest = urlparse(self.config.base_url or "").netloc or (self.config.provider or "?")
        log_egress(
            destination=dest,
            method=method,
            kind="llm",
            purpose=f"{self.config.provider}/{self.config.model}",
            tier=getattr(self.config, "tier_name", "") or "-",
        )

    def invoke_stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> Iterator[str]:
        """Stream chat-model output as text chunks.

        Yields plain-text deltas as the provider produces them. Final usage
        metadata (when the provider includes it in the last chunk) is recorded
        on the client so the caller can read it via ``get_token_usage_since``.

        Errors normalized via ``_normalize_invocation_error`` so rate limits
        surface the same friendly message as the non-streaming ``invoke``.
        """
        messages = [
            m.to_payload()
            for m in self.build_messages(system_prompt=system_prompt, user_prompt=user_prompt)
        ]
        decision, _ = self._governance_pre_llm(
            prompt_text=_messages_to_text(messages),
            run_id=str(uuid.uuid4()),
        )
        if decision is not None and decision.outcome in ("deny", "require_approval"):
            raise GovernanceBlockedError(decision)

        # Fail fast on a known-down local server (cloud bypasses). See invoke_turn.
        breaker_active = self._breaker_active()
        if breaker_active:
            self._circuit_breaker.before_call(self.config.base_url)

        model = self._model_factory(**self.model_kwargs())
        log_input = [
            {"role": str(m.get("role", "")), "content": str(m.get("content", ""))} for m in messages
        ]
        last_usage: dict[str, Any] | None = None
        accumulated: list[str] = []
        with llm_call_scope(
            model=self.config.model,
            provider=self.config.provider,
            input_messages=log_input,
            tier=self.config.tier_name,
        ) as log_state:
            self._log_llm_egress("STREAM")
            stream_iter = model.stream(messages)  # type: ignore[attr-defined]
            try:
                for chunk in stream_iter:
                    content = getattr(chunk, "content", "")
                    if isinstance(content, str) and content:
                        accumulated.append(content)
                        yield content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                text = item.get("text")
                                if isinstance(text, str) and text:
                                    accumulated.append(text)
                                    yield text
                            elif isinstance(item, str) and item:
                                accumulated.append(item)
                                yield item
                    usage = getattr(chunk, "usage_metadata", None)
                    if isinstance(usage, dict) and any(usage.values()):
                        last_usage = usage
            except Exception as exc:
                if breaker_active and _is_connection_error(exc):
                    self._circuit_breaker.record_failure(self.config.base_url)
                log_state["output_text"] = "".join(accumulated)
                normalized_error = _normalize_invocation_error(exc, self.config)
                if normalized_error is not None:
                    raise normalized_error from exc
                raise
            else:
                if breaker_active:
                    self._circuit_breaker.record_success(self.config.base_url)
            finally:
                # Explicit close lets keep_alive=0 actually fire on Ollama; a leaked
                # SSE connection would hold the model "active" indefinitely.
                close = getattr(stream_iter, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:  # noqa: BLE001, S110
                        pass
            log_state["output_text"] = "".join(accumulated)
            if last_usage is not None:
                log_state["tokens"] = {
                    "prompt_tokens": int(
                        last_usage.get("input_tokens", 0) or last_usage.get("prompt_tokens", 0)
                    ),
                    "completion_tokens": int(
                        last_usage.get("output_tokens", 0) or last_usage.get("completion_tokens", 0)
                    ),
                    "total_tokens": int(last_usage.get("total_tokens", 0)),
                }

        if last_usage is None:
            return
        prompt = int(last_usage.get("input_tokens", 0) or last_usage.get("prompt_tokens", 0))
        completion = int(
            last_usage.get("output_tokens", 0) or last_usage.get("completion_tokens", 0)
        )
        total = int(last_usage.get("total_tokens", 0)) or (prompt + completion)
        if not (prompt or completion or total):
            return
        usage_event = LLMTokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )
        self._record_usage(usage_event)
        self._cumulative_usage["prompt_tokens"] += prompt
        self._cumulative_usage["completion_tokens"] += completion
        self._cumulative_usage["total_tokens"] += total
        self._cumulative_usage["call_count"] += 1

    def invoke_turn(
        self,
        *,
        messages: Sequence[LLMMessage],
        bound_tools: Sequence[dict[str, Any]] = (),
        stop: Sequence[str] | None = None,
        json_schema: Mapping[str, Any] | None = None,
        think: bool | None = None,
    ) -> LLMInvocationResponse:
        """Invoke one model turn and normalize text plus any requested tool calls.

        ``stop``: optional completion-stop tokens. Forwarded as ``stop=``
        to LangChain's ``model.invoke(...)``. Both ChatOllama and
        ChatOpenAI accept the kwarg; providers without ``stop`` support
        will ignore it silently.

        ``json_schema``: the reply must be one JSON object fitting it (Ollama's
        ``format``). ``think`` overrides the config's reasoning flag for this call.
        Both are Ollama-only (the scripted fake also takes ``json_schema``); a
        ``json_schema`` on another provider is a ``ValueError``, raised before anything
        is sent. Callers want :meth:`invoke_json`.
        """
        if json_schema is not None and not supports_json_schema(self.config.provider):
            raise ValueError(
                f"a JSON-schema reply needs provider 'ollama' (or the scripted 'fake'); "
                f"this tier is on {self.config.provider!r}"
            )
        payload_messages = [message.to_payload() for message in messages]
        decision, _ = self._governance_pre_llm(
            prompt_text=_messages_to_text(payload_messages),
            run_id=str(uuid.uuid4()),
        )
        if decision is not None and decision.outcome in ("deny", "require_approval"):
            raise GovernanceBlockedError(decision)

        # Fail fast if the local model server has been seen down (skips the network
        # call + its 60-120s timeout). Cloud providers bypass.
        breaker_active = self._breaker_active()
        if breaker_active:
            self._circuit_breaker.before_call(self.config.base_url)

        factory_kwargs = self.model_kwargs()
        if json_schema is not None:
            factory_kwargs["format"] = dict(json_schema)
        if think is not None and self.config.provider == "ollama":
            factory_kwargs["think"] = think
        model = self._model_factory(**factory_kwargs)
        if bound_tools and hasattr(model, "bind_tools"):
            model = model.bind_tools(list(bound_tools))
        log_input = [
            {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
            for m in payload_messages
        ]
        invoke_kwargs: dict[str, Any] = {}
        if stop:
            invoke_kwargs["stop"] = list(stop)
        with (
            _llm_invoke_span(self) as span,
            llm_call_scope(
                model=self.config.model,
                provider=self.config.provider,
                input_messages=log_input,
                tier=self.config.tier_name,
            ) as log_state,
        ):
            self._log_llm_egress("POST")
            try:
                response = model.invoke(payload_messages, **invoke_kwargs)
            except Exception as exc:
                if breaker_active and _is_connection_error(exc):
                    self._circuit_breaker.record_failure(self.config.base_url)
                normalized_error = _normalize_invocation_error(exc, self.config)
                if normalized_error is not None:
                    raise normalized_error from exc
                raise
            if breaker_active:
                self._circuit_breaker.record_success(self.config.base_url)
            result = _coerce_invocation_response(response)
            log_state["output_text"] = result.content
            if result.tool_calls:
                log_state["tool_calls"] = [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in result.tool_calls
                ]
            if result.token_usage is not None:
                log_state["tokens"] = {
                    "prompt_tokens": result.token_usage.prompt_tokens,
                    "completion_tokens": result.token_usage.completion_tokens,
                    "total_tokens": result.token_usage.total_tokens,
                }
                _set_span_token_attributes(span, result.token_usage)
        if result.token_usage is not None:
            self._record_usage(result.token_usage)
            self._cumulative_usage["prompt_tokens"] += result.token_usage.prompt_tokens
            self._cumulative_usage["completion_tokens"] += result.token_usage.completion_tokens
            self._cumulative_usage["total_tokens"] += result.token_usage.total_tokens
            self._cumulative_usage["call_count"] += 1
        return result

    def invoke_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
        think: bool | None = False,
    ) -> JsonReply:
        """Ask for one JSON object matching ``schema``: the governed structured call.

        For a caller that needs a fixed-shape answer (a classifier, a judge), not a
        conversation. Each attempt is an ordinary :meth:`invoke_turn`, so the
        governance hooks, the audit row, the egress log and the circuit breaker all
        apply. ``think`` defaults to off: a reasoning model otherwise spends its budget
        thinking and may return empty content.

        One retry when the reply is not a JSON object; a second miss raises
        :class:`LLMBadReply`. A connection failure, a timeout or an open breaker raises
        :class:`LLMUnreachable`. A governance refusal is the usual
        :class:`CodingLLMInvocationError`. ``latency_ms`` covers every attempt.
        Ollama or the scripted fake only (``ValueError`` otherwise).
        """
        messages = self.build_messages(system_prompt=system_prompt, user_prompt=user_prompt)
        started = time.monotonic()
        last_content = ""
        for _attempt in range(2):
            try:
                response = self.invoke_turn(messages=messages, json_schema=schema, think=think)
            except CircuitBreakerOpenError as exc:
                raise LLMUnreachable(str(exc)) from exc
            except Exception as exc:
                if _is_connection_error(exc):
                    raise LLMUnreachable(
                        f"{self.config.base_url}: {type(exc).__name__}: {exc}"
                    ) from exc
                raise
            last_content = response.content
            data = _json_object(last_content)
            if data is not None:
                latency_ms = int((time.monotonic() - started) * 1000)
                return JsonReply(data=data, latency_ms=latency_ms, model=self.config.model)
        raise LLMBadReply(f"no JSON object after a retry: {last_content[:200]!r}")

    def _governance_pre_llm(
        self,
        *,
        prompt_text: str,
        run_id: str,
    ) -> tuple[HookDecision | None, str]:
        if self._governance_kernel is None:
            return None, prompt_text

        classify_ctx = HookContext(
            hook_point=HookPoint.PRE_CLASSIFY,
            run_id=run_id,
            agent_type=self._governance_agent_type,
            payload={"prompt": prompt_text},
        )
        classify_decision, classified_ctx = self._governance_kernel.fire_sync(
            HookPoint.PRE_CLASSIFY, classify_ctx
        )
        if classify_decision.outcome in ("deny", "require_approval"):
            return classify_decision, prompt_text

        # Every model call in a turn is governed at least as high as the turn's label (what
        # the user said, lifted by every governed call's result so far), not only by what
        # this prompt classifies as -- a planner, judge or narrator prompt can look tamer
        # than the data the turn holds. Outside a turn this is the prompt's own label.
        #
        # The one exemption is declared, not inferred: cloud search synthesis carries
        # public web content with the user's stored personal data stripped upstream, so a
        # "personal" hit there -- its prompt's or the turn's -- is PII inside the fetched
        # public content or the user's own search query, and is read as internal for the
        # egress gate. "secret" (real credentials) is NEVER lowered and still hard-blocks.
        classification = apply_turn_floor(
            classified_ctx.classification,
            stripped_public_content=self._governance_stripped_public_content,
        )
        if (
            self._governance_stripped_public_content
            and apply_turn_floor(classified_ctx.classification) == "personal"
        ):
            logger.info(
                "search synthesis: downgrading personal->internal for cloud egress "
                "(user data stripped; remaining is public web content)"
            )

        llm_ctx = HookContext(
            hook_point=HookPoint.PRE_LLM_CALL,
            run_id=run_id,
            agent_type=self._governance_agent_type,
            classification=classification,
            tier=self._governance_target_tier,
            payload={
                "prompt": prompt_text,
                "model": self.config.model,
                "provider": self.config.provider,
            },
        )
        decision, final_ctx = self._governance_kernel.fire_sync(HookPoint.PRE_LLM_CALL, llm_ctx)
        final_prompt = final_ctx.payload.get("prompt", prompt_text)
        return decision, final_prompt if isinstance(final_prompt, str) else prompt_text


class GovernedPromptCall:
    """A ``prompt -> text`` call through a governed :class:`CodingLLMClient`.

    For the components that take a bare ``llm_call`` (the conversation compactor, the task
    planner, the entity extractor, the keyword router's LLM fallback). Handed an opaque
    callable they fire ``PRE_CLASSIFY`` + ``PRE_LLM_CALL`` themselves, and cannot know
    where it goes. Handed one of these they fire nothing: the client below fires both at
    the tier the call actually goes to (the router's stamp, ``llm/locality.py``), floored
    at the turn's label. Firing again around it only added a second audit row and a
    second classifier run under a guessed tier.

    ``config_for`` is read per call, so a tier edit or a resource-governor downshift is
    the one the next call runs (and is governed) on. ``agent_type`` names the caller on
    the audit row, as the caller's own hooks did.
    """

    def __init__(
        self,
        config_for: Callable[[], CodingLLMConfig],
        *,
        agent_type: str,
        system_prompt: str = "",
    ) -> None:
        self._config_for = config_for
        self._agent_type = agent_type
        self._system_prompt = system_prompt

    def __call__(self, prompt: str) -> str:
        client = CodingLLMClient(self._config_for(), governance_agent_type=self._agent_type)
        return str(client.invoke(system_prompt=self._system_prompt, user_prompt=prompt))


@contextmanager
def _llm_invoke_span(client: CodingLLMClient) -> Iterator[Any]:
    """Span around one model invocation, via the global OTel tracer.

    Uses the globally registered tracer provider (set by Phoenix bootstrap)
    so no per-client wiring is needed; when tracing is off this yields a
    non-recording span at negligible cost. Made current so it parents under
    the active pipeline-stage span.
    """
    try:
        from opentelemetry import trace as otel_trace
    except ImportError:
        yield None
        return
    tracer = otel_trace.get_tracer("iris_harness.llm")
    with tracer.start_as_current_span("llm.invoke") as span:
        try:
            span.set_attribute("llm.model", client.config.model)
            span.set_attribute("llm.provider", client.config.provider)
            if client.config.tier_name is not None:
                span.set_attribute("llm.tier_name", client.config.tier_name)
            span.set_attribute("llm.governance_tier", client._governance_target_tier)
            span.set_attribute("llm.agent_type", client._governance_agent_type)
        except Exception:  # noqa: BLE001, S110 - tracing must never break the call
            pass
        yield span


def _set_span_token_attributes(span: Any, usage: LLMTokenUsage) -> None:
    """Record provider-reported token usage on the invocation span."""
    if span is None:
        return
    try:
        span.set_attribute("llm.token_count.prompt", usage.prompt_tokens)
        span.set_attribute("llm.token_count.completion", usage.completion_tokens)
        span.set_attribute("llm.token_count.total", usage.total_tokens)
    except Exception:  # noqa: BLE001, S110
        pass


def _messages_to_text(messages: Sequence[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = message.get("content")
        if isinstance(content, str):
            parts.append(f"{role}: {content}")
        elif isinstance(content, list):
            rendered: list[str] = []
            for item in content:
                if isinstance(item, str):
                    rendered.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        rendered.append(text)
            if rendered:
                parts.append(f"{role}: {' '.join(rendered)}")
    return "\n".join(parts).strip()


def _normalize_invocation_error(
    exc: Exception,
    config: CodingLLMConfig,
) -> CodingLLMInvocationError | None:
    """Convert provider errors that users can act on into stable CLI-facing failures."""
    if not _is_rate_limit_error(exc):
        return None

    retry_after_seconds = _extract_retry_after_seconds(exc)
    retry_after_label = _format_retry_after(retry_after_seconds)
    message = f"LLM rate limit reached for provider '{config.provider}' model '{config.model}'."
    if retry_after_seconds is not None:
        message += f" Retry after {retry_after_seconds} seconds ({retry_after_label})."

    message += " Switch to another configured backend in the coding pipeline config or wait for the quota window to reset."
    if config.provider == "github":
        message += " GitHub Models applies per-user daily quotas for some models."
    return CodingLLMInvocationError(message)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return whether the provider failure is a rate-limit rejection."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code == 429:
        return True

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int) and response_status == 429:
        return True

    normalized = str(exc).lower()
    return "rate limit" in normalized or "ratelimit" in normalized or "429" in normalized


def _extract_retry_after_seconds(exc: Exception) -> int | None:
    """Best-effort parse of a provider retry-after hint from headers or message text."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        getter = getattr(headers, "get", None)
        if callable(getter):
            retry_after = getter("retry-after") or getter("Retry-After")
            try:
                if retry_after is not None:
                    return int(str(retry_after).strip())
            except ValueError:
                pass

    match = _RATE_LIMIT_WAIT_SECONDS_RE.search(str(exc))
    if match is None:
        return None
    return int(match.group(1))


def _format_retry_after(seconds: int | None) -> str:
    """Render a compact duration for retry-after guidance."""
    if seconds is None or seconds <= 0:
        return "unknown"

    remaining = seconds
    hours, remaining = divmod(remaining, 3600)
    minutes, remaining = divmod(remaining, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if remaining or not parts:
        parts.append(f"{remaining}s")
    return " ".join(parts)


def _coerce_response_text(response: object) -> str:
    """Normalize a chat model response object into a plain string."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        normalized = "\n".join(part for part in parts if part.strip()).strip()
        if normalized:
            return normalized
    return str(content).strip()


def _extract_token_usage(response: object) -> LLMTokenUsage | None:
    """Extract token usage from a LangChain AIMessage response object."""
    # LangChain ≥0.3: usage_metadata attribute
    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        prompt = int(
            usage_metadata.get("input_tokens", 0) or usage_metadata.get("prompt_tokens", 0)
        )
        completion = int(
            usage_metadata.get("output_tokens", 0) or usage_metadata.get("completion_tokens", 0)
        )
        total = int(usage_metadata.get("total_tokens", 0)) or (prompt + completion)
        if prompt or completion or total:
            return LLMTokenUsage(
                prompt_tokens=prompt, completion_tokens=completion, total_tokens=total
            )

    # LangChain response_metadata (older convention)
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
        if isinstance(token_usage, dict):
            prompt = int(token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0))
            completion = int(
                token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0)
            )
            total = int(token_usage.get("total_tokens", 0)) or (prompt + completion)
            if prompt or completion or total:
                return LLMTokenUsage(
                    prompt_tokens=prompt, completion_tokens=completion, total_tokens=total
                )

    return None


# Thinking-by-default models (qwen3.x family, incl. the Tier-3
# qwen3.6:35b-a3b MoE) emit <think>…</think> blocks; older client libs
# inline them into content, and a truncated generation may never close the
# tag. Thinking must never reach user-facing content or downstream parsers
# (exp-003 + exp-004 spike 4 recommendations).
_THINK_BLOCK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


def _coerce_invocation_response(response: object) -> LLMInvocationResponse:
    """Normalize a chat model response into text plus tool calls."""
    content = _THINK_BLOCK_RE.sub("", _coerce_response_text(response)).strip()
    tool_calls = _coerce_tool_calls(response)

    # Fallback: some models (e.g. Llama variants) emit tool calls as XML text
    # rather than native function-calling structs.  Detect, parse, and strip them.
    if not tool_calls and _XML_FUNC_CALL_RE.search(content):
        xml_calls: list[LLMToolCall] = []
        for idx, m in enumerate(_XML_FUNC_CALL_RE.finditer(content), start=1):
            tool_name = m.group(1).strip()
            params_block = m.group(2)
            arguments = {k.strip(): v.strip() for k, v in _XML_PARAM_RE.findall(params_block)}
            xml_calls.append(LLMToolCall(id=f"xml-{idx}", name=tool_name, arguments=arguments))
        if xml_calls:
            tool_calls = tuple(xml_calls)
            content = _XML_FUNC_CALL_RE.sub("", content).strip()

    return LLMInvocationResponse(
        content=content,
        tool_calls=tool_calls,
        token_usage=_extract_token_usage(response),
    )


def _coerce_tool_calls(response: object) -> tuple[LLMToolCall, ...]:
    """Extract provider tool-call payloads into a stable internal shape."""
    raw_tool_calls = getattr(response, "tool_calls", None)
    if raw_tool_calls is None:
        raw_tool_calls = getattr(response, "additional_kwargs", {}).get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return ()

    tool_calls: list[LLMToolCall] = []
    for index, raw_tool_call in enumerate(raw_tool_calls, start=1):
        if not isinstance(raw_tool_call, dict):
            continue
        function_payload = raw_tool_call.get("function")
        if not isinstance(function_payload, dict):
            function_payload = {}

        tool_name = raw_tool_call.get("name") or function_payload.get("name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue

        raw_arguments = (
            raw_tool_call.get("args")
            if raw_tool_call.get("args") is not None
            else raw_tool_call.get("arguments")
        )
        if raw_arguments is None:
            raw_arguments = function_payload.get("arguments", {})

        arguments = _coerce_tool_arguments(raw_arguments)
        tool_calls.append(
            LLMToolCall(
                id=str(raw_tool_call.get("id") or f"tool-call-{index}"),
                name=tool_name,
                arguments=arguments,
            )
        )
    return tuple(tool_calls)


def _coerce_tool_arguments(raw_arguments: object) -> dict[str, Any]:
    """Normalize tool arguments from provider payloads."""
    if isinstance(raw_arguments, dict):
        return dict(raw_arguments)
    if isinstance(raw_arguments, str):
        loaded = json.loads(raw_arguments or "{}")
        if isinstance(loaded, dict):
            return loaded
        raise ValueError("tool arguments JSON must decode to a mapping")
    if raw_arguments is None:
        return {}
    raise ValueError("tool arguments must be a mapping or JSON object string")


LLMMessage.model_rebuild()
