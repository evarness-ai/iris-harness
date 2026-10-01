"""Maps intents to LLM tiers via llm_tiers.yaml use_for tags."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from iris_harness.foundation.settings.store import SettingsStore
    from iris_harness.kernel.governance import LLMTier

import yaml

from iris_harness.llm.locality import (
    Locality,
    declared_localities,
    parse_provider_localities,
    provider_locality,
)

logger = logging.getLogger(__name__)

# Put every tier on one declared provider (``TierRouter._force_provider``).
FORCED_PROVIDER_ENV = "IRIS_LLM_PROVIDER"


class ToolStrategy(StrEnum):
    """How a model should receive and emit tool calls."""

    NATIVE = "native"
    # Cloud models and well-trained local models: use OpenAI-style function calling.

    NATIVE_WITH_FALLBACK = "native_with_fallback"
    # Local models that claim tool-calling support: try native first, fall back to
    # ReAct text format if the model produces no tool_calls in the first turn.

    REACT = "react"
    # Small local models without reliable function-calling: use ReAct
    # (Thought / Action / Action Input / Observation) text format only.


class ModelTier(StrEnum):
    """How much deterministic scaffolding the planner should give a model.

    Orthogonal to :class:`ToolStrategy`:

    * ``ToolStrategy`` answers *how* tool calls are exchanged.
    * ``ModelTier`` answers *how much hand-holding* the planner needs around
      the tool call — task-spec rendering, verifier strictness, ``ask_user``
      gating, and prompt shape.
    """

    SMALL = "small"
    # Local non-reasoning models. Need a deterministic task spec rendered into
    # a tight slot-filled system prompt; verifier is strict; ``ask_user`` is
    # blocked unless explicitly allowed.

    MID = "mid"
    # Cloud or local tool-capable models without large-context reasoning. Get
    # the rich system prompt PLUS a structured REQUIREMENTS block from the
    # spec; verifier still runs.

    LARGE = "large"
    # Capable reasoning models with large context. Get the existing rich
    # prompt unchanged; the spec is used only as a cheap post-hoc verifier.


_LOCAL_PROVIDERS = frozenset({"ollama", "lmstudio"})
_LARGE_CONTEXT_THRESHOLD = 100_000


def tool_strategy_for_model(model: str, provider: str) -> ToolStrategy:
    """Return the best tool-calling strategy for a given model + provider pair.

    Decision logic:
      - Cloud providers → NATIVE (they handle function calling reliably).
      - Local providers with a known tool_calling=True metadata entry → NATIVE_WITH_FALLBACK.
      - Everything else local → REACT (text-based ReAct loop, most reliable for small models).
    """
    if provider not in _LOCAL_PROVIDERS:
        return ToolStrategy.NATIVE
    try:
        from iris_harness.llm.model_metadata import get_metadata

        meta = get_metadata(model)
        if meta and meta.tool_calling:
            return ToolStrategy.NATIVE_WITH_FALLBACK
    except Exception:  # noqa: BLE001, S110
        pass
    return ToolStrategy.REACT


def model_tier_for(model: str, provider: str) -> ModelTier:
    """Classify a model into a :class:`ModelTier` for planner scaffolding.

    Decision logic (cheap, deterministic, no LLM):

    * Local provider AND not a reasoning model  → SMALL
    * Reasoning + tool-calling + ≥100k context  → LARGE
    * Everything else                           → MID

    When metadata is missing we err on the safe side: local → SMALL, cloud →
    MID. Promote a model by adding it to MODEL_REGISTRY or by overriding in
    ``~/.iris/model_metadata.json``.
    """
    try:
        from iris_harness.llm.model_metadata import get_metadata

        meta = get_metadata(model)
    except Exception:  # noqa: BLE001
        meta = None

    is_local = provider in _LOCAL_PROVIDERS
    if meta is None:
        return ModelTier.SMALL if is_local else ModelTier.MID

    if is_local and not meta.reasoning:
        return ModelTier.SMALL
    if meta.reasoning and meta.tool_calling and meta.context_window >= _LARGE_CONTEXT_THRESHOLD:
        return ModelTier.LARGE
    return ModelTier.MID


def _ollama_base_url() -> str:
    """Resolve the Ollama base URL, honoring OLLAMA_BASE_URL when set.

    Users can point at a remote Ollama box without code changes by exporting
    ``OLLAMA_BASE_URL`` (with or without a trailing ``/v1``).
    """
    raw = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    return raw if raw.endswith("/v1") else f"{raw}/v1"


def _lmstudio_base_url() -> str:
    """Resolve the LM Studio base URL, honoring LM_STUDIO_BASE_URL when set.

    LM Studio defaults to ``http://localhost:1234`` and exposes an OpenAI-compatible
    API at ``/v1``. Users running a remote LM Studio box can override via env.
    """
    raw = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234").rstrip("/")
    return raw if raw.endswith("/v1") else f"{raw}/v1"


# The local providers' URLs come from the environment, so they are resolved when a
# config is built, not when this module is imported: a value set after import (a
# test's monkeypatch, a settings reload) must be the one the next client dials.
_ENV_BASE_URLS: dict[str, Callable[[], str]] = {
    "ollama": _ollama_base_url,
    "lmstudio": _lmstudio_base_url,
}

_FIXED_BASE_URLS: dict[str, str] = {
    # The scripted fake (llm/fake.py) dials nothing; its URL only names it in the logs.
    "fake": "fake://scripted",
    "anthropic": "https://api.anthropic.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "github": "https://models.inference.ai.azure.com",
}


def _provider_base_url(provider: str) -> str:
    """The base URL for ``provider``, read now; unknown providers fall back to Ollama."""
    resolver = _ENV_BASE_URLS.get(provider)
    if resolver is not None:
        return resolver()
    fixed = _FIXED_BASE_URLS.get(provider)
    return fixed if fixed is not None else _ollama_base_url()


def provider_root_url(provider: str) -> str:
    """The provider's server root (no ``/v1``), read now — for a reachability probe
    (``/api/version`` on Ollama) that must not load a model or touch the governor."""
    url = _provider_base_url(provider).rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


_PROVIDER_API_KEY_ENVS: dict[str, str | None] = {
    "ollama": None,
    "lmstudio": "LM_STUDIO_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "github": "GITHUB_TOKEN",
}


@dataclass(frozen=True)
class TierConfig:
    name: str
    provider: str
    model: str
    max_tokens: int
    temperature: float
    timeout_seconds: int
    use_for: tuple[str, ...]
    # Ollama-native tuning. Only honored when provider == "ollama".
    # ``num_ctx`` caps KV-cache growth; ``keep_alive`` controls how long the
    # model stays resident after the call (``"0"`` / ``0`` = unload immediately).
    num_ctx: int | None = None
    keep_alive: str | int | None = None
    num_thread: int | None = None
    # ``think``: a reasoning model's thinking (qwen3.x, ...). ``False`` turns it off,
    # ``True`` on; ``None`` (unset) leaves the model's default. Ollama only.
    think: bool | None = None
    # ``pinned: true`` in the YAML: this tier always runs its own model. The resource
    # governor never downshifts it and a learned start-tier prior never moves its
    # intents elsewhere — for a caller whose output is only comparable run to run
    # (a classifier's accuracy) or that must stay on one route.
    pinned: bool = False


class TierRouterService(Protocol):
    """The tier-router surface a plugin may call (``HarnessServices.tier_router``).

    Only the lookups plugins use today -- directly, or through ``make_narrative_llm_call``
    and ``governance_tier_for_intent``, which take this rather than ``TierRouter`` so
    the router a plugin was handed passes without a cast. Defined here, not beside
    ``HarnessServices``, because those two helpers live in this layer; the runtime and
    ``iris_harness.sdk.services`` re-export it.
    """

    def get_llm_config(self, intent: str) -> object:
        """A ``CodingLLMConfig`` for the tier matching ``intent``."""
        ...

    def get_tier(self, intent: str) -> TierConfig:
        """The tier ``intent`` runs on."""
        ...

    def get_tier_by_name(self, tier_name: str) -> TierConfig | None:
        """The tier configured under ``tier_name``, or None."""
        ...


@dataclass
class TierRouter:
    """Routes intents to the right LLM tier and returns a CodingLLMConfig."""

    _tiers: dict[str, TierConfig] = field(default_factory=dict)
    _intent_to_tier: dict[str, str] = field(default_factory=dict)
    # Learned start-tier priors (ADR-0068 L5): intents that reliably escalate get
    # a higher START tier so routing is predictive, not reactive. Empty until the
    # escalation-priors refresh applies recommendations (opt-in); a prior wins
    # over the static use_for mapping but the resource governor can still
    # downshift it under host pressure (D6).
    _intent_tier_priors: dict[str, str] = field(default_factory=dict)
    # Optional governor that pre-empts Ollama model evictions and (when its
    # ``adaptive`` flag is set) downshifts large local tiers under host
    # pressure. Production wires one in via ``IrisRuntime``; tests leave None.
    governor: object | None = None
    # ADR-0120: the file's own tiers and intent map (the defaults an owner's edit is
    # compared with and a reset returns to); the live ones above carry the edits.
    _declared_tiers: dict[str, TierConfig] = field(default_factory=dict)
    _declared_intents: dict[str, str] = field(default_factory=dict)
    # Where each provider runs (the file's ``providers:`` block, llm/locality.py). Empty
    # when the file declares none: the config dir's declarations answer then.
    _provider_localities: dict[str, Locality] = field(default_factory=dict)

    @classmethod
    def load_from_yaml(cls, path: Path, *, settings: SettingsStore | None = None) -> TierRouter:
        """Load ``llm_tiers.yaml`` and lay the owner's saved edits over it (ADR-0120).

        ``settings`` defaults to the store under the data dir, so every caller that
        loads the tiers — the runtime, the agent console, a helper — sees the edits.
        """
        router = cls()
        try:
            raw = yaml.safe_load(path.read_text())
            router._provider_localities = parse_provider_localities((raw or {}).get("providers"))
            tiers_raw = (raw or {}).get("tiers", {})
            for tier_name, cfg in tiers_raw.items():
                use_for = tuple(cfg.get("use_for") or [])
                num_ctx = cfg.get("num_ctx")
                num_thread = cfg.get("num_thread")
                think = cfg.get("think")
                if think is not None and not isinstance(think, bool):
                    logger.warning(
                        "tier %s: think must be true or false; ignoring %r", tier_name, think
                    )
                    think = None
                tier = TierConfig(
                    name=cfg.get("name", tier_name),
                    provider=cfg.get("provider", "ollama"),
                    model=cfg.get("model", "llama3.2:3b"),
                    max_tokens=int(cfg.get("max_tokens", 2048)),
                    temperature=float(cfg.get("temperature", 0.5)),
                    timeout_seconds=int(cfg.get("timeout_seconds", 30)),
                    use_for=use_for,
                    num_ctx=int(num_ctx) if num_ctx is not None else None,
                    keep_alive=cfg.get("keep_alive"),
                    num_thread=int(num_thread) if num_thread is not None else None,
                    think=think,
                    pinned=bool(cfg.get("pinned", False)),
                )
                router._tiers[tier_name] = tier
                for tag in use_for:
                    router._intent_to_tier.setdefault(tag, tier_name)
        except Exception:
            logger.exception("failed to load llm_tiers.yaml from %s; using defaults", path)
        router._declared_tiers = dict(router._tiers)
        router._declared_intents = dict(router._intent_to_tier)
        from iris_harness.foundation.settings.store import SettingsStore
        from iris_harness.llm.tier_edits import apply_saved

        apply_saved(router, settings or SettingsStore())
        router._force_provider(os.environ.get(FORCED_PROVIDER_ENV, ""))
        return router

    def _force_provider(self, raw: str) -> None:
        """``IRIS_LLM_PROVIDER``: every tier runs on this provider (its model, tuning and
        intents unchanged). The provider must be declared in the ``providers:`` block, so
        where it runs -- and so how every call on it is governed -- is on file; an
        undeclared name is ignored with a warning. For ``fake`` (the scripted model,
        llm/fake.py) this runs the whole harness offline and deterministically.
        """
        provider = raw.strip().lower()
        if not provider:
            return
        if provider not in self.provider_localities():
            logger.warning(
                "%s=%r names a provider llm_tiers.yaml does not declare; ignoring it",
                FORCED_PROVIDER_ENV,
                raw,
            )
            return
        self._tiers = {name: replace(tier, provider=provider) for name, tier in self._tiers.items()}

    def _resolve_tier_name(self, intent: str) -> str:
        """Tier for an intent: a learned escalation prior wins, else the static map."""
        static = self._intent_to_tier.get(intent)
        static_tier = self._tiers.get(static) if static is not None else None
        if static_tier is not None and static_tier.pinned:
            return str(static)
        prior = self._intent_tier_priors.get(intent)
        if prior is not None and prior in self._tiers:
            return prior
        return self._intent_to_tier.get(intent, "fallback")

    def set_intent_tier_priors(self, priors: Mapping[str, str]) -> None:
        """Replace the learned start-tier priors (L5). Only known tiers are kept."""
        self._intent_tier_priors = {i: t for i, t in priors.items() if t in self._tiers}

    def intent_tier_priors(self) -> dict[str, str]:
        return dict(self._intent_tier_priors)

    def intent_tier_map(self) -> dict[str, str]:
        """The static intent -> start-tier mapping (from llm_tiers.yaml use_for)."""
        return dict(self._intent_to_tier)

    def get_tier(self, intent: str) -> TierConfig:
        return self._tier_or_fallback(self._resolve_tier_name(intent))

    def _tier_or_fallback(self, tier_name: str) -> TierConfig:
        """The tier a call on ``tier_name`` runs on: it, else ``fallback``, else the inline
        last resort."""
        tier = self._tiers.get(tier_name) or self._tiers.get("fallback")
        if tier is None:
            # absolute last resort — inline default
            return TierConfig(
                name="Fallback",
                provider="ollama",
                model="llama3.2:3b",
                max_tokens=2048,
                temperature=0.5,
                timeout_seconds=30,
                use_for=(),
            )
        return tier

    def get_tier_by_name(self, tier_name: str) -> TierConfig | None:
        """Return the tier configured under ``tier_name`` in llm_tiers.yaml."""
        return self._tiers.get(tier_name)

    def provider_localities(self) -> dict[str, Locality]:
        """Where each provider runs: this file's declarations, else the config dir's."""
        return dict(self._provider_localities) or declared_localities()

    def governance_tier_for_tier(self, tier_name: str) -> LLMTier:
        """Governance's tier label for a call on the tier named ``tier_name``.

        An unknown name is labelled as the tier its call would run on (``fallback``, or
        the inline last resort when the file has none).
        """
        key = tier_name if tier_name in self._tiers else "fallback"
        return governance_tier_for(key, self._tier_or_fallback(key), self.provider_localities())

    def trace_metadata_for_intent(self, intent: str) -> dict[str, object]:
        """Return stable trace metadata describing the selected tier."""
        tier = self.get_tier(intent)
        return {
            "tier_name": tier.name,
            "tier_provider": tier.provider,
            "tier_model": tier.model,
        }

    def get_llm_config(self, intent: str) -> object:
        """Return a CodingLLMConfig for the tier matching this intent.

        Honors the resource governor when one is wired: a downshift swaps the
        intent's tier for the governor-recommended one before constructing
        the config and triggering arbiter eviction.
        """
        tier = self.get_tier(intent)
        resolved_tier_name = self._resolve_tier_name(intent)
        if resolved_tier_name not in self._tiers:
            resolved_tier_name = "fallback"
        return self._build_llm_config(tier, resolved_tier_name)

    def get_llm_config_for_tier(self, tier_name: str) -> object | None:
        """Return a CodingLLMConfig for the tier named ``tier_name``, or None if unknown.

        For callers that are configured with a tier rather than an intent (the
        escalation judge's ``judge_tier``). Same governor downshift as
        :meth:`get_llm_config`.
        """
        tier = self._tiers.get(tier_name)
        if tier is None:
            return None
        return self._build_llm_config(tier, tier_name)

    def _build_llm_config(self, tier: TierConfig, resolved_tier_name: str) -> object:
        from iris_harness.llm.client import CodingLLMConfig

        if self.governor is not None and tier.provider == "ollama" and not tier.pinned:
            current_name = resolved_tier_name
            recommended_name = self.governor.recommend_tier_name(current_name)  # type: ignore[attr-defined]
            if recommended_name != current_name:
                downshifted = self._tiers.get(recommended_name)
                if downshifted is not None and downshifted.provider == "ollama":
                    tier = downshifted
                    resolved_tier_name = recommended_name
        base_url = _provider_base_url(tier.provider)
        api_key_env = _PROVIDER_API_KEY_ENVS.get(tier.provider)
        if self.governor is not None and tier.provider == "ollama":
            try:
                self.governor.acquire(tier.model)  # type: ignore[attr-defined]
            except Exception:
                logger.debug("governor.acquire failed for %s", tier.model, exc_info=True)
        return CodingLLMConfig(
            provider=tier.provider,
            model=tier.model,
            tier_name=resolved_tier_name,
            # From the tier the call actually runs on (after any downshift): the client
            # governs the call by where it goes, not by a name.
            governance_tier=governance_tier_for(
                resolved_tier_name, tier, self.provider_localities()
            ),
            base_url=base_url,
            api_key_env=api_key_env,
            temperature=tier.temperature,
            max_tokens=tier.max_tokens,
            timeout_seconds=tier.timeout_seconds,
            num_ctx=tier.num_ctx,
            keep_alive=tier.keep_alive,
            num_thread=tier.num_thread,
            think=tier.think,
        )

    def tier_name_for_model(self, model: str) -> str | None:
        """Reverse-lookup the tier name configured for ``model`` (resolved tier).

        Lets the runtime record the tier that actually produced a response —
        the *resolved* tier, not the requested one (learning-observability.md
        §4.1 threat 4 / D4). Returns ``None`` when no tier serves the model.
        """
        for tier_name, tier in self._tiers.items():
            if tier.model == model:
                return tier_name
        return None

    def model_for_intent(self, intent: str) -> str:
        return self.get_tier(intent).model

    def provider_for_intent(self, intent: str) -> str:
        return self.get_tier(intent).provider


# How big a local tier's model is, which is not where it runs: a local tier is ``tier_2``
# when it is one of the larger local models, else ``tier_1``. The egress policy
# (``config/governance/egress.yaml``, design §6.3) may tell the two apart; as shipped it
# does not (``secret`` and ``personal`` may reach either). Local or cloud is never read
# from these names -- it is the provider's declared locality (llm/locality.py), so a
# local model named ``tier3`` governs as local and a cloud model under any name governs
# as cloud.
_SMALL_LOCAL_TIER_KEYS = frozenset({"tier1", "tier_1", "router", "code_exec", "fallback"})
_LARGE_LOCAL_TIER_KEYS = frozenset({"tier2", "tier_2", "tier3", "tier_3", "gemma"})


def governance_tier_for(
    tier_key: str,
    tier: TierConfig | None,
    localities: Mapping[str, Locality] | None = None,
) -> LLMTier:
    """Governance's tier label for a call on ``tier`` (configured under ``tier_key``).

    ``tier_3`` -- the prompt leaves the owner's machines -- exactly when the tier's
    provider is declared ``runs: cloud``, or is not declared at all (fail closed).
    ``localities`` defaults to the config dir's ``llm_tiers.yaml``. A local tier is
    ``tier_2`` when it is one of the larger local models (or its display name says
    "Advanced", as the cloud VM's ``private`` tier does), else ``tier_1``.
    """
    provider = tier.provider if tier is not None else ""
    if provider_locality(provider, localities) == "cloud":
        return "tier_3"
    key = tier_key.strip().lower()
    if key in _SMALL_LOCAL_TIER_KEYS:
        return "tier_1"
    if key in _LARGE_LOCAL_TIER_KEYS:
        return "tier_2"
    if tier is not None and "advanced" in tier.name.lower():
        return "tier_2"
    return "tier_1"


def governance_tier_for_intent(tier_router: TierRouterService, intent: str) -> LLMTier:
    """Governance's tier label for a call ``intent`` makes, from the tier it runs on.

    Lived in ``runtime/bootstrap`` until M6.1b. It moved here because a plugin needs
    it — the email agent left the core with its library (OSS plan M6, decision 2) and
    builds the same governed loop — and a plugin may not import the composition root
    (release gate 2). The mapping is the router's own vocabulary anyway.

    The tier is the one ``get_llm_config`` builds for ``intent`` (a learned start-tier
    prior included), labelled by :func:`governance_tier_for`.
    """
    if isinstance(tier_router, TierRouter):
        return tier_router.governance_tier_for_tier(tier_router._resolve_tier_name(intent))

    # A router that is not a TierRouter (a plugin's own, a test stub): the tier the
    # intent maps to, labelled by the config dir's provider declarations.
    intent_map = getattr(tier_router, "_intent_to_tier", None)
    key = str(intent_map.get(intent, "")) if isinstance(intent_map, dict) else ""
    tier: TierConfig | None
    try:
        tier = tier_router.get_tier(intent)
    except AttributeError:
        tier = None
    if tier is None:
        try:
            cfg_obj = tier_router.get_llm_config(intent)
        except Exception:  # noqa: BLE001
            cfg_obj = None
        tier = TierConfig(
            name=str(getattr(cfg_obj, "tier_name", "") or ""),
            provider=str(getattr(cfg_obj, "provider", "") or ""),
            model=str(getattr(cfg_obj, "model", "") or ""),
            max_tokens=1,
            temperature=0.0,
            timeout_seconds=1,
            use_for=(),
        )
        key = key or tier.name
    return governance_tier_for(key, tier)
