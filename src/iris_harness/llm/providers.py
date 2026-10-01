"""Provider profiles: the connection details each backend needs.

Profiles are stored in ~/.iris/providers.json. Built-in profiles are always
available as defaults and are merged with any user-defined overrides at load time.

Lived in ``cli/`` until M6.2 for the same reason as ``model_metadata``: the CLI edits
them, so they were filed with the editor rather than with the thing they configure.
``runtime.client_config`` builds a client from these, and the runtime importing the CLI
to do it was the edge (OSS plan M6, decision 6).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Override slot (``None``: resolved when a store is built, never frozen at import, so
# a relocated IRIS_HOME -- the suite, ``iris_harness.testing``'s harness -- never reads
# the owner's profile list).
_CONFIG_PATH: Path | None = None


def _default_config_path() -> Path:
    import os

    from iris_harness.foundation.plugin_dirs import iris_home

    if _CONFIG_PATH is not None:
        return _CONFIG_PATH
    env = os.environ.get("IRIS_PROVIDERS_CONFIG")
    if env is not None and env.strip():
        return Path(env)
    return iris_home() / "providers.json"


@dataclass
class ModelListing:
    """Result of fetching models from a provider's catalog."""

    usable: list[str]
    hidden: list[tuple[str, str]]  # (model_id, reason)
    error: str | None = None  # human-readable reason the live catalog was unavailable
    from_fallback: bool = False  # usable came from the static fallback, not a live fetch


# Curated known-good models per provider type, shown when the live catalog endpoint
# is unreachable, unauthorized, or unsupported — so `/model list` is still useful and
# the user can pick a model without a working /models endpoint. Kept intentionally
# short (current flagship + workhorse ids); the live catalog supersedes these whenever
# it succeeds.
_STATIC_FALLBACK_MODELS: dict[str, list[str]] = {
    "anthropic": [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ],
    "openrouter": [
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-opus-4.8",
        "openai/gpt-4o",
        "google/gemini-2.5-pro",
        "meta-llama/llama-3.3-70b-instruct",
    ],
}


_BUILTIN_DEFS: list[dict[str, object]] = [
    {
        "name": "github",
        "display_name": "GitHub Models",
        "provider_type": "github",
        "base_url": "https://models.inference.ai.azure.com",
        "model": "gpt-4o",
        "coding_model": "gpt-4o",
        "api_key_env": "GITHUB_TOKEN",
        "auth_mode": "static",
        "default_headers": {},
    },
    {
        "name": "anthropic",
        "display_name": "Anthropic Claude",
        "provider_type": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-5",
        "coding_model": "claude-sonnet-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "auth_mode": "static",
        "default_headers": {},
    },
    {
        "name": "openrouter",
        "display_name": "OpenRouter",
        "provider_type": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "~anthropic/claude-sonnet-latest",
        "coding_model": "~anthropic/claude-sonnet-latest",
        "api_key_env": "OPENROUTER_API_KEY",
        "auth_mode": "static",
        "default_headers": {},
    },
    {
        "name": "ollama",
        "display_name": "Ollama (Local)",
        "provider_type": "ollama",
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2:3b",
        # Coding stays on the qwen2.5-coder tier even when /model swaps the
        # general chat default. Keep in sync with tier2 in llm_tiers.yaml.
        "coding_model": "qwen2.5-coder:7b",
        "api_key_env": None,
        "auth_mode": "static",
        "default_headers": {},
    },
    {
        "name": "lmstudio",
        "display_name": "LM Studio (Local)",
        "provider_type": "lmstudio",
        "base_url": "http://localhost:1234/v1",
        "model": "qwen2.5-coder-14b-instruct",
        "coding_model": "qwen2.5-coder-14b-instruct",
        "api_key_env": "LM_STUDIO_API_KEY",
        "auth_mode": "static",
        "default_headers": {},
    },
    {
        "name": "copilot",
        "display_name": "GitHub Copilot",
        "provider_type": "copilot",
        "base_url": "https://api.githubcopilot.com",
        # NOTE: many Copilot models (e.g. gpt-4o, gpt-5.4-mini, gpt-4.1) are routed
        # through /responses and not /chat/completions, so they cannot be used by
        # this OpenAI-style client. Use `/model list` from the REPL to see the
        # current usable subset filtered by `_copilot_filter_reason`.
        "model": "gpt-5-mini",
        "coding_model": "gpt-5-mini",
        "api_key_env": None,
        "auth_mode": "copilot",
        "default_headers": {
            "Editor-Version": "iris-coding/0.1",
            "Copilot-Integration-Id": "vscode-chat",
            "Editor-Plugin-Version": "iris-coding/0.1",
        },
    },
]


@dataclass
class ProviderProfile:
    name: str
    display_name: str
    provider_type: str
    base_url: str
    model: str
    api_key_env: str | None
    auth_mode: str = "static"
    default_headers: dict[str, str] = field(default_factory=dict)
    # Coding-tier default for this provider. When set, the `coding` intent
    # uses this model instead of `model` (the general chat default). The
    # session-level `/model` override still wins.
    coding_model: str | None = None

    def is_available(self) -> bool:
        """True if the required env var is set (or no key is needed)."""
        if self.auth_mode == "copilot":
            return bool(os.environ.get("IRIS_ENABLE_COPILOT_BACKEND", "").strip())
        if self.api_key_env is None:
            return True
        return bool(os.environ.get(self.api_key_env, "").strip())

    def fetch_models(self) -> ModelListing:
        """Fetch available models from the provider's /models endpoint.

        Returns a `ModelListing` with usable model IDs and hidden entries
        (model_id, reason). Empty listing if the endpoint is unreachable or
        unsupported.
        """
        import json as _json
        import urllib.error
        import urllib.request

        api_key: str | None = None
        if self.auth_mode == "copilot":
            oauth_path = Path.home() / ".config" / "iris" / "copilot" / "oauth.json"
            if oauth_path.exists():
                try:
                    token_data = _json.loads(oauth_path.read_text(encoding="utf-8"))
                    api_key = token_data.get("access_token") or token_data.get("token")
                except Exception as exc:  # noqa: BLE001
                    logger.debug("could not read Copilot OAuth token: %s", exc)
        elif self.api_key_env:
            raw = os.environ.get(self.api_key_env, "").strip() or None
            if raw is None:
                api_key = None
            else:
                try:
                    from iris_harness.kernel.governance.vault import (
                        resolve_secret_value,
                    )

                    api_key = resolve_secret_value(raw)
                except Exception as exc:  # noqa: BLE001 - never let model fetch crash
                    logger.debug("could not resolve provider api key via vault: %s", exc)
                    api_key = raw

        url = self.base_url.rstrip("/") + "/models"
        headers = {"Accept": "application/json", **self.default_headers}
        if self.provider_type == "anthropic":
            if not api_key:
                return self._fallback_listing("ANTHROPIC_API_KEY not set")
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        elif api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Fetch + parse are BOTH guarded: some catalogs return a top-level JSON list
        # (e.g. GitHub Models) rather than {"data": [...]}, which previously crashed
        # the caller with AttributeError. Any failure degrades to the static fallback
        # with a human-readable reason instead of raising.
        try:
            req = urllib.request.Request(url, headers=headers)  # noqa: S310
            with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310
                data = _json.loads(resp.read().decode())
            items = self._extract_model_items(data)
        except urllib.error.HTTPError as exc:
            return self._fallback_listing(f"catalog HTTP {exc.code} from {self.display_name}")
        except Exception as exc:  # noqa: BLE001 - never let model fetch crash the REPL
            return self._fallback_listing(f"catalog unreachable: {type(exc).__name__}")

        seen: set[str] = set()
        usable: list[str] = []
        hidden: list[tuple[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            mid = self._clean_model_id(str(item.get("id") or item.get("name") or ""))
            if not mid or mid in seen:
                continue
            seen.add(mid)
            # Embedding models aren't chat-usable — keep them out of the chat picker.
            if "embed" in mid.lower():
                hidden.append((mid, "embedding model"))
                continue
            reason = self._copilot_filter_reason(item) if self.provider_type == "copilot" else None
            if reason:
                hidden.append((str(mid), reason))
            else:
                usable.append(str(mid))
        if not usable and not hidden:
            return self._fallback_listing(f"catalog from {self.display_name} returned no models")
        return ModelListing(usable=sorted(usable), hidden=sorted(hidden))

    @staticmethod
    def _clean_model_id(mid: str) -> str:
        """Normalize a catalog model id to the short usable name.

        The deprecated GitHub Models (Azure inference) catalog returns Azure asset paths
        like ``azureml://registries/azure-openai/models/gpt-4o-mini/versions/1``; the id
        you actually pass to chat completions is the short ``gpt-4o-mini``. Extract it so
        ``/model list`` shows usable ids, not asset URLs."""
        if mid.startswith("azureml://") and "/models/" in mid:
            tail = mid.split("/models/", 1)[1]
            return tail.split("/versions/", 1)[0].strip("/")
        return mid

    @staticmethod
    def _extract_model_items(data: object) -> list[object]:
        """Normalize a /models response into a list of model entries.

        Handles the three shapes seen in the wild: ``{"data": [...]}`` (OpenAI,
        Anthropic, OpenRouter), ``{"models": [...]}``, and a bare top-level list
        (GitHub Models catalog)."""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            inner = data.get("data") or data.get("models") or []
            return inner if isinstance(inner, list) else []
        return []

    def _fallback_listing(self, reason: str) -> ModelListing:
        """Static known-model listing when the live catalog is unavailable."""
        models = _STATIC_FALLBACK_MODELS.get(self.provider_type, [])
        return ModelListing(
            usable=list(models), hidden=[], error=reason, from_fallback=bool(models)
        )

    @staticmethod
    def _copilot_filter_reason(item: dict[str, object]) -> str | None:
        """Return a reason string if a Copilot model is genuinely UNUSABLE, else None.

        Usability = the model accepts ``/chat/completions`` and policy allows it.
        ``model_picker_enabled`` is only a VS Code UI hint — models absent from the
        picker (gpt-4o, gpt-4.1, gemini-2.5-pro, …) are still callable via the API, so
        it must NOT hide them (it previously hid 16 of the user's subscription models).
        """
        policy = item.get("policy")
        if isinstance(policy, dict) and policy.get("state") != "enabled":
            return "policy disabled"
        endpoints = item.get("supported_endpoints")
        # Only hide when the catalog explicitly lists endpoints AND omits chat completions.
        if isinstance(endpoints, list) and endpoints and "/chat/completions" not in endpoints:
            return "no /chat/completions"
        return None

    def readiness_issues(self) -> list[str]:
        """Return a list of human-readable setup issues for this provider."""
        issues: list[str] = []
        if self.auth_mode == "copilot":
            if not os.environ.get("IRIS_ENABLE_COPILOT_BACKEND", "").strip():
                issues.append(
                    "IRIS_ENABLE_COPILOT_BACKEND=1 not set — "
                    "restart the IRIS API with this env var"
                )
            oauth_path = Path.home() / ".config" / "iris" / "copilot" / "oauth.json"
            if not oauth_path.exists():
                issues.append("Copilot OAuth token missing — run: iris-code auth copilot login")
        elif self.api_key_env and not os.environ.get(self.api_key_env, "").strip():
            issues.append(f"{self.api_key_env} not set in the IRIS API server environment")
        return issues

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "provider_type": self.provider_type,
            "base_url": self.base_url,
            "model": self.model,
            "coding_model": self.coding_model,
            "api_key_env": self.api_key_env,
            "auth_mode": self.auth_mode,
            "default_headers": self.default_headers,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> ProviderProfile:
        raw_headers = d.get("default_headers") or {}
        if not isinstance(raw_headers, dict):
            raw_headers = {}
        coding_raw = d.get("coding_model")
        return cls(
            name=str(d["name"]),
            display_name=str(d.get("display_name") or d["name"]),
            provider_type=str(d.get("provider_type") or d["name"]),
            base_url=str(d["base_url"]),
            model=str(d["model"]),
            api_key_env=str(d["api_key_env"]) if d.get("api_key_env") else None,
            auth_mode=str(d.get("auth_mode") or "static"),
            default_headers={str(k): str(v) for k, v in raw_headers.items()},
            coding_model=str(coding_raw) if coding_raw else None,
        )

    def to_llm_overrides(self) -> dict[str, object]:
        """Return a dict of fields for overriding CodingLLMConfig from pipeline.yaml."""
        result: dict[str, object] = {
            "provider": self.provider_type,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "auth_mode": self.auth_mode,
        }
        if self.default_headers:
            result["default_headers"] = tuple(self.default_headers.items())
        return result


@dataclass
class ProviderConfig:
    active: str
    profiles: dict[str, ProviderProfile]


BUILTIN_PROFILES: dict[str, ProviderProfile] = {
    str(d["name"]): ProviderProfile.from_dict(d) for d in _BUILTIN_DEFS
}


class ProviderManager:
    """Manages provider profiles stored in ~/.iris/providers.json.

    Built-in profiles are always merged in at load time so they appear
    even if the JSON file doesn't exist yet.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self._path = config_path or _default_config_path()
        self._cfg: ProviderConfig | None = None

    def _ensure_loaded(self) -> ProviderConfig:
        if self._cfg is None:
            self._cfg = self.load()
        return self._cfg

    def load(self) -> ProviderConfig:
        profiles: dict[str, ProviderProfile] = dict(BUILTIN_PROFILES)
        active = "ollama"
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                active = str(raw.get("active") or "ollama")
                for name, pdata in (raw.get("profiles") or {}).items():
                    try:
                        if isinstance(pdata, dict):
                            pdata["name"] = name
                            profiles[name] = ProviderProfile.from_dict(pdata)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("skipping invalid provider profile %s: %s", name, exc)
                        continue
            except Exception as exc:  # noqa: BLE001
                logger.debug("could not load provider config %s: %s", self._path, exc)
        self._cfg = ProviderConfig(active=active, profiles=profiles)
        return self._cfg

    def save(self, cfg: ProviderConfig | None = None) -> None:
        target = cfg or self._ensure_loaded()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Only persist customised profiles; builtins are merged at load time.
        custom: dict[str, object] = {}
        for name, p in target.profiles.items():
            builtin = BUILTIN_PROFILES.get(name)
            if builtin is None or p.to_dict() != builtin.to_dict():
                custom[name] = p.to_dict()
        data: dict[str, object] = {"active": target.active, "profiles": custom}
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._cfg = target

    def get_active(self) -> ProviderProfile:
        cfg = self._ensure_loaded()
        return cfg.profiles.get(cfg.active) or BUILTIN_PROFILES["github"]

    def set_active(self, name: str) -> None:
        cfg = self._ensure_loaded()
        if name not in cfg.profiles:
            raise KeyError(f"Unknown provider: {name!r}")
        cfg.active = name
        self.save(cfg)

    def active_name(self) -> str:
        return self._ensure_loaded().active

    def list_profiles(self) -> list[ProviderProfile]:
        return list(self._ensure_loaded().profiles.values())

    def add_or_update(self, profile: ProviderProfile) -> None:
        cfg = self._ensure_loaded()
        cfg.profiles[profile.name] = profile
        self.save(cfg)
