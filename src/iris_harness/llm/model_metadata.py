"""Model metadata registry: what each model can do.

Maps model IDs to their capabilities: reasoning, tool-calling, and context window size.
Add new models by extending MODEL_REGISTRY or editing ~/.iris/model_metadata.json.

User-defined metadata in ~/.iris/model_metadata.json takes precedence over built-ins.

Lived in ``cli/`` until M6.2 because the CLI was its first reader, but nothing here is
presentation: ``llm.tier_router`` reads it to decide what a tier can be asked to do, and
the CLI only prints what it says. A tier router importing the CLI was the upward edge
that made the misplacement visible (OSS plan M6, decision 6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelMetadata:
    """Metadata for a single model."""

    model_id: str
    reasoning: bool = False  # True if model supports extended reasoning
    tool_calling: bool = False  # True if model supports tool/function calling
    context_window: int = 0  # Context window size in tokens (0 = unknown)

    def display_str(self) -> str:
        """Format metadata for display in the model list."""
        parts: list[str] = []
        if self.reasoning:
            parts.append("🧠")
        if self.tool_calling:
            parts.append("🔧")
        if self.context_window > 0:
            if self.context_window >= 1_000_000:
                ctx_str = f"{self.context_window / 1_000_000:.0f}M"
            elif self.context_window >= 1_000:
                ctx_str = f"{self.context_window / 1_000:.0f}k"
            else:
                ctx_str = str(self.context_window)
            parts.append(f"ctx:{ctx_str}")
        return " ".join(parts)


# Built-in model metadata.
#
# Local Ollama entries below were probed on this machine via /v1/chat/completions
# on May 1, 2026. Cloud entries are catalog-known names/aliases; they still
# require valid provider credentials before they can be used.
MODEL_REGISTRY: dict[str, ModelMetadata] = {
    # Verified local Ollama models.
    "qwen3-coder:30b": ModelMetadata(
        "qwen3-coder:30b", reasoning=False, tool_calling=True, context_window=0
    ),
    "qwen3.6:27b": ModelMetadata(
        "qwen3.6:27b", reasoning=True, tool_calling=True, context_window=0
    ),
    "qwen3.5:latest": ModelMetadata(
        "qwen3.5:latest", reasoning=True, tool_calling=True, context_window=0
    ),
    # The email judge's model (config/llm_tiers.yaml `email_judge`). A reasoning model:
    # its governed structured call (CodingLLMClient.invoke_json) turns thinking off.
    "qwen3.5:4b-q4_K_M": ModelMetadata(
        "qwen3.5:4b-q4_K_M", reasoning=True, tool_calling=True, context_window=0
    ),
    "llama3.2:3b": ModelMetadata(
        "llama3.2:3b", reasoning=False, tool_calling=True, context_window=131_072
    ),
    "llama3.2:latest": ModelMetadata(
        "llama3.2:latest", reasoning=False, tool_calling=True, context_window=131_072
    ),
    "llama3.2": ModelMetadata(
        "llama3.2", reasoning=False, tool_calling=True, context_window=131_072
    ),
    "phi4-mini:latest": ModelMetadata(
        "phi4-mini:latest", reasoning=False, tool_calling=True, context_window=131_072
    ),
    "gemma2:9b": ModelMetadata(
        "gemma2:9b", reasoning=False, tool_calling=False, context_window=8_192
    ),
    # OpenRouter catalog-known aliases/IDs. Auth was not valid in this shell,
    # so these are listed only as provider catalog entries, not locally verified.
    "~anthropic/claude-sonnet-latest": ModelMetadata(
        "~anthropic/claude-sonnet-latest",
        reasoning=False,
        tool_calling=True,
        context_window=200_000,
    ),
    "~anthropic/claude-haiku-latest": ModelMetadata(
        "~anthropic/claude-haiku-latest", reasoning=False, tool_calling=True, context_window=200_000
    ),
    "~openai/gpt-latest": ModelMetadata(
        "~openai/gpt-latest", reasoning=True, tool_calling=True, context_window=128_000
    ),
    "~openai/gpt-mini-latest": ModelMetadata(
        "~openai/gpt-mini-latest", reasoning=False, tool_calling=True, context_window=128_000
    ),
    "~google/gemini-flash-latest": ModelMetadata(
        "~google/gemini-flash-latest", reasoning=False, tool_calling=True, context_window=1_000_000
    ),
    "qwen/qwen3.6-flash": ModelMetadata(
        "qwen/qwen3.6-flash", reasoning=True, tool_calling=True, context_window=0
    ),
    "qwen/qwen3.6-27b": ModelMetadata(
        "qwen/qwen3.6-27b", reasoning=True, tool_calling=True, context_window=0
    ),
    "qwen/qwen3.6-35b-a3b": ModelMetadata(
        "qwen/qwen3.6-35b-a3b", reasoning=True, tool_calling=True, context_window=0
    ),
    "anthropic/claude-3.7-sonnet": ModelMetadata(
        "anthropic/claude-3.7-sonnet", reasoning=False, tool_calling=True, context_window=200_000
    ),
    "anthropic/claude-sonnet-4.5": ModelMetadata(
        "anthropic/claude-sonnet-4.5", reasoning=False, tool_calling=True, context_window=200_000
    ),
    # Conservative common cloud IDs used by provider defaults or user overrides.
    "gpt-4o": ModelMetadata("gpt-4o", reasoning=False, tool_calling=True, context_window=128_000),
    "gpt-4o-mini": ModelMetadata(
        "gpt-4o-mini", reasoning=False, tool_calling=True, context_window=128_000
    ),
    "claude-sonnet-4-5": ModelMetadata(
        "claude-sonnet-4-5", reasoning=False, tool_calling=True, context_window=200_000
    ),
}


def get_metadata(model_id: str) -> ModelMetadata | None:
    """Get metadata for a model, checking user config first, then built-ins.

    User metadata at ~/.iris/model_metadata.json is loaded and merged with built-ins.
    """
    # Try user-defined metadata first
    user_metadata = _load_user_metadata()
    if model_id in user_metadata:
        return user_metadata[model_id]

    # Fall back to built-in registry
    return MODEL_REGISTRY.get(model_id)


def _load_user_metadata() -> dict[str, ModelMetadata]:
    """Load user-defined model metadata from ~/.iris/model_metadata.json."""
    config_path = Path.home() / ".iris" / "model_metadata.json"
    if not config_path.exists():
        return {}

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        result: dict[str, ModelMetadata] = {}
        for model_id, meta in data.items():
            if isinstance(meta, dict):
                result[model_id] = ModelMetadata(
                    model_id=model_id,
                    reasoning=bool(meta.get("reasoning", False)),
                    tool_calling=bool(meta.get("tool_calling", False)),
                    context_window=int(meta.get("context_window", 0)),
                )
        return result
    except Exception:  # noqa: BLE001 — unknown model metadata is empty, never fatal
        return {}
