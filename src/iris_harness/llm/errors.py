"""User-readable text for an LLM/provider failure.

Lived in ``runtime/bootstrap.py``; it is core, not bootstrap-specific — every
agent that talks to a model wants the same sentence, and after the M4 extractions
some of those agents are plugins, which may not import runtime internals (OSS plan
release gate 2).

It never raises and never leaks a stack trace: the point is that a user sees
"Copilot OAuth token missing — run `iris-code auth copilot login`" instead of a
provider exception.
"""

from __future__ import annotations


def friendly_llm_error(exc: Exception) -> str:
    """Return a user-readable error string from an LLM call exception."""
    name = type(exc).__name__
    msg = str(exc)
    if "CopilotBackendDisabled" in name or "IRIS_ENABLE_COPILOT_BACKEND" in msg:
        return (
            "Copilot provider is disabled on the server.\n"
            "Restart the IRIS API with `IRIS_ENABLE_COPILOT_BACKEND=1` set, "
            "then run `iris-code auth copilot login` if you haven't already."
        )
    if "CopilotAuth" in name or "oauth.json" in msg.lower() or "device-flow" in msg.lower():
        return "Copilot OAuth token missing or expired.\n" "Run: `iris-code auth copilot login`"
    if "api_key" in msg.lower() or "authentication" in msg.lower() or "401" in msg:
        return f"LLM authentication failed — check your API key.\n({name}: {msg[:120]})"
    if "connect" in msg.lower() or "refused" in msg.lower() or "timeout" in msg.lower():
        return f"LLM provider unreachable — is it running?\n({name}: {msg[:120]})"
    if (
        "unsupported_api_for_model" in msg
        or "not accessible via the /chat/completions endpoint" in msg.lower()
    ):
        return (
            "This model isn't reachable via /chat/completions on the active provider "
            "(e.g. Copilot routes some models only through /responses).\n"
            "Run `/model list` to see models that are actually usable, "
            "or `/model reset` to fall back to the provider default."
        )
    if (
        "model_not_supported" in msg
        or "model is not supported" in msg.lower()
        or "requested model is not supported" in msg.lower()
    ):
        return (
            "Model not supported by the current provider.\n"
            "Use /model to reset, or /info to see the provider's default model name."
        )
    return f"LLM call failed ({name}: {msg[:200]})"
