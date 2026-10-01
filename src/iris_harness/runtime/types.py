"""Public runtime result shapes (Phase 2 decomposition).

The dataclasses ``IrisRuntime.chat`` / ``chat_stream`` / ``warmup`` return.
Extracted from ``bootstrap.py`` into their own module so extracted handler
modules can import them without a circular dependency on the huge bootstrap
module; re-exported there for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatResult:
    """Result returned by ``IrisRuntime.chat``."""

    response: str
    intent: str
    agent_type: str
    sources: tuple[str, ...]
    has_errors: bool
    error_summary: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StreamEvent:
    """One step of an ``IrisRuntime.chat_stream`` interaction.

    ``kind`` is one of:
      * ``"token"`` — intermediate prose text (user-visible, accumulates).
      * ``"activity"`` — short ephemeral status ("Running script…") for the
        live spinner; not appended to the final response.
      * ``"trace"`` — raw trace detail (cmd, stdout, stderr, JSON tool calls)
        buffered client-side for opt-in inspection (/trace).
      * ``"done"`` — final ``ChatResult`` after all post-processing.
      * ``"error"`` — terminal failure.
    """

    kind: str
    text: str = ""
    result: ChatResult | None = None
    error: str | None = None
    payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class WarmupResult:
    """Result returned by ``IrisRuntime.warmup``."""

    ok: bool
    role: str
    model: str
    latency_ms: float
    error: str | None = None
