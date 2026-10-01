"""Prompt-size budgeting for local-LLM inference.

Local models have small context windows (Tier 1 = 2048 tokens, Tier 2 = 4096).
Stuffing the full system prompt + identity blobs + conversation history easily
overruns ``num_ctx``, which causes Ollama to silently truncate or stall. The
helpers here cap prompt growth without needing a heavyweight tokenizer.

The estimator is char-based (``len // 4``), within ~15% of true token counts
for English chat text. If precision becomes load-bearing, swap in ``tiktoken``
behind the same interface — callers should not care.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class _HasContent(Protocol):
    role: str
    content: str


def estimate_tokens(text: str) -> int:
    """Cheap char-based token estimate. ~15% of true count for English chat text."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def budget_for(num_ctx: int, *, fraction: float = 0.75) -> int:
    """Return the prompt-token budget for a model with ``num_ctx`` context.

    Reserves ``1 - fraction`` of the window for the model's response so that
    ``num_predict`` is never silently clipped.
    """
    if num_ctx <= 0:
        return 0
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    return int(num_ctx * fraction)


def trim_text(text: str, *, max_chars: int) -> str:
    """Truncate ``text`` to ``max_chars`` characters, preserving the head."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text if max_chars > 0 else ""
    return text[: max_chars - 1] + "…"


def trim_messages(
    messages: Sequence[_HasContent],
    *,
    budget_tokens: int,
) -> list[_HasContent]:
    """Drop oldest non-pinned messages until the total token estimate fits the budget.

    The first message (typically a system prompt) and the last message
    (typically the current user turn) are always retained. Older context is
    dropped from the middle, oldest-first.
    """
    if not messages:
        return []
    msgs = list(messages)
    if budget_tokens <= 0 or len(msgs) <= 2:
        return msgs

    used = sum(estimate_tokens(m.content) for m in msgs)
    while used > budget_tokens and len(msgs) > 2:
        dropped = msgs.pop(1)
        used -= estimate_tokens(dropped.content)
    return msgs
