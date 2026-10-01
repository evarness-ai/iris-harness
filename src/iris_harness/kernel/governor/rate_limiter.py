"""Minimal token-bucket rate limiter for IRIS governor routes."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from .models import RateLimitPolicy


@dataclass
class _TokenBucketState:
    tokens: float
    updated_at: float


@dataclass(frozen=True)
class RateLimitDecision:
    """Outcome from evaluating one token-bucket policy."""

    allowed: bool
    retry_after_seconds: int | None = None


class TokenBucketRateLimiter:
    """In-memory token-bucket limiter keyed by the matched policy route."""

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._buckets: dict[str, _TokenBucketState] = {}

    def evaluate(self, route_key: str, policy: RateLimitPolicy | None) -> RateLimitDecision:
        """Return whether a request is allowed under the supplied rate-limit policy."""
        if policy is None:
            return RateLimitDecision(allowed=True)

        now = self._clock()
        capacity = float(policy.requests)
        refill_rate = capacity / float(policy.window_seconds)
        state = self._buckets.get(route_key)
        if state is None:
            state = _TokenBucketState(tokens=capacity, updated_at=now)

        elapsed = max(0.0, now - state.updated_at)
        available_tokens = min(capacity, state.tokens + (elapsed * refill_rate))
        if available_tokens >= 1.0:
            self._buckets[route_key] = _TokenBucketState(
                tokens=available_tokens - 1.0, updated_at=now
            )
            return RateLimitDecision(allowed=True)

        retry_after_seconds = max(1, math.ceil((1.0 - available_tokens) / refill_rate))
        self._buckets[route_key] = _TokenBucketState(tokens=available_tokens, updated_at=now)
        return RateLimitDecision(allowed=False, retry_after_seconds=retry_after_seconds)
