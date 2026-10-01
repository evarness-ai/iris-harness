"""OllamaArbiter + ResourceGovernor — bounded local-LLM hardware management.

Two concerns, two classes:

* :class:`OllamaArbiter` polls ``/api/ps`` *before* a call is dispatched and
  unloads larger residents first, so the next call starts against a free GPU
  instead of one mid-eviction. Ollama's own ``OLLAMA_MAX_LOADED_MODELS=1``
  evicts on demand but blocks the next request 20-40 s during the swap.

* :class:`ResourceGovernor` owns an arbiter and adds a *mode* — ``ACTIVE`` /
  ``IDLE`` / ``THERMAL`` — derived from live host pressure (free RAM, CPU
  load, macOS thermal throttling). When ``adaptive=True``, the tier router
  consults the mode to downshift large local tiers under pressure.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from urllib.parse import urlparse

import httpx

# The host sample itself is foundation's (M6.2): three subsystems read it, and
# none of what it does is LLM logic. This module keeps only the policy over it.
from iris_harness.foundation.observability.host_pressure import (
    PressureSnapshot,
    sample_pressure,
)
from iris_harness.foundation.observability.logging_setup import log_egress

logger = logging.getLogger(__name__)


# Approximate resident-set sizes (GB) for known Ollama models. Fuzzy-matched by
# base name when a tag isn't in the table — see ``_size_for``.
_DEFAULT_SIZES_GB: dict[str, float] = {
    "llama3.2:3b": 3.0,
    "qwen2.5-coder:7b": 5.0,
    "qwen3-coder:30b": 20.0,
    "qwen3.6:27b": 18.0,
}

# Safe peak for a 36 GB Mac after macOS + Chroma embeddings + IRIS process.
_DEFAULT_BUDGET_GB = 22.0


def _daemon_root(base_url: str) -> str:
    """Strip the OpenAI-compat ``/v1`` suffix to address Ollama's native API."""
    return base_url.rstrip("/").removesuffix("/v1")


class CircuitBreakerOpenError(RuntimeError):
    """Raised instead of dispatching a call when the breaker is OPEN (fail fast)."""


@dataclass
class _BreakerEntry:
    failures: int = 0
    opened_at: float | None = None  # monotonic seconds when opened; None = CLOSED


class OllamaCircuitBreaker:
    """Per-endpoint circuit breaker so a DOWN local model server fails fast.

    Without it, every call to an unreachable Ollama waits the full per-tier timeout
    (60-120s) before raising — so a down server makes each request hang for minutes.
    The breaker trips after ``failure_threshold`` consecutive CONNECTION failures and
    then fast-fails subsequent calls for ``cooldown_seconds`` (a single half-open probe
    is allowed once the cooldown elapses; success closes it, failure re-opens it).

    State is keyed by endpoint so independent servers don't interfere; thread-safe;
    effectively free when healthy (a dict lookup). Scope is the CALLER's concern — only
    local providers should route through it (cloud calls bypass entirely).
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = max(1, failure_threshold)
        self._cooldown = cooldown_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[str, _BreakerEntry] = {}

    def before_call(self, key: str) -> None:
        """Raise :class:`CircuitBreakerOpenError` if OPEN and still within cooldown."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.opened_at is None:
                return  # CLOSED
            remaining = self._cooldown - (self._clock() - entry.opened_at)
            if remaining > 0:
                raise CircuitBreakerOpenError(
                    f"local model server at {key} appears down "
                    f"(circuit open; retrying in ~{remaining:.0f}s)"
                )
            # Cooldown elapsed → allow one HALF-OPEN probe; state unchanged until the
            # caller reports the probe outcome via record_success / record_failure.

    def record_success(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)  # full reset → CLOSED

    def record_failure(self, key: str) -> None:
        with self._lock:
            entry = self._entries.setdefault(key, _BreakerEntry())
            entry.failures += 1
            if entry.failures >= self._threshold:
                entry.opened_at = self._clock()  # OPEN / re-open (refresh cooldown)

    def is_open(self, key: str) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            return bool(
                entry
                and entry.opened_at is not None
                and (self._clock() - entry.opened_at) < self._cooldown
            )


_DEFAULT_OLLAMA_BREAKER = OllamaCircuitBreaker()


def get_ollama_breaker() -> OllamaCircuitBreaker:
    """The process-wide breaker shared across the per-call LLM client instances."""
    return _DEFAULT_OLLAMA_BREAKER


def reset_ollama_breaker() -> None:
    """Forget every endpoint's failure state.

    Exists for tests. The breaker is process-global on purpose -- one DOWN endpoint
    should fail fast for every caller in the process -- but that makes it leak
    between tests: one test that trips it on a real connection failure fails every
    later test in the same pytest-xdist worker that builds a client, with an error
    about a model server rather than about anything the test asserts. That is an
    order-dependent failure, which is the worst kind to debug, so the suite resets
    it per test (``tests/conftest.py``).
    """
    with _DEFAULT_OLLAMA_BREAKER._lock:  # same module, private by convention
        _DEFAULT_OLLAMA_BREAKER._entries.clear()


@dataclass
class OllamaArbiter:
    """Pre-emptively evict resident Ollama models that won't fit alongside ``model``."""

    base_url: str
    http: httpx.Client | None = None
    budget_gb: float = _DEFAULT_BUDGET_GB
    sizes_gb: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_SIZES_GB))

    def __post_init__(self) -> None:
        if self.http is None:
            self.http = httpx.Client(timeout=10.0)

    def acquire(self, model: str) -> None:
        """Make room for ``model`` by unloading larger residents until it fits."""
        try:
            loaded = self._loaded_models()
        except Exception as exc:  # noqa: BLE001
            logger.debug("arbiter: failed to query /api/ps: %s", exc)
            return

        target_size = self._size_for(model)
        resident = [(name, size) for (name, size) in loaded if name != model]
        resident_total = sum(size for _, size in resident)

        if resident_total + target_size <= self.budget_gb:
            return

        resident.sort(key=lambda pair: pair[1], reverse=True)
        for name, size in resident:
            if resident_total + target_size <= self.budget_gb:
                break
            try:
                self._unload(name)
                resident_total -= size
                logger.info("arbiter: evicted %s (~%.1f GB) to make room for %s", name, size, model)
            except Exception as exc:  # noqa: BLE001
                logger.debug("arbiter: eviction of %s failed: %s", name, exc)

    def _loaded_models(self) -> list[tuple[str, float]]:
        assert self.http is not None
        url = f"{_daemon_root(self.base_url)}/api/ps"
        log_egress(
            destination=urlparse(url).netloc,
            method="GET",
            kind="llm",
            purpose="ollama.arbiter.ps",
        )
        resp = self.http.get(url)
        resp.raise_for_status()
        out: list[tuple[str, float]] = []
        for entry in resp.json().get("models") or []:
            name = entry.get("name") or entry.get("model") or ""
            size_bytes = entry.get("size") or entry.get("size_vram") or 0
            size_gb = float(size_bytes) / (1024**3) if size_bytes else self._size_for(name)
            if name:
                out.append((name, size_gb))
        return out

    def _unload(self, model: str) -> None:
        assert self.http is not None
        url = f"{_daemon_root(self.base_url)}/api/generate"
        log_egress(
            destination=urlparse(url).netloc,
            method="POST",
            kind="llm",
            purpose="ollama.arbiter.unload",
        )
        self.http.post(url, json={"model": model, "keep_alive": 0})

    def _size_for(self, model: str) -> float:
        if model in self.sizes_gb:
            return self.sizes_gb[model]
        base = model.split(":", 1)[0]
        for key, size in self.sizes_gb.items():
            if key.startswith(base):
                return size
        return 8.0


# ---------------------------------------------------------------------------
# ResourceGovernor — mode-aware tier downshift under host pressure
# ---------------------------------------------------------------------------


class Mode(StrEnum):
    ACTIVE = "active"  # default; no constraints
    IDLE = "idle"  # opt-in: prefer the small tier even for heavy intents
    THERMAL = "thermal"  # automatic: pin to small tier until pressure clears


# Hysteresis: how many consecutive polls before flipping mode. At a 30s
# heartbeat cadence: 90 s in, 150 s out. Slower exit on purpose — don't yank
# weights back the instant temp dips.
_THERMAL_ENTER_POLLS = 3
_THERMAL_EXIT_POLLS = 5

# Pressure thresholds. Free-RAM threshold sized for a 36 GB Mac; below ~4 GB
# headroom we're already swap-pressured before adding a model load.
_RAM_FREE_GB_THRESHOLD = 4.0

# What tier each non-active mode prefers. Keys reference tier names declared
# in ``config/llm_tiers.yaml``. Stale references are caught at runtime by the
# tier router (downshift no-ops if the target tier is missing).
_DOWNSHIFT_TARGETS: dict[Mode, str | None] = {
    Mode.ACTIVE: None,
    Mode.IDLE: "tier1",
    Mode.THERMAL: "tier1",
}


@dataclass
class ResourceGovernor:
    """Mode-aware wrapper around :class:`OllamaArbiter`.

    When ``adaptive`` is False (default), :meth:`recommend_tier_name` is a
    pass-through — the governor still polls and logs, so users can observe
    what *would* happen before opting in via ``IRIS_ADAPTIVE_TIERS=1``.
    """

    arbiter: OllamaArbiter
    adaptive: bool = False
    pin: Mode | None = None
    sampler: Callable[[], PressureSnapshot] = field(default=sample_pressure)
    _mode: Mode = field(default=Mode.ACTIVE, init=False)
    _consecutive_pressure: int = field(default=0, init=False)
    _consecutive_clear: int = field(default=0, init=False)
    _last_transition: datetime | None = field(default=None, init=False)
    _last_snapshot: PressureSnapshot | None = field(default=None, init=False)

    def poll(self) -> PressureSnapshot:
        """Sample pressure and update mode if hysteresis thresholds are met."""
        snapshot = self.sampler()
        self._last_snapshot = snapshot
        self._update_mode(snapshot)
        return snapshot

    def mode(self) -> Mode:
        """Return the active mode, honoring any ``/llm mode`` pin."""
        return self.pin if self.pin is not None else self._mode

    def snapshot(self) -> PressureSnapshot | None:
        """Return the most recent poll snapshot, or None if never polled."""
        return self._last_snapshot

    def set_pin(self, mode: Mode | None) -> None:
        """Pin (or release) the mode. Pin overrides automatic transitions."""
        old_pin = self.pin
        self.pin = mode
        if mode is not None:
            logger.info("governor pin set: %s (was %s)", mode, old_pin)
        else:
            logger.info("governor pin released (was %s)", old_pin)

    def recommend_tier_name(self, current: str) -> str:
        """Return the tier name to actually use, given current mode.

        Pass-through when ``adaptive`` is off, mode is ACTIVE, the current
        tier already *is* the downshift target, or no target is configured
        for the current mode.
        """
        if not self.adaptive:
            return current
        target = _DOWNSHIFT_TARGETS.get(self.mode())
        if target is None or target == current:
            return current
        return target

    def acquire(self, model: str) -> None:
        """Delegate to the underlying arbiter."""
        self.arbiter.acquire(model)

    def _update_mode(self, snapshot: PressureSnapshot) -> None:
        if self.pin is not None:
            return
        under_pressure = snapshot.thermal_throttled or snapshot.ram_free_gb < _RAM_FREE_GB_THRESHOLD
        if under_pressure:
            self._consecutive_pressure += 1
            self._consecutive_clear = 0
            if self._mode != Mode.THERMAL and self._consecutive_pressure >= _THERMAL_ENTER_POLLS:
                reason = (
                    f"cpu_speed_limit={snapshot.cpu_speed_limit}, "
                    f"ram_free={snapshot.ram_free_gb:.1f}GB"
                )
                self._set_mode(Mode.THERMAL, reason=reason)
        else:
            self._consecutive_clear += 1
            self._consecutive_pressure = 0
            if self._mode == Mode.THERMAL and self._consecutive_clear >= _THERMAL_EXIT_POLLS:
                self._set_mode(Mode.ACTIVE, reason="pressure cleared")

    def _set_mode(self, mode: Mode, *, reason: str) -> None:
        old = self._mode
        self._mode = mode
        self._last_transition = datetime.now(UTC)
        logger.info("governor mode: %s → %s (%s)", old, mode, reason)
