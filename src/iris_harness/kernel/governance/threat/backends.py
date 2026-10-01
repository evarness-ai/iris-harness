"""Threat-classifier backends (sub-phase 6a.1).

Three implementations of the ``ThreatClassifier`` protocol, all swappable via
``config/governance/threat-detection.yaml`` (D1):

- ``NullClassifier``     — always benign; the degraded/disabled fallback.
- ``PromptGuardClassifier`` — local transformers/ONNX runner for Prompt Guard 2
  (G1/G2 inbound + retrieved). Lazy-imports transformers so cold start stays
  cheap; degrades to a clear ``error`` verdict when unavailable.
- ``LlamaGuardClassifier`` — Llama Guard 3 over an injected ``invoke`` callable
  (G3 output). Parses the standard ``safe`` / ``unsafe\nS1,S9`` reply.

None of these raise from ``score`` — they return an ``error`` verdict so guards
fail-safe. Enforcement is the caller's job in later sub-phases.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from iris_harness.foundation.observability.logging_setup import log_egress
from iris_harness.kernel.governance.threat.types import ThreatSurface, ThreatVerdict

logger = logging.getLogger(__name__)

#: Llama Guard 3 hazard taxonomy: S-code -> the category names used in config
#: (``OutputConfig.enforce`` / ``log_only``). Source: Llama Guard 3 model card.
LLAMA_GUARD_CATEGORIES: dict[str, str] = {
    "S1": "violent_crimes",
    "S2": "non_violent_crimes",
    "S3": "sex_crimes",
    "S4": "child_exploitation",
    "S5": "defamation",
    "S6": "specialized_advice",
    "S7": "privacy",
    "S8": "intellectual_property",
    "S9": "indiscriminate_weapons",
    "S10": "hate",
    "S11": "self_harm",
    "S12": "sexual_content",
    "S13": "elections",
}


class NullClassifier:
    """Always-benign classifier. Used when a surface is disabled or a real
    backend could not be constructed (degraded mode)."""

    def __init__(self, *, name: str = "null", reason: str = "disabled") -> None:
        self.name = name
        self._reason = reason

    async def score(self, *, text: str, surface: ThreatSurface) -> ThreatVerdict:
        return ThreatVerdict.benign(surface=surface, backend=self.name)


class PromptGuardClassifier:
    """Prompt Guard 2 via a local transformers text-classification pipeline.

    The model + pipeline are built lazily on first use (transformers/torch are
    heavy) and cached. ``threshold`` decides the benign/malicious boundary; the
    raw malicious probability is always reported in ``score`` for shadow-mode
    tuning. Jailbreak-shaped labels map to ``jailbreak``, others to ``injection``.
    """

    def __init__(
        self,
        *,
        model_id: str,
        threshold: float = 0.8,
        name: str = "prompt_guard",
    ) -> None:
        self.name = name
        self._model_id = model_id
        self._threshold = threshold
        self._pipe: Callable[[str], Any] | None = None
        self._unavailable = False

    def _ensure_pipe(self) -> None:
        if self._pipe is not None or self._unavailable:
            return
        try:
            from transformers import pipeline

            self._pipe = pipeline(
                "text-classification",
                model=self._model_id,
                truncation=True,
                max_length=512,
                top_k=None,
            )
        except Exception:  # noqa: BLE001 - missing extra or model is a degrade, not a crash
            self._unavailable = True
            logger.warning(
                "PromptGuardClassifier unavailable (model=%s); guard will report 'error' "
                "until the transformers backend / weights are present",
                self._model_id,
            )

    def _classify_sync(self, text: str) -> tuple[float, str]:
        """Return ``(malicious_probability, raw_label)`` for ``text``."""
        assert self._pipe is not None  # guarded by caller
        results = self._pipe(text)
        # text-classification with top_k=None returns a list of {label, score}.
        rows = results[0] if results and isinstance(results[0], list) else results
        best_malicious = 0.0
        best_label = "benign"
        for row in rows:
            label = str(row.get("label", "")).lower()
            score = float(row.get("score", 0.0))
            if "benign" in label or label.endswith("_0") or label == "label_0":
                continue
            if score >= best_malicious:
                best_malicious = score
                best_label = label
        return best_malicious, best_label

    async def score(self, *, text: str, surface: ThreatSurface) -> ThreatVerdict:
        start = time.perf_counter()
        self._ensure_pipe()
        if self._unavailable or self._pipe is None:
            return ThreatVerdict.failure(
                surface=surface,
                backend=self.name,
                detail="transformers backend unavailable",
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        try:
            prob, raw_label = await asyncio.to_thread(self._classify_sync, text)
        except Exception as exc:  # fail-safe, never break the request path
            logger.debug("PromptGuardClassifier.score failed", exc_info=True)
            return ThreatVerdict.failure(
                surface=surface,
                backend=self.name,
                detail=f"classify error: {exc}",
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        if prob < self._threshold:
            return ThreatVerdict.benign(
                surface=surface, backend=self.name, score=prob, latency_ms=latency_ms
            )
        label = "jailbreak" if "jailbreak" in raw_label else "injection"
        return ThreatVerdict(
            label=label,
            score=prob,
            surface=surface,
            backend=self.name,
            categories=(raw_label,),
            latency_ms=latency_ms,
        )


class LlamaGuardClassifier:
    """Llama Guard 3 output guard over an injected sync ``invoke`` callable.

    ``invoke(system, user) -> str`` mirrors the existing curator judges
    (``_CuratorFaithfulnessLLMJudge``). The Ollama ``llama-guard3`` model carries
    its own chat template, so the content goes through as the user message and
    the reply is the standard ``safe`` / ``unsafe\nS1,S9`` format.
    """

    def __init__(
        self,
        *,
        invoke: Callable[[str, str], str],
        name: str = "llama_guard",
    ) -> None:
        self.name = name
        self._invoke = invoke

    @staticmethod
    def _parse(raw: str) -> tuple[bool, tuple[str, ...]]:
        """Parse a Llama Guard reply -> ``(unsafe, category_names)``."""
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        if not lines:
            raise ValueError("empty Llama Guard reply")
        head = lines[0].lower()
        if head.startswith("safe"):
            return False, ()
        if head.startswith("unsafe"):
            codes_line = lines[1] if len(lines) > 1 else ""
            names = tuple(
                LLAMA_GUARD_CATEGORIES.get(code.strip().upper(), code.strip())
                for code in codes_line.split(",")
                if code.strip()
            )
            return True, names
        raise ValueError(f"unrecognized Llama Guard reply: {raw[:80]!r}")

    async def score(self, *, text: str, surface: ThreatSurface) -> ThreatVerdict:
        start = time.perf_counter()
        try:
            raw = await asyncio.to_thread(self._invoke, "", text)
            unsafe, categories = self._parse(raw)
        except Exception as exc:  # fail-safe
            logger.debug("LlamaGuardClassifier.score failed", exc_info=True)
            return ThreatVerdict.failure(
                surface=surface,
                backend=self.name,
                detail=f"output guard error: {exc}",
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not unsafe:
            return ThreatVerdict.benign(surface=surface, backend=self.name, latency_ms=latency_ms)
        return ThreatVerdict(
            label="unsafe",
            score=1.0,
            surface=surface,
            backend=self.name,
            categories=categories,
            latency_ms=latency_ms,
        )


def default_ollama_invoke(
    *, endpoint: str, model: str, timeout_s: float
) -> Callable[[str, str], str]:
    """Build a sync ``invoke(system, user)`` that calls Ollama's ``/api/chat``.

    Used as the default output-guard transport when no ``invoke`` is injected.
    Kept minimal and dependency-light (httpx is already a core dep).
    """

    import httpx

    base = endpoint.rstrip("/")

    def _invoke(system: str, user: str) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        log_egress(
            destination=urlparse(base).netloc,
            method="POST",
            kind="llm",
            purpose="threat-classifier",
        )
        resp = httpx.post(
            f"{base}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("message", {}).get("content", ""))

    return _invoke
