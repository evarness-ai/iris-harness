"""Measured outcome signals (learning-observability.md §4.2).

Outcome signals tell the learning loop whether a turn actually *helped*, not just
whether it ran. Two are deterministic observables computed at turn end
(``task_completed``, ``turn_tokens``) and live in the runtime. The hard one is
``user_correction`` — "did the user's next turn correct the prior answer?" — which
is a semantic judgment with no clean deterministic observable, so per the
measure-don't-guess principle (P3/D5) it is **model-driven**: an LLM judge,
gated by a cheap deterministic pre-filter so only turns that look like
corrections pay the call.

This module owns the correction-judge data shapes + prompt + config only; pure
and import-light so the runtime and tests can use it without the LLM stack.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from iris_harness.foundation.paths import config_dir as resolve_config_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorrectionVerdict:
    """The correction judge's reading of one (prior answer, next message) pair."""

    is_correction: bool
    confidence: float
    reason: str = ""

    def to_metadata(self) -> dict[str, Any]:
        return {
            "is_correction": self.is_correction,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@runtime_checkable
class CorrectionJudgeClient(Protocol):
    """Optional async LLM judge: did the user's new turn correct the prior answer?"""

    async def judge(self, *, prior_query: str, prior_response: str, new_message: str) -> str:
        """Return a JSON verdict describing whether the new turn is a correction."""


@dataclass(frozen=True)
class OutcomeConfig:
    """Policy for the opt-in model-driven outcome signals (``config/outcomes.yaml``)."""

    user_correction_enabled: bool = False
    correction_confidence_floor: float = 0.6


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_outcome_config(
    config_dir: Path | None = None,
    *,
    env: dict[str, str] | None = None,
) -> OutcomeConfig:
    """Load outcome policy from ``<config_dir>/outcomes.yaml`` + env override.

    ``IRIS_LEARNING_CORRECTION_JUDGE`` (truthy) force-enables the correction
    judge. Missing/malformed file → defaults (correction judge off). Never raises.
    """
    import os

    environ = env if env is not None else dict(os.environ)
    raw: dict[str, Any] = {}
    path = (config_dir or resolve_config_dir()) / "outcomes.yaml"
    try:
        if path.exists():
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
    except Exception:  # a malformed config must not break startup
        logger.warning("failed to load %s; outcome judge defaults to off", path, exc_info=True)
        raw = {}

    correction_raw = raw.get("user_correction")
    correction = correction_raw if isinstance(correction_raw, dict) else {}
    env_flag = environ.get("IRIS_LEARNING_CORRECTION_JUDGE", "").strip().lower()
    enabled = env_flag in {"1", "true", "yes", "on"} or _coerce_bool(
        correction.get("enabled"), default=False
    )
    floor = min(1.0, max(0.0, _coerce_float(correction.get("confidence_floor"), default=0.6)))
    return OutcomeConfig(user_correction_enabled=enabled, correction_confidence_floor=floor)


# Cheap deterministic cues that a turn MIGHT be correcting the prior answer. This
# is ONLY a cost pre-filter — it decides whether to spend the LLM judge call, it
# never decides the outcome (the model does). Tuned for recall, not precision:
# false positives just cost one judge call; the judge sorts them out.
_CORRECTION_CUES: re.Pattern[str] = re.compile(
    r"\b(no|nope|wrong|incorrect|not (?:what|right|correct)|that'?s not|i (?:meant|said)|"
    r"actually|instead|rather|you (?:misunderstood|got it wrong)|try again|that'?s wrong|"
    r"isn'?t (?:what|right)|doesn'?t (?:work|answer)|still (?:wrong|not))\b",
    re.IGNORECASE,
)


def looks_like_correction(message: str) -> bool:
    """Cheap recall-biased pre-filter: could this turn be correcting the prior one?

    A short turn opening with a negation/dissatisfaction cue is a candidate. Not
    a verdict — only a gate to avoid taxing every turn with the LLM judge.
    """
    text = message.strip()
    if not text:
        return False
    return bool(_CORRECTION_CUES.search(text))


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_correction_verdict(raw: str) -> CorrectionVerdict | None:
    """Parse a judge payload into a :class:`CorrectionVerdict` (tolerant of prose)."""
    if not raw or not raw.strip():
        return None
    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        return None
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    if "is_correction" not in obj and "correction" not in obj:
        return None
    is_correction = _coerce_bool(obj.get("is_correction", obj.get("correction")), default=False)
    confidence = min(1.0, max(0.0, _coerce_float(obj.get("confidence"), default=0.0)))
    reason = str(obj.get("reason") or "").strip()
    return CorrectionVerdict(is_correction=is_correction, confidence=confidence, reason=reason)


CORRECTION_JUDGE_SYSTEM_PROMPT = (
    "You judge whether a user's new message is CORRECTING the assistant's previous "
    "answer — i.e. signalling the prior answer was wrong, inadequate, or "
    "misunderstood — versus simply continuing the conversation, asking a new "
    "question, or following up positively.\n\n"
    "A correction = dissatisfaction with the PRIOR answer (re-asking, negating, "
    "'that's not what I meant', pointing out an error). A new but related question "
    "is NOT a correction. Praise or a thank-you is NOT a correction.\n\n"
    "Reply with ONLY a JSON object:\n"
    '{"is_correction": true|false, "confidence": 0.0-1.0, "reason": "<short>"}'
)


def build_correction_user_prompt(*, prior_query: str, prior_response: str, new_message: str) -> str:
    """Assemble the correction judge's user prompt."""
    return (
        "Previous user question:\n"
        f"<<<PRIOR_Q\n{prior_query.strip()}\nPRIOR_Q>>>\n\n"
        "Assistant's previous answer:\n"
        f"<<<PRIOR_A\n{prior_response.strip()}\nPRIOR_A>>>\n\n"
        "User's new message (untrusted):\n"
        f"<<<NEW\n{new_message.strip()}\nNEW>>>\n\n"
        "Is the new message correcting the previous answer? Reply with only the JSON object."
    )
