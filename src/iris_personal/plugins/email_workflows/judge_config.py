"""The judge's vocabulary and settings (``judge.yaml``) and its event topics.

``judge.yaml`` ships beside this module; ``<config_dir>/email/judge.yaml`` overrides it
key by key. The code is mechanism only: bucket names, labels, the prompt and the chat
words all come from the YAML (config-driven plugin rule).

Settings (owner-editable, ADR-0120; declared in this plugin's ``manifest.yaml``):

- ``IRIS_EMAIL_JUDGE`` (default on): the ``email_judge`` job judges waiting mail; off,
  new mail still queues as ``waiting`` and stays hidden until judged.
- ``IRIS_EMAIL_JUDGE_LABELS`` (default on): write the IRIS/* Gmail labels.
- ``IRIS_EMAIL_JUDGE_MAX`` (default 100): emails judged per run; the rest wait.
- ``IRIS_EMAIL_JUDGE_UNSURE_BELOW`` (default: ``unsure_below`` in the YAML).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SHIPPED_CONFIG = Path(__file__).with_name("judge.yaml")
OVERRIDE_RELATIVE_PATH = Path("email") / "judge.yaml"

# The tier the judge calls (config/llm_tiers.yaml): the Mac's Ollama, direct, no failover.
JUDGE_TIER = "email_judge"

# Event topics (ADR-0013: plugin-private topics live with their producer).
EMAIL_JUDGED = "email.judged"
EMAIL_JUDGMENT_CORRECTED = "email.judgment_corrected"

_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class EmailJudgedPayload:
    """One email judged (a ``waiting`` row is not an event)."""

    message_id: str
    account_id: str
    bucket: str
    confidence: float | None


@dataclass(frozen=True)
class JudgmentCorrectedPayload:
    """The owner moved one email to another bucket (``bucket`` may be ``promo``)."""

    message_id: str
    account_id: str
    bucket: str
    previous: str | None
    source: str


@dataclass(frozen=True)
class Bucket:
    key: str
    name: str
    label: str
    definition: str


@dataclass(frozen=True)
class JudgeConfig:
    buckets: tuple[Bucket, ...]
    skip_labels: frozenset[str]
    unsure_below: float
    body_chars: int
    bucket_words: dict[str, tuple[str, ...]]
    prompt: str
    # The prompt's fill-ins (judge.yaml ``hint_line``, ``triage_hint_line``,
    # ``email_message``); empty in a hand-built config means "leave it out".
    hint_line: str = ""
    triage_hint_line: str = ""
    email_message: str = ""
    # judge.yaml ``queue_hours``: only mail received this recently is queued; 0 = all.
    queue_hours: int = 0
    # judge.yaml ``person_only``: buckets only a real person's email can be in, and the
    # sender-address words that mark an automated sender (such a verdict becomes
    # ``person_only_else``).
    person_only: tuple[str, ...] = ()
    automated_sender_words: tuple[str, ...] = ()
    person_only_else: str = "fyi"

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(b.key for b in self.buckets)

    def bucket(self, key: str) -> Bucket:
        for b in self.buckets:
            if b.key == key:
                return b
        raise KeyError(key)

    def name(self, key: str) -> str:
        """How surfaces say a bucket ("Needs reply"); ``promo`` reads "Promo"."""
        try:
            return self.bucket(key).name
        except KeyError:
            return key.replace("_", " ").capitalize()

    @property
    def labels(self) -> dict[str, str]:
        """bucket key → Gmail label name."""
        return {b.key: b.label for b in self.buckets}

    @classmethod
    def load(cls, config_dir: Path | None = None) -> JudgeConfig:
        raw: dict[str, Any] = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8")) or {}
        if config_dir is not None:
            override = Path(config_dir) / OVERRIDE_RELATIVE_PATH
            if override.is_file():
                raw.update(yaml.safe_load(override.read_text(encoding="utf-8")) or {})
        buckets_raw = raw.get("buckets") or {}
        if not isinstance(buckets_raw, dict) or "unsure" not in buckets_raw:
            raise ValueError("judge.yaml: buckets must be a mapping that includes 'unsure'")
        buckets = tuple(
            Bucket(
                key=key,
                name=str(spec["name"]),
                label=str(spec["label"]),
                definition=" ".join(str(spec["definition"]).split()),
            )
            for key, spec in buckets_raw.items()
        )
        words = {
            str(k): tuple(str(w).lower() for w in (v or []))
            for k, v in (raw.get("bucket_words") or {}).items()
        }
        return cls(
            buckets=buckets,
            skip_labels=frozenset(str(x) for x in raw.get("skip_labels") or []),
            unsure_below=float(raw.get("unsure_below", 0.7)),
            body_chars=int(raw.get("body_chars", 4000)),
            bucket_words=words,
            prompt=str(raw.get("prompt") or ""),
            queue_hours=int(raw.get("queue_hours") or 0),
            person_only=tuple(str(b) for b in (raw.get("person_only") or {}).get("buckets") or ()),
            automated_sender_words=tuple(
                str(w).lower()
                for w in (raw.get("person_only") or {}).get("automated_sender_words") or ()
            ),
            person_only_else=str((raw.get("person_only") or {}).get("else") or "fyi"),
            **{
                key: str(raw[key])
                for key in ("hint_line", "triage_hint_line", "email_message")
                if raw.get(key)
            },
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUE


def judge_enabled() -> bool:
    return _env_bool("IRIS_EMAIL_JUDGE", True)


def labels_enabled() -> bool:
    return _env_bool("IRIS_EMAIL_JUDGE_LABELS", True)


def judge_max_per_run() -> int:
    try:
        return max(1, int(os.environ.get("IRIS_EMAIL_JUDGE_MAX", "100")))
    except ValueError:
        return 100


def unsure_below(config: JudgeConfig) -> float:
    raw = os.environ.get("IRIS_EMAIL_JUDGE_UNSURE_BELOW")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return config.unsure_below
