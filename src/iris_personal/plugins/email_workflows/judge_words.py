"""The words of the judge's correction surfaces and digest lines (``judge.yaml``).

``judge.yaml``'s ``rebucket:`` section is what the Action Center card, the chat
intercept and the web list say; its ``digest:`` section is the morning digest's
"Needs reply" section, the "Judged yesterday" line and the footer phrases. The code is
mechanism only (config-driven plugin rule): every phrase and every chat word is here.

Loaded the way :meth:`.judge_config.JudgeConfig.load` loads the rest of the file: the
shipped YAML, then ``<config_dir>/email/judge.yaml`` replacing top-level keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .judge_config import OVERRIDE_RELATIVE_PATH, SHIPPED_CONFIG

_CARD_KEYS = (
    "tag",
    "title",
    "promo_choice",
    "guess",
    "note",
    "answered",
    "answered_no_labels",
    "answered_promo",
)
_REPLY_KEYS = (
    "changed",
    "changed_no_labels",
    "changed_promo",
    "already",
    "ambiguous",
    "which_bucket",
    "not_found",
    "counts",
    "counts_none",
)
_DIGEST_KEYS = (
    "needs_reply_title",
    "needs_reply_empty_title",
    "needs_reply_empty",
    "needs_reply_item",
    "judged_line",
    "judged_part",
    "unsure_part",
    "waiting_part",
    "learned",
)


def _read(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: judge config must be a mapping")
    return raw


def _texts(section: Any, keys: tuple[str, ...], where: str) -> dict[str, str]:
    if not isinstance(section, dict):
        raise ValueError(f"judge.yaml: {where} must be a mapping")
    missing = [k for k in keys if not str(section.get(k) or "").strip()]
    if missing:
        raise ValueError(f"judge.yaml: {where} is missing {', '.join(missing)}")
    return {k: " ".join(str(v).split()) for k, v in section.items() if isinstance(v, str)}


def _words(section: dict[str, Any], key: str) -> tuple[str, ...]:
    raw = section.get(key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"judge.yaml: rebucket.chat.{key} must be a non-empty list")
    return tuple(" ".join(str(w).lower().split()) for w in raw if str(w).strip())


@dataclass(frozen=True)
class ChatWords:
    """The chat grammar's vocabulary (``rebucket.chat``)."""

    window_days: int
    email_words: tuple[str, ...]
    filler_words: tuple[str, ...]
    negations: tuple[str, ...]
    links: tuple[str, ...]
    question_starts: tuple[str, ...]
    judged_today: tuple[str, ...]
    replies: dict[str, str]


@dataclass(frozen=True)
class SurfaceWords:
    """Everything the correction surfaces and the digest say."""

    promo_name: str
    card: dict[str, str]
    chat: ChatWords
    digest: dict[str, str]
    needs_reply_max: int
    sent_labels: frozenset[str]
    learned_how: dict[str, str]

    @classmethod
    def load(cls, config_dir: Path | None = None) -> SurfaceWords:
        raw = _read(SHIPPED_CONFIG)
        if config_dir is not None:
            override = Path(config_dir) / OVERRIDE_RELATIVE_PATH
            if override.is_file():
                raw.update(_read(override))
        rebucket = raw.get("rebucket")
        digest = raw.get("digest")
        if not isinstance(rebucket, dict) or not isinstance(digest, dict):
            raise ValueError("judge.yaml: rebucket and digest must be mappings")
        chat = rebucket.get("chat")
        if not isinstance(chat, dict):
            raise ValueError("judge.yaml: rebucket.chat must be a mapping")
        how = digest.get("learned_how")
        if not isinstance(how, dict) or not how:
            raise ValueError("judge.yaml: digest.learned_how must be a mapping")
        return cls(
            promo_name=str(rebucket.get("promo_name") or "Promo"),
            card=_texts(rebucket.get("card"), _CARD_KEYS, "rebucket.card"),
            chat=ChatWords(
                window_days=int(chat.get("window_days", 14)),
                email_words=_words(chat, "email_words"),
                filler_words=_words(chat, "filler_words"),
                negations=_words(chat, "negations"),
                links=_words(chat, "links"),
                question_starts=_words(chat, "question_starts"),
                judged_today=_words(chat, "judged_today"),
                replies=_texts(chat.get("replies"), _REPLY_KEYS, "rebucket.chat.replies"),
            ),
            digest=_texts(digest, _DIGEST_KEYS, "digest"),
            needs_reply_max=int(digest.get("needs_reply_max", 10)),
            sent_labels=frozenset(str(x) for x in digest.get("sent_labels") or []),
            learned_how={str(k): str(v) for k, v in how.items()},
        )


def bucket_name(config: Any, words: SurfaceWords, key: str | None) -> str:
    """How surfaces say a bucket: judge.yaml's name, ``promo`` as ``promo_name``."""
    from .judgments import PROMO

    if key is None:
        return ""
    if key == PROMO:
        return words.promo_name
    return str(config.name(key))


__all__ = ["ChatWords", "SurfaceWords", "bucket_name"]
