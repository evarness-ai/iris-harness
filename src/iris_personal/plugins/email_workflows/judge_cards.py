"""The email judge's Action Center card: "What is this email?" (loop-proof PR 5).

Only an email whose effective bucket is Unsure gets a card — there is no card per
email. The card shows the sender, the subject and the snippet, and IRIS's guess with
its confidence; the owner answers Bill / Event / Needs reply / FYI / "Promo — hide it".
The answer is :func:`.judge_corrections.apply_correction` with ``source="card"`` — the
path every correction takes — so the Gmail label moves (the label step listens for
``email.judgment_corrected``) and the sender's next mail is judged with the hint. The
row is no longer Unsure, so the reconcile closes the card.

D18: every dated item ends. An unanswered card closes after ``unsure_card_days``
(``config/digest.yaml`` ``expiry:``) local days from its judgment; the row stays Unsure
in the store and on the web list, where it can still be corrected.

Cards come up to date on ``email.judged`` and ``email.judgment_corrected`` (a Gmail
relabel closes the card too) and on the Action Center's normal sync. Every word is in
``judge.yaml`` (``rebucket.card``).
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import Any

from iris_harness.sdk.pending_actions import (
    DesiredAction,
    PendingActionsSummary,
    reconcile,
)
from iris_harness.sdk.tasks import (
    ActionCard,
    ActionChoice,
    ActionFact,
    SourceKind,
    Task,
    TaskAction,
    TaskStore,
)

from .judge_config import JudgeConfig, labels_enabled
from .judge_corrections import apply_correction, valid_buckets
from .judge_view import JudgedEmail, default_emit, open_stores, with_emails
from .judge_words import SurfaceWords, bucket_name
from .judgments import PROMO

logger = logging.getLogger(__name__)

#: ``execute`` target of an Unsure email's card: ``email-judge:<message id>``.
TARGET_PREFIX = "email-judge:"
#: Dedup key of one email's card.
_DEDUP_PREFIX = "email-judge-unsure:"
#: The bucket that gets a card.
UNSURE = "unsure"
#: How many Unsure cards one pass looks at (newest first).
_SCAN_LIMIT = 500
_FACT_MAX = 200


def _fact(label: str, value: str) -> ActionFact | None:
    text = " ".join(str(value or "").split())
    if not text:
        return None
    if len(text) > _FACT_MAX:
        text = text[: _FACT_MAX - 1].rstrip() + "…"
    return ActionFact(label=label, value=text)


def _aware(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def days_expired(start: datetime, days: int, now: datetime, tz: tzinfo) -> bool:
    """Whether an item dated ``start`` has aged out after ``days`` local days: it lives
    through its own local day plus ``days`` more, and is expired from the next midnight
    (the digest's day-based expiry, ``sdk.digest.expiry_days``)."""
    last = start.astimezone(tz).date() + timedelta(days=days)
    return now >= datetime.combine(last + timedelta(days=1), time.min, tzinfo=tz)


class EmailJudgeActionProvider:
    """The ``email`` slice of the Action Center: one card per Unsure judged email."""

    source_kind: SourceKind = "email"

    def __init__(
        self,
        data_dir: Path,
        *,
        config_dir: Path | None = None,
        now: Any = None,
        tz: tzinfo | None = None,
        emit: Any = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._config_dir = config_dir
        self._now = now  # a callable returning an aware datetime (tests); None = the clock
        self._tz = tz
        self._emit = emit if emit is not None else default_emit

    # -- helpers ---------------------------------------------------------------

    def _clock(self) -> tuple[datetime, tzinfo]:
        tz = self._tz
        if tz is None:
            from iris_harness.sdk.time import iris_timezone

            tz = iris_timezone()
        now = self._now() if self._now is not None else datetime.now(tz)
        return now, tz

    def _config(self) -> tuple[JudgeConfig, SurfaceWords]:
        return JudgeConfig.load(self._config_dir), SurfaceWords.load(self._config_dir)

    # -- PendingActionProvider -------------------------------------------------

    def desired_actions(self) -> list[DesiredAction]:
        from iris_harness.sdk.digest import expiry_days

        stores = open_stores(self._data_dir, create=False)
        if stores is None:
            return []
        judgments, emails = stores
        config, words = self._config()
        now, tz = self._clock()
        days = expiry_days("unsure_card_days")  # declared in this plugin's manifest
        out: list[DesiredAction] = []
        for item in with_emails(emails, judgments.recent(bucket=UNSURE, limit=_SCAN_LIMIT)):
            judged = _aware(item.judgment.judged_at)
            if judged is None or days_expired(judged, days, now, tz):
                continue
            out.append(self._card(item, config, words))
        return out

    @staticmethod
    def _card(item: JudgedEmail, config: JudgeConfig, words: SurfaceWords) -> DesiredAction:
        j = item.judgment
        card = words.card
        confidence = f"{j.confidence:.2f}" if j.confidence is not None else "?"
        # The model's own pick when it was not sure enough (judge_one keeps it as
        # ``guess``); plain "Unsure" when it picked unsure itself.
        picked = str(j.fields.get("guess") or "")
        guess = card["guess"].format(
            bucket=config.name(picked if picked in config.keys else UNSURE),
            confidence=confidence,
        )
        choices = tuple(
            ActionChoice(value=key, label=config.name(key)) for key in config.keys if key != UNSURE
        ) + (
            ActionChoice(value=PROMO, label=card["promo_choice"]),
        )
        facts = tuple(
            f
            for f in (
                _fact("From", item.sender or (item.email.from_address if item.email else "")),
                _fact("Subject", item.subject),
                _fact("Says", item.email.snippet if item.email else ""),
                _fact("IRIS's guess", guess),
            )
            if f is not None
        )
        sender = item.sender or "an unknown sender"
        return DesiredAction(
            dedup_key=f"{_DEDUP_PREFIX}{j.message_id}",
            title=card["title"],
            description=f'{sender} · "{item.subject}"',
            source_id=j.message_id,
            action=TaskAction(
                kind="execute",
                label="Answer",
                target_id=f"{TARGET_PREFIX}{j.message_id}",
                safe=True,
                choices=choices,
                card=ActionCard(tag=card["tag"], facts=facts, note=card["note"]),
            ),
        )

    def invoke(self, task: Task) -> str:
        raise ValueError("an email card is answered with a choice")

    def invoke_choice(self, task: Task, choice: str, option: str | None) -> str:
        """The owner's answer: the email's bucket, through ``apply_correction``."""
        del option
        target = (task.action.target_id if task.action else None) or ""
        if not target.startswith(TARGET_PREFIX):
            raise ValueError(f"email judge provider has no card for {target!r}")
        message_id = target[len(TARGET_PREFIX) :]
        config, words = self._config()
        if choice not in valid_buckets(config):
            raise ValueError(f"unknown bucket {choice!r}")
        stores = open_stores(self._data_dir)
        assert stores is not None  # create=True always opens
        judgments, _ = stores
        correction = apply_correction(
            judgments, config, message_id, choice, source="card", emit=self._emit
        )
        if correction is None:
            raise ValueError("that email is no longer in IRIS")
        card = words.card
        if choice == PROMO:
            return card["answered_promo"]
        name = bucket_name(config, words, choice)
        if not labels_enabled():
            return card["answered_no_labels"].format(bucket=name)
        return card["answered"].format(bucket=name, label=config.bucket(choice).label)


def sync_judge_cards(data_dir: Path, task_store: TaskStore, **kwargs: Any) -> PendingActionsSummary:
    """Reconcile the Unsure cards against the judgments."""
    return reconcile(EmailJudgeActionProvider(data_dir, **kwargs), task_store)


def refresh_judge_cards(
    data_dir: Path, *, config_dir: Path | None = None
) -> PendingActionsSummary | None:
    """Bring the cards in line after a judgment or a correction. The task store sits
    beside ``email.db``. Best-effort: ``None`` on failure (the next sync catches up)."""
    try:
        if not (Path(data_dir) / "email.db").exists():
            return None
        task_store = TaskStore(db_path=Path(data_dir) / "tasks.db")
        task_store.ensure_schema()
        return sync_judge_cards(data_dir, task_store, config_dir=config_dir)
    except Exception:  # cards are advisory; the next sync catches up
        logger.warning("email judge: card refresh failed", exc_info=True)
        return None


__all__ = [
    "TARGET_PREFIX",
    "EmailJudgeActionProvider",
    "days_expired",
    "refresh_judge_cards",
    "sync_judge_cards",
]
