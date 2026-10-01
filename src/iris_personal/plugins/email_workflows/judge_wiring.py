"""Where the email judge runs (loop-proof PR 5): queue on sweep, judge as its own job.

Unjudged mail is invisible to the rest of IRIS, so ``email.new_arrived`` — the topic
triage, followup auto-resolution, the semantic index and every other reader of new mail
subscribe to — now means *released*, not *fetched*:

* ``email.swept`` (the sweep stored new mail) -> :func:`build_queue_handler`: each email
  the judge will judge becomes a ``waiting`` row; the rest (a skipped tab, a promo
  sender) is released at once as ``email.new_arrived``. It queues whether or not
  judging is on, so with the judge off new mail waits, hidden.
* The ``email_judge`` heartbeat -> :class:`EmailJudgeJob`: judge the waiting queue
  oldest first up to the cap, release what it judged (``email.new_arrived`` per
  account), then the label step. The Mac unreachable stops the run and leaves the rest
  waiting for the next one. The job never raises: a failure is a FAILED run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_harness.sdk.types import HeartbeatRun, HeartbeatStatus
from iris_personal.email.events import EMAIL_NEW_ARRIVED, EmailNewArrivedPayload

from .judge import JudgeRunReport, LLMCall, admit_swept_mail, llm_from_router, run_judge
from .judge_config import JudgeConfig, labels_enabled
from .judgments import JudgmentStore

logger = logging.getLogger(__name__)

Emit = Callable[[str, Any], None]


def _stores(db_path: Path | None) -> tuple[JudgmentStore, Any]:
    from iris_personal.email.store import EmailStore

    email_store = EmailStore(db_path=db_path) if db_path is not None else EmailStore()
    email_store.ensure_schema()
    store = JudgmentStore(db_path=email_store.db_path)
    store.ensure_schema()
    return store, email_store


def _config_dir(api: Any) -> Path | None:
    """The runtime's config dir, read when a handler runs (not when setup wires it)."""
    services = getattr(api, "services", None) if api is not None else None
    return getattr(services, "config_dir", None)


def _default_emit() -> Emit:
    from iris_harness.sdk.events import get_default_bus

    return get_default_bus().emit_sync


def release(emit: Emit, released: Mapping[str, list[str]] | Mapping[str, tuple[str, ...]]) -> int:
    """Emit ``email.new_arrived`` per account for mail now visible; returns the count."""
    total = 0
    for account_id, ids in released.items():
        if not ids:
            continue
        emit(
            EMAIL_NEW_ARRIVED,
            EmailNewArrivedPayload(
                account_id=account_id,
                new_message_ids=tuple(ids),
                count=len(ids),
                fell_back_to_cold_start=False,
            ),
        )
        total += len(ids)
    return total


def build_queue_handler(
    api: Any = None,
    *,
    db_path: Path | None = None,
    emit: Emit | None = None,
    clock: Callable[[], datetime] | None = None,
    config_dir: Path | None = None,
) -> Callable[[Any], None]:
    """The ``email.swept`` handler: queue what the judge will judge, release the rest.

    ``judge.yaml`` is read from the runtime's config dir (``api``), else ``config_dir``
    (a caller with no runtime, such as email setup's first fetch)."""

    def handle(payload: Any) -> None:
        try:
            store, email_store = _stores(db_path)
            admitted = admit_swept_mail(
                store,
                email_store,
                JudgeConfig.load(_config_dir(api) or config_dir),
                str(payload.account_id),
                list(payload.new_message_ids),
                now=clock() if clock is not None else None,
            )
        except Exception:  # never lose new mail to a queue failure
            logger.exception("email judge: queueing failed; releasing the mail unjudged")
            admitted_released: tuple[str, ...] = tuple(payload.new_message_ids)
        else:
            admitted_released = admitted.released
            logger.debug(
                "email judge: %d waiting, %d released",
                len(admitted.waiting),
                len(admitted_released),
            )
        if admitted_released:
            (emit or _default_emit())(
                EMAIL_NEW_ARRIVED,
                EmailNewArrivedPayload(
                    account_id=str(payload.account_id),
                    new_message_ids=admitted_released,
                    count=len(admitted_released),
                    fell_back_to_cold_start=bool(payload.fell_back_to_cold_start),
                ),
            )

    return handle


def mail_providers(account_ids: set[str]) -> dict[str, Any]:
    """account id -> its mounted mail provider (accounts with none are left out)."""
    from iris_personal.email.providers import mail_provider_for

    out: dict[str, Any] = {}
    for account_id in sorted(account_ids):
        provider = mail_provider_for(account_id)
        if provider is not None:
            out[account_id] = provider
    return out


def sync_labels_after(store: JudgmentStore, config: JudgeConfig) -> Any:
    """The label step, when labels are on (``IRIS_EMAIL_JUDGE_LABELS``)."""
    if not labels_enabled():
        return None
    from .judge_labels import sync_labels

    providers: Mapping[str, Any] = mail_providers({j.account_id for j in store.labels_due()})
    return sync_labels(store, config, providers)


def judge_and_release(
    *,
    llm: LLMCall | None,
    config_dir: Path | None = None,
    db_path: Path | None = None,
    emit: Emit | None = None,
    labels: bool = True,
    **run_kwargs: Any,
) -> tuple[JudgeRunReport, str]:
    """One judge run, then the release of what it judged, then the label step.
    Returns the report and a one-line note for the heartbeat run.

    ``labels=False`` leaves the label step for later: email setup judges before the
    owner has seen the label preview, so its labels wait for that approval (R4)."""
    store, email_store = _stores(db_path)
    config = JudgeConfig.load(config_dir)
    send = emit or _default_emit()
    report = run_judge(store, email_store, config, llm=llm, emit=send, **run_kwargs)
    oldest = store.waiting(limit=1)
    report.oldest_waiting = oldest[0].created_at if oldest else None
    note = f"judge: {report.summary()}"
    if report.released:
        note += f"; released {release(send, report.released)}"
    if report.enabled and not report.dry_run and labels:
        try:
            labels = sync_labels_after(store, config)
            if labels is not None:
                note += f"; labels: {labels}"
        except Exception as exc:  # labels never fail the run
            logger.exception("email judge: label step failed")
            note += f"; labels failed: {type(exc).__name__}"
    return report, note


@dataclass
class EmailJudgeJob:
    """The ``email_judge`` heartbeat handler. Never raises."""

    judge: Callable[[], tuple[JudgeRunReport, str]]

    def __call__(self, definition: Any) -> HeartbeatRun:
        name = getattr(definition, "name", "email_judge")
        try:
            report, note = self.judge()
        except Exception as exc:  # a judge failure never crashes the heartbeat
            logger.exception("email judge: run failed")
            return HeartbeatRun(
                name=name,
                status=HeartbeatStatus.FAILED,
                finished_at=datetime.now(UTC),
                output="judge failed",
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
        idle = not report.enabled or report.no_model or (report.unreachable and not report.judged)
        return HeartbeatRun(
            name=name,
            status=HeartbeatStatus.SKIPPED if idle else HeartbeatStatus.SUCCESS,
            finished_at=datetime.now(UTC),
            output=note,
            error=report.unreachable_error[:500] if report.unreachable else "",
            result=report.result(),
        )


def build_judge_job(api: Any) -> EmailJudgeJob:
    """Production wiring: the tier router's ``email_judge`` tier (looked up every run, so
    an owner's tier edit applies) and the process bus."""

    def judge() -> tuple[JudgeRunReport, str]:
        services = getattr(api, "services", None)
        return judge_and_release(
            llm=llm_from_router(getattr(services, "tier_router", None)),
            config_dir=_config_dir(api),
        )

    return EmailJudgeJob(judge=judge)


__all__ = [
    "EmailJudgeJob",
    "build_judge_job",
    "build_queue_handler",
    "judge_and_release",
    "mail_providers",
    "release",
    "sync_labels_after",
]
