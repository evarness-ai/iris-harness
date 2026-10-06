"""The email judge (loop-proof PR 5, plan D10-D12): one bucket per non-promo email.

The sweep stores new mail; the judge's queue (:func:`admit_swept_mail`) makes each email
it will judge a ``waiting`` row — hidden from the rest of IRIS until judged — and
releases the rest at once. The ``email_judge`` job (three times a day) reads the
waiting rows oldest first, up to ``IRIS_EMAIL_JUDGE_MAX`` per run; the rest keep
waiting. For each it reads the sender, the subject and up to ``body_chars`` of the body
(fetched live from the mailbox, never stored; the stored snippet when that fails), asks
the ``email_judge`` tier for one JSON object — a bucket from ``judge.yaml``, a
confidence, a reason and any figures or dates literally in the email — records the
verdict on the email's row and reports it as released (``email.new_arrived``).

* Below ``unsure_below`` confidence, or a bucket ``judge.yaml`` does not know, the row
  is ``unsure`` and the judge's pick is kept as ``fields["guess"]``.
* One email failing (a bad reply, a store error) makes that email ``unsure`` with
  ``error`` set; the run goes on.
* The model unreachable (the Mac asleep, the breaker open) stops the run: every email
  not yet judged stays ``waiting`` (and hidden). The judge never tries another model or
  the cloud.

What the judge is told comes from ``judge.yaml``: today's date in the owner's zone
(without it the model dates things in the wrong year), the bucket definitions, the
owner's past corrections for this sender (the learning effect) and the classifier's
learned category as a hint. A Bill records its figures on the row only: the finance
reader keeps creating dues.
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, tzinfo
from email.utils import parseaddr
from typing import TYPE_CHECKING, Any

from iris_harness.sdk.content import wrap_external_content
from iris_harness.sdk.llm import (
    CodingLLMClient,
    CodingLLMConfig,
    JsonReply,
    LLMUnreachable,
    supports_json_schema,
)

from .judge_config import (
    EMAIL_JUDGED,
    JUDGE_TIER,
    EmailJudgedPayload,
    JudgeConfig,
    judge_enabled,
    judge_max_per_run,
    unsure_below,
)
from .judgments import PROMO, WAITING, JudgmentStore

if TYPE_CHECKING:
    from iris_harness.sdk.services import TierRouterService
    from iris_personal.email.contracts import EmailMessage
    from iris_personal.email.provider_api import ProgressFn
    from iris_personal.email.store import EmailStore

logger = logging.getLogger(__name__)

# The bucket every doubt lands in; judge.yaml must define it (JudgeConfig.load checks).
UNSURE = "unsure"

# What the judge may copy out of an email: amounts, then ISO dates.
MONEY_FIELDS = ("min_due", "statement_balance", "credit_balance")
DATE_FIELDS = ("due_date", "event_start", "reply_by")
FIGURE_FIELDS = MONEY_FIELDS + DATE_FIELDS

# How the email was read (``fields["read"]``).
READ_BODY = "body"
READ_SNIPPET = "snippet"

# (system prompt, user message, JSON schema) -> the model's JSON reply.
LLMCall = Callable[[str, str, Mapping[str, Any]], JsonReply]
# (message, max_chars) -> the body text, read live.
BodyFetcher = Callable[["EmailMessage", int], str]
Emit = Callable[[str, Any], None]


# -- the call -----------------------------------------------------------------------


def build_schema(config: JudgeConfig) -> dict[str, Any]:
    """The JSON schema the model's reply must fit; the bucket enum is judge.yaml's keys."""
    properties: dict[str, Any] = {
        "bucket": {"type": "string", "enum": list(config.keys)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    }
    for name in MONEY_FIELDS:
        properties[name] = {"type": ["number", "null"]}
    for name in DATE_FIELDS:
        properties[name] = {"type": ["string", "null"]}
    return {"type": "object", "properties": properties, "required": list(properties)}


def make_tier_llm(config: CodingLLMConfig, *, llm_client: CodingLLMClient | None = None) -> LLMCall:
    """The judge's governed call on ``config`` (the ``email_judge`` tier's): its local
    Ollama only, never another route -- or the scripted fake model, which plays Ollama's
    schema-constrained decoder in-process (the demo, tests).

    Every email is one governed :meth:`CodingLLMClient.invoke_json`, the way the finance
    sweep reads each email in full (``make_narrative_llm_call``, agent type ``chat``):
    the pre-LLM hooks, the audit row and the egress log fire per email, and the shared
    circuit breaker applies. ``llm_client`` is for tests.
    """
    if not supports_json_schema(config.provider):
        raise ValueError(
            f"the {JUDGE_TIER} tier must be on provider 'ollama' (the Mac, direct) or the "
            f"scripted 'fake'; it is {config.provider!r}"
        )
    if llm_client is None:
        llm_client = CodingLLMClient(config, governance_agent_type="chat")

    def call(system: str, user: str, schema: Mapping[str, Any]) -> JsonReply:
        return llm_client.invoke_json(system_prompt=system, user_prompt=user, schema=schema)

    return call


def llm_from_router(router: TierRouterService | None) -> LLMCall | None:
    """The judge's call from the tier router, or ``None`` when there is no usable tier.

    The tier must exist by name and the router must resolve the ``email_judge`` intent
    to it: ``get_llm_config`` falls back to another tier for an unknown intent, and the
    judge's accuracy is only comparable run to run on its own pinned model.
    """
    if router is None or router.get_tier_by_name(JUDGE_TIER) is None:
        return None
    config = router.get_llm_config(JUDGE_TIER)
    if not isinstance(config, CodingLLMConfig) or config.tier_name != JUDGE_TIER:
        logger.warning(
            "email judge: intent %s does not resolve to its tier; not judging", JUDGE_TIER
        )
        return None
    try:
        return make_tier_llm(config)
    except ValueError:
        logger.warning("email judge: tier %s is not a local Ollama tier; not judging", JUDGE_TIER)
        return None


# -- the prompt ---------------------------------------------------------------------


def _fill(template: str, values: Mapping[str, str]) -> str:
    """``{name}`` placeholders only: a stray brace in a template or an email is kept."""
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    return out


def owner_zone() -> tzinfo:
    from iris_harness.sdk.time import iris_timezone

    return iris_timezone()


def build_prompt(
    config: JudgeConfig,
    *,
    now: datetime,
    tz: tzinfo,
    hints: Sequence[tuple[str, str]] = (),
    triage_category: str | None = None,
) -> str:
    """The system prompt: ``judge.yaml``'s ``prompt`` with its placeholders filled."""
    local = now.astimezone(tz)
    definitions = "\n".join(f"- {b.key}: {b.definition}" for b in config.buckets)
    hint_lines = (
        [_fill(config.hint_line, {"subject": s, "bucket": b}) for s, b in hints]
        if config.hint_line
        else []
    )
    hints_text = "".join(f"{line}\n" for line in hint_lines)
    triage_text = (
        _fill(config.triage_hint_line, {"category": triage_category}) + "\n"
        if triage_category and config.triage_hint_line
        else ""
    )
    return _fill(
        config.prompt,
        {
            "today": f"{local:%A} {local:%Y-%m-%d}",
            # Weekday names resolve against a list, not the model's arithmetic
            # ("by Thursday" was dated a Friday without it).
            "week": ", ".join(
                f"{d:%A} {d:%Y-%m-%d}" for d in (local + timedelta(days=i) for i in range(1, 8))
            ),
            "tz": str(getattr(tz, "key", None) or tz),
            "bucket_definitions": definitions,
            "triage_hint": triage_text,
            "hints": hints_text,
        },
    ).rstrip()


def build_email_message(config: JudgeConfig, message: EmailMessage, body: str) -> str:
    """The user message: the sender, subject and body, all text a third party wrote, so the
    whole of it goes in one untrusted-content envelope with instruction-like spans redacted
    (issue #148). The model reads it as data; its reply is the schema-constrained verdict."""
    values = {"sender": message.from_address, "subject": message.subject, "body": body}
    text = (
        "\n\n".join(values.values())
        if not config.email_message
        else _fill(config.email_message, values)
    )
    return wrap_external_content(text, source="email", tool="judge")


# -- one email ----------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    bucket: str
    confidence: float | None
    fields: dict[str, Any]
    reason: str = ""
    model: str = ""
    latency_ms: int | None = None
    error: str = ""


def _confidence(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, value))


def _figures(data: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in MONEY_FIELDS:
        value = data.get(name)
        if isinstance(value, int | float) and not isinstance(value, bool):
            out[name] = value
    for name in DATE_FIELDS:
        value = data.get(name)
        if isinstance(value, str) and value.strip():
            out[name] = value.strip()
    return out


def triage_category(message: EmailMessage) -> str | None:
    """The category the classifier learned for this email (a mailbox tab is not one)."""
    if message.classified_category and message.classified_source != "vendor":
        return str(message.classified_category)
    return None


def is_automated_sender(from_address: str, config: JudgeConfig) -> bool:
    """The address's local part carries one of judge.yaml's automated-sender words
    ("noreply", "alerts", "updates"…)."""
    address = parseaddr(from_address or "")[1] or from_address or ""
    local = address.split("@", 1)[0].lower()
    return any(word in local for word in config.automated_sender_words)


def judge_one(
    message: EmailMessage,
    *,
    config: JudgeConfig,
    llm: LLMCall,
    body: str,
    read: str = READ_BODY,
    hints: Sequence[tuple[str, str]] = (),
    now: datetime | None = None,
    tz: tzinfo | None = None,
    threshold: float | None = None,
) -> Verdict:
    """Ask the model about one email. :class:`LLMUnreachable` propagates (the caller
    stops the run); any other failure is the caller's to record as ``unsure``."""
    system = build_prompt(
        config,
        now=now or datetime.now(UTC),
        tz=tz or owner_zone(),
        hints=hints,
        triage_category=triage_category(message),
    )
    user = build_email_message(config, message, body)
    reply = llm(system, user, build_schema(config))
    data = reply.data
    picked = str(data.get("bucket") or "")
    confidence = _confidence(data.get("confidence"))
    fields = _figures(data)
    fields["read"] = read
    limit = unsure_below(config) if threshold is None else threshold
    bucket = picked
    if picked not in config.keys or confidence is None or confidence < limit:
        bucket = UNSURE
        if picked and picked != UNSURE:
            fields["guess"] = picked
    elif bucket in config.person_only and is_automated_sender(message.from_address, config):
        # Only a real person's email can need a reply: a system sender's "action
        # required" never does (the model says so ~1 in 3 times without this rule).
        fields["guess"] = bucket
        fields["rule"] = "automated sender"
        bucket = config.person_only_else
    return Verdict(
        bucket=bucket,
        confidence=confidence,
        fields=fields,
        reason=str(data.get("reason") or "")[:500],
        model=reply.model,
        latency_ms=reply.latency_ms,
    )


# -- which emails -------------------------------------------------------------------


def load_message(email_store: EmailStore, message_id: str) -> EmailMessage | None:
    """The one read of a queued email. A ``waiting`` email is held — hidden from every
    other reader of the store — so the judge must ask for it explicitly.
    """
    return email_store.get(message_id, include_held=True)


def skip_reason(message: EmailMessage, config: JudgeConfig, store: JudgmentStore) -> str | None:
    """Why this email is not judged (a skipped tab, a promo sender), or ``None``."""
    tabs = config.skip_labels.intersection(message.labels)
    if tabs:
        return f"tab {sorted(tabs)[0]}"
    if store.is_promo_sender(message.from_address):
        return "promo sender"
    return None


@dataclass(frozen=True)
class Admitted:
    """What the queue did with one sweep's new mail."""

    waiting: tuple[str, ...]
    released: tuple[str, ...]


def admit_swept_mail(
    store: JudgmentStore,
    email_store: EmailStore,
    config: JudgeConfig,
    account_id: str,
    message_ids: Sequence[str],
    *,
    now: datetime | None = None,
) -> Admitted:
    """Queue what the sweep stored: each email the judge will judge becomes a
    ``waiting`` row (hidden until judged); the rest is released at once — a skipped tab,
    a promo sender, an id the store does not have, an email already judged, or one
    received before the ``queue_hours`` window (a cold start re-fetching old mail).

    Queues whether or not judging is on: turning it on later drains the queue."""
    cutoff = (
        (now or datetime.now(UTC)) - timedelta(hours=config.queue_hours)
        if config.queue_hours > 0
        else None
    )
    waiting: list[str] = []
    released: list[str] = []
    for mid in message_ids:
        message = load_message(email_store, mid)
        if (
            message is None
            or (cutoff is not None and message.received_at < cutoff)
            or skip_reason(message, config, store) is not None
        ):
            released.append(mid)
            continue
        row = store.get(mid)
        if row is not None and row.status != WAITING:
            released.append(mid)
            continue
        waiting.append(mid)
    if waiting:
        store.mark_waiting(account_id, waiting)
    return Admitted(waiting=tuple(waiting), released=tuple(released))


def fetch_body_live(message: EmailMessage, max_chars: int) -> str:
    """The body from the account's mail provider (read live, never stored)."""
    from iris_personal.email.providers import mail_provider_for

    provider = mail_provider_for(message.account_id)
    if provider is None:
        raise LookupError(f"no mail provider mounted for {message.account_id}")
    return provider.fetch_message_body(message.account_id, message.id, max_chars=max_chars)


def _read(
    message: EmailMessage, config: JudgeConfig, fetch_body: BodyFetcher | None
) -> tuple[str, str]:
    if fetch_body is not None:
        try:
            body = fetch_body(message, config.body_chars)
        except Exception:  # noqa: BLE001 — a failed read falls back to the snippet
            logger.info("email judge: body read failed for %s; using the snippet", message.id)
        else:
            if body and body.strip():
                return body[: config.body_chars], READ_BODY
    return message.snippet, READ_SNIPPET


# -- a run --------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgedItem:
    message_id: str
    account_id: str
    sender: str
    subject: str
    verdict: Verdict


@dataclass
class JudgeRunReport:
    run_id: str
    enabled: bool = True
    dry_run: bool = False
    judged: int = 0
    unsure: int = 0
    errors: int = 0
    skipped: int = 0
    waiting: int = 0
    unreachable: bool = False
    unreachable_error: str = ""
    no_model: bool = False
    # When the oldest email still waiting was queued (ISO), for the health check.
    oldest_waiting: str | None = None
    counts: dict[str, int] = field(default_factory=dict)
    items: list[JudgedItem] = field(default_factory=list)
    # account id -> the ids this run took off the waiting queue (judged, or closed as
    # skipped): the ones to release as ``email.new_arrived``. Empty on a dry run.
    released: dict[str, list[str]] = field(default_factory=dict)

    def result(self) -> dict[str, Any]:
        """The structured result the heartbeat run keeps (``job_watch`` reads it)."""
        return {
            "judged": self.judged,
            "waiting": self.waiting,
            "oldest_waiting": self.oldest_waiting,
            "unsure": self.unsure,
            "errors": self.errors,
            "unreachable": self.unreachable,
        }

    def summary(self) -> str:
        if not self.enabled:
            return f"judge off; {self.waiting} waiting"
        if self.no_model:
            return f"judge has no local model tier; {self.waiting} waiting"
        parts = [f"judged {self.judged}"]
        parts += [f"{k} {v}" for k, v in sorted(self.counts.items())]
        if self.errors:
            parts.append(f"errors {self.errors}")
        if self.skipped:
            parts.append(f"skipped {self.skipped}")
        parts.append(f"waiting {self.waiting}")
        if self.unreachable:
            parts.append("model unreachable")
        return ", ".join(parts)


def _candidates(
    store: JudgmentStore,
    email_store: EmailStore,
    config: JudgeConfig,
    cap: int,
    backfill: int,
) -> list[tuple[str, str, bool]]:
    """``(message_id, account_id, was_waiting)`` to judge: waiting rows first (oldest
    first), then — only when asked (the CLI) — the newest emails that have no row yet
    (already visible, so never released again)."""
    picked = [(j.message_id, j.account_id, True) for j in store.waiting(limit=cap)]
    room = min(cap - len(picked), backfill)
    if room <= 0:
        return picked
    recent: list[EmailMessage] = []
    for account_id in email_store.list_accounts():
        recent.extend(email_store.list_recent(account_id, limit=room * 3))
    recent.sort(key=lambda m: m.received_at, reverse=True)
    have = {mid for mid, _, _ in picked}
    fresh = set(store.missing([m.id for m in recent if m.id not in have]))
    for message in recent:
        if len(picked) >= cap or room <= 0:
            break
        if message.id in fresh and skip_reason(message, config, store) is None:
            picked.append((message.id, message.account_id, False))
            room -= 1
    return picked


def run_judge(
    store: JudgmentStore,
    email_store: EmailStore,
    config: JudgeConfig,
    *,
    llm: LLMCall | None,
    fetch_body: BodyFetcher | None = fetch_body_live,
    emit: Emit | None = None,
    limit: int | None = None,
    backfill: int = 0,
    dry_run: bool = False,
    now: datetime | None = None,
    tz: tzinfo | None = None,
    run_id: str | None = None,
    progress: ProgressFn | None = None,
) -> JudgeRunReport:
    """Judge what is waiting, up to ``limit`` (default ``IRIS_EMAIL_JUDGE_MAX``).

    ``dry_run`` writes nothing (no rows, no events): the verdicts are only in the
    report. ``backfill`` also takes up to that many recent emails that have no row yet.
    ``progress``, when given, is called before each candidate is judged.
    """
    report = JudgeRunReport(run_id=run_id or uuid.uuid4().hex[:12], dry_run=dry_run)
    if not judge_enabled():
        report.enabled = False
        report.waiting = store.count_waiting()
        return report
    if llm is None:
        report.no_model = True
        report.waiting = store.count_waiting()
        return report
    cap = limit if limit is not None else judge_max_per_run()
    moment = now or datetime.now(UTC)
    zone = tz or owner_zone()
    threshold = unsure_below(config)
    counts: Counter[str] = Counter()

    released: dict[str, list[str]] = {}

    def _release(was_waiting: bool, account: str, mid: str) -> None:
        if was_waiting and not dry_run:
            released.setdefault(account, []).append(mid)

    candidates = _candidates(store, email_store, config, cap, backfill)
    total = len(candidates)
    for i, (message_id, account_id, was_waiting) in enumerate(candidates, start=1):
        if progress is not None:
            progress(i / total, f"judging {i}/{total}")
        message = load_message(email_store, message_id)
        reason = (
            "email not in the store" if message is None else skip_reason(message, config, store)
        )
        if message is None or reason is not None:
            # Queued, then gone or skipped (the owner marked the sender promo since):
            # closed as promo, so it never waits forever and never gets an IRIS/* label.
            report.skipped += 1
            if not dry_run:
                store.record(
                    message_id,
                    account_id,
                    bucket=PROMO,
                    confidence=None,
                    tier=JUDGE_TIER,
                    run_id=report.run_id,
                    error=f"skipped: {reason}",
                )
            _release(was_waiting, account_id, message_id)
            continue
        try:
            body, read = _read(message, config, fetch_body)
            verdict = judge_one(
                message,
                config=config,
                llm=llm,
                body=body,
                read=read,
                hints=store.owner_buckets_for_sender(message.from_address),
                now=moment,
                tz=zone,
                threshold=threshold,
            )
        except LLMUnreachable as exc:
            logger.warning("email judge: model unreachable, stopping this run: %s", exc)
            report.unreachable = True
            report.unreachable_error = str(exc)
            break
        except Exception as exc:  # one bad email never stops the run
            logger.exception("email judge: judging %s failed", message_id)
            report.errors += 1
            verdict = Verdict(
                bucket=UNSURE,
                confidence=None,
                fields={},
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
        counts[verdict.bucket] += 1
        report.judged += 1
        report.items.append(
            JudgedItem(
                message_id=message_id,
                account_id=message.account_id,
                sender=message.from_address,
                subject=message.subject,
                verdict=verdict,
            )
        )
        if dry_run:
            continue
        store.record(
            message_id,
            message.account_id,
            bucket=verdict.bucket,
            confidence=verdict.confidence,
            fields=verdict.fields,
            reason=verdict.reason,
            model=verdict.model,
            tier=JUDGE_TIER,
            latency_ms=verdict.latency_ms,
            run_id=report.run_id,
            error=verdict.error,
        )
        _release(was_waiting, account_id, message_id)
        if emit is not None:
            try:
                emit(
                    EMAIL_JUDGED,
                    EmailJudgedPayload(
                        message_id=message_id,
                        account_id=message.account_id,
                        bucket=verdict.bucket,
                        confidence=verdict.confidence,
                    ),
                )
            except Exception:  # a subscriber's failure is not the judge's
                logger.exception("email judge: emitting %s failed", EMAIL_JUDGED)

    report.released = released
    report.counts = dict(counts)
    report.unsure = counts.get(UNSURE, 0)
    report.waiting = store.count_waiting()
    return report


__all__ = [
    "DATE_FIELDS",
    "FIGURE_FIELDS",
    "MONEY_FIELDS",
    "UNSURE",
    "JudgeRunReport",
    "JudgedItem",
    "LLMCall",
    "Verdict",
    "build_email_message",
    "build_prompt",
    "build_schema",
    "fetch_body_live",
    "judge_one",
    "load_message",
    "llm_from_router",
    "make_tier_llm",
    "Admitted",
    "admit_swept_mail",
    "run_judge",
    "skip_reason",
]
