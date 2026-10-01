"""Email followup detection + auto-resolution (Phase 2 Track 2B).

Two halves:

  Detection — user-invoked via ``iris email detect-followups``. For
  each classified email in the requested window, asks Tier 3 local
  ("does the user need to send a personal reply?") and, on yes,
  upserts a Task whose ``wait_for`` carries the thread identity. The
  CLI is the trigger because each call costs a Tier 3 invocation —
  same architectural pattern as ``triage-batch`` (ADR-0022 §1).

  Auto-resolution — subscriber on ``email.new_arrived``. For each
  newly-arrived message that belongs to a thread we have an open
  followup for, calls ``TaskStore.resolve_wait``. No LLM call. The
  resolved task does NOT auto-complete (ADR-0014 #10) — the user
  decides whether to mark it done after seeing the resolved followup
  in the morning brief.

Dedup contract: ``followup_key(provider, thread_id)`` — exactly one
followup task per email thread. Re-running detection over the same
thread is a no-op (TaskStore.upsert first-write wins).

Tier 3 client: ``LlamaServerClient`` transitional shim, same as
``iris_personal.plugins.email_workflows.discovery`` and ``iris_personal.plugins.email_workflows.triage`` use. Retires when
ADR-0022 §5 / Track 1H lands.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from iris_harness.sdk.events import EventBus, get_default_bus
from iris_harness.sdk.tasks import TaskStore, WaitFor
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.events import (
    EMAIL_CLASSIFIED,
    EMAIL_NEW_ARRIVED,
    EmailClassifiedPayload,
    EmailNewArrivedPayload,
)
from iris_personal.email.feedback_keys import email_followup_dims_from
from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows.discovery import (
    LLAMA_SERVER_BASE_DEFAULT,
    LlamaServerClient,
)

logger = logging.getLogger(__name__)


def followup_key(provider: str, thread_id: str) -> str:
    """Task dedup key for an email followup waiting on a reply (ADR-0005 convention).

    Colon-separated, lowercase namespace; callers canonicalize the ids first. Lived in
    ``iris_harness.services.tasks.dedup_keys`` until the core/SDK boundary plan (PR 2):
    this module is its only producer.

    Args:
        provider: 'gmail', 'outlook', etc.
        thread_id: provider-native thread identifier.
    """
    return f"followup:{provider}:{thread_id}"


DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# Gmail's own "tab" labels for bulk, one-to-many mail. Personal
# correspondence lands in CATEGORY_PERSONAL (or carries no CATEGORY_*
# label at all); promotions/newsletters/notifications never warrant a
# personal *reply*, even when the body personalizes the greeting
# ("Rahul, ...") — that is marketing, not a question to the user.
# Honoring the provider's bulk-mail signal is more reliable than the
# topical classifier, which can bucket vendor blasts into a benign
# subject category (e.g. "learning/finance"). See issue 0028.
BULK_MAIL_LABELS = frozenset(
    {
        "CATEGORY_PROMOTIONS",
        "CATEGORY_SOCIAL",
        "CATEGORY_UPDATES",
        "CATEGORY_FORUMS",
    }
)


def _is_bulk_mail(email: EmailMessage) -> bool:
    """True when the provider tagged this as bulk/promotional mail.

    Gmail sets ``CATEGORY_*`` tab labels on every message; the four in
    ``BULK_MAIL_LABELS`` mark mail that is never a personal reply
    obligation. Other providers (IMAP/Outlook) carry no such label and
    fall through to the LLM verdict.
    """
    return any(label in BULK_MAIL_LABELS for label in email.labels)


# Category roots (ADR-0017: email/root/branch/leaf) whose mail can plausibly
# warrant a personal reply. The user's framing: "only act on important,
# personal, financial, or genuine person-to-person emails." Roots outside this
# set — learning, news, social, shopping, transactional, tools, automotive,
# other — are filed correspondence, not a reply obligation, so a confident
# classification into one of them suppresses followup creation. Tunable via
# ``IRIS_FOLLOWUP_ACTIONABLE_ROOTS`` (comma-separated) without a code change.
DEFAULT_ACTIONABLE_ROOTS = frozenset({"personal", "work", "finance", "jobs", "travel", "community"})


def _actionable_roots(override: frozenset[str] | None = None) -> frozenset[str]:
    if override is not None:
        return override
    raw = os.getenv("IRIS_FOLLOWUP_ACTIONABLE_ROOTS", "").strip()
    if raw:
        return frozenset(r.strip().lower() for r in raw.split(",") if r.strip())
    return DEFAULT_ACTIONABLE_ROOTS


def _category_root(category_path: str | None) -> str | None:
    """Extract the root from an ``email/root/branch/leaf`` path, or None."""
    if not category_path:
        return None
    parts = category_path.split("/")
    # Tolerate both "email/root/..." and a bare "root/..." shape.
    if parts and parts[0] == "email":
        parts = parts[1:]
    return parts[0].strip().lower() if parts and parts[0].strip() else None


def email_followup_dims(email: EmailMessage, category_path: str | None = None) -> dict[str, str]:
    """Suppression-key dimensions for an email followup message.

    ``category_path`` is accepted for call-site symmetry but intentionally
    unused — the key is sender-scoped (see :func:`email_followup_dims_from`).
    """
    return email_followup_dims_from(email.account_id, email.from_address)


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FollowupDecision:
    """LLM verdict on whether one email warrants a followup task.

    ``confidence`` is the LLM's self-reported number; downstream code
    gates on ``DEFAULT_CONFIDENCE_THRESHOLD`` before creating tasks.
    """

    needs_reply: bool
    rationale: str
    confidence: float


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are an email triage assistant. Given one inbound email and its "
    "category, decide whether the user needs to send a personal reply.\n"
    "\n"
    "Reply NEEDED when:\n"
    "  - A person addresses the user directly with a question, request, "
    "or open thread that expects a response.\n"
    "  - The user owes a personal follow-up commitment (a promise to "
    "send something, a meeting confirmation, etc.).\n"
    "\n"
    "Reply NOT needed when:\n"
    "  - Automated notifications, receipts, OTPs, statements.\n"
    "  - Newsletters, marketing, mass announcements.\n"
    "  - Confirmations of an action the user already completed.\n"
    "  - System / no-reply / do-not-reply senders.\n"
    "\n"
    'Output strict JSON: {"needs_reply": true|false, '
    '"rationale": "<one short sentence>", "confidence": 0.0-1.0}.'
)


def _build_user_prompt(email: EmailMessage, category_path: str | None) -> str:
    return (
        f"From: {email.from_address}\n"
        f"Subject: {email.subject}\n"
        f"Category: {category_path or '(unclassified)'}\n"
        f"Snippet: {email.snippet}\n"
    )


@dataclass
class FollowupDetector:
    """Tier-3-local detector for "does this email need a reply?".

    Pass a custom ``client`` for tests (any object with
    ``complete_json(system, user) -> str`` works). The default uses
    the same transitional ``LlamaServerClient`` shim as triage and
    discovery.
    """

    client: Any = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = LlamaServerClient(base_url=LLAMA_SERVER_BASE_DEFAULT)

    def detect(self, email: EmailMessage, *, category_path: str | None = None) -> FollowupDecision:
        """Ask the LLM and parse the JSON verdict.

        Soft-fails into a ``needs_reply=False`` decision with the
        error in ``rationale`` so the caller can log and skip rather
        than raising. Network or model errors must not block other
        emails in a batch.
        """
        try:
            raw = self.client.complete_json(
                _SYSTEM_PROMPT, _build_user_prompt(email, category_path)
            )
        except Exception as exc:  # noqa: BLE001 — soft-fail per-email
            logger.warning("followup detect: LLM call failed for %s: %s", email.id, exc)
            return FollowupDecision(
                needs_reply=False, rationale=f"llm-error: {exc}", confidence=0.0
            )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "followup detect: invalid JSON from LLM for %s: %s (raw=%r)",
                email.id,
                exc,
                raw,
            )
            return FollowupDecision(
                needs_reply=False,
                rationale=f"json-decode-error: {exc}",
                confidence=0.0,
            )

        needs = bool(data.get("needs_reply", False))
        rationale = str(data.get("rationale", "") or "")
        confidence = float(data.get("confidence", 0.0) or 0.0)
        confidence = max(0.0, min(1.0, confidence))
        return FollowupDecision(needs_reply=needs, rationale=rationale, confidence=confidence)


# ---------------------------------------------------------------------------
# Task creation (CLI-invoked path)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FollowupOutcome:
    """One per-email result from ``detect_and_persist``.

    ``action`` ∈ {"created", "skipped:no-thread", "skipped:bulk-category",
    "skipped:non-actionable-category", "skipped:suppressed-by-feedback",
    "skipped:low-confidence", "skipped:no-reply-needed",
    "skipped:already-exists", "soft-failed"}.
    """

    email_id: str
    action: str
    detail: str = ""
    task_id: str | None = None


def detect_and_persist(
    email: EmailMessage,
    *,
    category_path: str | None,
    detector: FollowupDetector,
    task_store: TaskStore,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    feedback_store: Any = None,
    actionable_roots: frozenset[str] | None = None,
) -> FollowupOutcome:
    """Run detection for one email and, on positive verdict, upsert a Task.

    Idempotent — re-invocation for the same thread returns the
    existing task via the ``followup_key`` dedup contract.

    Gates, cheapest-first, before the Tier 3 call:
      1. no thread_id (no anchor for auto-resolution);
      2. provider bulk/promotional label (issue 0028);
      3. classified into a non-actionable category root;
      4. suppressed by prior user "not useful" feedback (``feedback_store``,
         the generic surface-feedback spine — optional).

    ``category_path`` falls back to ``email.classified_category`` when not
    given, so the LLM and the gates see the real triage verdict. A vendor
    classification (the mailbox's tab, not a triage verdict) is not used; bulk
    tabs are already caught by the label gate below.
    """
    if not email.thread_id:
        return FollowupOutcome(email.id, "skipped:no-thread")

    triage_verdict = email.classified_category if email.classified_source != "vendor" else None
    category = category_path or triage_verdict

    if _is_bulk_mail(email):
        # Provider tabbed this as bulk/promotional — never a personal
        # reply obligation. Skip before the Tier 3 call (cheaper, and
        # the LLM is easily fooled by personalized marketing greetings).
        return FollowupOutcome(
            email.id,
            "skipped:bulk-category",
            detail=f"labels={sorted(set(email.labels) & BULK_MAIL_LABELS)}",
        )

    root = _category_root(category)
    if root is not None and root not in _actionable_roots(actionable_roots):
        # Confidently filed under a non-actionable root (learning, news,
        # social, shopping, ...). Not a person-to-person reply obligation.
        return FollowupOutcome(
            email.id,
            "skipped:non-actionable-category",
            detail=f"root={root!r}",
        )

    if feedback_store is not None:
        dims = email_followup_dims(email, category)
        try:
            suppressed = feedback_store.should_suppress("email", "followup", dims)
        except Exception:  # suppression must never block detection
            logger.debug("followup suppression check failed for %s", email.id, exc_info=True)
            suppressed = False
        if suppressed:
            return FollowupOutcome(
                email.id,
                "skipped:suppressed-by-feedback",
                detail=f"user marked {dims['from_domain']} not useful",
            )

    key = followup_key(email.provider, email.thread_id)
    existing = task_store.get_by_dedup_key(key)
    if existing is not None:
        return FollowupOutcome(
            email.id,
            "skipped:already-exists",
            detail=f"task {existing.id[:8]} already tracks this thread",
            task_id=existing.id,
        )

    decision = detector.detect(email, category_path=category)
    if not decision.needs_reply:
        return FollowupOutcome(email.id, "skipped:no-reply-needed", detail=decision.rationale)
    if decision.confidence < confidence_threshold:
        return FollowupOutcome(
            email.id,
            "skipped:low-confidence",
            detail=f"{decision.confidence:.2f} < {confidence_threshold:.2f}",
        )

    wait = WaitFor(
        kind="reply_from",
        payload={
            "thread_id": email.thread_id,
            "account_id": email.account_id,
            "provider": email.provider,
            "from": email.from_address,
        },
    )
    title = (
        f"Reply to {email.from_address}: {email.subject}"
        if email.subject
        else (f"Reply to {email.from_address}")
    )
    task = task_store.upsert(
        dedup_key=key,
        title=title[:500],
        description=decision.rationale,
        source_kind="email",
        source_id=f"email/{email.account_id}/{email.id}",
        wait_for=wait,
    )
    return FollowupOutcome(email.id, "created", detail=decision.rationale, task_id=task.id)


# ---------------------------------------------------------------------------
# Auto-resolution subscriber
# ---------------------------------------------------------------------------


def _resolve_thread_followup(
    *,
    new_message: EmailMessage,
    task_store: TaskStore,
) -> str | None:
    """If a live followup tracks ``new_message.thread_id``, mark its
    wait resolved. Returns the task id resolved, or None.
    """
    if not new_message.thread_id:
        return None
    key = followup_key(new_message.provider, new_message.thread_id)
    task = task_store.get_by_dedup_key(key)
    if task is None:
        return None
    if task.wait_for is None or task.wait_for_resolved_at is not None:
        return None
    if task.status not in ("open", "doing"):
        return None
    task_store.resolve_wait(task.id, by_event=f"email.new_arrived:{new_message.id}")
    return task.id


def _handle_email_new_arrived(
    payload: Any, *, email_store: EmailStore, task_store: TaskStore
) -> None:
    """Subscriber: scan newly-arrived messages and resolve any open
    followup whose thread they belong to.

    Soft-fails per-message — never raises.
    """
    if not isinstance(payload, EmailNewArrivedPayload):
        logger.warning(
            "email-followup: unexpected payload %r on %s",
            type(payload),
            EMAIL_NEW_ARRIVED,
        )
        return

    for message_id in payload.new_message_ids:
        try:
            message = email_store.get(message_id)
        except Exception:
            logger.exception("email-followup: load failed for %s", message_id)
            continue
        if message is None:
            continue
        try:
            resolved = _resolve_thread_followup(new_message=message, task_store=task_store)
        except Exception:
            logger.exception("email-followup: resolve failed for %s", message_id)
            continue
        if resolved is not None:
            logger.info(
                "email-followup: resolved task=%s by message=%s thread=%s",
                resolved[:8],
                message_id,
                message.thread_id,
            )


def subscribe_email_followup(
    bus: EventBus | None = None,
    *,
    email_store: EmailStore | None = None,
    task_store: TaskStore | None = None,
) -> None:
    """Wire the auto-resolution subscriber to ``email.new_arrived``.

    Detection is intentionally NOT wired here — it lives behind the
    ``iris email detect-followups`` CLI per ADR-0022's "Tier 3 only
    on demand" stance.
    """
    target_bus = bus if bus is not None else get_default_bus()
    target_bus.on(
        EMAIL_NEW_ARRIVED,
        build_followup_handler(bus=target_bus, email_store=email_store, task_store=task_store),
    )
    logger.info("email-followup auto-resolver subscribed to %s", EMAIL_NEW_ARRIVED)


def build_followup_handler(
    *,
    bus: EventBus | None = None,
    email_store: EmailStore | None = None,
    task_store: TaskStore | None = None,
) -> Any:
    """The store-bound ``email.new_arrived`` handler, without subscribing it.

    Factored out of :func:`subscribe_email_followup` at OSS plan M3.2 so the
    email_workflows plugin can hand the *same* callable to ``api.subscribe`` and get
    the fault boundary around it, while ``subscribe_email_followup`` stays the entry
    point for callers that own their own bus (tests, and any CLI path).

    The stores are built once and closed over, exactly as before — opening them per
    event was never the behaviour.
    """
    target_bus = bus if bus is not None else get_default_bus()
    estore = email_store if email_store is not None else EmailStore()
    estore.ensure_schema()
    tstore = task_store if task_store is not None else TaskStore(bus=target_bus)
    tstore.ensure_schema()

    def _handler(payload: Any) -> None:
        _handle_email_new_arrived(payload, email_store=estore, task_store=tstore)

    return _handler


# ---------------------------------------------------------------------------
# Re-exports for convenience
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "EMAIL_CLASSIFIED",
    "EmailClassifiedPayload",
    "FollowupDecision",
    "FollowupDetector",
    "FollowupOutcome",
    "detect_and_persist",
    "subscribe_email_followup",
]
