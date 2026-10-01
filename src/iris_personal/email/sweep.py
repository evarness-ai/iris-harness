"""email-sweep heartbeat handler — Phase 1 Track 1D.

Iterates active Gmail accounts on each tick; calls fetch_new_emails for
each; emits EMAIL_SWEPT on the runtime bus when new messages land (the email
judge's queue turns that into EMAIL_NEW_ARRIVED as each email is released,
loop-proof PR 5). Per-account failures are collected, never crash the heartbeat.

An account whose email setup has not turned the sweep on is left alone
(``sweep_gate``: owner decision 2026-09-30); an account setup never touched is swept
as it always was.

Cadence + max_messages are tunable via ``config/heartbeats.yaml``.

Per canonical §3.1 sweep model + ADR-Q9 (event bus convention) +
ADR-0013 (subsystem topics live with producer).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from iris_harness.sdk.events import EventBus, get_default_bus
from iris_harness.sdk.types import HeartbeatDefinition, HeartbeatRun, HeartbeatStatus
from iris_personal.email.accounts import (
    EmailAccountStore,
    is_non_mailbox_provider,
    ledger_account_label,
)
from iris_personal.email.events import (
    EMAIL_LABELS_CHANGED,
    EMAIL_SWEPT,
    EmailLabelsChangedPayload,
    EmailNewArrivedPayload,
)
from iris_personal.email.providers import mail_provider_for
from iris_personal.email.store import EmailStore
from iris_personal.email.sweep_gate import SweepGate

logger = logging.getLogger(__name__)

DEFAULT_MAX_MESSAGES_PER_TICK = 100


@dataclass
class EmailSweepHandler:
    """HeartbeatHandler that drives mailbox fetches on a schedule.

    Provider-agnostic since OSS plan M5.7 track A: each active account is fetched
    through the ``MailProvider`` registered for its ``provider`` (``email.providers``),
    so the sweep itself talks to no mailbox app and stays core as the mechanism that
    keeps the store every read queries current. An account whose provider plugin is
    not mounted is skipped and named in the run output, never silently dropped —
    unless its plugin declared the provider is not a mailbox (a calendar or drive
    account in the shared table), which the sweep leaves out without comment.

    Construct with ``bus=None`` for silent operation (handy in tests);
    construct via ``build_email_sweep_handler()`` for production wiring
    against the module-level singleton bus.
    """

    bus: EventBus | None = None
    accounts_store: EmailAccountStore | None = None
    email_store: EmailStore | None = None
    # Test seam — a fake fetcher stands in for every registered provider's ``fetch_new``.
    # Accounts are still selected by the registry: a provider must be registered for
    # the account's kind for the sweep to reach it at all.
    fetcher: object = field(default=None)

    def __call__(self, definition: HeartbeatDefinition) -> HeartbeatRun:
        accounts_store = self.accounts_store or EmailAccountStore()
        accounts_store.ensure_schema()
        email_store = self.email_store or EmailStore()
        email_store.ensure_schema()

        max_messages = int(
            cast("Any", definition.params.get("max_messages", DEFAULT_MAX_MESSAGES_PER_TICK))
        )
        # Calendar/Drive rows share the accounts table; their plugins declare them
        # non-mailboxes, and the sweep passes them by without a word.
        active = [
            a
            for a in accounts_store.list(active_only=True)
            if not is_non_mailbox_provider(a.provider)
        ]
        unserved = sorted({a.provider for a in active if mail_provider_for(a.provider) is None})
        served = [a for a in active if mail_provider_for(a.provider) is not None]
        # An account whose email setup has not turned the sweep on yet waits for it
        # (``sweep_gate``); an account setup never touched has no row and is swept.
        gate = SweepGate(db_path=email_store.db_path)
        held = gate.held()
        waiting = [a.id for a in served if a.id in held]
        for account_id in waiting:
            if gate.note_skip(account_id):
                _note_first_skip(account_id, held[account_id].reason)
        accounts = [a for a in served if a.id not in held]
        suffix = (f"; no provider mounted for: {', '.join(unserved)}" if unserved else "") + (
            f"; waiting for email setup: {', '.join(waiting)}" if waiting else ""
        )
        if not accounts:
            return HeartbeatRun(
                name=definition.name,
                status=HeartbeatStatus.SKIPPED,
                finished_at=datetime.now(UTC),
                output=(
                    "no account to sweep"
                    if waiting
                    else "no active accounts with a mounted mail provider"
                )
                + suffix,
            )

        total_fetched = 0
        per_account_summary: list[str] = []
        errors: list[str] = []

        for account in accounts:
            provider = mail_provider_for(account.provider)
            assert provider is not None  # selected above
            fetch = self.fetcher or provider.fetch_new
            try:
                result = fetch(  # type: ignore[operator]
                    account.id,
                    store=email_store,
                    max_messages=max_messages,
                )
            except Exception as exc:  # heartbeat must not crash
                logger.exception("email-sweep: fetch failed for %s", account.id)
                errors.append(f"{account.id}: {exc}")
                per_account_summary.append(f"{account.id}: error")
                continue

            total_fetched += result.fetched
            per_account_summary.append(f"{account.id}: +{result.fetched}")

            if result.fetched > 0 and self.bus is not None:
                self.bus.emit_sync(
                    EMAIL_SWEPT,
                    EmailNewArrivedPayload(
                        account_id=account.id,
                        new_message_ids=result.new_message_ids,
                        count=result.fetched,
                        fell_back_to_cold_start=result.fell_back_to_cold_start,
                    ),
                )
            # Label changes on mail already in the store (the email judge's read-back of
            # the owner's Gmail relabels). A separate topic: no new mail arrived.
            label_changes = getattr(result, "label_changes", ())
            if label_changes and self.bus is not None:
                self.bus.emit_sync(
                    EMAIL_LABELS_CHANGED,
                    EmailLabelsChangedPayload(account_id=account.id, changes=label_changes),
                )

        status = HeartbeatStatus.FAILED if errors else HeartbeatStatus.SUCCESS
        return HeartbeatRun(
            name=definition.name,
            status=status,
            finished_at=datetime.now(UTC),
            output=f"swept {len(accounts)} account(s), total +{total_fetched}: "
            + "; ".join(per_account_summary)
            + suffix,
            error="; ".join(errors),
        )


def _note_first_skip(account_id: str, reason: str) -> None:
    """The first time the sweep passes a held account by: one log line and one audit
    row, so the owner can see why it is not fetched. The row's reason names the
    account's provider only (:func:`ledger_account_label`); the account id is in the
    payload."""
    import uuid

    from iris_harness.sdk.audit import AuditLog, audit_db_path

    logger.info("email-sweep: %s waits for email setup (%s); not swept", account_id, reason)
    try:
        AuditLog(db_path=audit_db_path()).record(
            run_id=f"email-sweep-{uuid.uuid4().hex[:12]}",
            step_id=None,
            agent_type="heartbeat",
            hook_point="email_sweep",
            plugin="email_sweep",
            decision="deny",
            severity="info",
            reason=(
                f"email sweep: {ledger_account_label(account_id)} waits for email setup "
                "to turn the sweep on"
            ),
            payload={"account": account_id, "why": reason},
        )
    except Exception:  # a ledger hiccup never crashes the heartbeat
        logger.warning("email-sweep: audit write failed for %s", account_id, exc_info=True)


def build_email_sweep_handler() -> EmailSweepHandler:
    """Factory used by bootstrap to wire the handler with the singleton bus."""
    return EmailSweepHandler(bus=get_default_bus())
