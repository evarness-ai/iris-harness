"""Manually trigger the email-followup auto-resolution subscriber.

Picks the first open followup with `wait_for` set, injects a synthetic
inbound email on the same thread, and emits `email.new_arrived`. The
subscriber should call `TaskStore.resolve_wait` and set
`wait_for_resolved_at` without auto-completing the task (ADR-0014 #10).

Usage:
    IRIS_AUTH_SECRET=test-secret-for-testing IRIS_DISABLE_WARMUP=1 \\
        poetry run python scripts/phase2_test_auto_resolve.py
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

from iris_harness.foundation.eventbus import EventBus
from iris_harness.services.tasks.store import TaskStore
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.events import EMAIL_NEW_ARRIVED, EmailNewArrivedPayload
from iris_personal.email.followup import subscribe_email_followup
from iris_personal.email.store import EmailStore


def main() -> None:
    tstore = TaskStore()
    tstore.ensure_schema()
    followups = [t for t in tstore.list(status="open") if t.wait_for is not None]
    if not followups:
        print("no open followups; create some via `iris email detect-followups` first")
        sys.exit(1)

    task = followups[0]
    thread_id = str(task.wait_for.payload["thread_id"])  # type: ignore[union-attr]
    account_id = str(task.wait_for.payload.get("account_id", "gmail:test"))  # type: ignore[union-attr]
    print(f"using followup task={task.id[:8]} thread={thread_id}")
    print(f"  current wait_for_resolved_at: {task.wait_for_resolved_at}")

    # Inject a synthetic 'reply' message in the same thread.
    estore = EmailStore()
    estore.ensure_schema()
    fake_id = f"synthetic-reply-{task.id[:8]}"
    estore.upsert(
        EmailMessage(
            id=fake_id,
            provider="gmail",
            account_id=account_id,
            thread_id=thread_id,
            from_address="synthetic-replier@example.com",
            subject="Re: synthetic test reply",
            snippet="synthetic body for auto-resolve test",
            received_at=datetime.now(UTC),
        )
    )

    # Wire the subscriber on a local bus and emit. We pass the real
    # EmailStore and TaskStore so the subscriber writes against the
    # actual files (not in-memory copies).
    bus = EventBus()
    subscribe_email_followup(bus, email_store=estore, task_store=tstore)
    bus.emit_sync(
        EMAIL_NEW_ARRIVED,
        EmailNewArrivedPayload(
            account_id=account_id,
            new_message_ids=(fake_id,),
            count=1,
            fell_back_to_cold_start=False,
        ),
    )

    # Re-read the task; wait_for_resolved_at should now be set.
    after = tstore.get(task.id)
    assert after is not None
    print(f"  new wait_for_resolved_at: {after.wait_for_resolved_at}")
    print(f"  task status: {after.status} (should still be 'open' per ADR-0014 #10)")
    if after.wait_for_resolved_at is None:
        print("ERROR: auto-resolution did not fire")
        sys.exit(2)
    print("OK — followup auto-resolution worked")


if __name__ == "__main__":
    main()
