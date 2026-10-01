"""Phase 3 — live Gmail scenarios (READ-ONLY), graduating from the synthetic corpus.

Safeguards, stated up front:

- **Read-only against Gmail**: the only remote call is ``fetch_new_emails``
  (the same sync the production heartbeat runs). Nothing is sent, modified,
  labeled, or deleted on the account.
- **Classification is a dry run**: ``classify`` only — no ``mark_classified``
  writes, no events emitted. The production triage tick remains the sole
  writer of classifications.
- **No LLM sees mail content**: the pure-kNN path embeds locally
  (sentence-transformers). The governance scenario asserts this by checking
  the audit log for the run window.

Three scenarios:

1. ``live-sync``   — real incremental fetch through the production path.
2. ``live-triage`` — dry-run classify the store's unclassified backlog;
   gate behavior + confidence stats on real mail (info verdicts: no labels).
3. ``holdout``     — classify the 50 human-labeled holdout messages and
   grade predicted root vs ``true_root``: a real accuracy number on real mail.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scenarios.common import REPO_ROOT, ScenarioRecord, Timer

ACCOUNT_ID = "gmail:user@example.com"
WORKSPACE = Path.home() / ".iris" / "workspace"
AUDIT_DB = Path.home() / ".local" / "share" / "iris" / "audit.db"
MAX_SYNC = 50
TRIAGE_LIMIT = 50


def _classifier() -> Any:
    from iris_harness.email.triage import EmailTriageClassifier

    return EmailTriageClassifier(
        workspace_dir=WORKSPACE,
        db_path=REPO_ROOT / "data" / "iris.db",
        email_db_path=REPO_ROOT / "data" / "email.db",
    )


def run_live_sync() -> list[ScenarioRecord]:
    from iris_harness.email.gmail_fetch import fetch_new_emails
    from iris_harness.email.store import EmailStore

    store = EmailStore(db_path=REPO_ROOT / "data" / "email.db")
    try:
        with Timer() as timer:
            result = fetch_new_emails(ACCOUNT_ID, store=store, max_messages=MAX_SYNC)
    except Exception as exc:  # noqa: BLE001 — OAuth expiry etc. must not kill the run
        return [
            ScenarioRecord(
                scenario="email-live-sync",
                case_id="incremental-fetch",
                verdict="fail",
                expected=f"read-only sync, <= {MAX_SYNC} messages",
                actual=f"{type(exc).__name__}: {str(exc)[:140]}",
                latency_ms=0.0,
                detail={"remediation": "iris auth gmail login --user <address>"},
            )
        ]
    return [
        ScenarioRecord(
            scenario="email-live-sync",
            case_id="incremental-fetch",
            verdict="pass",
            expected=f"read-only sync, <= {MAX_SYNC} messages",
            actual=f"fetched {result.fetched} new messages",
            latency_ms=round(timer.elapsed_ms, 2),
            detail={
                "fetched": result.fetched,
                "cold_start_fallback": result.fell_back_to_cold_start,
            },
        )
    ]


def run_live_triage() -> list[ScenarioRecord]:
    from iris_harness.email.store import EmailStore

    classifier = _classifier()
    store = EmailStore(db_path=REPO_ROOT / "data" / "email.db")
    with Timer() as load_timer:
        centroids = classifier.load_categories(ACCOUNT_ID)
    unclassified = store.list_unclassified(ACCOUNT_ID, limit=TRIAGE_LIMIT)

    records = [
        ScenarioRecord(
            scenario="email-live-triage",
            case_id="load-categories",
            verdict="pass" if centroids else "fail",
            expected="centroids from real accepted categories",
            actual=f"{len(centroids)} centroids, {len(unclassified)} unclassified to score",
            latency_ms=round(load_timer.elapsed_ms, 2),
        )
    ]
    for mail in unclassified:
        with Timer() as timer:
            result = classifier.classify(mail)  # dry run — never persisted
        records.append(
            ScenarioRecord(
                scenario="email-live-triage",
                case_id=mail.id[:16],
                verdict="info",  # no ground truth for fresh mail
                expected="classified-or-queued (gate decides)",
                actual=result.category_path or "<queued>",
                latency_ms=round(timer.elapsed_ms, 2),
                detail={
                    "queued": result.queued,
                    "confidence": result.confidence,
                    "from_domain": mail.from_domain or "",
                },
            )
        )
    return records


def run_holdout() -> list[ScenarioRecord]:
    from iris_harness.email.holdout import holdout_path, load_holdout
    from iris_harness.skills.contracts.email import EmailMessage

    classifier = _classifier()
    classifier.load_categories(ACCOUNT_ID)
    labels = load_holdout(holdout_path(WORKSPACE, ACCOUNT_ID))

    records: list[ScenarioRecord] = []
    for label in labels:
        message = EmailMessage(
            id=label.message_id,
            provider="gmail",
            account_id=label.account_id,
            from_address=label.from_address,
            subject=label.subject,
            snippet=label.snippet,
            received_at=label.received_at,
        )
        with Timer() as timer:
            result = classifier.classify(message)
        predicted_root = (
            (result.category_path or "").split("/")[1] if result.category_path else None
        )
        ok = predicted_root == label.true_root and not result.queued
        records.append(
            ScenarioRecord(
                scenario="email-live-holdout",
                case_id=label.message_id[:16],
                verdict="pass" if ok else "fail",
                expected=label.true_root,
                actual=predicted_root or "<queued>",
                latency_ms=round(timer.elapsed_ms, 2),
                detail={
                    "queued": result.queued,
                    "confidence": result.confidence,
                    "predicted_path": result.category_path or "",
                },
            )
        )
    return records


def audit_window_check(start_iso: str) -> list[ScenarioRecord]:
    """Prove no LLM call carried mail content during the live run."""
    import sqlite3

    with sqlite3.connect(AUDIT_DB) as conn:
        (llm_calls,) = conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE ts >= ? AND hook_point = 'pre_llm_call'",
            (start_iso,),
        ).fetchone()
    return [
        ScenarioRecord(
            scenario="email-live-governance",
            case_id="no-llm-during-triage",
            verdict="pass" if llm_calls == 0 else "fail",
            expected="0 pre_llm_call audit rows in the run window (pure-kNN is local)",
            actual=f"{llm_calls} pre_llm_call rows",
            latency_ms=0.0,
        )
    ]


def run() -> list[ScenarioRecord]:
    window_start = datetime.now(UTC).isoformat()
    records: list[ScenarioRecord] = []
    records += run_live_sync()
    records += run_live_triage()
    records += run_holdout()
    records += audit_window_check(window_start)
    return records


def extra_summary(records: list[ScenarioRecord]) -> str:
    lines: list[str] = []
    triage = [r for r in records if r.scenario == "email-live-triage" and r.verdict == "info"]
    if triage:
        queued = sum(1 for r in triage if r.detail.get("queued"))
        lines.append(
            f"  live triage: {len(triage)} fresh messages, "
            f"{len(triage) - queued} classified / {queued} queued "
            f"({(len(triage) - queued) / len(triage):.0%} gate pass-rate)"
        )
    holdout = [r for r in records if r.scenario == "email-live-holdout"]
    if holdout:
        passed = sum(1 for r in holdout if r.verdict == "pass")
        queued = sum(1 for r in holdout if r.detail.get("queued"))
        wrong = len(holdout) - passed - queued
        lines.append(
            f"  holdout (real labels): {passed}/{len(holdout)} correct root "
            f"({passed / len(holdout):.0%}), {queued} abstained, {wrong} wrong"
        )
    confidences = [
        r.detail["confidence"] for r in records if r.detail.get("confidence") is not None
    ]
    if confidences:
        lines.append(
            f"  confidence: mean={sum(confidences) / len(confidences):.3f} "
            f"min={min(confidences):.3f} max={max(confidences):.3f}"
        )
    return "\n".join(lines)
