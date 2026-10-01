"""LangChain BaseTool wrappers for the email-triage skill (Track 1G).

Thin wrappers over ``iris_personal.plugins.email_workflows.triage.EmailTriageClassifier``. The
classifier is constructed lazily per call to keep the agent's tool
registration cheap — MiniLM only loads when triage actually runs.

Two tools per ADR-0021:

  classify_email_by_id   — single-message classification (look up by id,
                            classify, persist, emit event)
  run_email_triage       — batch over the backlog of unclassified
                            emails for an account

Both soft-fail per ADR-0021 §7 — return a status string instead of
raising on per-email errors.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows.triage import EmailTriageClassifier

_DEFAULT_LIMIT = 20


def _get_classifier() -> EmailTriageClassifier:
    """Per-call construction. The classifier carries its own per-process
    centroid cache; if the LangChain agent invokes the tool twice in one
    request, the second call rebuilds the cache. Acceptable for Phase 1.
    The module-level singleton in iris_harness.runtime.bootstrap covers the
    event-subscriber hot path."""
    return EmailTriageClassifier()


class ClassifyEmailByIdInput(BaseModel):
    """Input arguments for classify_email_by_id."""

    message_id: str = Field(
        ...,
        description="The provider-native message id, e.g. Gmail's payload.id.",
    )


class ClassifyEmailByIdTool(BaseTool):
    """Classify a single locally-stored email via the hybrid pipeline."""

    name: str = "classify_email_by_id"
    description: str = (
        "Classify a single email already persisted in data/email.db. Looks "
        "the message up by id, runs the hybrid kNN+LLM pipeline, writes the "
        "result back to the row, and fires email.classified. Soft-fails on "
        "per-email errors."
    )
    args_schema: type[BaseModel] = ClassifyEmailByIdInput

    def _run(self, message_id: str) -> str:
        store = EmailStore()
        store.ensure_schema()
        message = store.get(message_id)
        if message is None:
            return f"no email with id={message_id} in email.db"
        classifier = _get_classifier()
        result = classifier.classify_and_persist(message, store=store)
        if result.error is not None:
            return f"soft-fail for {message_id}: {result.error}"
        return (
            f"classified {message_id} → {result.category_path} "
            f"(confidence {result.confidence:.2f})"
        )


class RunEmailTriageInput(BaseModel):
    """Input arguments for run_email_triage."""

    account_id: str = Field(
        ...,
        description="The email_accounts.id slug, e.g. 'gmail:user.in@example.com'.",
    )
    limit: int = Field(
        default=_DEFAULT_LIMIT,
        ge=1,
        le=500,
        description="Cap on unclassified emails processed in this single call.",
    )


class RunEmailTriageTool(BaseTool):
    """Process the backlog of unclassified emails for one account."""

    name: str = "run_email_triage"
    description: str = (
        "Process the backlog of unclassified emails for one account. Returns "
        "a summary including the number processed, succeeded, and soft-failed. "
        "Idempotent — already-classified rows are skipped."
    )
    args_schema: type[BaseModel] = RunEmailTriageInput

    def _run(self, account_id: str, limit: int = _DEFAULT_LIMIT) -> str:
        classifier = _get_classifier()
        results = classifier.classify_unclassified(account_id, limit=limit)
        if not results:
            return f"no unclassified emails for {account_id}"
        succeeded = sum(1 for r in results if r.category_path is not None)
        failed = len(results) - succeeded
        return (
            f"triaged {len(results)} email(s) for {account_id} "
            f"({succeeded} classified, {failed} soft-failed)"
        )


class EmailInboxSummaryInput(BaseModel):
    """Input arguments for email_inbox_summary.

    Optional ``account_id`` filters to one account; when omitted, the
    tool aggregates across all accounts present in email.db.
    """

    account_id: str | None = Field(
        default=None,
        description=("email_accounts.id slug; when omitted, totals span all accounts."),
    )
    view: Literal["triage", "last_24h"] = Field(
        default="triage",
        description=(
            "'triage' (default): one row of classified / pending / unclassified counts. "
            "'last_24h': one row per mailbox — its last 24 h of mail grouped by triage "
            "category, largest first (the morning digest's Inbox summary)."
        ),
    )


class EmailInboxSummaryTool(BaseTool):
    """Per-account counts of classified vs queued vs unclassified mail."""

    name: str = "email_inbox_summary"
    description: str = (
        "Summarise the inbox. view='triage' (default) returns one row of triage "
        "state — counts of classified, pending_review (kNN-ambiguous), and "
        "unclassified rows. view='last_24h' returns one row per mailbox with its "
        "last 24 hours of mail grouped by triage category, largest first."
    )
    args_schema: type[BaseModel] = EmailInboxSummaryInput

    def _run(self, account_id: str | None = None, view: str = "triage") -> list[dict[str, str]]:
        import sqlite3

        from iris_harness.sdk.persistence import data_path

        # Under IRIS_DATA_DIR like every store (a cwd-relative path read nothing on a
        # deployment whose data dir is not ./data — the digest said "no accounts").
        db_path = data_path("email.db")
        if not db_path.exists():
            return []
        if view == "last_24h":
            from datetime import UTC, datetime

            from iris_personal.plugins.email_workflows.inbox_mix import inbox_mix

            store = EmailStore(db_path=db_path)
            store.ensure_schema()
            return inbox_mix(store, now=datetime.now(UTC), account_id=account_id)
        clauses: list[str] = []
        params: list[str] = []
        if account_id:
            clauses.append("account_id = ?")
            params.append(account_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        # `where` only ever contains "WHERE account_id = ?"; user input flows
        # via the parameterized `params` list, never the SQL string.
        select_clause = (
            "SELECT "
            "  SUM(CASE WHEN classified_category IS NOT NULL THEN 1 ELSE 0 END) AS classified, "
            "  SUM(CASE WHEN triage_state = 'pending_review' THEN 1 ELSE 0 END) AS pending, "
            "  SUM(CASE WHEN classified_category IS NULL "
            "           AND (triage_state IS NULL OR triage_state = 'error') "
            "           THEN 1 ELSE 0 END) AS unclassified, "
            "  COUNT(*) AS total "
            "FROM emails "
        )
        sql = select_clause + where
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(sql, params).fetchone()
        if not row or (row[3] or 0) == 0:
            return []
        return [
            {
                "scope": account_id or "all accounts",
                "classified": str(row[0] or 0),
                "pending": str(row[1] or 0),
                "unclassified": str(row[2] or 0),
                "total": str(row[3] or 0),
            }
        ]


class _NoArgs(BaseModel):
    """Empty input schema for parameterless tools."""


class EmailFocusTool(BaseTool):
    """The morning digest's Focus section (loop-proof PR 2, D17).

    Last-24h mail in the owner's Focus categories (Settings → Digest): the newest
    ``focus_per_account`` per inbox, at most ``focus_limit`` in all, minus senders
    marked "not useful". Returns the section as
    markdown — its own heading (which names the categories) plus one bullet per email,
    each ending in the 👎 link. Logic lives in
    ``iris_personal.plugins.email_workflows.digest_focus``.
    """

    name: str = "email_focus"
    description: str = (
        "The digest's Focus section: the last 24 hours of mail in the owner's focus "
        "categories, newest first, minus senders marked not useful."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        import os
        from datetime import UTC, datetime
        from pathlib import Path

        from iris_harness.sdk.digest import load_digest_settings
        from iris_harness.sdk.learning import SurfaceFeedbackStore
        from iris_personal.plugins.email_workflows.digest_focus import render_focus

        settings = load_digest_settings(Path(os.environ.get("IRIS_DATA_DIR") or "data"))
        store = EmailStore()
        store.ensure_schema()
        suppression = SurfaceFeedbackStore()
        suppression.ensure_schema()
        return render_focus(
            store,
            suppression,
            settings.focus_categories,
            limit=settings.focus_limit,
            per_account=settings.focus_per_account,
            now=datetime.now(UTC),
        )

    async def _arun(self) -> str:
        return self._run()


def _judge_config_dir() -> Path | None:
    raw = os.environ.get("IRIS_CONFIG_DIR")
    return Path(raw).expanduser() if raw else None


class EmailNeedsReplyTool(BaseTool):
    """The morning digest's "Needs reply (N)" (loop-proof PR 5).

    Emails the judge (or the owner) put in Needs reply, newest first, as "Sender:
    subject": gone once the owner replies in the thread, re-buckets it, or it is older
    than digest.yaml ``expiry.needs_reply_days``. Logic lives in
    ``iris_personal.plugins.email_workflows.judge_digest``.
    """

    name: str = "email_needs_reply"
    description: str = (
        "The digest's Needs reply section: judged emails waiting on the owner's answer, "
        "newest first, as '## Needs reply (N)' then 'Sender: subject' lines."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        from datetime import datetime

        from iris_harness.sdk.digest import expiry_days
        from iris_harness.sdk.persistence import data_path
        from iris_harness.sdk.time import iris_timezone
        from iris_personal.plugins.email_workflows.judge_digest import render_needs_reply
        from iris_personal.plugins.email_workflows.judge_view import open_stores
        from iris_personal.plugins.email_workflows.judge_words import SurfaceWords

        words = SurfaceWords.load(_judge_config_dir())
        tz = iris_timezone()
        stores = open_stores(data_path("email.db").parent)
        assert stores is not None
        judgments, emails = stores
        return render_needs_reply(
            judgments,
            emails,
            words,
            days=expiry_days("needs_reply_days"),
            now=datetime.now(tz),
            tz=tz,
        )

    async def _arun(self) -> str:
        return self._run()


class EmailJudgedYesterdayTool(BaseTool):
    """The morning digest's "Judged yesterday: …" line (loop-proof PR 5)."""

    name: str = "email_judged_yesterday"
    description: str = (
        "One line: how many emails the judge sorted yesterday, by bucket, how many are "
        "Unsure (answer them in the Action Center) and how many still wait."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        from datetime import datetime

        from iris_harness.sdk.persistence import data_path
        from iris_harness.sdk.time import iris_timezone, previous_local_day
        from iris_personal.plugins.email_workflows.judge_config import JudgeConfig
        from iris_personal.plugins.email_workflows.judge_digest import judged_line
        from iris_personal.plugins.email_workflows.judge_view import open_stores
        from iris_personal.plugins.email_workflows.judge_words import SurfaceWords

        db = data_path("email.db")
        if not db.exists():
            return ""
        stores = open_stores(db.parent)
        assert stores is not None
        judgments, _ = stores
        config_dir = _judge_config_dir()
        tz = iris_timezone()
        start, end = previous_local_day(datetime.now(tz), tz)
        return judged_line(
            judgments,
            JudgeConfig.load(config_dir),
            SurfaceWords.load(config_dir),
            start=start,
            end=end,
        )

    async def _arun(self) -> str:
        return self._run()


SKILL_TOOLS = [
    ClassifyEmailByIdTool,
    RunEmailTriageTool,
    EmailInboxSummaryTool,
    EmailFocusTool,
    EmailNeedsReplyTool,
    EmailJudgedYesterdayTool,
]
