"""LangChain BaseTool wrappers for the email-followup skill (Track 2B).

Two tools per the manifest:

  detect_followups       — user-invoked Tier 3 detection over recent
                           classified mail for one account.
  list_open_followups    — read-only surface over the TaskStore,
                           filtered to open followup Tasks. Used by
                           the morning-brief slot.

The runtime separately wires ``subscribe_email_followup`` for
auto-resolution on ``email.new_arrived`` — no tool here for that
because it has no user-invokable surface; it's a pure subscriber.
"""

from __future__ import annotations

from pathlib import Path

from iris_harness.sdk.tasks import TaskStore
from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows.discovery import LlamaServerClient
from iris_personal.plugins.email_workflows.followup import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    FollowupDetector,
    detect_and_persist,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

_DEFAULT_LIMIT = 50


class DetectFollowupsInput(BaseModel):
    """Input arguments for detect_followups."""

    account_id: str = Field(
        ...,
        description="The email_accounts.id slug, e.g. 'gmail:user@gmail.com'.",
    )
    limit: int = Field(
        default=_DEFAULT_LIMIT,
        ge=1,
        le=500,
        description="Cap on classified emails scanned.",
    )


class DetectFollowupsTool(BaseTool):
    """Detect followups over recent classified mail (Tier 3 local)."""

    name: str = "detect_followups"
    description: str = (
        "Scan recent classified emails for one account, ask the LLM whether "
        "each needs a reply, and upsert a followup Task per positive verdict. "
        "Idempotent — re-running over the same thread is a no-op."
    )
    args_schema: type[BaseModel] = DetectFollowupsInput

    def _run(self, account_id: str, limit: int = _DEFAULT_LIMIT) -> str:
        from iris_harness.sdk.learning import SurfaceFeedbackStore

        estore = EmailStore(db_path=Path("data/email.db"))
        estore.ensure_schema()
        tstore = TaskStore(db_path=Path("data/tasks.db"))
        tstore.ensure_schema()
        feedback = SurfaceFeedbackStore()
        feedback.ensure_schema()
        detector = FollowupDetector(client=LlamaServerClient())

        candidates = [m for m in estore.list_recent(account_id, limit=limit) if m.thread_id]
        if not candidates:
            return f"no thread-bearing emails for {account_id}"

        created = 0
        skipped = 0
        for email in candidates:
            # category_path=None → detect_and_persist falls back to the
            # email's own classified_category (the real triage verdict).
            outcome = detect_and_persist(
                email,
                category_path=None,
                detector=detector,
                task_store=tstore,
                confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                feedback_store=feedback,
            )
            if outcome.action == "created":
                created += 1
            else:
                skipped += 1
        return (
            f"scanned {len(candidates)} email(s) for {account_id} "
            f"({created} new followups, {skipped} skipped)"
        )


class ListOpenFollowupsInput(BaseModel):
    """Input arguments for list_open_followups."""

    limit: int = Field(
        default=_DEFAULT_LIMIT,
        ge=1,
        le=200,
        description="Cap on rows returned.",
    )


class ListOpenFollowupsTool(BaseTool):
    """Return open followup Tasks (wait_for set, not yet resolved)."""

    name: str = "list_open_followups"
    description: str = (
        "Return open followup Tasks for the morning brief. A followup is a "
        "Task with wait_for set; only status=open|doing and "
        "wait_for_resolved_at IS NULL are returned."
    )
    args_schema: type[BaseModel] = ListOpenFollowupsInput

    def _run(self, limit: int = _DEFAULT_LIMIT) -> list[dict[str, str]]:
        tstore = TaskStore(db_path=Path("data/tasks.db"))
        tstore.ensure_schema()
        # No filter argument on TaskStore.list for "has wait_for"; Phase 2
        # uses a coarse pull then a Python filter. Acceptable at MVP volumes;
        # future TaskStore can take a wait_for_only kwarg.
        candidates = tstore.list(status="open", limit=limit * 2) + tstore.list(
            status="doing", limit=limit * 2
        )
        followups = [
            t for t in candidates if t.wait_for is not None and t.wait_for_resolved_at is None
        ][:limit]
        return [
            {
                "task_id": t.id,
                "title": t.title,
                "wait_for_kind": t.wait_for.kind if t.wait_for else "",
                "from": str(t.wait_for.payload.get("from", "")) if t.wait_for else "",
            }
            for t in followups
        ]


SKILL_TOOLS = [DetectFollowupsTool, ListOpenFollowupsTool]
