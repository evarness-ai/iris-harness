"""LangChain BaseTool wrapper for the gmail-inbox skill (Track 1C).

Thin wrapper over ``iris_personal.plugins.gmail.gmail_fetch.fetch_new_emails``. The
domain logic — Gmail API, cursor management, MIME parsing, store
persistence — lives in the src/iris/ module and is tested there.
"""

from __future__ import annotations

from iris_personal.plugins.gmail.gmail_fetch import DEFAULT_MAX_MESSAGES, fetch_new_emails
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class FetchNewEmailsInput(BaseModel):
    """Input arguments for fetch_new_emails."""

    account_id: str = Field(
        ...,
        description="The email_accounts.id slug, e.g. 'gmail:user.in@example.com'.",
    )
    max_messages: int = Field(
        default=DEFAULT_MAX_MESSAGES,
        ge=1,
        le=1000,
        description="Cap on messages fetched in this single call.",
    )


class FetchNewEmailsTool(BaseTool):
    """Sync new Gmail messages for one account into data/email.db."""

    name: str = "fetch_new_emails"
    description: str = (
        "Sync new Gmail messages for one connected account since the last "
        "cursor. Returns a brief summary including the count fetched and "
        "the new sync cursor. Idempotent — re-running is a no-op when no "
        "new mail has arrived."
    )
    args_schema: type[BaseModel] = FetchNewEmailsInput

    def _run(self, account_id: str, max_messages: int = DEFAULT_MAX_MESSAGES) -> str:
        result = fetch_new_emails(account_id, max_messages=max_messages)
        marker = " (fell back to cold-start)" if result.fell_back_to_cold_start else ""
        cursor_hint = f" → cursor={result.new_cursor}" if result.new_cursor else ""
        return f"fetched {result.fetched} email(s) for {result.account_id}{cursor_hint}{marker}"


SKILL_TOOLS = [FetchNewEmailsTool]
