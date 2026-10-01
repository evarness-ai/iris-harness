"""The ``research`` tool, registered on the governed loop (OSS plan M4.7).

The tool body is the closure that lived in the built-in tool set: refuse a
personal-finance query, strip the user's identifiers, then run the engine. The
only change is where its two inputs come from — the rerank embedder is
``HarnessServices.embed`` instead of a ``SemanticIndex`` the core happened to
hold, and the owner identifiers are read once at setup, as before.
"""

from __future__ import annotations

import logging
from typing import Any

from iris_harness.sdk import PluginAPI

from .guard import (
    REDIRECT_TO_LOCAL_TOOLS,
    is_personal_finance_web_query,
    owner_identity_identifiers,
)

logger = logging.getLogger(__name__)

DESCRIPTION = (
    "Search the public web and read pages for CURRENT facts: news, rankings, "
    "prices, weather, docs, release notes, or anything after your training "
    "cutoff. A news question already uses the news lens and a recent window - do "
    "NOT pass search_type for one. Returns ranked results with sources. For a quick fact pass "
    '"fetch_content": false (snippets only, faster, no page crawl); omit it '
    "for depth. Call it ONCE per question, then answer from what it returns. "
    'Args: {"query": str, "search_type"?: "web"|"news"|"github"|"docs", '
    '"fetch_content"?: bool, "max_results"?: int, '
    '"freshness"?: "day"|"week"|"month"|"year"}.'
)


def build_research_call(embed: Any = None) -> Any:
    """The guarded ``research`` callable. Shared by the plugin and its tests."""
    from .privacy import strip_emails, strip_owner_identifiers
    from .tool import run_research

    # Derived from the identity files, not hardcoded, so the guard tracks whatever
    # the profile records (issue 0031).
    owner_names = owner_identity_identifiers()

    def _research(args: dict[str, Any]) -> str:
        raw_query = str(args.get("query") or args.get("input") or "")
        if is_personal_finance_web_query(raw_query):
            logger.info("research: refused a personal-finance query; redirecting to local tools")
            return REDIRECT_TO_LOCAL_TOOLS
        cleaned, name_stripped = strip_owner_identifiers(raw_query, owner_names)
        cleaned, email_stripped = strip_emails(cleaned)
        if name_stripped or email_stripped:
            logger.info("research query: stripped personal identifiers before web egress")
            args = {**args, "query": cleaned}
        return run_research(args, embed=embed)

    return _research


def register(api: PluginAPI) -> None:
    """Register ``research`` on the governed ReAct tool pool."""
    api.register_tool("research", DESCRIPTION, build_research_call(api.services.embed))
