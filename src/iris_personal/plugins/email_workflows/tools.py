"""The email tools on the shared ReAct loop — ``search_inbox`` / ``read_email`` / … .

``build_runtime._domain_tools`` built these until M6.1b, when the email library left
the core with this plugin (OSS plan M6, decision 2). They join the same governed pool
through ``PluginAPI.register_tool``, exactly as ``finance_lookup``, ``daily_plan`` and
``calendar_lookup`` do, so one turn can still compose email with finance and calendar.

The turn's question reaches them through ``services.current_query()`` — a tool
registered once at setup has no closure to read it from (the M4.2 seam).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from iris_harness.sdk.llm import make_narrative_llm_call

if TYPE_CHECKING:
    from iris_harness.sdk import PluginAPI

logger = logging.getLogger(__name__)


def register(api: PluginAPI) -> None:
    """Register the email tool set on the loop."""
    services = api.services
    # The same tier-2 instruct narrator the core built for the domain tools.
    narrative_llm = make_narrative_llm_call(
        services.tier_router, intent="communication", max_tokens=512
    )

    from iris_personal.email.agent_tools import build_email_tools

    for tool in build_email_tools(
        data_dir=services.data_dir,
        llm_call=narrative_llm,
        summarize_llm=narrative_llm,
        current_query=services.current_query,
        current_session_id=services.current_session_id,
        continuations=services.continuations,
    ):
        api.register_tool(
            tool.name,
            tool.description,
            tool.call,
            describe=tool.describe,
            validate=tool.validate,
        )


__all__ = ["register"]
