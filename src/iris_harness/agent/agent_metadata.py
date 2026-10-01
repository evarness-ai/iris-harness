"""Presentation metadata for the per-agent console (ADR-0074).

The live ``AgentExecutor.registered_agents()`` is the source of truth for *which*
agents exist in this process; this catalog only supplies how to *describe* them in
the console — a title, a one-line description, and the optional ``source_kind`` that
ties an agent to its pending actions (ADR-0073 provider seam). Agents registered
without an entry here surface name-only; entries here that aren't registered are
omitted by the endpoint (the registry wins, never this file).

This is reference data, not behaviour — it drives no routing.
"""

from __future__ import annotations

from dataclasses import dataclass

from iris_harness.services.tasks import SourceKind


@dataclass(frozen=True)
class AgentMeta:
    """How the console describes one agent."""

    title: str
    description: str
    source_kind: SourceKind | None = None  # joins to its pending actions, if any
    heartbeats: tuple[str, ...] = ()  # the background tick(s) that drive this agent
    intents: tuple[str, ...] = ()  # the routing intents it serves -> its LLM tier(s)


# Keyed by the agent_type used in AgentExecutor.register(...).
AGENT_CATALOG: dict[str, AgentMeta] = {
    "finance": AgentMeta(
        title="Finance",
        description="Net-worth, portfolio, dues & balances over the local finance store",
        source_kind="finance-statements",
        heartbeats=("finance_ingest_tick", "finance_monitor"),
        intents=("finance",),
    ),
    "email": AgentMeta(
        title="Email",
        description="Inbox digest, topic search & triage over the connected mailboxes",
        source_kind="email",
        heartbeats=("email_sweep", "email_judge"),
        intents=("communication", "email_summary"),
    ),
    "planner": AgentMeta(
        title="Planner",
        description="Daily plan: calendar, tasks, follow-ups & bills",
        source_kind="calendar-prep",
        heartbeats=("meeting_prep", "calendar_mirror"),
        intents=("task_planning",),
    ),
    "calendar": AgentMeta(
        title="Calendar",
        description="Upcoming events & meetings over the local calendar store",
        heartbeats=("calendar_mirror", "meeting_prep"),
        intents=("calendar",),
    ),
    "filemanager": AgentMeta(
        title="Files",
        description="File custody, search, organize plans, vault & photo catalog over allowed roots",
        source_kind="filemanager-organize",
        heartbeats=("filemanager_retention", "filemanager_index", "filemanager_photos"),
        intents=("files",),
    ),
    "research": AgentMeta(
        title="Research",
        description="Web research, synthesis & document drafting over the provider chain",
        intents=("search",),
    ),
    "coding_agent": AgentMeta(
        title="Coding",
        description="The iris-code SDLC pipeline (multi-stage, persona sub-agents)",
        intents=("coding",),
    ),
    "code_exec": AgentMeta(
        title="Code Exec",
        description="Sandboxed code execution loop (Docker-gated)",
        intents=("code_exec",),
    ),
    "clarify": AgentMeta(
        title="Clarify",
        description="Grounded clarifying question when the router is uncertain",
    ),
    "system": AgentMeta(
        title="System",
        description="General ReAct handler — tools, search, system & catch-all turns",
        intents=("system", "general", "search"),
    ),
}


def agent_meta(name: str) -> AgentMeta | None:
    """Catalog entry for an agent type, or None if undescribed."""
    return AGENT_CATALOG.get(name)
