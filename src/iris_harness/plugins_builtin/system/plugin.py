"""The ``system`` reference plugin (OSS plan M1 tracer bullet).

Three registrations, three kinds, zero runtime imports:

* intercept ``time_date`` — answers "what time is it" style turns from the local
  clock, deterministically, before any classifier or model runs.
* tool ``system_health`` — the ReAct tool the model calls to report IRIS's own
  service / credential / hardware / plugin health (ADR-0069).
* heartbeat ``health_tick`` — refreshes the cached health snapshot in the
  background so the tool and ``GET /health`` read a recent view, then runs the
  health watch (ADR-0116): repair red checks, notify the owner, record incidents.
"""

from __future__ import annotations

from typing import Any

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.parsing import _deterministic_time_date_reply

SYSTEM_HEALTH_DESCRIPTION = (
    "Report IRIS's OWN runtime health — whether its services and the "
    "local model backend are up, host load/thermal, AND the state of "
    "connected credentials (Gmail/Calendar/Drive tokens, cloud-LLM API "
    "keys: connected / expiring / expired / not-connected). CALL THIS "
    "(not memory) when the user asks 'is everything working / running', "
    "'are you healthy', 'why is my email/calendar not working', 'is my "
    "Gmail connected', 'is the server/ollama up', or 'check your status'. "
    "Returns a per-target green/yellow/red report and any items needing "
    "attention, each with the exact fix command when known. No arguments."
)


def setup(api: PluginAPI) -> None:
    services = api.services

    def time_date(message: str, *, session_id: str, span: Any = None) -> Any:
        deterministic = _deterministic_time_date_reply(message)
        if deterministic is None:
            return None
        return services.deterministic_reply(
            message=message,
            session_id=session_id,
            response=deterministic,
            metadata={"deterministic_time_date": True},
            span=span,
        )

    api.register_intercept("time_date", time_date, trace_text="deterministic time/date response")

    def system_health(args: dict[str, Any]) -> str:
        # Reads the background-cached snapshot (refreshed by health_tick),
        # building one on a cold cache. The fault boundary catches anything else.
        from iris_harness.sdk.health import (
            current_snapshot,
            render_text,
        )

        return render_text(current_snapshot())

    api.register_tool("system_health", SYSTEM_HEALTH_DESCRIPTION, system_health)

    from iris_harness.sdk.health import build_health_tick_handler

    api.register_heartbeat(
        "health_tick",
        build_health_tick_handler(heartbeat_diagnostics_provider=services.heartbeat_diagnostics),
        description="Refresh the cached System Health snapshot.",
    )

    # The health watch (ADR-0116): after each refresh, repair red checks and tell the
    # owner on every channel when repair runs out. Installed per process, like the
    # snapshot cache; a second runtime replaces it.
    from iris_harness.sdk.health import build_watcher, install_watcher

    install_watcher(
        build_watcher(
            config_dir=services.config_dir,
            data_dir=services.data_dir,
            channels=services.channels,
            heartbeats=services.heartbeats,
            diagnostics_provider=services.heartbeat_diagnostics,
        )
    )
