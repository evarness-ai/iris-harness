"""Agents and plugins: the per-agent console (ADR-0074), agent settings, and the read-only plugin inventory.

    GET    /agents
    GET    /plugins
    GET    /plugins/{name}
    GET    /api/v1/webui/nav
    GET    /agents/{name}
    GET    /agents/{name}/metrics
    PATCH  /agents/{name}/settings

Moved out of ``create_app`` unchanged (review item: split the god function); the route
table and OpenAPI schema are identical before and after. The write guard in ``main``
still gates the mutating routes.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from iris_harness.foundation.settings import SETTINGS_DB_NAME, SettingsStore
from iris_harness.runtime import IrisRuntime
from iris_harness.server.iris_api.settings_routes import actor_of

WEB_NAV_PATH = "/api/v1/webui/nav"


class AgentSettingsPatch(BaseModel):
    """Request body for ``PATCH /agents/{name}/settings`` (ADR-0074 §4).

    ``toggles`` maps an env-flag key to its desired boolean. Only toggles the agent
    owns are accepted; the server persists + hot-applies them."""

    toggles: dict[str, bool] = Field(default_factory=dict)


def _agent_stores(name: str) -> dict[str, Any] | None:
    """Agent-specific 'key stores' panel for the dashboard (ADR-0074), or None.

    The owning plugin registers it (finance: account count + statement inventory);
    the dashboard names no agent's stores itself."""
    from iris_harness.runtime.agent_panels import agent_panel

    return agent_panel(name)


def _agent_settings(name: str, rt: IrisRuntime) -> dict[str, Any] | None:
    """Read-only reflection of an agent's EXISTING settings (ADR-0074) — shared with
    the chat tool + CLI via the harness composer. Honours existing config."""
    from iris_harness.runtime.agent_console import agent_settings
    from iris_harness.runtime.settings_catalog import registry_catalog

    return agent_settings(
        name,
        tier_router=rt.tier_router,
        heartbeat_defs=rt.heartbeats.list_definitions(),
        catalog=registry_catalog(getattr(rt, "plugin_registry", None)),
    )


def install_agent_routes(app: FastAPI, runtime: Callable[[], Any]) -> None:
    """Register these routes. ``runtime`` returns the live runtime or raises 503."""

    # ── Per-agent console (ADR-0074) ───────────────────────────────────────────

    @app.get("/agents")
    def list_agents() -> dict[str, Any]:
        """Registered agents with presentation metadata (ADR-0074). The live
        runtime registry is the source of truth for which agents exist; the catalog
        only supplies title/description/source_kind. Pure read."""
        from iris_harness.agent.agent_metadata import agent_meta
        from iris_harness.runtime.plugin_host.inventory import agent_plugins

        rt = runtime()
        names = sorted(rt.agent_executor.registered_agents())
        owners = agent_plugins(getattr(rt, "plugin_registry", None))
        agents: list[dict[str, Any]] = []
        for name in names:
            meta = agent_meta(name)
            agents.append(
                {
                    "name": name,
                    "title": meta.title if meta else name,
                    "description": meta.description if meta else "",
                    "source_kind": meta.source_kind if meta else None,
                    # The plugin that registered this agent; None = the core's own.
                    "plugin": owners.get(name),
                }
            )
        return {"count": len(agents), "agents": agents}

    # ── Plugins (OSS plan; read-only inventory) ────────────────────────────────

    @app.get("/plugins")
    def list_plugins() -> dict[str, Any]:
        """The effective profile and every plugin it named: load status, what each
        registered, and its declared surface. Pure read over the live registry."""
        from iris_harness.runtime.plugin_host.inventory import plugins_inventory

        rt = runtime()
        return plugins_inventory(
            getattr(rt, "plugin_registry", None),
            getattr(rt, "profile", None),
            config_dir=getattr(rt, "config_dir", None),
        )

    @app.get(WEB_NAV_PATH)
    def web_nav_route() -> dict[str, Any]:
        """The console's navigation (OSS plan R17): the core's screens plus those of
        every mounted plugin. An installed plugin that is not mounted is listed under
        ``unavailable`` with why, never in the nav. Pure read."""
        from iris_harness.runtime.plugin_host.nav import web_nav

        rt = runtime()
        return web_nav(
            getattr(rt, "plugin_registry", None), config_dir=getattr(rt, "config_dir", None)
        )

    @app.get("/plugins/{name}")
    def get_plugin(name: str) -> dict[str, Any]:
        """One plugin: manifest, registrations, declared-vs-registered drift, and the
        YAML files in its directory. Pure read. 404 when the profile has no such plugin."""
        from iris_harness.runtime.plugin_host.inventory import plugin_detail

        rt = runtime()
        detail = plugin_detail(
            getattr(rt, "plugin_registry", None), getattr(rt, "profile", None), name
        )
        if detail is None:
            raise HTTPException(status_code=404, detail=f"no such plugin: {name}")
        return detail

    @app.get("/agents/{name}")
    def agent_dashboard(name: str) -> dict[str, Any]:
        """Ops snapshot for one agent (ADR-0074): its pending actions, recent runs,
        and key stores. Composed from existing surfaces — no new persistence. 404
        for an unregistered agent."""
        from iris_harness.agent.agent_metadata import agent_meta
        from iris_harness.services.tasks import TaskStore, pending_action_from_task

        rt = runtime()
        if name not in rt.agent_executor.registered_agents():
            raise HTTPException(status_code=404, detail=f"no such agent: {name}")
        meta = agent_meta(name)

        # Pending actions — the ADR-0073 provider seam, filtered to this agent.
        pending: list[dict[str, Any]] = []
        if meta and meta.source_kind:
            ts = TaskStore(db_path=rt.data_dir / "tasks.db")
            ts.ensure_schema()
            pending = [
                pending_action_from_task(t).as_dict()
                for t in ts.list(source_kind=meta.source_kind, has_action=True, limit=200)
                if t.status in ("open", "doing")
            ]

        # Recent runs — the heartbeat ticks that drive this agent (newest first).
        recent_runs: list[dict[str, Any]] = []
        hb_names = set(meta.heartbeats) if meta else set()
        if hb_names:
            runs = [r for r in rt.heartbeats.runs() if r.name in hb_names]
            for r in reversed(runs[-10:]):
                recent_runs.append(
                    {
                        "name": r.name,
                        "status": r.status.value,
                        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                        "output": r.output,
                        "error": r.error,
                    }
                )

        from iris_harness.runtime.plugin_host.inventory import agent_plugins

        return {
            "name": name,
            "title": meta.title if meta else name,
            "description": meta.description if meta else "",
            "source_kind": meta.source_kind if meta else None,
            # The plugin that registered this agent; None = the core's own.
            "plugin": agent_plugins(getattr(rt, "plugin_registry", None)).get(name),
            "pending_actions": {"count": len(pending), "actions": pending},
            "recent_runs": recent_runs,
            "stores": _agent_stores(name),
            "settings": _agent_settings(name, rt),
        }

    @app.get("/agents/{name}/metrics")
    def agent_metrics_endpoint(name: str, window_days: int = 7) -> dict[str, Any]:
        """Per-agent metrics (ADR-0074) — measured success/correction/token telemetry
        rolled up from the learning-intelligence matrix over the agent's intents.
        Reuses existing data; no Phoenix. 404 for an unregistered agent."""

        from iris_harness.runtime.agent_console import agent_metrics
        from iris_harness.services.learning.intelligence import build_intelligence

        rt = runtime()
        if name not in rt.agent_executor.registered_agents():
            raise HTTPException(status_code=404, detail=f"no such agent: {name}")
        store = getattr(rt, "learning_store", None)
        if store is None:
            return {"name": name, "available": False, "metrics": None}
        report = build_intelligence(store, window=timedelta(days=max(1, window_days)))
        return {"name": name, "available": True, "metrics": agent_metrics(name, report)}

    @app.patch("/agents/{name}/settings")
    def patch_agent_settings(
        name: str, request: AgentSettingsPatch, http_request: Request
    ) -> dict[str, Any]:
        """Edit an agent's toggles (ADR-0074 §4) — the catalog's on/off settings that name
        this agent (ADR-0120). Write-gated. Saved in the settings store + hot-applied. A
        guarded one is refused here (409): only Settings asks for the owner's confirm."""
        from iris_harness.runtime.agent_settings_store import set_toggle, toggles_for_agent
        from iris_harness.runtime.settings_catalog import registry_catalog

        rt = runtime()
        if name not in rt.agent_executor.registered_agents():
            raise HTTPException(status_code=404, detail=f"no such agent: {name}")
        catalog = registry_catalog(getattr(rt, "plugin_registry", None))
        owned = toggles_for_agent(name, catalog)
        applied: list[dict[str, Any]] = []
        for key, enabled in request.toggles.items():
            if key not in owned:
                raise HTTPException(
                    status_code=400, detail=f"{name} has no editable toggle {key!r}"
                )
            if owned[key].declaration.guarded:
                raise HTTPException(
                    status_code=409,
                    detail=f"{key} is guarded; change it in Settings, which asks you to confirm",
                )
            applied.append(
                set_toggle(
                    key,
                    bool(enabled),
                    catalog=catalog,
                    store=SettingsStore(db_path=rt.data_dir / SETTINGS_DB_NAME),
                    actor=actor_of(http_request),
                )
            )
        return {"name": name, "applied": applied, "settings": _agent_settings(name, rt)}
