"""Health and status: capabilities, System Health, connectors, incidents, a health-watch pass, and the runtime inventory.

    GET    /capabilities
    GET    /health
    GET    /health/connectors
    GET    /health/doctor
    GET    /health/incidents
    POST   /health/watch
    GET    /runtime/inventory

Moved out of ``create_app`` unchanged (review item: split the god function); the route
table and OpenAPI schema are identical before and after. The write guard in ``main``
still gates the mutating routes.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from iris_harness.server.iris_api.governance_routes import _env_flag
from iris_harness.server.iris_api.write_guard import _may_write


def _feedback_capture_enabled() -> bool:
    return _env_flag("IRIS_FEEDBACK_CAPTURE", default=False)


def install_health_routes(app: FastAPI, runtime: Callable[[], Any]) -> None:
    """Register these routes. ``runtime`` returns the live runtime or raises 503."""

    @app.get("/capabilities")
    def capabilities(request: Request) -> dict[str, Any]:
        """What the web UI is allowed to do (so it can hide write controls), and
        which deployment it is looking at.

        ``writes_enabled`` answers for *this caller*, not for the process: a
        console opened on a paired control device is editable even where the
        global switch is off, and one on a read device is not. Without it the
        "read-only" chip contradicted what the API actually allowed.

        ``deployment_label`` is ``IRIS_DEPLOYMENT_LABEL``, or ``""`` when unset. The
        console served from an image is built once for every deployment, so the
        build cannot carry the name; the badge reads it from here at runtime."""
        return {
            "writes_enabled": _may_write(getattr(request.state, "principal", None)),
            "feedback_capture": _feedback_capture_enabled(),
            "deployment_label": os.environ.get("IRIS_DEPLOYMENT_LABEL", "").strip(),
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Current System Health snapshot (ADR-0069): services, credentials, and
        hardware as green/yellow/red/grey, plus the red-check alert projection.

        Reads the background-cached snapshot (refreshed by the health_tick
        heartbeat), building one on a cold cache. Pure read — no governed call.
        Also backs the existing `iris status` CLI, which calls this path."""
        from iris_harness.services.health.service import current_snapshot
        from iris_harness.services.heartbeat.diagnostics import diagnose_heartbeats

        rt = runtime()
        heartbeats = getattr(rt, "heartbeats", None)
        diagnostics = (
            diagnose_heartbeats(
                heartbeats.list_definitions(),
                heartbeats.runs(),
                created_at=heartbeats.created_at(),
            )
            if heartbeats is not None
            else []
        )
        return current_snapshot(heartbeat_diagnostics=diagnostics).as_dict()

    def _heartbeat_diagnostics(rt: Any) -> list[Any]:
        from iris_harness.services.heartbeat.diagnostics import diagnose_heartbeats

        heartbeats = getattr(rt, "heartbeats", None)
        if heartbeats is None:
            return []
        return diagnose_heartbeats(
            heartbeats.list_definitions(), heartbeats.runs(), created_at=heartbeats.created_at()
        )

    @app.get("/health/connectors")
    def health_connectors(live: bool = False) -> dict[str, Any]:
        """Connectivity of every external connection (ADR-0116): the credential rows
        of the health snapshot — Gmail/Calendar/Drive accounts, cloud-LLM keys — plus
        any open incident on them.

        ``live=true`` rebuilds the snapshot with the network refresh-probe, so each
        token is actually exercised against its provider (the only egress here);
        the default reads the cached snapshot."""
        from iris_harness.services.health.models import CheckKind
        from iris_harness.services.health.service import current_snapshot, refresh
        from iris_harness.services.health.watch import current_watcher

        rt = runtime()
        diagnostics = _heartbeat_diagnostics(rt)
        snapshot = (
            refresh(net_probe=True, heartbeat_diagnostics=diagnostics)
            if live
            else current_snapshot(heartbeat_diagnostics=diagnostics)
        )
        rows = [c for c in snapshot.checks if c.kind is CheckKind.CREDENTIAL]
        watcher = current_watcher()
        open_incidents = (
            {i.key: i.as_dict() for i in watcher.store.open_incidents()} if watcher else {}
        )
        connectors = [{**c.as_dict(), "incident": open_incidents.get(c.key)} for c in rows]
        worst = max((c.state for c in rows), key=lambda st: st.rank, default=None)
        return {
            "state": worst.value if worst else "grey",
            "live": live,
            "sampled_at": snapshot.sampled_at,
            "connectors": connectors,
        }

    @app.get("/health/doctor")
    def health_doctor() -> dict[str, Any]:
        """The install preflight ``iris doctor`` prints (OSS plan R5): Python, platform,
        RAM, disk, IRIS_HOME, the vault master key, Ollama and the starter models, with a
        verdict. Read-only -- the fixes are the CLI's, where the owner can answer them.

        The OS keyring is not read here: a server must not raise a Keychain dialog. This
        process's own audit-key state answers instead once a governed call resolved it."""
        from iris_harness.services.system.doctor import run_doctor

        return run_doctor(read_keyring=False).as_dict()

    @app.get("/health/incidents")
    def health_incidents(open_only: bool = False, limit: int = 50) -> dict[str, Any]:
        """What the health watch noticed, tried, and told the owner (ADR-0116),
        newest first. ``open_only=true`` lists what is still broken."""
        from iris_harness.services.health.watch import current_watcher

        watcher = current_watcher()
        if watcher is None:
            return {"enabled": False, "count": 0, "incidents": []}
        incidents = watcher.store.recent(limit=limit, open_only=open_only)
        return {
            "enabled": watcher.config.enabled,
            "count": len(incidents),
            "incidents": [i.as_dict() for i in incidents],
        }

    @app.post("/health/watch")
    def run_health_watch() -> dict[str, Any]:
        """Run one health-watch pass now: refresh, repair, notify (ADR-0116).
        Write-gated by IRIS_WEBUI_ALLOW_WRITES — a pass can restart services."""
        from iris_harness.services.health.service import refresh
        from iris_harness.services.health.watch import current_watcher

        rt = runtime()
        watcher = current_watcher()
        if watcher is None:
            raise HTTPException(status_code=503, detail="health watch not installed")
        snapshot = refresh(heartbeat_diagnostics=_heartbeat_diagnostics(rt))
        return {
            "state": snapshot.worst().value,
            "summary": snapshot.summary(),
            "events": watcher.observe(snapshot),
            "open": [i.as_dict() for i in watcher.store.open_incidents()],
        }

    @app.get("/runtime/inventory")
    def runtime_inventory_endpoint() -> dict[str, Any]:
        """What IRIS is currently running (fast-follow #5): version + git rev,
        Python, key package versions, locally-pulled Ollama models. Local reads,
        zero egress; no "newer available" check. Pure read — no governed call."""
        from dataclasses import asdict

        from iris_harness.services.system.inventory import runtime_inventory

        return asdict(runtime_inventory())

    # ── Self-learning: experiments, proposals, metrics, flags ─────────────────
