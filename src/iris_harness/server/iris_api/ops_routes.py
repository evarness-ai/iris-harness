"""Operations: heartbeats (list, trigger, runs) and the LLM mode (state, pin, pressure).

    GET    /heartbeat
    POST   /heartbeat/trigger/{name}
    GET    /heartbeat/runs
    GET    /llm/mode
    POST   /llm/mode/pin
    DELETE /llm/mode/pin
    GET    /llm/pressure

Moved out of ``create_app`` unchanged (review item: split the god function); the route
table and OpenAPI schema are identical before and after. The write guard in ``main``
still gates the mutating routes.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from iris_harness.llm.arbiter import Mode
from iris_harness.runtime import IrisRuntime
from iris_harness.server.iris_api.settings_routes import heartbeat_payload


class LLMModePinRequest(BaseModel):
    """Request body for ``POST /llm/mode/pin``."""

    mode: Mode


def _governor_or_503(rt: IrisRuntime) -> Any:
    governor = getattr(rt.tier_router, "governor", None)
    if governor is None:
        raise HTTPException(status_code=503, detail="resource governor not configured")
    return governor


def _governor_state(governor: Any) -> dict[str, Any]:
    snapshot = governor.snapshot()
    pin = governor.pin
    return {
        "mode": str(governor.mode()),
        "auto_mode": str(getattr(governor, "_mode", governor.mode())),
        "pin": str(pin) if pin is not None else None,
        "adaptive": bool(getattr(governor, "adaptive", False)),
        "snapshot": _snapshot_payload(snapshot),
    }


def _snapshot_payload(snapshot: Any) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "ram_free_gb": snapshot.ram_free_gb,
        "cpu_percent": snapshot.cpu_percent,
        "cpu_speed_limit": snapshot.cpu_speed_limit,
        "thermal_throttled": snapshot.thermal_throttled,
        "sampled_at": snapshot.sampled_at.isoformat(),
    }


def install_ops_routes(app: FastAPI, runtime: Callable[[], Any]) -> None:
    """Register these routes. ``runtime`` returns the live runtime or raises 503."""

    @app.get("/heartbeat")
    def list_heartbeats() -> dict[str, Any]:
        rt = runtime()
        from iris_harness.services.heartbeat.diagnostics import diagnose_heartbeats

        diagnostics = diagnose_heartbeats(
            rt.heartbeats.list_definitions(),
            rt.heartbeats.runs(),
            created_at=rt.heartbeats.created_at(),
        )
        # Every heartbeat, disabled ones included, so the app can turn one back on
        # (ADR-0120); diagnostics above still judge only the scheduled ones.
        return {
            "heartbeats": [
                heartbeat_payload(rt.heartbeats, d) for d in rt.heartbeats.all_definitions()
            ],
            "timezone": rt.heartbeats.timezone(),
            "diagnostic_count": len(diagnostics),
            "diagnostics": [d.as_dict() for d in diagnostics],
        }

    @app.post("/heartbeat/trigger/{name}")
    def trigger_heartbeat(name: str) -> dict[str, Any]:
        rt = runtime()
        run = rt.heartbeats.trigger_by_name(name)
        if run is None:
            raise HTTPException(status_code=404, detail=f"heartbeat '{name}' not found")
        return {
            "name": run.name,
            "status": run.status.value,
            "output": run.output,
            "error": run.error,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        }

    @app.get("/heartbeat/runs")
    def list_heartbeat_runs(
        name: str | None = None,
        limit: int = 50,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Heartbeat runs, oldest first. With a run store (loop-proof D13) these are the
        kept runs — every cron run, only the status changes of a fast tick — which
        survive a restart, each with its slot, trigger and structured ``result``;
        without one, this process's in-memory runs."""
        rt = runtime()
        if getattr(rt.heartbeats, "run_store", None) is not None:
            if since is not None and since.tzinfo is None:
                since = since.replace(tzinfo=UTC)
            kept = rt.heartbeats.kept_runs(name=name or None, since=since, limit=max(limit, 0))
            return {
                "count": len(kept),
                "total": len(kept),
                "source": "kept",
                "runs": [r.as_dict() for r in reversed(kept)],
            }
        runs = rt.heartbeats.runs()
        if name:
            runs = [r for r in runs if r.name == name]
        clipped = runs[-limit:] if limit > 0 else runs
        return {
            "count": len(clipped),
            "total": len(runs),
            "runs": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                    "output": r.output,
                    "error": r.error,
                }
                for r in clipped
            ],
        }

    @app.get("/llm/mode")
    def get_llm_mode() -> dict[str, Any]:
        rt = runtime()
        governor = _governor_or_503(rt)
        return _governor_state(governor)

    @app.post("/llm/mode/pin")
    def pin_llm_mode(request: LLMModePinRequest) -> dict[str, Any]:
        rt = runtime()
        governor = _governor_or_503(rt)
        governor.set_pin(request.mode)
        return _governor_state(governor)

    @app.delete("/llm/mode/pin")
    def unpin_llm_mode() -> dict[str, Any]:
        rt = runtime()
        governor = _governor_or_503(rt)
        governor.set_pin(None)
        return _governor_state(governor)

    @app.get("/llm/pressure")
    def get_llm_pressure() -> dict[str, Any]:
        rt = runtime()
        governor = _governor_or_503(rt)
        snapshot = governor.poll()
        return {
            "mode": str(governor.mode()),
            "snapshot": _snapshot_payload(snapshot),
        }

    # NOTE: registered BEFORE the greedy ``/memory/{session_id}`` route below, which
    # would otherwise match ``/memory/contradictions`` as session_id="contradictions".
