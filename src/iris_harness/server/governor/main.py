"""Thin FastAPI wrapper around the embedded IRIS governor kernel."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request

from iris_harness.foundation.observability.logging_setup import configure_logging

# Install the root log handler (IRIS_LOG_LEVEL, default INFO) before anything else imports
# so every iris.* INFO line — including the ingress/egress audit trail — actually appears
# in the service log instead of being dropped at the default WARNING level.
from iris_harness.foundation.paths import config_path, repo_root

configure_logging(service="governor")

from iris_harness.kernel.governor import IRISGovernorService  # noqa: E402
from iris_harness.server.auth import install_bearer_auth, routed_path  # noqa: E402

REPO_ROOT = repo_root()


def create_app(*, governor_service: IRISGovernorService | None = None) -> FastAPI:
    """Create the IRIS governor HTTP wrapper."""
    service = governor_service or _build_governor_service()
    app = FastAPI(title="IRIS Governor", version="0.1.0")

    # Registered before the ingress log so the log stays outermost (Starlette
    # runs the last-registered middleware first) and refused attempts are
    # still recorded in the ingress trail.
    install_bearer_auth(app)

    @app.middleware("http")
    async def _ingress_log(request: Request, call_next: Any) -> Any:
        # Log EVERY inbound request crossing the service boundary (no allowlist),
        # so there is a complete ingress trail. Health probes are demoted to DEBUG.
        import time as _time

        from iris_harness.foundation.observability.logging_setup import (
            ingress_logger,
            log_ingress,
        )

        start = _time.monotonic()
        path = routed_path(request)
        is_probe = path in {"/healthz", "/health"}
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            log_ingress(
                method=request.method,
                path=path,
                source=request.client.host if request.client else "",
                status=500,
                duration_ms=(_time.monotonic() - start) * 1000,
            )
            raise
        dur_ms = (_time.monotonic() - start) * 1000
        if is_probe:
            ingress_logger.debug("INGRESS %s %s status=%s", request.method, path, status)
        else:
            log_ingress(
                method=request.method,
                path=path,
                source=request.client.host if request.client else "",
                status=status,
                duration_ms=dur_ms,
            )
        return response

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return service.health_report().model_dump(mode="json")

    @app.get("/routes")
    def list_routes() -> dict[str, Any]:
        report = service.health_report()
        return {
            "route_count": report.route_count,
            "routes": report.routes,
            "policy_version": report.policy_version,
        }

    @app.post("/guard/{route:path}")
    def guard_route(
        route: str,
        payload: dict[str, Any] | None = Body(default=None),  # noqa: B008 - FastAPI idiom
    ) -> dict[str, Any]:
        try:
            decision = service.guard(route, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return decision.model_dump(mode="json")

    return app


def _build_governor_service() -> IRISGovernorService:
    """Build the default governor kernel: the policy from the resolved config
    (``foundation.paths.config_dir()``), the audit log under the repo root as before."""
    policy_path = _read_path_env("IRIS_GOVERNOR_POLICY_PATH") or config_path(
        "governor", "policy.yaml"
    )
    audit_db_path = _read_path_env("IRIS_GOVERNOR_AUDIT_DB")
    return IRISGovernorService.from_repo_root(
        REPO_ROOT,
        policy_path=policy_path,
        audit_db_path=audit_db_path,
    )


def _read_path_env(env_var: str) -> Path | None:
    """Resolve an optional filesystem path from the environment."""
    raw_value = os.getenv(env_var)
    if raw_value is None or not raw_value.strip():
        return None
    return Path(raw_value).expanduser().resolve()


app = create_app()
