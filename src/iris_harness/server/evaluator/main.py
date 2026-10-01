"""Thin FastAPI wrapper around the in-process evaluator registry.

Run with:
    poetry run uvicorn iris_harness.server.evaluator.main:app --port 8090

The agent talks to this service over HTTP loopback when
``IRIS_GOVERNANCE_EVALUATOR_MODE=remote``. Endpoint:

- ``GET /healthz``     — liveness probe + signal count
- ``POST /evaluate``   — body: ``StepRecord`` JSON → ``{signals: [...]}``

The endpoint contract MATCHES the in-process registry: same
``SignalResult`` shape (one per registered signal) and same
worst-wins precedence applied by the caller. Out-of-process mode is
a pure transport swap, not a policy change.

The service runs the same default signals as the in-process registry
when started without flags. Future work (a follow-up story) wires
loop_detect / goal_drift / cost_budget into this service via the
same env vars the in-process wiring honours.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Body, FastAPI, Request

# Install the root log handler (IRIS_LOG_LEVEL, default INFO) before anything else imports
# so every iris.* INFO line — including the ingress/egress audit trail — actually appears
# in the service log instead of being dropped at the default WARNING level.
from iris_harness.foundation.observability.logging_setup import configure_logging

configure_logging(service="evaluator")

from iris_harness.kernel.governance.evaluator import EvaluatorRegistry, StepRecord  # noqa: E402
from iris_harness.kernel.governance.evaluator.signals import (  # noqa: E402
    ActionRepeatSignal,
    ClassificationViolationSignal,
    StepCapSignal,
    ToolFailureStreakSignal,
)
from iris_harness.server.auth import install_bearer_auth, routed_path  # noqa: E402

logger = logging.getLogger(__name__)

_STEP_BODY = Body(...)  # module-level Body() so the function signature isn't a call-default


def build_default_registry() -> EvaluatorRegistry:
    """Match the in-process default — the four cheap signals."""
    registry = EvaluatorRegistry()
    registry.register(ClassificationViolationSignal())
    registry.register(StepCapSignal())
    registry.register(ActionRepeatSignal())
    registry.register(ToolFailureStreakSignal())
    registry.init_lock()
    return registry


def create_app(*, registry: EvaluatorRegistry | None = None) -> FastAPI:
    reg = registry or build_default_registry()
    app = FastAPI(title="IRIS Evaluator", version="0.1.0")

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
        return {
            "ok": True,
            "signal_count": reg.signal_count(),
            "locked": reg.is_locked,
        }

    @app.post("/evaluate")
    def evaluate(payload: dict[str, Any] = _STEP_BODY) -> dict[str, Any]:
        step = StepRecord.model_validate(payload)
        results = reg.evaluate(step)
        return {
            "signals": [r.model_dump(mode="json") for r in results],
        }

    @app.post("/reset/{run_id}")
    def reset(run_id: str) -> dict[str, Any]:
        reg.reset_run_state(run_id)
        return {"ok": True, "run_id": run_id}

    return app


app = create_app()
