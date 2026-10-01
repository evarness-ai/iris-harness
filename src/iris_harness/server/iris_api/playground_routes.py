"""The playground: list suites, run one, and the drift panel.

    GET    /playground/suites
    POST   /playground/run
    GET    /playground/drift

Moved out of ``create_app`` unchanged (review item: split the god function); the route
table and OpenAPI schema are identical before and after. The write guard in ``main``
still gates the mutating routes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class PlaygroundRunRequest(BaseModel):
    """Request body for ``POST /playground/run`` — run a suite by name."""

    suite: str = Field(..., min_length=1)


def install_playground_routes(app: FastAPI, runtime: Callable[[], Any]) -> None:
    """Register these routes. ``runtime`` returns the live runtime or raises 503."""

    # ── Playground (Phase 1) ──────────────────────────────────────────────
    # The harness test bench: list/run YAML scenarios and see config-vs-runtime
    # drift. GETs are open reads; POST /playground/run executes scenarios (which
    # drive chat) so it is write-gated like other active operations.
    @app.get("/playground/suites")
    def playground_suites() -> dict[str, Any]:
        from iris_harness.playground.loader import discover_suites, load_suite

        suites: list[dict[str, Any]] = []
        for path in discover_suites():
            try:
                suite = load_suite(path)
            except ValueError as exc:
                suites.append({"name": path.stem, "error": str(exc), "path": str(path)})
                continue
            suites.append(
                {
                    "name": suite.name,
                    "description": suite.description,
                    "path": str(path),
                    "scenarios": [
                        {"name": s.name, "message": s.message, "tags": list(s.tags)}
                        for s in suite.scenarios
                    ],
                }
            )
        return {"suites": suites}

    @app.post("/playground/run")
    def playground_run(request: PlaygroundRunRequest) -> dict[str, Any]:
        from iris_harness.playground.loader import discover_suites, load_suite
        from iris_harness.playground.runner import PlaygroundRunner

        rt = runtime()
        match = next((p for p in discover_suites() if p.stem == request.suite), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"no such suite: {request.suite}")
        suite = load_suite(match)
        # Interactive mode: run against the live runtime (its current flags),
        # not a fresh build. The CLI is the rebuild/CI path.
        result = PlaygroundRunner(rt.chat).run_suite(suite)
        return {
            "suite": result.suite_name,
            "ok": result.ok,
            "passed": result.passed,
            "total": result.total,
            "results": [
                {
                    "name": r.scenario_name,
                    "passed": r.passed,
                    "intent": r.intent,
                    "handler": r.handler,
                    "sources": list(r.sources),
                    "duration_ms": r.duration_ms,
                    "error": r.error,
                    "response": r.response,
                    "failed_assertions": [
                        {"field": a.field, "expected": a.expected, "actual": a.actual}
                        for a in r.failed_assertions
                    ],
                }
                for r in result.results
            ],
        }

    @app.get("/playground/drift")
    def playground_drift() -> dict[str, Any]:
        from iris_harness.playground.drift import build_drift_report

        rt = runtime()
        return build_drift_report(rt).to_dict()
