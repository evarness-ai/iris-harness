"""/readyz: the API is ready only once its runtime is built.

Regression (found 2026-09-26): /healthz answers 200 as soon as the process serves, so an
API whose runtime failed to build (every data route then 503s) still looked healthy to
Docker, and roll_vm.sh would pass that roll. The container's healthcheck now asks
/readyz; /healthz keeps meaning "the process is up".
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from iris_harness.server.iris_api import main

ROOT = Path(__file__).resolve().parents[5]


def _failing_build() -> Any:
    raise RuntimeError("simulated runtime build failure")


def test_ready_once_the_runtime_is_built() -> None:
    app = main.create_app(runtime=SimpleNamespace(tracer=None), auto_start_runtime=False)  # type: ignore[arg-type]
    with TestClient(app) as client:  # no token: a probe, like /healthz
        resp = client.get(main.READY_PATH)
    assert resp.status_code == 200
    assert resp.json() == {"ready": True}


def test_not_ready_when_the_runtime_failed_to_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "build_runtime", _failing_build)
    app = main.create_app(auto_start_runtime=False)
    with TestClient(app) as client:
        ready = client.get(main.READY_PATH)
        alive = client.get("/healthz")
    assert ready.status_code == 503
    assert ready.json() == {"ready": False, "reason": "runtime not built"}
    # Liveness keeps its meaning for the health watch and the other probes.
    assert alive.status_code == 200
    assert alive.json()["runtime_ready"] is False
