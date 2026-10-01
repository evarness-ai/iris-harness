"""The one way a route reaches the live runtime: ``runtime_or_503(app)``.

``create_app`` stores the runtime on ``app.state.runtime`` (``None`` until it is built,
or when the build failed). Every route and the route modules split out of
``create_app`` go through this, so "no runtime" is always the same 503.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException

if TYPE_CHECKING:
    from iris_harness.runtime import IrisRuntime


def runtime_or_503(app: FastAPI) -> IrisRuntime:
    rt: IrisRuntime | None = app.state.runtime
    if rt is None:
        raise HTTPException(status_code=503, detail="runtime unavailable")
    return rt
