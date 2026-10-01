"""``POST /api/digest/not-useful`` — the 👎 on a morning-digest Focus line (loop-proof D17).

Every Focus line ends with ``[👎](iris:not-useful/<quoted sender>)``; the web
rendering turns it into a button that posts ``{"sender": "..."}`` here. The verdict
goes into the existing surface-suppression ledger (``learning.db``, issue 0028) under
``email/focus`` keyed by the sender's address — the same store and the same effect as
any other "not useful": the Focus slot consults ``should_suppress`` and the sender is
gone from tomorrow's Focus, and the digest footer's ``learned yesterday`` line names it.

Who may call: any authenticated principal. Like ``/surface-feedback`` it is local
learning telemetry with no external effect, so it is not a governed write — a
read-only paired phone can still hide noise from its own digest.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from iris_harness.services.learning.suppression import (
    EMAIL_FOCUS_SURFACE,
    EMAIL_SEARCH_SUBSYSTEM,
    NOT_USEFUL,
    SurfaceFeedbackStore,
    email_focus_dims,
)

NOT_USEFUL_PATH = "/api/digest/not-useful"


class DigestNotUsefulRequest(BaseModel):
    """The sender of the Focus line the owner marked not useful."""

    sender: str = Field(..., min_length=3, max_length=512)


def install_digest_feedback_routes(
    app: FastAPI, store: Callable[[], SurfaceFeedbackStore] = SurfaceFeedbackStore
) -> None:
    """Mount the digest 👎 route. ``store`` is injectable for tests."""

    @app.post(NOT_USEFUL_PATH)
    def digest_not_useful(request: DigestNotUsefulRequest) -> dict[str, Any]:
        dims = email_focus_dims(request.sender)
        sender = dims["sender"]
        if "@" not in sender or sender.startswith("@") or sender.endswith("@"):
            raise HTTPException(status_code=422, detail="sender must be an email address")
        ledger = store()
        ledger.ensure_schema()
        ledger.record(EMAIL_SEARCH_SUBSYSTEM, EMAIL_FOCUS_SURFACE, dims, NOT_USEFUL)
        return {"ok": True, "sender": sender, "hidden_from": EMAIL_FOCUS_SURFACE}


__all__ = ["NOT_USEFUL_PATH", "DigestNotUsefulRequest", "install_digest_feedback_routes"]
