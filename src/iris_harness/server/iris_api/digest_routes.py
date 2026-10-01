"""Read routes for stored digests (loop-proof plan PR 2, graph §7).

    GET /api/digest             recent digests, newest first (summaries)
    GET /api/digest/latest      the newest digest, full
    GET /api/digest/{id}        one digest, full — where a push notification lands

The capability is ``services.digests.DigestStore``, written by the
``skill_brief`` handler on every render; these routes and the web console's
``/digest`` view are surfaces. Read-only, so any authenticated principal may
call them and the write gate never applies.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from iris_harness.services.digests import DigestStore, StoredDigest

DIGEST_PATH = "/api/digest"

_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _summary(digest: StoredDigest) -> dict[str, Any]:
    return {
        "id": digest.id,
        "created_at": digest.created_at,
        "subject": digest.subject,
        "skill_id": digest.skill_id,
        "failed_sections": len(digest.failed_sections),
    }


def install_digest_routes(app: FastAPI, store: Callable[[], DigestStore]) -> None:
    """Register the digest read routes. ``store`` returns the shared digest store."""

    @app.get(DIGEST_PATH)
    def digest_list(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
        return {"digests": [_summary(d) for d in store().list(limit=limit)]}

    # Registered before the ``{digest_id}`` route so "latest" is never read as an id.
    @app.get(f"{DIGEST_PATH}/latest")
    def digest_latest(skill_id: str | None = None) -> dict[str, Any]:
        digest = store().latest(skill_id=skill_id or None)
        if digest is None:
            raise HTTPException(status_code=404, detail="no digest has been sent yet")
        return digest.as_dict()

    @app.get(f"{DIGEST_PATH}/{{digest_id}}")
    def digest_get(digest_id: str) -> dict[str, Any]:
        digest = store().get(digest_id) if _ID_RE.match(digest_id) else None
        if digest is None:
            raise HTTPException(status_code=404, detail=f"no digest {digest_id!r}")
        return digest.as_dict()


__all__ = ["DIGEST_PATH", "install_digest_routes"]
