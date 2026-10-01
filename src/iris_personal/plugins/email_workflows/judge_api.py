"""``/api/v1/email/judgments`` — the email judge's API (loop-proof PR 5).

The owner's rule: every capability has an API, never UI-only. The web list (Inbox →
Judged) reads and corrects through these; so can anything else::

    GET  /api/v1/email/judgments?bucket=&limit=&since=
         -> {"judgments": [row, ...], "buckets": [{"key", "name"}, ...]}
    GET  /api/v1/email/judgments/summary?day=YYYY-MM-DD
         -> {"day", "total", "counts": {bucket: n}, "unsure", "waiting"}
    POST /api/v1/email/judgments/{message_id}/bucket   {"bucket": "needs_reply"}
         -> {"judgment": row, "previous": "fyi", "changed": true}

A row: ``message_id``, ``account_id``, ``sender``, ``from_address``, ``subject``,
``snippet``, ``received_at``, ``bucket`` (the effective one) and ``bucket_name``,
``judge_bucket``, ``owner_bucket``, ``owner_source``, ``confidence``, ``figures`` (what
the judge read), ``judged_at``, ``corrected_at``. ``buckets`` is judge.yaml's, plus
``promo`` — what the owner may pick.

``bucket`` filters by the effective bucket; ``since`` is an ISO date or datetime
(judged at or after). ``day`` is the owner's local day (``IRIS_TZ``; default today).

The POST is :func:`.judge_corrections.apply_correction` with ``source="web"``: 404 when
the email was never judged, 422 when the bucket is not one of ``buckets``. Setting the
bucket it already has answers 200 with ``changed: false``. It is an ordinary gated
write: a read-only paired device cannot make it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

PATH = "/api/v1/email/judgments"
_LIMIT_MAX = 500


class BucketBody(BaseModel):
    bucket: str = Field(min_length=1, max_length=40)


def _since(raw: str | None, tz: Any) -> datetime | None:
    if not raw:
        return None
    text = raw.strip()
    try:
        if len(text) == 10:
            return datetime.combine(date.fromisoformat(text), time.min, tzinfo=tz)
        value = datetime.fromisoformat(text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"since: not a date: {raw!r}") from exc
    return value if value.tzinfo else value.replace(tzinfo=tz)


def day_summary(
    data_dir: Path, day: date, tz: Any, *, config_dir: Path | None = None
) -> dict[str, Any]:
    """Counts per effective bucket for one local day, plus Unsure and waiting."""
    from .judge_config import JudgeConfig
    from .judge_view import open_stores
    from .judgments import PROMO

    config = JudgeConfig.load(config_dir)
    stores = open_stores(data_dir)
    assert stores is not None  # create=True always opens
    judgments, _ = stores
    start = datetime.combine(day, time.min, tzinfo=tz)
    rows = judgments.judged_between(start, start + timedelta(days=1))
    counts: dict[str, int] = {}
    for key in (*config.keys, PROMO):
        n = sum(1 for j in rows if j.effective_bucket == key)
        if n:
            counts[key] = n
    return {
        "day": day.isoformat(),
        "total": len(rows),
        "counts": counts,
        "unsure": counts.get("unsure", 0),
        "waiting": judgments.count_waiting(),
    }


def build_router(
    data_dir: Callable[[], Path], config_dir: Callable[[], Path | None] | None = None
) -> APIRouter:
    """The router; ``data_dir`` / ``config_dir`` are read per request."""
    router = APIRouter(tags=["email"])

    def _cfg_dir() -> Path | None:
        return config_dir() if config_dir is not None else None

    def _zone() -> Any:
        from iris_harness.sdk.time import iris_timezone

        return iris_timezone()

    def _loaded() -> tuple[Any, Any]:
        from .judge_config import JudgeConfig
        from .judge_words import SurfaceWords

        return JudgeConfig.load(_cfg_dir()), SurfaceWords.load(_cfg_dir())

    @router.get(PATH)
    def list_judgments(
        bucket: str | None = Query(default=None, max_length=40),
        limit: int = Query(default=100, ge=1, le=_LIMIT_MAX),
        since: str | None = Query(default=None, max_length=40),
    ) -> dict[str, Any]:
        from .judge_corrections import valid_buckets
        from .judge_view import judged_emails, open_stores, row_view
        from .judge_words import bucket_name

        config, words = _loaded()
        allowed = valid_buckets(config)
        wanted = (bucket or "").strip() or None
        if wanted is not None and wanted not in allowed:
            raise HTTPException(status_code=422, detail=f"unknown bucket {wanted!r}")
        stores = open_stores(data_dir())
        assert stores is not None
        judgments, emails = stores
        items = judged_emails(
            judgments, emails, bucket=wanted, limit=limit, since=_since(since, _zone())
        )
        return {
            "judgments": [row_view(item, config, words) for item in items],
            "buckets": [{"key": k, "name": bucket_name(config, words, k)} for k in allowed],
        }

    @router.get(f"{PATH}/summary")
    def summary(day: str | None = Query(default=None, max_length=10)) -> dict[str, Any]:
        tz = _zone()
        try:
            wanted = date.fromisoformat(day) if day else datetime.now(tz).date()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"day: not a date: {day!r}") from exc
        return day_summary(data_dir(), wanted, tz, config_dir=_cfg_dir())

    @router.post(f"{PATH}/{{message_id}}/bucket")
    def set_bucket(message_id: str, body: BucketBody) -> dict[str, Any]:
        from .judge_corrections import UnknownBucket, apply_correction
        from .judge_view import default_emit, open_stores, row_view, with_emails

        config, words = _loaded()
        stores = open_stores(data_dir())
        assert stores is not None
        judgments, emails = stores
        try:
            correction = apply_correction(
                judgments,
                config,
                message_id,
                body.bucket.strip(),
                source="web",
                emit=default_emit,
            )
        except UnknownBucket as exc:
            raise HTTPException(status_code=422, detail=f"unknown bucket {body.bucket!r}") from exc
        if correction is None:
            raise HTTPException(status_code=404, detail="no judged email with that id")
        (item,) = with_emails(emails, [correction.judgment])
        return {
            "judgment": row_view(item, config, words),
            "previous": correction.previous,
            "changed": correction.changed,
        }

    return router


__all__ = ["PATH", "build_router", "day_summary"]
