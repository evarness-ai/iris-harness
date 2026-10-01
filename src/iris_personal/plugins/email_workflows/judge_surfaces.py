"""Wiring for the email judge's correction surfaces and digest lines (loop-proof PR 5).

``setup()`` calls :func:`register` once. Nothing here opens a store at setup: every
piece reads ``email.db`` when it runs, so a harness with no mail yet starts clean.

* the ``email`` Action Center slice — "What is this email?" on Unsure emails
  (:mod:`.judge_cards`), brought up to date on ``email.judged`` and
  ``email.judgment_corrected`` (process bus);
* ``/api/v1/email/judgments`` (:mod:`.judge_api`) — every capability has an API;
* the ``email_rebucket`` chat intercept (:mod:`.judge_chat`);
* the digest footer's ``email_judgment_corrections`` learned source
  (:mod:`.judge_digest`; the Needs reply section and the Judged line are skill tools
  of ``email-triage``).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from iris_harness.sdk import PluginAPI

logger = logging.getLogger(__name__)

INTERCEPT = "email_rebucket"


def register(api: PluginAPI) -> None:
    from iris_harness.sdk.pending_actions import register_provider

    from .judge_api import build_router
    from .judge_cards import EmailJudgeActionProvider, refresh_judge_cards
    from .judge_config import EMAIL_JUDGED, EMAIL_JUDGMENT_CORRECTED
    from .judge_digest import learned_source

    services = api.services
    data_dir = services.data_dir
    config_dir = services.config_dir

    register_provider(EmailJudgeActionProvider(data_dir, config_dir=config_dir))

    def _on_judgment(payload: object) -> None:
        del payload
        refresh_judge_cards(data_dir, config_dir=config_dir)

    api.subscribe(EMAIL_JUDGED, _on_judgment, scope="process")
    api.subscribe(EMAIL_JUDGMENT_CORRECTED, _on_judgment, scope="process")

    api.register_api_router(
        "email_judgments", lambda: build_router(lambda: data_dir, lambda: config_dir)
    )
    api.register_learned_source("email_judgment_corrections", learned_source(data_dir, config_dir))
    api.register_intercept(
        INTERCEPT,
        build_rebucket_intercept(services),
        trace_text="email re-bucketed",
        trace_fields=("email_rebucket", "message_id"),
        guard_output=True,  # names the email by its subject and sender (parity decision B)
    )


def build_rebucket_intercept(services: Any) -> Any:
    """``handler(message, *, session_id, span)`` for the ``email_rebucket`` intercept."""

    def _handle(message: str, *, session_id: str, span: Any = None) -> Any:
        from iris_harness.sdk.time import iris_timezone

        from .judge_chat import handle_rebucket_turn
        from .judge_config import JudgeConfig
        from .judge_view import default_emit, open_stores
        from .judge_words import SurfaceWords

        try:
            stores = open_stores(services.data_dir, create=False)
            if stores is None:
                return None
            judgments, emails = stores
            tz = iris_timezone()
            turn = handle_rebucket_turn(
                message,
                judgments,
                emails,
                config=JudgeConfig.load(services.config_dir),
                words=SurfaceWords.load(services.config_dir),
                now=datetime.now(tz),
                tz=tz,
                emit=default_emit,
            )
        except Exception:  # never break the turn; fall through to routing
            logger.exception("email rebucket intercept failed; falling through")
            return None
        if turn is None:
            return None
        return services.deterministic_reply(
            message=message,
            session_id=session_id,
            response=turn.reply,
            metadata={"email_rebucket": turn.kind, "message_id": turn.message_id},
            span=span,
        )

    return _handle


__all__ = ["INTERCEPT", "build_rebucket_intercept", "register"]
