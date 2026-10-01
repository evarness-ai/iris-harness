"""The ``email_semantic_index`` heartbeat: keep the semantic email index filled.

New mail is indexed as it arrives (``email.new_arrived``). This fills in the rest —
mail from before the index existed, or indexed by an older version — at most
``max_per_run`` emails per run (heartbeat param; 300 by default), newest first, so a
small host (the 1 GiB cloud VM embeds about six emails a second) catches up over a
few runs instead of in one long burst. Skipped when semantic search is off.
"""

from __future__ import annotations

import logging
from typing import Any

from iris_harness.sdk.types import (
    HeartbeatDefinition,
    HeartbeatHandler,
    HeartbeatRun,
    HeartbeatStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_PER_RUN = 300


def build_semantic_index_handler(api: Any) -> HeartbeatHandler:
    """``api`` is the plugin API (or anything with ``.services.data_dir``); its services
    are read when the heartbeat runs, not at setup."""

    def handler(definition: HeartbeatDefinition) -> HeartbeatRun:
        services = api.services
        from iris_personal.email.semantic_index import (
            EmailSemanticIndex,
            refresh_semantic_index,
            semantic_search_enabled,
        )
        from iris_personal.email.store import EmailStore

        if not semantic_search_enabled():
            return HeartbeatRun(
                name=definition.name,
                status=HeartbeatStatus.SKIPPED,
                output="semantic email search is off (IRIS_EMAIL_SEMANTIC_SEARCH=0)",
            )
        params = definition.params or {}
        try:
            max_per_run = int(str(params.get("max_per_run", DEFAULT_MAX_PER_RUN)))
        except (TypeError, ValueError):
            max_per_run = DEFAULT_MAX_PER_RUN
        try:
            store = EmailStore(db_path=services.data_dir / "email.db")
            store.ensure_schema()
            index = EmailSemanticIndex(persist_dir=services.data_dir / "email_semantic")
            if not index.is_ready:
                return HeartbeatRun(
                    name=definition.name,
                    status=HeartbeatStatus.FAILED,
                    error="semantic index unavailable (chromadb or the embedding model)",
                )
            summary = refresh_semantic_index(
                email_store=store, index=index, max_per_run=max_per_run
            )
        except Exception as exc:  # never crash the heartbeat loop
            logger.warning("semantic index refresh failed", exc_info=True)
            return HeartbeatRun(name=definition.name, status=HeartbeatStatus.FAILED, error=str(exc))
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            output=f"{summary}; index holds {index.count()} vector(s)",
        )

    return handler


__all__ = ["build_semantic_index_handler"]
