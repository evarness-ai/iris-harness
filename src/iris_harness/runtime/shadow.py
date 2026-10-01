"""Shadow mode: run the candidate loop beside the live one and compare.

Release gate 1 -- `bootstrap.py` is the composition root only. Deciding what to
compare and how to report the difference is not composition; `build_runtime` just
asks for the handlers.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from iris_harness.agent.agent_executor import (
    AgentTask,
    HandlerResult,
    StreamChunk,
)
from iris_harness.foundation.observability.session_log import bind_context
from iris_harness.runtime.handlers.general_support import (
    _normalize_handler_result,
)

logger = logging.getLogger(__name__)


def _run_shadow_candidate(
    handler: Callable[[AgentTask], HandlerResult],
    task: AgentTask,
) -> dict[str, object]:
    started = time.monotonic()
    try:
        output, metadata = _normalize_handler_result(handler(task))
        return {
            "success": True,
            "output": output,
            "metadata": metadata,
            "latency_ms": (time.monotonic() - started) * 1000,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "output": "",
            "metadata": {},
            "latency_ms": (time.monotonic() - started) * 1000,
            "error": str(exc),
        }


def _shadow_compare_payload(
    *,
    legacy: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    legacy_output = str(legacy.get("output") or "")
    candidate_output = str(candidate.get("output") or "")
    legacy_success = bool(legacy.get("success"))
    candidate_success = bool(candidate.get("success"))
    comparison = {
        "legacy_success": legacy_success,
        "candidate_success": candidate_success,
        "success_match": legacy_success == candidate_success,
        "output_match": legacy_output.strip() == candidate_output.strip(),
        "legacy_latency_ms": round(float(cast(Any, legacy.get("latency_ms")) or 0.0), 3),
        "candidate_latency_ms": round(float(cast(Any, candidate.get("latency_ms")) or 0.0), 3),
        "latency_delta_ms": round(
            float(cast(Any, candidate.get("latency_ms")) or 0.0)
            - float(cast(Any, legacy.get("latency_ms")) or 0.0),
            3,
        ),
        "candidate_error": candidate.get("error"),
    }
    if not comparison["output_match"] or not comparison["success_match"]:
        logger.info(
            "AgenticCore shadow divergence detected: legacy_success=%s candidate_success=%s",
            legacy_success,
            candidate_success,
        )
    return comparison


def _make_shadow_handlers(
    *,
    legacy_handler: Callable[[AgentTask], HandlerResult],
    legacy_stream_handler: Callable[[AgentTask], Iterator[StreamChunk]],
    candidate_handler: Callable[[AgentTask], HandlerResult],
) -> tuple[Callable[[AgentTask], HandlerResult], Callable[[AgentTask], Iterator[StreamChunk]]]:
    """Run legacy + AgenticCore side by side while preserving legacy output."""

    def handler(task: AgentTask) -> HandlerResult:
        with ThreadPoolExecutor(max_workers=2) as pool:
            legacy_future = pool.submit(bind_context(_run_shadow_candidate), legacy_handler, task)
            candidate_future = pool.submit(
                bind_context(_run_shadow_candidate), candidate_handler, task
            )
            legacy = legacy_future.result()
            candidate = candidate_future.result()

        comparison = _shadow_compare_payload(legacy=legacy, candidate=candidate)
        if not bool(legacy.get("success")):
            error = str(legacy.get("error") or "legacy shadow handler failed")
            raise RuntimeError(error)

        metadata = dict(cast(Any, legacy.get("metadata")) or {})
        metadata.update(
            {
                "agentic_core_shadow_mode": "shadow",
                "agentic_core_shadow_response_source": "legacy",
                "agentic_core_shadow_compare": comparison,
                "agentic_core_shadow_candidate_metadata": dict(
                    cast(Any, candidate.get("metadata")) or {}
                ),
            }
        )
        return str(legacy.get("output") or ""), metadata

    def stream_handler(task: AgentTask) -> Iterator[StreamChunk]:
        legacy_meta: dict[str, object] = {}
        for item in legacy_stream_handler(task):
            if isinstance(item, dict):
                legacy_meta.update(item)
            else:
                yield item
        yield {
            **legacy_meta,
            "agentic_core_shadow_mode": "shadow",
            "agentic_core_shadow_response_source": "legacy",
            "agentic_core_shadow_pending": True,
        }

    return handler, stream_handler
