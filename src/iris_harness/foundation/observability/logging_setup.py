"""Central logging configuration + ingress/egress audit loggers.

Two problems this solves:

1. **INFO was invisible.** Services start via ``uvicorn`` with no ``--log-level`` and
   nothing configures the root logger, so every ``iris.*`` logger fell back to WARNING and
   all INFO-level lines (including egress audit lines) were silently dropped. ``configure_logging``
   installs a root handler at ``IRIS_LOG_LEVEL`` (default INFO) so they appear in the service
   log file.

2. **No single ingress/egress trail.** ``log_ingress`` / ``log_egress`` emit one structured
   line per inbound request and per outbound network call on dedicated ``iris.ingress`` /
   ``iris.egress`` loggers, so every boundary crossing is visible in one place with no
   loopholes. These complement (not replace) the per-turn session JSONL and the governance
   audit ledger.

Call ``configure_logging(service=...)`` once at each service entrypoint (before the app
starts handling traffic). It is idempotent.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from iris_harness.foundation.logsafe import log_safe

_CONFIGURED = False

# Dedicated boundary loggers. Named so they can be filtered/routed independently
# (e.g. `grep "iris.egress"` for the complete outbound trail).
ingress_logger = logging.getLogger("iris.ingress")
egress_logger = logging.getLogger("iris.egress")

# Third-party loggers that are noisy at INFO and would drown the signal.
_NOISY_LIBS = ("httpx", "httpcore", "urllib3", "openai", "asyncio", "watchfiles")


def configure_logging(*, service: str = "iris", force: bool = False) -> None:
    """Install a root stdout handler at IRIS_LOG_LEVEL (default INFO). Idempotent.

    uvicorn redirects the service's stdout to its ``*.log`` file, so a stdout handler is
    what lands the lines on disk. ``service`` is embedded in the format for multi-service
    grepping. Set ``IRIS_LOG_LEVEL=DEBUG`` for more, ``WARNING`` to quiet INFO again.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    _CONFIGURED = True

    level_name = os.getenv("IRIS_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Reuse an existing iris handler if configure_logging is somehow called twice.
    have_iris_handler = any(getattr(h, "_iris_handler", False) for h in root.handlers)
    if not have_iris_handler:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt=f"%(asctime)s %(levelname)s [{service}] %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        handler._iris_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)

    # Keep chatty libraries from burying the app + boundary logs.
    for name in _NOISY_LIBS:
        logging.getLogger(name).setLevel(logging.WARNING)

    egress_logger.info("logging configured: service=%s level=%s", service, level_name)


def log_ingress(
    *,
    method: str,
    path: str,
    source: str = "",
    status: int | None = None,
    duration_ms: float | None = None,
    session_id: str | None = None,
) -> None:
    """One line per inbound request crossing a service boundary (HTTP / WS / channel)."""
    # method/path/source/session id come off the wire: the ASGI path is percent-decoded, so
    # %0a in a URL is a real newline here. One request must stay one line.
    parts = [f"INGRESS {log_safe(method)} {log_safe(path, 500)}"]
    if source:
        parts.append(f"from={log_safe(source)}")
    if session_id:
        parts.append(f"session={log_safe(session_id)}")
    if status is not None:
        parts.append(f"status={status}")
    if duration_ms is not None:
        parts.append(f"dur_ms={duration_ms:.0f}")
    ingress_logger.info(" ".join(parts))


def log_egress(
    *,
    destination: str,
    method: str = "",
    purpose: str = "",
    kind: str = "network",
    detail: str = "",
    status: int | str | None = None,
    **fields: Any,
) -> None:
    """One line per outbound network call crossing a service boundary.

    ``kind`` groups the call (``llm`` / ``search`` / ``crawl`` / ``mcp`` / ``channel`` /
    ``service`` / ``network``). ``destination`` is the host/URL (no secrets — callers must
    pass a sanitized target). Never raises.
    """
    try:
        parts = [f"EGRESS {kind}"]
        if method:
            parts.append(method.upper())
        parts.append(f"-> {destination}")
        if purpose:
            parts.append(f"purpose={purpose}")
        if status is not None:
            parts.append(f"status={status}")
        if detail:
            parts.append(detail)
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        egress_logger.info(" ".join(parts))
    except Exception:  # noqa: BLE001, S110 - logging must never break a real call
        pass


__all__ = ["configure_logging", "egress_logger", "ingress_logger", "log_egress", "log_ingress"]
