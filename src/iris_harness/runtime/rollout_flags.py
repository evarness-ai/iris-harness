"""The rollout flags `build_runtime` branches on.

Each answers one question the composition root asks once -- is the agentic loop on,
in shadow, or off; does the filemanager persona mount -- by reading an env var and
normalising it. Reading a flag is not composition, and gate 1 wants the composition
root to be composition.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _agentic_core_rollout_mode() -> str:
    """The AgenticCore rollout flag: ``off`` / ``shadow`` / ``on``.

    The rule itself lives with the loop (``core.agentic_core.rollout_mode``) as of
    M6.1b, so a plugin whose agent runs on that loop reads the same flag without
    importing the composition root.
    """
    from iris_harness.agent.agentic_core import rollout_mode

    return rollout_mode()


def _filemanager_agent_enabled() -> bool:
    """FMX5: answer the ``files`` intent as a first-class ``filemanager`` agent.

    Opt-in and off by default: with it off the agent stays unregistered and ``files``
    routing falls back exactly as before, because ``allowed_agent_types`` gates on
    ``registered_agents()``. Requires the base loop (rollout on/shadow).
    """
    return os.getenv("IRIS_FILEMANAGER_AGENT", "").strip().lower() in {"1", "true", "yes", "on"}
