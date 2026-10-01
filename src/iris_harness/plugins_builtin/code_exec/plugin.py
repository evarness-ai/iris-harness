"""``setup(api)`` for the code-exec reference plugin (OSS plan M4.6, decision 9).

Two registrations, and the mount is conditional on the sandbox actually being
reachable — the same check ``build_runtime`` used to make before registering the
agent. That condition is the plugin's own now, which is the point of the split:
a capability decides whether it can serve, instead of the composition root
knowing how to ask.

With Docker down, nothing registers. The ReAct pool simply has no ``code_exec``
tool and the ``code_exec`` agent type is unregistered, so routing falls back
exactly as it did before — rather than offering a tool whose every call returns
"code_exec unavailable".
"""

from __future__ import annotations

import logging
from typing import Any

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.types import AgentTask

from .handler import _is_docker_available, _make_code_exec_handler

logger = logging.getLogger(__name__)

TOOL_DESCRIPTION = (
    "Execute a short coding task inside a sandboxed Docker container. " 'Args: {"task": str}.'
)


def setup(api: PluginAPI) -> None:
    if not _is_docker_available():
        logger.warning("code_exec not mounted — the Docker daemon is not reachable")
        return

    handler, stream_handler = _make_code_exec_handler(
        api.services.tier_router,
        # Prior attempts at a similar task, and where this run's outcome is
        # recorded. None when lesson capture is off; the loop runs without it.
        lesson_capture=api.services.lessons,
        repo_root=_repo_root(api),
    )
    api.register_intent_handler("code_exec", handler, stream_handler=stream_handler)

    def _code_exec(args: dict[str, Any]) -> str:
        task_str = str(args.get("task") or args.get("script") or args.get("input") or "").strip()
        if not task_str:
            return "Error: code_exec requires a 'task' argument."
        try:
            result = handler(AgentTask(query=task_str, agent_type="code_exec"))
        except Exception as exc:  # noqa: BLE001 — a tool returns an observation, never raises
            return f"code_exec failed: {exc}"
        if isinstance(result, tuple):
            return str(result[0])
        return str(result)

    api.register_tool("code_exec", TOOL_DESCRIPTION, _code_exec)
    logger.debug("code_exec: tool + agent registered")


def _repo_root(api: PluginAPI) -> Any:
    """The repo root, for ``config/governance/sandbox.yaml``.

    ``HarnessServices`` exposes ``config_dir``, and the sandbox policy is read
    relative to the repo, so derive the root from it rather than asking for a
    second path the harness would have to publish.
    """
    config_dir = api.services.config_dir
    return config_dir.parent if config_dir.name == "config" else None
