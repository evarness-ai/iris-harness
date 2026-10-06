"""The MCP protocol half of ``iris mcp serve``: a stdlib JSON-RPC stdio server.

A spec-subset Model Context Protocol server implemented on the stdlib only
(newline-delimited JSON-RPC 2.0 over stdin/stdout) -- no `mcp`/`fastmcp`
dependency, so it is trivially embeddable. Implements the methods real clients use to
drive tools: ``initialize``, ``notifications/initialized``, ``ping``, ``tools/list``,
``tools/call``.

This module only speaks the protocol. It never decides how a tool runs: each
:class:`McpTool` carries an ``invoke`` its builder supplies, and ``iris mcp serve``
supplies one that runs the call through the governed tool runner
(``runtime/mcp_serve.py``) -- ``PRE_TOOL_USE``, approvals, ``POST_TOOL_USE`` and the
audit rows, as the caller ``mcp:<client>``. It used to load skill tools here and call
them directly, which skipped all of that (OSS plan L2 finding).

:func:`load_skill_tools` reads skill packages for that layer: each tool with its JSON
schema and the governor route its manifest declares, and nothing that runs it.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TextIO

from pydantic import BaseModel

from iris_harness.foundation.paths import config_root

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "iris"
SERVER_VERSION = "0.2.0"
# JSON-RPC error codes
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602


def default_repo_root() -> Path:
    """Repo root. Asked for, not counted.

    This used to be ``Path(__file__).resolve().parents[4]``, and layer 3 of M6.2 made it
    ``parents[3]``-wrong: the skills failed to load and the MCP server exposed no tools at
    all, silently. ``foundation.paths.repo_root`` walks up for the ``pyproject.toml``
    marker instead, so the next tree move cannot break it. The skills it serves are read
    from ``<root>/config/skills``, so the root is now ``foundation.paths.config_root()``:
    the checkout's root there, the packaged ``iris_harness/_data`` from a wheel.
    """
    return config_root()


@dataclass(frozen=True)
class McpReply:
    """What one ``tools/call`` answers: the text, and whether it is an error."""

    text: str
    is_error: bool = False


@dataclass
class McpTool:
    """One exposed tool: name, description, JSON-Schema, and an invoker.

    ``invoke`` returns the result text, or an :class:`McpReply` -- with ``is_error`` set
    when the call produced no usable result (refused, held, withheld).
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    invoke: Callable[[dict[str, Any]], str | McpReply]


@dataclass(frozen=True)
class SkillTool:
    """A skill package's tool, as its manifest declares it. Nothing here runs it."""

    name: str
    description: str
    input_schema: dict[str, Any]
    # The manifest's ``governor_route`` (``<domain>/<verb>``, e.g. ``calendar/read``).
    governor_route: str
    skill: str
    # The LangChain ``BaseTool`` instance; ``invoke(args)`` runs it.
    instance: Any
    # What the tool's output is, from its manifest (``external``: text a third party wrote),
    # so a served skill tool is marked and scanned like the same tool in the agent loop.
    content: Literal["internal", "external"] = "internal"

    @property
    def verb(self) -> str:
        """The route's verb: ``read`` for ``calendar/read``, empty when undeclared."""
        return self.governor_route.rsplit("/", 1)[-1] if self.governor_route else ""


def _schema_of(instance: Any) -> dict[str, Any]:
    raw_schema = instance.args_schema
    if raw_schema is None:
        return {"type": "object", "properties": {}}
    if isinstance(raw_schema, dict):
        return raw_schema
    if issubclass(raw_schema, BaseModel):
        return dict(raw_schema.model_json_schema())
    return dict(raw_schema.schema())  # a pydantic.v1 model, which LangChain 1.x accepts


def load_skill_tools(
    skills: Iterable[str],
    *,
    repo_root: Path | None = None,
) -> list[SkillTool]:
    """The tools of the named skill packages, each with its declared governor route."""
    from iris_harness.tools.skills.registry import SkillRegistry

    wanted = set(skills)
    if not wanted:
        return []
    registry = SkillRegistry(repo_root or default_repo_root())
    registry.discover()

    tools: list[SkillTool] = []
    for package in registry.list_packages(only_loadable=True):
        if package.manifest.name not in wanted:
            continue
        routes = {t.name: t.governor_route for t in package.manifest.tools}
        contents = {t.name: t.content for t in package.manifest.tools}
        for tool_class in package.tool_classes:
            instance = tool_class()
            tools.append(
                SkillTool(
                    name=instance.name,
                    description=instance.description,
                    input_schema=_schema_of(instance),
                    governor_route=routes.get(instance.name, ""),
                    skill=package.manifest.name,
                    instance=instance,
                    content=contents.get(instance.name, "internal"),
                )
            )
    return tools


@dataclass
class IrisMCPServer:
    """JSON-RPC handler exposing a fixed set of tools.

    ``client_info`` is what the client said about itself at ``initialize``. It is a
    claim, kept for the log: nothing decides a permission by it.
    """

    tools: list[McpTool]
    client_info: dict[str, Any] = field(default_factory=dict)

    def _by_name(self, name: str) -> McpTool | None:
        return next((t for t in self.tools if t.name == name), None)

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Handle one JSON-RPC request. Returns None for notifications."""
        method = request.get("method")
        req_id = request.get("id")
        # Notifications (no id) get no response.
        if req_id is None and method != "initialize":
            return None

        if method == "initialize":
            params = request.get("params") or {}
            claimed = params.get("clientInfo")
            self.client_info = dict(claimed) if isinstance(claimed, dict) else {}
            client_version = params.get("protocolVersion")
            return self._ok(
                req_id,
                {
                    "protocolVersion": client_version or PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            )
        if method == "ping":
            return self._ok(req_id, {})
        if method == "tools/list":
            return self._ok(
                req_id,
                {
                    "tools": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.input_schema,
                        }
                        for t in self.tools
                    ]
                },
            )
        if method == "tools/call":
            return self._call_tool(req_id, request.get("params") or {})

        return self._err(req_id, _METHOD_NOT_FOUND, f"method not found: {method}")

    def _call_tool(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not name:
            return self._err(req_id, _INVALID_PARAMS, "tools/call requires 'name'")
        tool = self._by_name(str(name))
        if tool is None:
            return self._tool_result(req_id, f"unknown tool: {name}", is_error=True)
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return self._err(req_id, _INVALID_PARAMS, "tools/call 'arguments' must be an object")
        try:
            reply = tool.invoke(arguments)
        except Exception as exc:  # surface tool errors in-band
            logger.exception("mcp tools/call failed for %s", name)
            return self._tool_result(req_id, f"{type(exc).__name__}: {exc}", is_error=True)
        if isinstance(reply, McpReply):
            return self._tool_result(req_id, reply.text, is_error=reply.is_error)
        return self._tool_result(req_id, reply, is_error=False)

    @staticmethod
    def _ok(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _err(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    def _tool_result(self, req_id: Any, text: str, *, is_error: bool) -> dict[str, Any]:
        return self._ok(req_id, {"content": [{"type": "text", "text": text}], "isError": is_error})


def serve_stdio(
    server: IrisMCPServer,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> None:
    """Run the newline-delimited JSON-RPC loop until stdin closes."""
    src = stdin or sys.stdin
    dst = stdout or sys.stdout
    for line in src:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("mcp: dropping non-JSON line")
            continue
        if not isinstance(request, dict):
            logger.warning("mcp: dropping a JSON line that is not a request object")
            continue
        response = server.handle(request)
        if response is not None:
            dst.write(json.dumps(response) + "\n")
            dst.flush()


__all__ = [
    "IrisMCPServer",
    "McpReply",
    "McpTool",
    "SkillTool",
    "default_repo_root",
    "load_skill_tools",
    "serve_stdio",
]
