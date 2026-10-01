"""A minimal MCP client over stdio: what Claude Desktop, an IDE or another agent does.

MCP's stdio transport is newline-delimited JSON-RPC 2.0 on a child process's stdin and
stdout. This client starts a server, performs the ``initialize`` handshake, and then
lists and calls tools. Standard library only; it works against ``iris mcp serve`` and
against any other MCP stdio server (``weather_server.py`` here).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Self

PROTOCOL_VERSION = "2024-11-05"


class McpError(RuntimeError):
    """The server answered a request with a JSON-RPC error."""


class StdioMcpClient:
    """``with StdioMcpClient([...]) as client: client.list_tools()``."""

    def __init__(self, command: Sequence[str], *, env: Mapping[str, str] | None = None) -> None:
        self._command = list(command)
        self._env = dict(env) if env is not None else None
        self._process: subprocess.Popen[str] | None = None
        self._next_id = 0
        self.server_info: dict[str, Any] = {}

    def __enter__(self) -> Self:
        self._process = subprocess.Popen(  # noqa: S603 -- the caller's own server command
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=self._env,
            text=True,
        )
        result = self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "example-client", "version": "0.1.0"},
            },
        )
        self.server_info = dict(result.get("serverInfo") or {})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        process = self._process
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _send(self, message: dict[str, Any]) -> None:
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send one request and return its ``result``."""
        assert self._process is not None and self._process.stdout is not None
        self._next_id += 1
        self._send(
            {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}}
        )
        line = self._process.stdout.readline()
        if not line:
            raise McpError(f"the server closed the connection during {method!r}")
        reply = json.loads(line)
        if "error" in reply:
            raise McpError(str(reply["error"].get("message", reply["error"])))
        return dict(reply["result"])

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self.request("tools/list")["tools"])

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Call a tool; returns its text, raising :class:`McpError` if it failed."""
        result = self.request("tools/call", {"name": name, "arguments": arguments or {}})
        text = "\n".join(
            part.get("text", "") for part in result.get("content", []) if part.get("type") == "text"
        )
        if result.get("isError"):
            raise McpError(text)
        return text
