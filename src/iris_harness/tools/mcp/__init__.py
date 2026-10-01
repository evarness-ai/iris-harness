"""IRIS as an MCP *server* -- the protocol half of ``iris mcp serve``.

IRIS consumes external MCPs via ``iris_harness.tools.mcp_bridge``; this package is the
other direction: a zero-dependency stdlib JSON-RPC stdio server. It speaks the protocol
only. How a served tool runs -- through the governed tool runner, as ``mcp:<client>`` --
is ``iris_harness.runtime.mcp_serve``'s.
"""

from .server import (
    IrisMCPServer,
    McpReply,
    McpTool,
    SkillTool,
    load_skill_tools,
    serve_stdio,
)

__all__ = ["IrisMCPServer", "McpReply", "McpTool", "SkillTool", "load_skill_tools", "serve_stdio"]
