"""Tools — everything the agent can *do*, as opposed to what it knows or decides.

One layer, four parts (OSS plan M6, decision 6): the MCP bridge and tool adapters here
at the root (their import paths are unchanged), plus ``skills/`` (the code-first skill
system and its semantic router), ``sandbox/`` (the isolated execution environment a tool
may need) and ``mcp/`` (the MCP server IRIS exposes).

``llm/`` is this layer's sibling: neither imports the other. A tool that needs a model
gets one from its caller, and a model call that needs a tool is handed it -- the two meet
in the layer above, not in each other.
"""

from .mcp_bridge import (
    MCPApprovalRequired,
    MCPBridge,
    MCPBridgeConfig,
    MCPInvocationResult,
    MCPServerConfig,
    MCPToolDefinition,
    load_mcp_bridge_config,
)
from .propose_skill_from_sandbox import propose_skill_from_sandbox

__all__ = [
    "MCPApprovalRequired",
    "MCPBridge",
    "MCPBridgeConfig",
    "MCPInvocationResult",
    "MCPServerConfig",
    "MCPToolDefinition",
    "load_mcp_bridge_config",
    "propose_skill_from_sandbox",
]
