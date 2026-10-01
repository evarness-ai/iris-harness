"""An external MCP server, for the consuming half: ``python weather_server.py``.

Stands in for any third-party MCP server IRIS might consume. It speaks MCP's stdio
transport (newline-delimited JSON-RPC 2.0) with the standard library only, and serves
one tool, ``forecast``, from a fixed table -- no network, so it runs anywhere.
"""

from __future__ import annotations

import json
import sys
from typing import Any

FORECASTS = {"lisbon": "Sunny, 24 C", "oslo": "Light rain, 11 C"}
TOOLS = [
    {
        "name": "forecast",
        "description": "Today's forecast for a city.",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


def handle(request: dict[str, Any]) -> dict[str, Any] | None:
    method, req_id = request.get("method"), request.get("id")
    if req_id is None:
        return None  # a notification
    if method == "initialize":
        result: dict[str, Any] = {
            "protocolVersion": (request.get("params") or {}).get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "weather", "version": "0.1.0"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = request.get("params") or {}
        city = str((params.get("arguments") or {}).get("city", "")).lower()
        known = city in FORECASTS
        text = FORECASTS[city] if known else f"no forecast for {city!r}"
        result = {"content": [{"type": "text", "text": text}], "isError": not known}
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        reply = handle(json.loads(line))
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
