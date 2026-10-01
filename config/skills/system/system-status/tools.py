"""LangChain tool for the system-status skill (System agent S0).

Thin wrapper over ``iris_harness.services.system`` — a deterministic snapshot of host resources
and IRIS's own state (connections, skills, heartbeats, DBs). Read-only; domain
logic + tests live in src/iris/system.
"""

from __future__ import annotations

from iris_harness.services.system.status import host_status, iris_status
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class _NoArgs(BaseModel):
    pass


class SystemStatusTool(BaseTool):
    name: str = "system_status"
    description: str = (
        "Report IRIS's system status: host resources (RAM/CPU/thermal) and "
        "what IRIS has connected and running (accounts, skills, heartbeats, "
        "databases, FileManager roots)."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> dict[str, object]:
        host = host_status()
        iris = iris_status()
        return {
            "ram_free_gb": round(host.ram_free_gb, 1),
            "ram_total_gb": round(host.ram_total_gb, 1),
            "cpu_percent": round(host.cpu_percent, 0),
            "thermal_throttled": host.thermal_throttled,
            "accounts": iris.accounts,
            "skills": iris.skill_count,
            "heartbeats": iris.heartbeat_count,
            "filemanager_roots": iris.filemanager_roots,
            "databases": dict(iris.database_sizes),
        }


SKILL_TOOLS = [SystemStatusTool]
