"""System agent (7th PAIA agent) — host status + IRIS self-introspection.

The conversational "system" surface (chat, time/date, weather, identity)
already lives in the system ReAct handler; this package adds the net-new
user-facing **status** surface: host resources (reusing the tier governor's
pressure sampler) and IRIS's own state (connected accounts, skills, enabled
heartbeats, DB sizes, FileManager roots). Deterministic; no OS mutation.
"""

from .status import (
    HostStatus,
    IrisStatus,
    SystemReport,
    host_status,
    iris_status,
    system_report,
)

__all__ = [
    "HostStatus",
    "IrisStatus",
    "SystemReport",
    "host_status",
    "iris_status",
    "system_report",
]
