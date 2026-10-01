"""Who may call which tool, supplied from above — the permission contract's seam.

A plugin cannot call another plugin's tool unless the governance contract allows it
(docs/architecture/plugin-capabilities.md §4). The allow-list is config — every mounted
manifest's ``uses: tools`` plus the operator's ``config/governance/tool-access.yaml`` — and
the plugin host, which sits above the kernel, is what knows the manifests. So the kernel
asks: the runtime registers a policy once plugins have mounted, and ``CallerPolicyHook``
reads it on every ``PRE_TOOL_USE``.

Fail closed, and say so: with no policy registered, a ``plugin:`` call is denied with a
reason naming the missing policy — never allowed because nobody was asked.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from iris_harness.foundation.process_state import track_globals

#: ``policy(caller, tool) -> None`` to allow, or the reason it is denied.
CallerPolicy = Callable[[str, str], str | None]

#: The caller prefix of a client outside IRIS served over MCP (``iris mcp serve``):
#: ``mcp:<client>``, where ``<client>`` is the name the operator gave the server, never
#: one the client sent. Such a caller cannot answer an approval or a confirmation.
MCP_CALLER_PREFIX = "mcp:"


def is_mcp_caller(caller: str | None) -> bool:
    """Whether ``caller`` is a client outside IRIS, served over MCP."""
    return bool(caller) and str(caller).startswith(MCP_CALLER_PREFIX)


_lock = threading.Lock()
_policy: CallerPolicy | None = None


def register_caller_policy(policy: CallerPolicy | None) -> None:
    """Install (or, with ``None``, remove) the policy the caller hook enforces."""
    global _policy
    with _lock:
        _policy = policy


def caller_policy() -> CallerPolicy | None:
    """The registered policy, or ``None`` when no runtime has registered one."""
    with _lock:
        return _policy


__all__ = [
    "MCP_CALLER_PREFIX",
    "CallerPolicy",
    "caller_policy",
    "is_mcp_caller",
    "register_caller_policy",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_policy")
