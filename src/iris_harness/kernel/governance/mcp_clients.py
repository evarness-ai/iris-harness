"""Which MCP clients the owner declared local -- the seam ``McpClientEgressHook`` reads.

``iris mcp serve`` serves IRIS's tools to a client outside IRIS, as the caller
``mcp:<client>``. Whether a result carrying the owner's personal data may reach that
client depends on where the client runs, and only the owner can say: the declaration is
the owner's MCP serve config (``config/mcp-serve.yaml``, overlaid by
``$IRIS_HOME/mcp-serve.yaml``; ``runtime/mcp_serve.py`` reads it), never the client.

The serve process registers the clients declared local here once; the hook asks. With
nothing registered every client is non-local: fail closed, never "local because nobody
said otherwise".
"""

from __future__ import annotations

import threading
from collections.abc import Iterable

from iris_harness.foundation.process_state import track_globals
from iris_harness.kernel.governance.caller_policy import is_mcp_caller

_lock = threading.Lock()
_local: frozenset[str] = frozenset()


def register_local_mcp_clients(callers: Iterable[str]) -> None:
    """Declare the ``mcp:<client>`` callers that run on the owner's machine (replaces any
    earlier declaration; an empty iterable declares none)."""
    global _local
    with _lock:
        _local = frozenset(callers)


def mcp_client_is_local(caller: str | None) -> bool:
    """Whether ``caller`` is an MCP client the owner declared local."""
    if not is_mcp_caller(caller):
        return False
    with _lock:
        return str(caller) in _local


__all__ = ["mcp_client_is_local", "register_local_mcp_clients"]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_local")
