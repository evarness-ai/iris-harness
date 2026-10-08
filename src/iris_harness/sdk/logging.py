"""Logging a plugin may use, including the two calls that are not optional.

``log_egress`` and ``log_ingress`` are how a plugin that talks to the outside
world records that it did. They are re-exported here because a plugin that
reaches for them should not have to reach into ``foundation/`` to find them --
and because an egress a plugin makes without logging it is an egress the
governance ledger cannot see.

``agent_scope`` and ``log_tool_run`` are the session-scoped pair: they put a
plugin's tool runs in the same per-session trail the harness's own writes, so a
session log does not simply go quiet where a plugin was doing the work.

``log_safe`` makes a value that came off the wire (a name, a key, an id) safe to put in a log
line: control characters, CR and LF are escaped, so one request cannot forge a second entry.
"""

from __future__ import annotations

from iris_harness.foundation.logsafe import log_safe
from iris_harness.foundation.observability.logging_setup import (
    egress_logger,
    ingress_logger,
    log_egress,
    log_ingress,
)
from iris_harness.foundation.observability.session_log import agent_scope, log_tool_run

__all__ = [
    "agent_scope",
    "egress_logger",
    "ingress_logger",
    "log_egress",
    "log_ingress",
    "log_safe",
    "log_tool_run",
]
