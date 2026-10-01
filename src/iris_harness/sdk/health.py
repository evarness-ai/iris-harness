"""Contributing to `iris system status`.

A plugin that owns something checkable registers a provider here and the harness
folds its rows into the one snapshot the CLI, the API and the web UI all render.
`build_health_tick_handler` is the heartbeat side of the same surface. A plugin that
can fix what it checks (refresh its own token) registers a repairer beside its check
provider; the health watch runs it before asking the owner (ADR-0116).

A check is a `HealthCheck` of a `CheckKind` in a `HealthState` (green / yellow / red),
optionally carrying a `Reconnect` the UI offers when the owner must re-authorise.
`net_probe_enabled()` says whether the owner opted in to checks that touch the network.

`register_account_counter` and `register_file_root_counter` feed the counts the status
report shows (connected accounts per provider, file roots under management).
"""

from __future__ import annotations

from iris_harness.services.health import render_text
from iris_harness.services.health.heartbeat import build_health_tick_handler
from iris_harness.services.health.models import CheckKind, HealthCheck, HealthState, Reconnect
from iris_harness.services.health.repair import (
    RepairOutcome,
    credential_refresh_repairer,
    register_repairer,
)
from iris_harness.services.health.service import (
    current_snapshot,
    net_probe_enabled,
    register_check_provider,
)
from iris_harness.services.health.watch import build_watcher, current_watcher, install_watcher
from iris_harness.services.system.status import (
    register_account_counter,
    register_file_root_counter,
)

__all__ = [
    "CheckKind",
    "HealthCheck",
    "HealthState",
    "Reconnect",
    "RepairOutcome",
    "build_health_tick_handler",
    "build_watcher",
    "credential_refresh_repairer",
    "current_snapshot",
    "current_watcher",
    "install_watcher",
    "net_probe_enabled",
    "register_account_counter",
    "register_check_provider",
    "register_file_root_counter",
    "register_repairer",
    "render_text",
]
