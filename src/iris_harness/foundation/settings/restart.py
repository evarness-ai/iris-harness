"""Restart the harness from the app, so a setting that applies "after restart" can (ADR-0120).

The app cannot restart containers — no Docker socket is mounted, on purpose — so a
restart is a request every server process honours by exiting, and the supervisor
brings each back (on the VM: Docker Compose's ``restart: unless-stopped``). The request
is a timestamp in the settings store, which every process already reads; each server
watches it and exits once it is newer than the process's own start.

Only where something brings the processes back: ``IRIS_SUPERVISED=1`` says so (the VM's
compose sets it). A harness started by hand has nothing to restart it, so the API
refuses there rather than stopping the owner's harness for good.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

from iris_harness.foundation.env import env_flag
from iris_harness.foundation.settings.store import SettingsStore

logger = logging.getLogger(__name__)

SYSTEM_SECTION = "system"
RESTART_KEY = "restart_requested_at"
SUPERVISED_ENV = "IRIS_SUPERVISED"
# A supervised process that must NOT exit on the app's restart request. The VM's
# governor owns the network namespace api, channel-gateway and proxy share: when it
# exited too, the proxy stayed on the dead namespace (every proxied model call failed)
# and channel-gateway, restarting before it, could not join and stayed down (2026-09-28).
RESTART_ON_REQUEST_ENV = "IRIS_RESTART_ON_REQUEST"
PROCESS_STARTED_AT = datetime.now(UTC)
_POLL_SECONDS = 3.0


def supervised() -> bool:
    """Whether a supervisor restarts this process when it exits."""
    return env_flag(SUPERVISED_ENV, default=False)


def exits_on_restart_request() -> bool:
    """Whether this process honours the app's restart request (supervised, not opted out)."""
    return supervised() and env_flag(RESTART_ON_REQUEST_ENV, default=True)


def request_restart(store: SettingsStore, *, actor: str) -> datetime:
    """Ask every server process to restart; returns the request's time."""
    now = datetime.now(UTC)
    previous = store.get(SYSTEM_SECTION, RESTART_KEY)
    store.set(SYSTEM_SECTION, RESTART_KEY, now.isoformat(), old=previous, actor=actor)
    logger.warning("restart requested by %s", actor)
    return now


def restart_requested_after(store: SettingsStore, started_at: datetime) -> bool:
    raw = store.get(SYSTEM_SECTION, RESTART_KEY)
    if not isinstance(raw, str):
        return False
    try:
        return datetime.fromisoformat(raw) > started_at
    except ValueError:
        return False


def watch_for_restart(
    store: SettingsStore | None = None,
    *,
    started_at: datetime = PROCESS_STARTED_AT,
    exit_process: Callable[[int], object] = os._exit,
    poll_seconds: float = _POLL_SECONDS,
    name: str = "server",
) -> threading.Thread:
    """Start a daemon thread that exits this process when a restart is requested."""
    store = store or SettingsStore()

    def _watch() -> None:
        while True:
            try:
                if restart_requested_after(store, started_at):
                    logger.warning("%s: restart requested; exiting for the supervisor", name)
                    exit_process(0)
                    return
            except Exception:  # a store hiccup must not kill the watcher
                logger.debug("restart watch: store read failed", exc_info=True)
            time.sleep(poll_seconds)

    thread = threading.Thread(target=_watch, name="iris-restart-watch", daemon=True)
    thread.start()
    return thread
