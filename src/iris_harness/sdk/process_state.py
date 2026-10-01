"""Declaring a plugin's process-wide state, so a test harness can put it back.

A plugin (or a domain library) that keeps a module-level registry or cache -- anything
its ``setup()`` or its first use fills and a second runtime in the same process must not
inherit -- declares it once, at the bottom of the module that owns it::

    from iris_harness.sdk.process_state import track_globals

    _providers: dict[str, Provider] = {}
    ...
    track_globals(__name__, "_providers")

``iris_harness.testing.harness`` snapshots every declared piece of state before it
builds a runtime and restores it on exit. :func:`register_process_state` takes an
explicit ``save`` / ``restore`` pair for state that is not a plain global.
"""

from __future__ import annotations

from iris_harness.foundation.process_state import register_process_state, track_globals

__all__ = ["register_process_state", "track_globals"]
