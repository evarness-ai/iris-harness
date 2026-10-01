"""Who may perform a gated control write (the web console's write gate).

A paired device may when it holds ``control``; any other caller only when the operator
opted in with ``IRIS_WEBUI_ALLOW_WRITES``. The write-guard middleware in ``main`` asks
this (beside its service-principal rule), and ``/capabilities`` reports it so the
console knows whether to show write controls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris_harness.foundation.env import env_flag

if TYPE_CHECKING:
    from iris_harness.foundation.auth import Principal


def _webui_writes_enabled() -> bool:
    return env_flag("IRIS_WEBUI_ALLOW_WRITES", default=False)


def _may_write(principal: Principal | None) -> bool:
    """May this caller perform a gated control write?"""
    if principal is not None and principal.kind == "device":
        return principal.can_control
    return _webui_writes_enabled()
