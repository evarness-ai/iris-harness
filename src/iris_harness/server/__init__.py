"""Runtime service entry points for IRIS.

Every server under this package reads settings, so the owner's saved setting changes
(ADR-0120) are applied here, when the package is first imported — before any server
module, or anything it imports, reads an ``IRIS_*`` value. Under a supervisor, each
server also watches for the app's restart request and exits for the supervisor to
bring it back with the new settings — unless ``IRIS_RESTART_ON_REQUEST=0`` (the VM's
governor, which owns the network namespace the other services share).
"""

from iris_harness.foundation.settings.env_overrides import apply_env_overrides
from iris_harness.foundation.settings.restart import exits_on_restart_request, watch_for_restart

apply_env_overrides()
if exits_on_restart_request():
    watch_for_restart()
