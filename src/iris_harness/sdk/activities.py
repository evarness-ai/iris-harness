"""The background Activity spine, for a plugin that has no running ``IrisRuntime`` to
hand a long job to the harness's shared one (a bare CLI process, say) and so needs to
run its own small one instead -- same ``activities.db``, same shape.

A plugin *inside* a turn or a mounted host reaches the harness's own shared runner
through the typed ``HarnessServices.submit_activity`` field instead (never this
module): that one runner is also wired for in-chat and channel completion notices,
which a plugin-built one is not.

``ActivityStore`` is the durable record (``get``, ``list``, the ``mark_*`` transitions)
a plugin may also use read-only, from any process, to poll a job someone else
submitted. ``ActivityRunner`` submits work to a small in-process thread pool and
drives it through the same transitions, handing ``work`` a ``progress(frac, message)``
callback. ``ActivityOutcome`` is what ``work`` returns on success.
"""

from __future__ import annotations

from iris_harness.services.activities import ActivityOutcome, ActivityRunner, ActivityStore

__all__ = [
    "ActivityOutcome",
    "ActivityRunner",
    "ActivityStore",
]
