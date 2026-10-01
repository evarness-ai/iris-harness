"""The owner's clock.

`iris_timezone()` is the zone every "today", "tomorrow at 9" and quiet-hours check
runs in: ``IRIS_TZ``, or UTC when it is unset or not a zone name. A plugin that
reads the system zone instead would disagree with the digest and the reminders
the moment the host runs in UTC (the VM does).

`local_now()` / `local_today()` are the owner's wall clock and date for "now" and
"today": ``IRIS_TZ`` when set, else the machine's own zone (a run without ``IRIS_TZ``
keeps meaning the machine's clock). Use them instead of ``datetime.now().astimezone()``
or ``date.today()``, which read the machine's zone even when ``IRIS_TZ`` says otherwise.

`previous_local_day(now, tz)` is the ``[start, end)`` of the owner's yesterday, the
window the digest's "learned yesterday" line covers.
"""

from __future__ import annotations

from iris_harness.foundation.clock import iris_timezone, local_now, local_today
from iris_harness.services.digest.learned import previous_local_day

__all__ = ["iris_timezone", "local_now", "local_today", "previous_local_day"]
