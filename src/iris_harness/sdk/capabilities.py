"""Capabilities: typed service interfaces plugins share in code, without importing each other.

A capability is a ``domain.verb`` name plus a ``Protocol`` published here, so provider and
consumer agree on one stable shape, versioned with the SDK. Declare it in your manifest:

.. code-block:: yaml

    capabilities:
      provides: [mail.read]   # you implement it
      uses: [mail.read]       # you work without it, degraded
      requires: []            # you are not loaded at all without these

A provider registers its implementation in ``setup``: ``api.provide("mail.read", impl)``.
A consumer asks for it: ``api.capability("mail.read")`` returns the implementation, or
``None`` when nothing provides it (take your degraded path). When several plugins provide
one capability you still get ONE implementation, which fans out to all of them. You reach
it only through the Protocol's methods; a call into it is attributed to the provider (a
failure there is the provider's, in System Health), and a call after the provider stopped
being mounted raises ``CapabilityUnavailable``.

Every call is governed like a tool call, named ``capability:<name>.<method>``: the kernel
checks your manifest allows it, applies the method's declared effect (a ``confirm: once``
write is refused with ``require_approval`` until the approval executor is extended to
capability calls), audits it, and masks the owner's identity literals out of
the result's declared text fields before you see it -- you get a masked *copy*. A call
governance stops raises ``CapabilityDenied`` (a ``CapabilityUnavailable``).

Undeclared use is refused both ways: ``provide`` of a capability your manifest does not
list under ``capabilities: provides``, or ``capability`` of one not under ``uses`` /
``requires``, is recorded as your plugin's failure and has no effect. Providers mount
before consumers. Design: docs/architecture/plugin-capabilities.md §2.

``CAPABILITIES`` is every capability this SDK version publishes (``weather.forecast`` so
far; more with rollout step 4). The catalogue is closed: a new capability ships in an SDK
release. Re-exports: the catalogue is defined in ``iris_harness.foundation.capabilities``
because the host and the core's own consumers sit below the SDK and must import it too.
"""

from __future__ import annotations

from iris_harness.foundation.capabilities import (
    CAPABILITIES,
    CapabilityDenied,
    CapabilitySpec,
    CapabilityUnavailable,
    Forecast,
    ForecastPeriod,
    MethodSpec,
    WeatherForecast,
)

__all__ = [
    "CAPABILITIES",
    "CapabilityDenied",
    "CapabilitySpec",
    "CapabilityUnavailable",
    "Forecast",
    "ForecastPeriod",
    "MethodSpec",
    "WeatherForecast",
]
