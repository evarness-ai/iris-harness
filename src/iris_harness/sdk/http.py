"""The governed HTTP client: a plugin's outbound requests, declared, allowed and recorded.

Stable tier. A plugin that talks to a web service uses this instead of ``httpx`` or
``requests`` directly (docs/architecture/plugin-egress.md, issue #103). Every request:

1. is checked against the hosts the plugin's manifest declares under ``egress:`` (scheme,
   host, port), and against the data class the run holds -- an undeclared host raises
   :class:`EgressDenied` and nothing is sent;
2. is recorded in the governance ledger -- a ``pre_egress`` row (allowed or denied) and a
   ``post_egress`` row (status, bytes each way, duration, or the error's class) -- with the
   plugin, the tool that was running, the caller, the run and the session. Never the path,
   query, headers, body or any secret;
3. has the strings it carries read by the owner-PII guards' egress column.

Get one from ``api.http`` in ``setup(api)`` (bound to your plugin by the harness: you cannot
claim another's name), or from :func:`current_http` inside a declarative plugin's function.

    http = api.http
    reply = http.get("https://api.open-meteo.com/v1/forecast", params={"latitude": 52.5})

Redirects are not followed: a 3xx comes back as the response, and following it is a new
request, governed like the first. In a test, ``iris_harness.testing.fake_http`` replaces the
transport only, so the declaration, the hooks and the rows run as they do in production.

What this does not do: it cannot stop a plugin from opening its own socket. It governs the
calls made through it (docs/architecture/plugin-egress.md).
"""

from __future__ import annotations

from iris_harness.runtime.governed_http import EgressDenied, GovernedHttp, current_http

__all__ = ["EgressDenied", "GovernedHttp", "current_http"]
