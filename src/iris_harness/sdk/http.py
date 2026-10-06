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

Get one from ``api.http`` in ``setup(api)`` (bound to your plugin's name by the harness), or
from :func:`current_http` inside a declarative plugin's function. That binding is a
convention: ``GovernedHttp("other")`` constructs, and an in-process plugin that does so is
recorded under the name it passed. The boundary that stops a plugin acting as another is the
out-of-process one (issue #111-#114), not this class.

    http = api.http
    reply = http.get("https://api.open-meteo.com/v1/forecast", params={"latitude": 52.5})

Redirects are not followed: a 3xx comes back as the response, and following it is a new
request, governed like the first. In a test, ``iris_harness.testing.fake_http`` replaces the
transport only, so the declaration, the hooks and the rows run as they do in production.

Bounds, per request: the whole transfer has one time budget (``timeout`` seconds in total, not
per operation; default 10, at most 60; ``None``, zero or a negative value mean the default) and
the decoded response body is read up to 10 MiB; past either, the request is cut off, recorded
and raises :class:`EgressDenied`. The environment is not consulted (no proxy variables, netrc
or CA-bundle variables), and a ``Host`` or ``Proxy-*`` header is refused. The connection
resolves the host name once, refuses an address that is loopback, private, link-local, shared
or otherwise internal, and connects to the address it checked. Names such as ``localhost``,
``*.local`` and ``*.internal`` are never contacted, even under ``open_web``.

What this does not do: it cannot stop a plugin from opening its own socket. It governs the
calls made through it (docs/architecture/plugin-egress.md). A request whose pre-egress ledger
row cannot be written is not sent; but a plugin that never goes through the client leaves no
row at all. The name lookup itself has no timeout.
"""

from __future__ import annotations

from iris_harness.runtime.governed_http import EgressDenied, GovernedHttp, current_http

__all__ = ["EgressDenied", "GovernedHttp", "current_http"]
