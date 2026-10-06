"""Testing a plugin, or IRIS itself, without a model server or a network.

Stable tier (OSS plan R16): what a test imports to run the harness deterministically.

* :func:`use_fake_model` puts every tier on the scripted fake model for the duration of
  a ``with`` block. The fake replaces only the transport: governance hooks, audit rows,
  the egress log and JSON parsing run exactly as for a real model. A script is data --
  an ordered list of ``match`` -> ``reply`` rules (see :mod:`iris_harness.llm.fake` for
  the format) -- given as a :class:`Script`, a mapping, or a YAML file.
* :func:`transcript` lists every call the fake answered, so a test can assert on what
  the model was asked and what the audit ledger holds for it.
* :func:`fake_http` answers the requests a plugin makes through the governed HTTP client
  (``iris_harness.sdk.http``) from a script, replacing only the transport: the manifest's
  ``egress`` declaration, the hooks and the audit rows run as in production, so a test
  proves a declared host is allowed and an undeclared one is denied without a socket.
* :func:`no_network` refuses every outbound socket connection for a ``with`` block, so
  a test proves a path is offline instead of hoping it is.
* :func:`harness` builds a real, governed IRIS in a throwaway home -- the composition
  root, not a look-alike -- with those three in place, the owner's settings, keyring and
  network out of reach, and plugins supplied in-process (:func:`plugin`). Its
  ``chat`` / ``chat_stream`` run the full turn pipeline and return a
  :class:`TurnResult` (the answer, the agent, the turn's audit-row ids and, streamed,
  its :class:`TurnEvent` steps) -- the harness's own types, so the runtime's internal
  result shapes are not part of the stable tier; ``audit_rows`` reads the ledger as
  :class:`TurnAuditRow` (the ledger's own row type is internal too) and ``audit_gaps``
  checks that every model call and every answer was audited (R14). Process-wide state the run
  filled is put back on exit (:mod:`iris_harness.foundation.process_state`). See
  :mod:`iris_harness.testing.harness`.
* :func:`check_network_imports` reports every import of a raw network library (``httpx``,
  ``requests``, ``socket``, ...) in a plugin's source: its outbound calls belong on
  ``api.http``, where each is declared and recorded.
* :func:`check_stable_imports` reports every import of IRIS code outside the stable
  tier (``iris_harness/sdk/stable_tier.yaml``); the examples and the scaffold are held
  to it, and a plugin's CI can hold itself to it the same way.

Plugin *code* never imports this package (the SDK-only import contracts say so): a
plugin that ships a demo selects the fake through ``iris_harness.sdk.llm``'s
``FORCED_PROVIDER_ENV`` and ``FAKE_MODEL_SCRIPT_ENV``.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

from iris_harness.llm.fake import (
    FAKE_PROVIDER,
    FakeCall,
    FakeModelError,
    Reply,
    Rule,
    Script,
    ScriptedChatModel,
    install_script,
    reset_transcript,
    transcript,
)
from iris_harness.llm.tier_router import FORCED_PROVIDER_ENV
from iris_harness.runtime.governed_http import _use_transport
from iris_harness.testing.conformance import (
    ConformanceError,
    Violation,
    assert_conformant,
    check_conformance,
)
from iris_harness.testing.harness import (
    Harness,
    PluginState,
    TurnAuditRow,
    TurnEvent,
    TurnRecord,
    TurnResult,
    harness,
    plugin,
)
from iris_harness.testing.network_imports import (
    NETWORK_MODULES,
    NetworkImportViolation,
    check_network_imports,
)
from iris_harness.testing.stable import (
    StableImportViolation,
    StableTier,
    check_stable_imports,
    stable_tier,
)


class NetworkBlockedError(OSError):
    """An outbound connection was attempted inside :func:`no_network`."""


def _as_script(script: Script | Mapping[str, Any] | Path | str) -> Script:
    if isinstance(script, Script):
        return script
    if isinstance(script, Mapping):
        return Script.from_mapping(script)
    return Script.load(script)


@contextmanager
def use_fake_model(script: Script | Mapping[str, Any] | Path | str) -> Iterator[Script]:
    """Every tier a ``TierRouter`` loads inside the block runs on the scripted fake.

    Installs ``script`` for this process, sets ``IRIS_LLM_PROVIDER=fake`` and clears the
    transcript; restores all three on exit. A router loaded *before* the block keeps its
    providers: load the tiers (or build the runtime) inside it.
    """
    loaded = _as_script(script)
    previous = os.environ.get(FORCED_PROVIDER_ENV)
    os.environ[FORCED_PROVIDER_ENV] = FAKE_PROVIDER
    install_script(loaded)
    reset_transcript()
    try:
        yield loaded
    finally:
        install_script(None)
        if previous is None:
            os.environ.pop(FORCED_PROVIDER_ENV, None)
        else:
            os.environ[FORCED_PROVIDER_ENV] = previous


HttpRoute = Mapping[str, Any]


@contextmanager
def fake_http(
    routes: Callable[[httpx.Request], httpx.Response] | Mapping[str, HttpRoute | httpx.Response],
) -> Iterator[list[httpx.Request]]:
    """Answer governed HTTP requests from ``routes`` inside the block; yield what was sent.

    ``routes`` is either a function ``request -> httpx.Response`` or a mapping from a URL
    without its query (``"https://api.open-meteo.com/v1/forecast"``, optionally prefixed with
    a method, ``"POST https://..."``) to a reply: an :class:`httpx.Response`, or a mapping of
    ``status`` (default 200) and one of ``json`` / ``text`` / ``content`` / ``headers``. A
    request no route matches gets a 404, so a plugin that reaches an unexpected URL fails
    visibly. The returned list holds every request that was actually sent (a request the
    egress policy denied never reaches the transport, so it is not in it).

    Only the transport is replaced: ``api.http`` still checks the manifest's declaration and
    writes the ledger rows. Real sockets stay refused by :func:`no_network`.
    """

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if callable(routes):
            return routes(request)
        bare = str(request.url.copy_with(query=None, fragment=None))
        reply = routes.get(f"{request.method} {bare}", routes.get(bare))
        if reply is None:
            return httpx.Response(404, json={"error": f"no fake route for {request.method} {bare}"})
        if isinstance(reply, httpx.Response):
            return reply
        spec = dict(reply)
        status = int(spec.pop("status", 200))
        return httpx.Response(status, **spec)

    sent: list[httpx.Request] = []
    with _use_transport(httpx.MockTransport(answer)):
        yield sent


@contextmanager
def no_network() -> Iterator[list[Any]]:
    """Refuse every outbound socket connection inside the block.

    Yields the list of addresses that were attempted (each raised
    :class:`NetworkBlockedError`), so a test can assert it stayed empty. Unix-domain
    sockets are left alone: they never leave the machine.
    """
    attempted: list[Any] = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection

    def _refuse(address: Any) -> None:
        attempted.append(address)
        raise NetworkBlockedError(f"network access blocked in this test: {address!r}")

    def connect(self: socket.socket, address: Any) -> None:
        if self.family == getattr(socket, "AF_UNIX", None):
            real_connect(self, address)
            return
        _refuse(address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        if self.family == getattr(socket, "AF_UNIX", None):
            return real_connect_ex(self, address)
        _refuse(address)
        return 1  # pragma: no cover - _refuse raised

    def create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        _refuse(address)
        raise AssertionError("unreachable")  # pragma: no cover

    socket.socket.connect = connect  # type: ignore[method-assign,assignment]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign,assignment]
    socket.create_connection = create_connection
    try:
        yield attempted
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = real_connect_ex  # type: ignore[method-assign]
        socket.create_connection = real_create_connection


__all__ = [
    "ConformanceError",
    "FakeCall",
    "FakeModelError",
    "Harness",
    "NETWORK_MODULES",
    "NetworkBlockedError",
    "NetworkImportViolation",
    "PluginState",
    "Reply",
    "Rule",
    "Script",
    "ScriptedChatModel",
    "StableImportViolation",
    "StableTier",
    "TurnAuditRow",
    "TurnEvent",
    "TurnRecord",
    "TurnResult",
    "Violation",
    "assert_conformant",
    "check_conformance",
    "check_network_imports",
    "check_stable_imports",
    "harness",
    "no_network",
    "plugin",
    "stable_tier",
    "transcript",
    "fake_http",
    "use_fake_model",
]
