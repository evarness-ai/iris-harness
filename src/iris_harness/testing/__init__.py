"""Testing a plugin, or IRIS itself, without a model server or a network.

Stable tier (OSS plan R16): what a test imports to run the harness deterministically.

* :func:`use_fake_model` puts every tier on the scripted fake model for the duration of
  a ``with`` block. The fake replaces only the transport: governance hooks, audit rows,
  the egress log and JSON parsing run exactly as for a real model. A script is data --
  an ordered list of ``match`` -> ``reply`` rules (see :mod:`iris_harness.llm.fake` for
  the format) -- given as a :class:`Script`, a mapping, or a YAML file.
* :func:`transcript` lists every call the fake answered, so a test can assert on what
  the model was asked and what the audit ledger holds for it.
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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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
from iris_harness.testing.harness import (
    Harness,
    TurnAuditRow,
    TurnEvent,
    TurnRecord,
    TurnResult,
    harness,
    plugin,
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
    "FakeCall",
    "FakeModelError",
    "Harness",
    "NetworkBlockedError",
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
    "check_stable_imports",
    "harness",
    "no_network",
    "plugin",
    "stable_tier",
    "transcript",
    "use_fake_model",
]
