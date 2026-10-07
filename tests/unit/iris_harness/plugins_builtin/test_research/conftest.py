"""Shared fixtures for the research engine tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest

import iris_harness.plugins_builtin.research.extract as extract_mod
from iris_harness.foundation.process_state import restore_process_state, snapshot_process_state
from iris_harness.plugins_builtin.research.providers import BUILTIN_PROVIDERS
from iris_harness.services.research.providers import register_search_provider

# A public address (example.com's). Tests that fetch a page resolve every host to it,
# so the extractor's public-address check passes with no DNS.
PUBLIC_ADDRESS = "93.184.215.14"


class FakeClient:
    """Stands in for the governed client a provider asks for (``provider._http``).

    Answers every request with ``reply`` (a status and a body), or raises ``error``;
    ``calls`` holds ``(method, url, kwargs)`` for each. The governed client itself, with
    its declaration, its ledger and its address checks, is exercised in
    ``test_governed_egress.py``."""

    def __init__(self, *, status: int = 200, body: Any = None, error: Exception | None = None):
        self.status = status
        self.body = {} if body is None else body
        self.error = error
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def _answer(self, method: str, url: str, kwargs: dict[str, Any]) -> httpx.Response:
        self.calls.append((method, url, kwargs))
        if self.error is not None:
            raise self.error
        return httpx.Response(self.status, json=self.body)

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self._answer("GET", url, kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self._answer("POST", url, kwargs)


@pytest.fixture(autouse=True)
def _no_real_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every host to a public address: hermetic, and the SSRF check passes.

    A test about the check itself patches ``_resolve_host`` again with its own answer."""
    monkeypatch.setattr(extract_mod, "_resolve_host", lambda host: [PUBLIC_ADDRESS])


@pytest.fixture(autouse=True)
def _builtin_search_chain() -> Iterator[None]:
    """The chain the plugin's ``setup`` registers, and the process-wide registry put back
    after each test (a test that registers a provider leaves nothing behind)."""
    snapshot = snapshot_process_state()
    for provider_class in BUILTIN_PROVIDERS:
        register_search_provider(provider_class.name, provider_class(), owner="plugin:research")
    yield
    restore_process_state(snapshot)
