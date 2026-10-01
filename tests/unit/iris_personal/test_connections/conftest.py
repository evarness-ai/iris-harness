"""Shared setup: a public URL, throwaway data + audit dirs, Gmail's real reconnect
registration, and the fake Google behind the module's HTTP client.

The connections library is part of the email slice (OSS plan R2), so this conftest ships
with it and registers only the email slice's provider. Tests that need the calendar and
Drive providers too (the private domains') extend ``fake`` in their own module.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from iris_harness.runtime.api_routes import clear_api_routers
from iris_harness.runtime.plugin_host.api import HarnessServices, PluginAPI
from iris_harness.runtime.plugin_host.registry import PluginRegistry
from iris_personal.connections import google

from .fake_google import FakeGoogle

PUBLIC_URL = "https://iris-vm.example.ts.net"


def plugin_api(tmp_path: Path, plugin: str) -> PluginAPI:
    """A bare ``PluginAPI`` for one plugin's reconnect registration."""
    services = HarnessServices(
        config_dir=tmp_path / "config",
        data_dir=tmp_path / "data",
        tier_router=None,
        agent_executor=None,
        heartbeats=None,
        channels=None,
        deterministic_reply=lambda **kw: None,
    )
    return PluginAPI(plugin=plugin, services=services, registry=PluginRegistry())


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeGoogle]:
    from iris_personal.plugins.gmail.plugin import _register_web_reconnect as gmail

    monkeypatch.setenv("IRIS_PUBLIC_URL", PUBLIC_URL)
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "audit.db"))
    monkeypatch.delenv(google.TEST_BASE_ENV, raising=False)
    clear_api_routers()
    google.clear_providers()
    google.pending_starts.clear()
    gmail(plugin_api(tmp_path, "gmail"))
    fake_google = FakeGoogle()
    monkeypatch.setattr(google, "http_client", fake_google.client)
    yield fake_google
    clear_api_routers()
    google.clear_providers()
    google.pending_starts.clear()
