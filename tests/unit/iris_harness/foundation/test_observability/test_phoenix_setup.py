"""Tests for Phoenix bootstrap helpers."""

from __future__ import annotations

import logging
import os
import sys
import types
import warnings

from iris_harness.foundation.observability.instruments import _reset_instrumentation_state
from iris_harness.foundation.observability.phoenix_setup import (
    PhoenixSetupConfig,
    _silence_phoenix_noise,
    initialize_phoenix,
)


def test_silence_phoenix_noise_filters_known_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _silence_phoenix_noise()
        warnings.warn(
            "authlib.jose module is deprecated, please use joserfc", UserWarning, stacklevel=2
        )
        warnings.warn(
            "Skipped unsupported reflection of expression-based index ix_latency",
            UserWarning,
            stacklevel=2,
        )
    assert caught == []  # both benign warnings suppressed
    # The aioboto3 notice is a logger.warning — its logger floor is raised.
    assert logging.getLogger("phoenix.server.app").level >= logging.ERROR


def _install_module(monkeypatch, name: str, module: types.ModuleType) -> None:
    parts = name.split(".")
    for idx in range(1, len(parts)):
        parent_name = ".".join(parts[:idx])
        if parent_name not in sys.modules:
            monkeypatch.setitem(sys.modules, parent_name, types.ModuleType(parent_name))
    monkeypatch.setitem(sys.modules, name, module)


def test_initialize_phoenix_returns_disabled_result() -> None:
    result = initialize_phoenix(PhoenixSetupConfig(enabled=False))

    assert result.enabled is False
    assert result.tracer is None
    assert result.phoenix_url is None


def test_initialize_phoenix_embedded_mode(monkeypatch) -> None:
    class _Session:
        url = "http://localhost:6111"

    class _Provider:
        def get_tracer(self, name: str):
            return {"name": name}

    phoenix_module = types.ModuleType("phoenix")
    phoenix_module.launch_app = lambda: _Session()
    phoenix_otel_module = types.ModuleType("phoenix.otel")
    phoenix_otel_module.register = lambda **kwargs: _Provider()
    _install_module(monkeypatch, "phoenix", phoenix_module)
    _install_module(monkeypatch, "phoenix.otel", phoenix_otel_module)
    _reset_instrumentation_state()

    result = initialize_phoenix(
        PhoenixSetupConfig(
            enabled=True,
            port=6111,
            launch_sleep_seconds=0.0,
            enable_langchain=False,
            enable_httpx=False,
        )
    )

    assert result.enabled is True
    assert result.phoenix_url == "http://localhost:6111"
    assert result.tracer == {"name": "iris.runtime"}
    assert result.instrumented_targets == ()
    assert result.error is None


def test_initialize_phoenix_external_mode_does_not_launch(monkeypatch) -> None:
    # External mode must export spans WITHOUT importing/launching the Phoenix
    # server in-process — that is the whole point of moving it out.
    class _Provider:
        def get_tracer(self, name: str):
            return {"name": name}

    def _boom() -> object:
        raise AssertionError("px.launch_app must not be called in external mode")

    phoenix_module = types.ModuleType("phoenix")
    phoenix_module.launch_app = _boom
    phoenix_otel_module = types.ModuleType("phoenix.otel")
    phoenix_otel_module.register = lambda **kwargs: _Provider()
    _install_module(monkeypatch, "phoenix", phoenix_module)
    _install_module(monkeypatch, "phoenix.otel", phoenix_otel_module)
    _reset_instrumentation_state()

    result = initialize_phoenix(
        PhoenixSetupConfig(
            enabled=True,
            mode="external",
            port=6006,
            endpoint="http://127.0.0.1:6006/v1/traces",
            endpoint_base="http://127.0.0.1:6006",
            enable_langchain=False,
            enable_httpx=False,
        )
    )

    assert result.enabled is True
    assert result.mode == "external"
    assert result.phoenix_url == "http://127.0.0.1:6006"
    assert result.tracer == {"name": "iris.runtime"}
    assert result.error is None


def test_initialize_phoenix_rejects_unknown_mode() -> None:
    result = initialize_phoenix(PhoenixSetupConfig(enabled=True, mode="bogus"))
    assert result.enabled is True
    assert result.error is not None
    assert "bogus" in result.error


def test_initialize_phoenix_handles_launch_failure(monkeypatch) -> None:
    phoenix_module = types.ModuleType("phoenix")

    def _explode() -> object:
        raise RuntimeError("boom")

    phoenix_module.launch_app = _explode
    _install_module(monkeypatch, "phoenix", phoenix_module)
    _reset_instrumentation_state()

    result = initialize_phoenix(PhoenixSetupConfig(enabled=True, launch_sleep_seconds=0.0))

    assert result.enabled is True
    assert result.tracer is None
    assert result.error == "boom"


def test_initialize_phoenix_sets_working_dir(monkeypatch, tmp_path) -> None:
    class _Session:
        url = "http://localhost:6006"

    class _Provider:
        def get_tracer(self, name: str):
            return {"name": name}

    phoenix_module = types.ModuleType("phoenix")
    phoenix_module.launch_app = lambda: _Session()
    phoenix_otel_module = types.ModuleType("phoenix.otel")
    phoenix_otel_module.register = lambda **kwargs: _Provider()
    _install_module(monkeypatch, "phoenix", phoenix_module)
    _install_module(monkeypatch, "phoenix.otel", phoenix_otel_module)
    _reset_instrumentation_state()

    working_dir = tmp_path / "phoenix-store"
    monkeypatch.delenv("PHOENIX_WORKING_DIR", raising=False)

    result = initialize_phoenix(
        PhoenixSetupConfig(
            enabled=True,
            launch_sleep_seconds=0.0,
            enable_langchain=False,
            enable_httpx=False,
            working_dir=str(working_dir),
        )
    )

    # The dir is created and exported so embedded Phoenix persists traces on disk.
    assert working_dir.exists()
    assert os.environ["PHOENIX_WORKING_DIR"] == str(working_dir)
    assert result.working_dir == str(working_dir)
