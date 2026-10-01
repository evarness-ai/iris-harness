"""``iris_harness.testing``: the stable-tier helpers a plugin's tests use (OSS plan R16)."""

from __future__ import annotations

import os
import socket

import pytest

from iris_harness.llm.fake import FAKE_PROVIDER
from iris_harness.llm.tier_router import FORCED_PROVIDER_ENV
from iris_harness.testing import NetworkBlockedError, no_network, transcript, use_fake_model


def test_use_fake_model_selects_the_fake_and_restores_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iris_harness.llm.client import CodingLLMClient, CodingLLMConfig

    monkeypatch.setenv(FORCED_PROVIDER_ENV, "ollama")
    with use_fake_model({"default": {"content": "scripted"}}):
        assert os.environ[FORCED_PROVIDER_ENV] == FAKE_PROVIDER
        client = CodingLLMClient(CodingLLMConfig(provider="fake", model="m"))
        assert client.invoke(system_prompt="", user_prompt="hi") == "scripted"
        assert [c.rule for c in transcript()] == ["default"]
    assert os.environ[FORCED_PROVIDER_ENV] == "ollama"


def test_no_network_refuses_outbound_connections_and_restores_them() -> None:
    real = socket.socket.connect
    with no_network() as attempted:
        with pytest.raises(NetworkBlockedError):
            socket.create_connection(("192.0.2.1", 80), timeout=1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(NetworkBlockedError):
                sock.connect(("192.0.2.1", 80))
        finally:
            sock.close()
    assert len(attempted) == 2
    assert socket.socket.connect is real
