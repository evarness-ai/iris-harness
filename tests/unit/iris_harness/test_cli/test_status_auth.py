"""``iris status`` reads ``GET /health``, which takes the secret like any data route
(ADR-0117) — it used to be an open path, and the command sent no header."""

from __future__ import annotations

import io
import json
import urllib.request
from typing import Any

import pytest
from typer.testing import CliRunner

from iris_harness.main import app

runner = CliRunner()


def test_status_sends_the_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_AUTH_SECRET", "s3cret")
    seen: list[urllib.request.Request] = []

    def fake_urlopen(request: Any, timeout: float = 0) -> io.BytesIO:
        seen.append(request)
        return io.BytesIO(json.dumps({"state": "green"}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = runner.invoke(app, ["status", "--api", "http://iris.test"])

    assert result.exit_code == 0, result.output
    assert "IRIS API online" in result.output
    (request,) = seen
    assert request.full_url == "http://iris.test/health"
    assert request.get_header("Authorization") == "Bearer s3cret"
    assert "s3cret" not in result.output
