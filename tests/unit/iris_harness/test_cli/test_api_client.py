"""``cli/api_client.py``: the CLI's calls to its own services log host-only egress."""

from __future__ import annotations

import io
import logging
import urllib.request
from typing import Any

import httpx
import pytest

from iris_harness.cli import api_client
from iris_harness.cli.api_client import api_host, harness_api_client, harness_urlopen

SECRET = "test-secret-for-testing"
# Userinfo, a path with an ID in it, a query string and a fragment: none may be logged.
URL = "http://operator:hunter2@127.0.0.1:8003/governance/approvals/ap-123/respond?token=abc#f"


@pytest.fixture()
def egress(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("IRIS_AUTH_SECRET", SECRET)
    caplog.set_level(logging.INFO, logger="iris.egress")

    def lines() -> list[str]:
        return [r.getMessage() for r in caplog.records if r.name == "iris.egress"]

    return lines


def _assert_host_only(line: str) -> None:
    assert "-> 127.0.0.1:8003 " in line + " ", line
    for leak in ("hunter2", "operator", "/governance", "ap-123", "token", "abc", "#f", SECRET):
        assert leak not in line, (leak, line)


def test_api_host_keeps_host_and_port_only() -> None:
    assert api_host(URL) == "127.0.0.1:8003"
    assert api_host("https://api.example.test/v1/x?key=1") == "api.example.test"
    assert api_host("http://[::1]:8003/chat") == "[::1]:8003"
    assert api_host("not a url") == "unknown"
    assert api_host("http://host:notaport/") == "unknown"


def test_client_logs_each_request_host_only_and_sends_the_secret(egress: Any) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    with harness_api_client(purpose="approvals.respond", timeout=1.0, transport=transport) as c:
        c.post(URL, json={"status": "approved"})
        c.get("http://127.0.0.1:8003/plugins", params={"session_id": "s-9"})

    lines = egress()
    assert len(lines) == 2, lines
    assert lines[0].startswith("EGRESS service POST -> 127.0.0.1:8003")
    assert "purpose=approvals.respond" in lines[0]
    assert lines[1].startswith("EGRESS service GET -> 127.0.0.1:8003")
    assert "s-9" not in lines[1] and "/plugins" not in lines[1]
    for line in lines:
        _assert_host_only(line)
    # The secret travels in the header, where it belongs, and never in the log line.
    # (seen[0] carried URL userinfo, which httpx turns into Basic auth for that request.)
    assert seen[1].headers["authorization"] == f"Bearer {SECRET}"


def test_client_without_auth_sends_no_secret(egress: Any) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    with harness_api_client(purpose="probe", timeout=1.0, auth=False, transport=transport) as c:
        c.get("http://127.0.0.1:8090/healthz")
    assert "authorization" not in seen[0].headers
    assert egress() == ["EGRESS service GET -> 127.0.0.1:8090 purpose=probe"]


def test_a_failed_request_is_still_logged(egress: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    transport = httpx.MockTransport(handler)
    with harness_api_client(purpose="plugins", timeout=1.0, transport=transport) as c:
        with pytest.raises(httpx.ConnectError):
            c.get(URL)
    assert len(egress()) == 1


def test_urlopen_logs_host_only_then_opens(egress: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[tuple[urllib.request.Request, float]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float = 0) -> io.BytesIO:
        opened.append((request, timeout))
        return io.BytesIO(b"{}")

    monkeypatch.setattr(api_client.urllib.request, "urlopen", fake_urlopen)
    request = urllib.request.Request(
        URL, data=b"{}", headers={"Authorization": f"Bearer {SECRET}"}, method="POST"
    )
    with harness_urlopen(request, purpose="chat-stream", timeout=600) as resp:
        assert resp.read() == b"{}"

    assert opened == [(request, 600)]
    (line,) = egress()
    assert line.startswith("EGRESS service POST -> 127.0.0.1:8003 purpose=chat-stream")
    _assert_host_only(line)
