"""The governed HTTP client's network hardening (issue #103, review findings).

Everything runs against a loopback server this test starts; no external network and no real
DNS (the resolver is a seam, faked below). Loopback is a *blocked* address in production, so
the tests that need to reach the server patch the address check to permit exactly 127.0.0.1
and leave every other rule (resolve once, check every address, pin the connect) running.
"""

from __future__ import annotations

import gzip
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from iris_harness.kernel.governance import GovernanceKernel, build_default_kernel
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.plugin_egress import (
    HostRule,
    PluginEgress,
    PluginEgressPolicy,
    bind_egress_kernel,
    normalize_host_pattern,
    register_egress_policy,
)
from iris_harness.runtime import egress_transport, governed_http
from iris_harness.runtime.governed_http import MAX_RESPONSE_BYTES, EgressDenied, GovernedHttp
from iris_harness.testing import fake_http

NAME = "good.test"
CHUNK = b"x" * 65536


class _Server:
    def __init__(self) -> None:
        self.seen: list[dict[str, str]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                outer.seen.append({"path": self.path, "host": self.headers.get("Host", "")})
                try:
                    getattr(self, "_" + self.path.strip("/"))()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True

            def _ok(self) -> None:
                body = b"hello"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _big(self) -> None:
                total = MAX_RESPONSE_BYTES + 3 * len(CHUNK)
                self.send_response(200)
                self.send_header("Content-Length", str(total))
                self.end_headers()
                for _ in range(total // len(CHUNK)):
                    self.wfile.write(CHUNK)

            def _bomb(self) -> None:
                body = gzip.compress(b"\0" * (MAX_RESPONSE_BYTES + 2_000_000))
                self.send_response(200)
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _trickle(self) -> None:
                self.send_response(200)
                self.send_header("Content-Length", "1000")
                self.end_headers()
                for _ in range(40):
                    self.wfile.write(b"y")
                    self.wfile.flush()
                    time.sleep(0.2)

            def _slowheaders(self) -> None:
                self.wfile.write(b"HTTP/1.1 200 OK\r\n")
                for _ in range(40):
                    self.wfile.write(b"X-Pad: 1")
                    self.wfile.flush()
                    time.sleep(0.2)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = int(self.httpd.server_address[1])
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def server() -> Iterator[_Server]:
    srv = _Server()
    try:
        yield srv
    finally:
        srv.close()


class _Dns:
    """Fake DNS: ``good.test`` -> 127.0.0.1 unless a test sets other answers."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.answers: dict[str, list[str]] = {NAME: ["127.0.0.1"]}

    def __call__(self, host: str, port: int, **_: Any) -> list[tuple[Any, ...]]:
        self.calls.append(host)
        return [(2, 1, 6, "", (a, port)) for a in self.answers[host]]


@pytest.fixture()
def resolver(monkeypatch: pytest.MonkeyPatch) -> _Dns:
    dns = _Dns()
    monkeypatch.setattr(egress_transport, "_resolve", dns)
    return dns


@pytest.fixture()
def loopback_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    real = egress_transport.blocked_address
    monkeypatch.setattr(
        egress_transport, "blocked_address", lambda a: False if a == "127.0.0.1" else real(a)
    )


@pytest.fixture()
def audit(tmp_path: Path) -> Iterator[AuditLog]:
    log = AuditLog(db_path=tmp_path / "audit.db")
    yield log


@contextmanager
def _bound(kernel: GovernanceKernel, port: int) -> Iterator[None]:
    register_egress_policy(
        PluginEgressPolicy(
            {
                "p": PluginEgress(
                    hosts=(HostRule(NAME, schemes=("http", "https"), ports=(port,)),)
                ),
                "web": PluginEgress(open_web=True),
            }
        )
    )
    bind_egress_kernel(lambda: kernel)
    try:
        yield
    finally:
        register_egress_policy(None)
        bind_egress_kernel(None)


@pytest.fixture()
def wired(audit: AuditLog, server: _Server) -> Iterator[AuditLog]:
    with _bound(build_default_kernel(audit_log=audit), server.port):
        yield audit


def _rows(audit: AuditLog, point: str) -> list[dict[str, Any]]:
    return [
        {"decision": r.decision, "reason": r.reason, **json.loads(r.payload_json)}
        for r in audit.query()
        if r.hook_point == point
    ]


# -- F1a: local-network names, even under open_web ---------------------------------------


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "localhost.",
        "x.localhost",
        "good.com.localhost",
        "printer.local",
        "metadata.internal",
        "a.b.internal.",
        "host.localdomain",
    ],
)
def test_a_local_network_name_is_denied_even_for_open_web(host: str) -> None:
    policy = PluginEgressPolicy({"web": PluginEgress(open_web=True)})
    verdict = policy.decide("web", scheme="https", host=host, port=443)
    assert not verdict.allowed and "local-network name" in verdict.reason


def test_an_ordinary_name_is_still_allowed_for_open_web() -> None:
    policy = PluginEgressPolicy({"web": PluginEgress(open_web=True)})
    assert policy.decide("web", scheme="https", host="example.org", port=443).allowed


@pytest.mark.parametrize("host", ["*.internal", "corp.internal", "*.local", "x.localdomain"])
def test_a_local_network_name_cannot_be_declared(host: str) -> None:
    with pytest.raises(ValueError):
        normalize_host_pattern(host)


# -- F1b: resolve once, check, connect to the checked address ----------------------------


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "192.168.0.9",
        "172.16.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",  # noqa: S104 - an address under test, not a bind
        "::1",
        "::",
        "fd00::1",
        "fc00::5",
        "fe80::1%en0",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
        "64:ff9b::7f00:1",
        "2002:7f00:1::1",
        "not-an-address",
    ],
)
def test_internal_addresses_are_blocked(address: str) -> None:
    assert egress_transport.blocked_address(address)


@pytest.mark.parametrize("address", ["8.8.8.8", "93.184.216.34", "2606:4700:4700::1111"])
def test_public_addresses_are_not_blocked(address: str) -> None:
    assert not egress_transport.blocked_address(address)


def test_a_name_that_resolves_inside_is_not_connected_to(
    wired: AuditLog, server: _Server, resolver: _Dns
) -> None:
    with pytest.raises(EgressDenied, match="not sent"):
        GovernedHttp("p").get(f"http://{NAME}:{server.port}/ok")
    assert server.seen == []
    [row] = _rows(wired, "post_egress")
    assert row["egress"]["aborted"] == "address" and row["egress"]["error"] == "EgressDenied"


def test_one_internal_answer_among_public_ones_blocks_the_name(
    wired: AuditLog, server: _Server, resolver: _Dns
) -> None:
    resolver.answers[NAME] = ["8.8.8.8", "127.0.0.1"]
    with pytest.raises(EgressDenied, match="not sent"):
        GovernedHttp("p").get(f"http://{NAME}:{server.port}/ok")
    assert server.seen == []


def test_the_checked_address_is_the_one_connected_to_and_the_name_is_resolved_once(
    wired: AuditLog, server: _Server, resolver: _Dns, loopback_allowed: None
) -> None:
    reply = GovernedHttp("p").get(f"http://{NAME}:{server.port}/ok")
    assert (reply.status_code, reply.content) == (200, b"hello")
    assert resolver.calls == [NAME]  # one lookup, no second one at connect time
    assert server.seen[0]["host"] == f"{NAME}:{server.port}"  # the Host header keeps the name


# -- F4: size, time ----------------------------------------------------------------------


def test_a_response_over_the_cap_is_cut_off_and_recorded(
    wired: AuditLog, server: _Server, resolver: _Dns, loopback_allowed: None
) -> None:
    with pytest.raises(EgressDenied, match="larger than"):
        GovernedHttp("p").get(f"http://{NAME}:{server.port}/big")
    [row] = _rows(wired, "post_egress")
    assert row["egress"]["aborted"] == "max_bytes"
    assert MAX_RESPONSE_BYTES < row["egress"]["bytes_in"] < MAX_RESPONSE_BYTES + 1_000_000


def test_the_cap_counts_decoded_bytes(
    wired: AuditLog, server: _Server, resolver: _Dns, loopback_allowed: None
) -> None:
    with pytest.raises(EgressDenied, match="larger than"):
        GovernedHttp("p").get(f"http://{NAME}:{server.port}/bomb")
    [row] = _rows(wired, "post_egress")
    assert row["egress"]["bytes_in"] > MAX_RESPONSE_BYTES  # decoded, not the ~10 KB on the wire


def test_a_slow_body_is_cut_off_at_the_total_deadline(
    wired: AuditLog, server: _Server, resolver: _Dns, loopback_allowed: None
) -> None:
    started = time.monotonic()
    with pytest.raises(EgressDenied, match="time limit"):
        GovernedHttp("p").get(f"http://{NAME}:{server.port}/trickle", timeout=1)
    assert time.monotonic() - started < 3
    [row] = _rows(wired, "post_egress")
    assert row["egress"]["aborted"] == "deadline" and 0 < row["egress"]["bytes_in"] < 20


def test_slow_headers_are_cut_off_at_the_total_deadline(
    wired: AuditLog, server: _Server, resolver: _Dns, loopback_allowed: None
) -> None:
    started = time.monotonic()
    with pytest.raises(EgressDenied, match="time limit"):
        GovernedHttp("p").get(f"http://{NAME}:{server.port}/slowheaders", timeout=1)
    assert time.monotonic() - started < 3


class _Drip(httpx.SyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        for _ in range(30):
            time.sleep(0.2)
            yield b"z"


def test_the_deadline_also_holds_on_the_fake_transport_between_chunks(
    wired: AuditLog, server: _Server
) -> None:
    started = time.monotonic()
    with fake_http(lambda request: httpx.Response(200, stream=_Drip())):
        with pytest.raises(EgressDenied, match="time limit"):
            GovernedHttp("p").get(f"http://{NAME}:{server.port}/x", timeout=1)
    assert time.monotonic() - started < 3


@pytest.mark.parametrize(
    "asked, default, expected",
    [
        (None, 10.0, 10.0),
        (0, 10.0, 10.0),
        (-5, 10.0, 10.0),
        (float("nan"), 10.0, 10.0),
        ("soon", 10.0, 10.0),
        (2.5, 10.0, 2.5),
        (1e9, 10.0, governed_http._MAX_TIMEOUT),
        (float("inf"), 10.0, governed_http._MAX_TIMEOUT),
    ],
)
def test_the_callers_timeout_is_clamped(asked: Any, default: float, expected: float) -> None:
    assert governed_http._seconds(asked, default) == expected


def test_the_client_constructor_timeout_is_clamped_too() -> None:
    assert GovernedHttp("p", timeout=1e9)._timeout == governed_http._MAX_TIMEOUT
    assert GovernedHttp("p", timeout=0)._timeout == governed_http._DEFAULT_TIMEOUT


async def test_the_async_path_has_the_same_pinned_connect_and_cap(
    wired: AuditLog, server: _Server, resolver: _Dns, loopback_allowed: None
) -> None:
    ok = await GovernedHttp("p").arequest("GET", f"http://{NAME}:{server.port}/ok")
    assert ok.content == b"hello"
    with pytest.raises(EgressDenied, match="larger than"):
        await GovernedHttp("p").arequest("GET", f"http://{NAME}:{server.port}/big")


# -- F2: nothing from the environment ----------------------------------------------------


def test_the_environment_cannot_reroute_a_request(
    wired: AuditLog,
    server: _Server,
    resolver: _Dns,
    loopback_allowed: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")  # a dead proxy
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert GovernedHttp("p").get(f"http://{NAME}:{server.port}/ok").status_code == 200


# -- F6, F7: the request is refused, recorded, and never echoed --------------------------


def test_an_invalid_url_is_a_recorded_denial_that_does_not_echo_it(wired: AuditLog) -> None:
    with pytest.raises(EgressDenied) as caught:
        GovernedHttp("p").get(f"http://{NAME}:notaport/")
    assert "notaport" not in str(caught.value)
    [row] = _rows(wired, "pre_egress")
    assert row["decision"] == "deny" and row["egress"]["malformed"] == "the URL is not valid"
    assert "notaport" not in json.dumps(row)


@pytest.mark.parametrize(
    "header", ["Host", "host", "HOST", "Proxy-Authorization", "proxy-connection"]
)
def test_a_host_or_proxy_header_is_refused_and_recorded(
    wired: AuditLog, server: _Server, resolver: _Dns, loopback_allowed: None, header: str
) -> None:
    with pytest.raises(EgressDenied, match="Host or Proxy"):
        GovernedHttp("p").get(f"http://{NAME}:{server.port}/ok", headers={header: "other.test"})
    assert server.seen == []
    [row] = _rows(wired, "pre_egress")
    assert row["decision"] == "deny"


def test_ordinary_headers_still_go(
    wired: AuditLog, server: _Server, resolver: _Dns, loopback_allowed: None
) -> None:
    ok = GovernedHttp("p").get(f"http://{NAME}:{server.port}/ok", headers={"Accept": "text/plain"})
    assert ok.status_code == 200


# -- F5: no ledger row, no request -------------------------------------------------------


class _BrokenLedger(AuditLog):
    def record(self, **_: Any) -> int:
        raise OSError("disk full")


def test_a_failed_pre_egress_ledger_write_stops_the_request(
    tmp_path: Path, server: _Server, resolver: _Dns, loopback_allowed: None
) -> None:
    broken = _BrokenLedger(db_path=tmp_path / "b.db")
    with _bound(build_default_kernel(audit_log=broken), server.port):
        with pytest.raises(EgressDenied, match="ledger write"):
            GovernedHttp("p").get(f"http://{NAME}:{server.port}/ok")
        assert server.seen == []
        with fake_http({}) as sent, pytest.raises(EgressDenied, match="ledger write"):
            GovernedHttp("p").get(f"http://{NAME}:{server.port}/x")
        assert sent == []
