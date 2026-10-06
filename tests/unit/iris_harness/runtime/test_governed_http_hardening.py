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
import tracemalloc
import zlib
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
    EgressScope,
    HostRule,
    PluginEgress,
    PluginEgressPolicy,
    bind_egress_kernel,
    egress_scope,
    normalize_host_pattern,
    register_egress_policy,
)
from iris_harness.runtime import egress_transport, governed_http
from iris_harness.runtime.governed_http import MAX_RESPONSE_BYTES, EgressDenied, GovernedHttp
from iris_harness.testing import fake_http

NAME = "good.test"
_ZEROS = 48 * 1024 * 1024  # decoded size of the bombs: well over the 10 MiB cap


def _gz(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=9)


_LAYERS = _gz(_gz(b"\0" * _ZEROS))
_BOMB = _gz(b"\0" * _ZEROS)  # ~50 KB on the wire: one wire chunk decodes to 48 MiB
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
                outer.seen.append(
                    {
                        "path": self.path,
                        "host": self.headers.get("Host", ""),
                        "ae": self.headers.get("Accept-Encoding", ""),
                    }
                )
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

            def _send(self, body: bytes, encoding: str = "") -> None:
                self.send_response(200)
                if encoding:
                    self.send_header("Content-Encoding", encoding)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _bomb(self) -> None:
                self._send(_BOMB, "gzip")

            def _gz(self) -> None:
                self._send(_gz(b"hello gzip"), "gzip")

            def _deflate(self) -> None:
                self._send(zlib.compress(b"hello deflate"), "deflate")

            def _rawdeflate(self) -> None:
                comp = zlib.compressobj(wbits=-zlib.MAX_WBITS)
                self._send(comp.compress(b"hello raw") + comp.flush(), "deflate")

            def _layers(self) -> None:
                self._send(_LAYERS, "gzip, gzip")

            def _zstd(self) -> None:
                self._send(b"\x28\xb5\x2f\xfd" + b"\0" * 64, "zstd")

            def _br(self) -> None:
                self._send(b"\x1b" + b"\0" * 64, "br")

            def _badgz(self) -> None:
                self._send(b"this is not gzip", "gzip")

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


@pytest.fixture(autouse=True)
def in_a_call() -> Iterator[None]:
    """Every request is made inside a governed call of plugin ``p``, as the runner stamps it."""
    with egress_scope(EgressScope(run_id="r1", agent_type="chat", tool="t", tool_plugin="p")):
        yield


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
        "::7f00:1",
        "::a00:5",
        "::ffff:0:7f00:1",
        "::ffff:0:a00:5",
        "64:ff9b:1::1",
        "5f00::1",
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


def _peak(fn: Any) -> tuple[Any, int]:
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        try:
            result: Any = fn()
        except EgressDenied as exc:
            result = exc
        return result, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_a_decompression_bomb_is_bounded_in_memory_and_counted_truthfully(
    wired: AuditLog, server: _Server, resolver: _Dns, loopback_allowed: None
) -> None:
    result, peak = _peak(lambda: GovernedHttp("p").get(f"http://{NAME}:{server.port}/bomb"))
    assert isinstance(result, EgressDenied) and "larger than" in str(result)
    assert peak < MAX_RESPONSE_BYTES + 8 * 1024 * 1024  # the cap plus a step, not 48 MiB
    [row] = _rows(wired, "post_egress")
    assert row["egress"]["aborted"] == "max_bytes"
    cap_and_a_step = MAX_RESPONSE_BYTES + governed_http._DECODE_STEP
    assert MAX_RESPONSE_BYTES < row["egress"]["bytes_in"] <= cap_and_a_step


@pytest.mark.parametrize("path", ["layers", "zstd", "br"])
def test_an_encoding_that_cannot_be_bounded_is_refused_before_it_is_read(
    wired: AuditLog,
    server: _Server,
    resolver: _Dns,
    loopback_allowed: None,
    path: str,
) -> None:
    result, peak = _peak(lambda: GovernedHttp("p").get(f"http://{NAME}:{server.port}/{path}"))
    assert isinstance(result, EgressDenied) and "encoding" in str(result)
    assert peak < 4 * 1024 * 1024
    [row] = _rows(wired, "post_egress")
    assert row["egress"]["aborted"] == "encoding" and row["egress"]["bytes_in"] == 0


def test_only_the_identity_encoding_is_asked_for(
    wired: AuditLog, server: _Server, resolver: _Dns, loopback_allowed: None
) -> None:
    GovernedHttp("p").get(f"http://{NAME}:{server.port}/ok", headers={"Accept-Encoding": "br"})
    assert server.seen[0]["ae"] == "identity"


@pytest.mark.parametrize(
    "path, body",
    [("gz", b"hello gzip"), ("deflate", b"hello deflate"), ("rawdeflate", b"hello raw")],
)
def test_a_single_gzip_or_deflate_layer_is_decoded(
    wired: AuditLog, server: _Server, resolver: _Dns, loopback_allowed: None, path: str, body: bytes
) -> None:
    reply = GovernedHttp("p").get(f"http://{NAME}:{server.port}/{path}")
    assert reply.content == body and "content-encoding" not in reply.headers


def test_a_body_that_is_not_what_its_encoding_says_is_a_recorded_abort(
    wired: AuditLog, server: _Server, resolver: _Dns, loopback_allowed: None
) -> None:
    with pytest.raises(EgressDenied, match="could not be decoded"):
        GovernedHttp("p").get(f"http://{NAME}:{server.port}/badgz")
    [row] = _rows(wired, "post_egress")
    assert row["egress"]["aborted"] == "decode"


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
        (True, 10.0, 10.0),
        (False, 10.0, 10.0),
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


_BAD_HEADERS: list[Any] = [
    {"Host": "other.test"},
    {"host": "other.test"},
    {"HOST": "other.test"},
    {"Proxy-Authorization": "x"},
    {"proxy-connection": "x"},
    {"Transfer-Encoding": "chunked"},
    {"Connection": "upgrade"},
    {"Upgrade": "websocket"},
    {"TE": "trailers"},
    {"Content-Length": "5"},
    [("Host", "other.test")],
    (("Host", "other.test"),),
    [("Proxy-Authorization", "x")],
    {b"Host": b"other.test"},
    [(b"host", b"other.test")],
    [("Accept", "text/plain"), ("hOsT", "other.test")],
]


@pytest.mark.parametrize("headers", _BAD_HEADERS)
def test_a_host_framing_or_proxy_header_is_refused_in_any_spelling_and_recorded(
    wired: AuditLog, server: _Server, resolver: _Dns, loopback_allowed: None, headers: Any
) -> None:
    with pytest.raises(EgressDenied, match="header is not sent"):
        GovernedHttp("p").get(f"http://{NAME}:{server.port}/ok", headers=headers)
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


# -- S1, S2: the client acts for the running tool's plugin, and only inside a call ------


@pytest.mark.parametrize("name", ["p", "web"])
def test_a_plugin_cannot_use_another_plugins_declaration(
    wired: AuditLog, server: _Server, resolver: _Dns, loopback_allowed: None, name: str
) -> None:
    """Plugin ``evil`` runs the tool; ``p`` declares the host and ``web`` is open_web."""
    scope = EgressScope(run_id="r2", agent_type="chat", tool="steal", tool_plugin="evil")
    with egress_scope(scope):
        with pytest.raises(EgressDenied, match="different plugin"):
            GovernedHttp(name).get(f"http://{NAME}:{server.port}/ok")
    assert server.seen == []
    [row] = _rows(wired, "pre_egress")
    assert row["decision"] == "deny"
    assert row["tool_plugin"] == "evil" and row["tool_name"] == "steal"
    assert row["egress"]["plugin"] == name


def test_a_request_outside_a_governed_call_is_denied(
    wired: AuditLog, server: _Server, resolver: _Dns, loopback_allowed: None
) -> None:
    """A thread the tool started has no scope: no run, no data class, no parent call."""
    outcome: list[BaseException | httpx.Response] = []

    def work() -> None:
        try:
            outcome.append(GovernedHttp("p").get(f"http://{NAME}:{server.port}/ok"))
        except EgressDenied as exc:
            outcome.append(exc)

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    [denied] = outcome
    assert isinstance(denied, EgressDenied) and "no governed" in str(denied)
    assert server.seen == []
    [row] = _rows(wired, "pre_egress")
    assert row["decision"] == "deny"


def test_a_denial_is_not_an_oserror_so_except_oserror_cannot_swallow_it() -> None:
    assert not issubclass(EgressDenied, OSError)
    with pytest.raises(EgressDenied):
        try:
            raise EgressDenied("denied")
        except OSError:  # a plugin's network code
            pytest.fail("a governance denial was caught as an OSError")


def test_a_resolver_that_raises_anything_is_a_recorded_refusal(
    wired: AuditLog, server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: Any, **__: Any) -> Any:
        raise ValueError("resolver bug")

    monkeypatch.setattr(egress_transport, "_resolve", broken)
    with pytest.raises(EgressDenied, match="not sent"):
        GovernedHttp("p").get(f"http://{NAME}:{server.port}/ok")
    [row] = _rows(wired, "post_egress")
    assert row["egress"]["aborted"] == "address"


def test_the_httpx_this_runs_on_has_the_pool_and_httpcore_the_pinned_transport_needs() -> None:
    """Fails loudly if an httpx or httpcore upgrade moves what ``PinnedTransport`` swaps."""
    import httpcore

    assert httpcore.__version__.startswith("1.")
    assert type(httpx.HTTPTransport()._pool) is httpcore.ConnectionPool
    egress_transport.PinnedTransport(egress_transport.PinnedBackend(1.0)).close()
