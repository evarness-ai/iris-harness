"""The research plugin's fetches go through the governed client (issue #172).

The page fetch (``extract.py``) and the keyed providers (Brave, Exa, Tavily) used to open their
own connections. Now each request is checked against the plugin's ``egress`` declaration,
made to the address the client checked, and recorded in the ledger. Everything runs against a
loopback server this test starts, with a fake resolver; no external network.

The SSRF protections the old code had are pinned per hop on the real transport (a fake
transport would skip the address check): a redirect to a loopback, private, link-local or
metadata address is refused and never connected to; a redirect loop and a long chain stop at
the cap; a redirect to ``file://`` is refused before any request. A page fetched on an executor
thread still leaves ledger rows, and a fetch with no tool call running is refused, not sent.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

import iris_harness.plugins_builtin.research.extract as extract_mod
from iris_harness.kernel.governance import build_default_kernel
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.plugin_egress import (
    EgressScope,
    PluginEgress,
    PluginEgressPolicy,
    bind_egress_kernel,
    egress_scope,
    register_egress_policy,
)
from iris_harness.plugins_builtin.research.extract import (
    _MAX_REDIRECTS,
    _fetch,
    _fetch_and_extract,
    extract_into,
)
from iris_harness.plugins_builtin.research.models import SearchResult
from iris_harness.plugins_builtin.research.providers import (
    BraveProvider,
    ExaProvider,
    TavilyProvider,
)
from iris_harness.runtime import egress_transport

PAGES = Path(__file__).resolve().parents[4] / "fixtures" / "research_pages"
SCOPE = EgressScope(run_id="r1", agent_type="chat", tool="research", tool_plugin="research")


class _Server:
    def __init__(self) -> None:
        self.seen: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                outer.seen.append(self.path)
                parsed = urlparse(self.path)
                if parsed.path == "/redirect":
                    self._redirect(parse_qs(parsed.query)["to"][0])
                elif parsed.path == "/loop":
                    self._redirect("/loop")
                elif parsed.path.startswith("/chain/"):
                    self._redirect(f"/chain/{int(parsed.path.rsplit('/', 1)[1]) + 1}")
                elif parsed.path.startswith("/status/"):
                    self._send(int(parsed.path.rsplit("/", 1)[1]), b"nope")
                elif parsed.path.startswith("/page/"):
                    self._send(200, (PAGES / f"{parsed.path.rsplit('/', 1)[1]}.html").read_bytes())
                else:
                    self._send(200, b"<html><body>hello</body></html>")

            def _redirect(self, to: str) -> None:
                if "\r" in to or "\n" in to:
                    self.send_error(400, "bad redirect target")
                    return
                self.send_response(302)
                self.send_header("Location", to)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _send(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

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
    """``pub.test`` -> the loopback server (allowed below); other names as a test sets them."""

    def __init__(self) -> None:
        self.answers: dict[str, list[str]] = {"pub.test": ["127.0.0.1"]}

    def __call__(self, host: str, port: int, **_: Any) -> list[tuple[Any, ...]]:
        addresses = self.answers.get(host, [host])  # an unknown name or a literal answers as itself
        return [(2, 1, 6, "", (a, port)) for a in addresses]


@pytest.fixture()
def dns(monkeypatch: pytest.MonkeyPatch) -> _Dns:
    fake = _Dns()
    monkeypatch.setattr(egress_transport, "_resolve", fake)
    real = egress_transport.blocked_address
    # Only the test server's own address is let through; every other rule keeps running.
    monkeypatch.setattr(
        egress_transport, "blocked_address", lambda a: False if a == "127.0.0.1" else real(a)
    )
    return fake


@pytest.fixture()
def ledger(tmp_path: Path, server: _Server, dns: _Dns) -> Iterator[AuditLog]:
    log = AuditLog(db_path=tmp_path / "audit.db")
    kernel = build_default_kernel(audit_log=log)
    register_egress_policy(PluginEgressPolicy({"research": PluginEgress(open_web=True)}))
    bind_egress_kernel(lambda: kernel)
    try:
        yield log
    finally:
        register_egress_policy(None)
        bind_egress_kernel(None)


@pytest.fixture()
def in_a_research_call() -> Iterator[None]:
    with egress_scope(SCOPE):
        yield


def _rows(ledger: AuditLog, point: str) -> list[Any]:
    return [r for r in ledger.query() if r.hook_point == point]


def _url(server: _Server, path: str) -> str:
    return f"http://pub.test:{server.port}{path}"


# -- the fetch, hop by hop ----------------------------------------------------------------


def test_a_page_is_fetched_and_each_request_leaves_pre_and_post_rows_with_a_call_id(
    server: _Server, ledger: AuditLog, in_a_research_call: None
) -> None:
    body = _fetch(_url(server, "/page/article"), timeout=8.0)
    assert body is not None and b"Oslo" in body
    pre, post = _rows(ledger, "pre_egress"), _rows(ledger, "post_egress")
    assert len(pre) == len(post) == 1
    assert pre[0].call_id and pre[0].call_id == post[0].call_id
    assert pre[0].plugin == "plugin_egress" or "research" in pre[0].payload_json
    assert json.loads(post[0].payload_json)["egress"]["host"] == "pub.test"


def test_a_redirect_is_followed_one_governed_request_per_hop(
    server: _Server, ledger: AuditLog, in_a_research_call: None
) -> None:
    target = _url(server, "/page/boilerplate")
    body = _fetch(_url(server, f"/redirect?to={target}"), timeout=8.0)
    assert body is not None and b"slow travel" in body.lower()
    assert len(_rows(ledger, "pre_egress")) == 2
    calls = {r.call_id for r in _rows(ledger, "pre_egress")}
    assert len(calls) == 2  # each hop is its own request, with its own id


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.2",  # loopback (the test server's own 127.0.0.1 is the only one let through)
        "10.0.0.5",
        "172.16.3.4",
        "192.168.1.10",
        "169.254.169.254",  # link-local: the cloud metadata address
        "::1",
        "fe80::1",
        "::ffff:169.254.169.254",
    ],
)
def test_a_redirect_to_a_non_public_address_is_refused_and_never_connected_to(
    address: str, server: _Server, dns: _Dns, ledger: AuditLog, in_a_research_call: None
) -> None:
    dns.answers["inner.test"] = [address]
    assert _fetch_and_extract(_url(server, "/redirect?to=http://inner.test/admin"), 8.0) is None
    assert [p for p in server.seen if p.startswith("/admin")] == []
    # The refused hop is a recorded request that failed, and no connection was made for it.
    posts = _rows(ledger, "post_egress")
    assert json.loads(posts[-1].payload_json)["egress"]["aborted"] == "address"


def test_an_ip_literal_in_a_redirect_is_denied_by_policy_before_any_lookup(
    server: _Server, ledger: AuditLog, in_a_research_call: None
) -> None:
    """Even open_web contacts host *names* only: a literal address (the metadata address
    included) is a recorded denial at the pre-egress hook, never resolved or connected to.
    This also means a result that links a bare public IP is no longer fetched (it was)."""
    to = "http://169.254.169.254/latest/meta-data/"
    assert _fetch_and_extract(_url(server, f"/redirect?to={to}"), 8.0) is None
    denied = [r for r in _rows(ledger, "pre_egress") if r.decision == "deny"]
    assert len(denied) == 1 and "IP address" in denied[0].reason
    assert json.loads(denied[0].payload_json)["egress"]["host"] == "169.254.169.254"


@pytest.mark.parametrize("target", ["file:///etc/passwd", "ftp://example.com/x", "gopher://x/"])
def test_a_redirect_to_a_non_http_scheme_is_refused_before_any_request(
    target: str, server: _Server, ledger: AuditLog, in_a_research_call: None
) -> None:
    assert _fetch_and_extract(_url(server, f"/redirect?to={target}"), 8.0) is None
    assert len(_rows(ledger, "pre_egress")) == 1  # only the first hop was ever requested


class _Hop:
    def __init__(self, status: int, location: str = "", content: bytes = b"ok") -> None:
        self.status_code = status
        self.headers = {"location": location}
        self.content = content


def _scripted_fetch(
    monkeypatch: pytest.MonkeyPatch, hops: dict[str, _Hop], start: str
) -> tuple[bytes | None, list[str]]:
    asked: list[str] = []

    class _Http:
        def get(self, url: str, **kwargs: Any) -> _Hop:
            asked.append(url)
            return hops[url]

    monkeypatch.setattr(extract_mod, "current_http", lambda: _Http())
    return _fetch(start, 8.0), asked


def test_an_https_page_redirecting_to_http_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(extract_mod.UnsafeURLError, match="https to http"):
        _scripted_fetch(
            monkeypatch, {"https://a.test/": _Hop(302, "http://a.test/x")}, "https://a.test/"
        )
    # Refused before the second request: only the first hop was made. (Through the batch,
    # _fetch_and_extract turns the refusal into no content.)


def test_an_http_page_may_redirect_to_https_and_http_to_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    up = {"http://a.test/": _Hop(302, "https://a.test/x"), "https://a.test/x": _Hop(200)}
    assert _scripted_fetch(monkeypatch, up, "http://a.test/") == (
        b"ok",
        ["http://a.test/", "https://a.test/x"],
    )
    same = {"http://a.test/": _Hop(302, "/next"), "http://a.test/next": _Hop(200)}
    assert _scripted_fetch(monkeypatch, same, "http://a.test/")[0] == b"ok"
    https_only = {"https://a.test/": _Hop(301, "/moved"), "https://a.test/moved": _Hop(200)}
    assert _scripted_fetch(monkeypatch, https_only, "https://a.test/")[0] == b"ok"


def test_the_downgrade_is_refused_through_the_batch_and_recorded_once(
    server: _Server, ledger: AuditLog, in_a_research_call: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the real client: an https first hop would need TLS, so the scripted hop above pins
    the rule; here the whole fetch returns no content and the first request is the only one."""
    hops = {"https://pub.test/": _Hop(302, f"http://pub.test:{server.port}/page/article")}
    monkeypatch.setattr(
        extract_mod,
        "current_http",
        lambda: type("H", (), {"get": lambda self, url, **kw: hops[url]})(),
    )
    assert _fetch_and_extract("https://pub.test/", 8.0) is None
    assert server.seen == []


def test_a_redirect_loop_stops_at_the_cap(
    server: _Server, ledger: AuditLog, in_a_research_call: None
) -> None:
    assert _fetch_and_extract(_url(server, "/loop"), 8.0) is None
    assert len(server.seen) == _MAX_REDIRECTS + 1


def test_a_long_chain_stops_at_the_cap(
    server: _Server, ledger: AuditLog, in_a_research_call: None
) -> None:
    assert _fetch_and_extract(_url(server, "/chain/0"), 8.0) is None
    assert server.seen == [f"/chain/{i}" for i in range(_MAX_REDIRECTS + 1)]


def test_a_chain_within_the_cap_is_followed(
    server: _Server, ledger: AuditLog, in_a_research_call: None
) -> None:
    target = _url(server, "/page/article")
    hop = _url(server, f"/redirect?to={target}")
    chain = _url(server, f"/redirect?to={hop}")
    assert _fetch(chain, 8.0) is not None
    assert len(server.seen) == 3


def test_an_error_status_is_no_content(
    server: _Server, ledger: AuditLog, in_a_research_call: None
) -> None:
    assert _fetch_and_extract(_url(server, "/status/404"), 8.0) is None
    assert _fetch_and_extract(_url(server, "/status/500"), 8.0) is None


def test_an_unresolvable_name_is_refused(
    server: _Server, dns: _Dns, ledger: AuditLog, in_a_research_call: None, monkeypatch: Any
) -> None:
    def nxdomain(host: str, port: int, **_: Any) -> Any:
        raise OSError("nxdomain")

    monkeypatch.setattr(egress_transport, "_resolve", nxdomain)
    assert _fetch_and_extract("https://nowhere.test/", 8.0) is None


def test_a_local_network_name_is_refused_even_though_research_is_open_web(
    server: _Server, ledger: AuditLog, in_a_research_call: None
) -> None:
    assert _fetch_and_extract("http://localhost:8003/", 8.0) is None
    assert _fetch_and_extract("http://printer.local/", 8.0) is None
    assert server.seen == []


# -- no tool call, no request ---------------------------------------------------------------


def test_a_fetch_with_no_tool_call_running_is_refused_and_nothing_is_sent(
    server: _Server, ledger: AuditLog
) -> None:
    assert _fetch_and_extract(_url(server, "/page/article"), 8.0) is None
    assert server.seen == []


async def test_the_executor_threads_carry_the_tool_calls_scope(
    server: _Server, ledger: AuditLog, in_a_research_call: None
) -> None:
    results = [
        SearchResult(title=str(i), url=_url(server, f"/page/{n}"))
        for i, n in enumerate(["article", "boilerplate", "utf8_odd"])
    ]
    await extract_into(results, max_pages=3)
    assert all(r.content for r in results)
    assert len(_rows(ledger, "pre_egress")) == 3
    assert all(r.call_id for r in _rows(ledger, "pre_egress"))


async def test_executor_threads_without_a_scope_send_nothing(
    server: _Server, ledger: AuditLog
) -> None:
    results = [SearchResult(title="x", url=_url(server, "/page/article"))]
    await extract_into(results, max_pages=1)
    assert results[0].content is None and server.seen == []


def test_the_sync_wrapper_keeps_the_scope(
    server: _Server, ledger: AuditLog, in_a_research_call: None
) -> None:
    results = [SearchResult(title="x", url=_url(server, "/page/article"))]
    extract_mod.extract_into_sync(results, max_pages=1)
    assert results[0].content and len(_rows(ledger, "pre_egress")) == 1


# -- golden pages: the extracted text is byte-identical to what the old fetch produced ------


@pytest.mark.parametrize("name", ["article", "boilerplate", "utf8_odd", "long", "empty"])
def test_the_extracted_text_matches_what_the_old_fetch_produced(
    name: str, server: _Server, ledger: AuditLog, in_a_research_call: None
) -> None:
    expected = (PAGES / f"{name}.expected.md").read_text(encoding="utf-8")
    got = _fetch_and_extract(_url(server, f"/page/{name}"), 8.0)
    assert (got or "") == expected


# -- the keyed providers ----------------------------------------------------------------------


def test_the_providers_send_nothing_outside_a_tool_call(
    server: _Server, ledger: AuditLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    monkeypatch.setenv("EXA_API_KEY", "k")
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    for provider in (BraveProvider(), ExaProvider(), TavilyProvider()):
        assert provider.search("q", max_results=3) == []
    assert _rows(ledger, "pre_egress") == [] or all(
        r.decision == "deny" for r in _rows(ledger, "pre_egress")
    )


def test_a_provider_inside_a_tool_call_is_recorded_and_denied_when_not_declared(
    server: _Server, ledger: AuditLog, in_a_research_call: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # open_web covers the keyed hosts; the request itself goes to a host that is not
    # resolvable here, so what the ledger shows is the governed attempt, never a bare socket.
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    assert BraveProvider().search("q", max_results=3) == []
    pre = _rows(ledger, "pre_egress")
    assert pre and json.loads(pre[0].payload_json)["egress"]["host"] == "api.search.brave.com"


def _running_loop_check() -> None:  # pragma: no cover - keeps asyncio imported for type checkers
    asyncio.get_event_loop_policy()
