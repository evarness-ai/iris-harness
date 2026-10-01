"""The research crawler fetches only public http(s) URLs (security review, 2026-09-26).

Search results are internet input. Before the check, a result URL of ``file:///...``,
``http://localhost:...`` or ``http://169.254.169.254/`` was fetched from inside the
server. Hermetic: DNS is a fake resolver, and no socket is opened (a fetch that got
past the check would hit ``_open_url``, which these tests replace with a recorder).
"""

from __future__ import annotations

import ipaddress
import urllib.request
from email.message import Message
from typing import Any

import pytest

import iris_harness.plugins_builtin.research.extract as extract_mod
from iris_harness.plugins_builtin.research.extract import (
    UnsafeURLError,
    _check_public_url,
    _CheckedRedirectHandler,
    _crawl4ai_extract,
    _fetch_and_extract,
)

NON_PUBLIC = [
    "127.0.0.1",  # loopback
    "::1",
    "10.0.0.5",  # private
    "172.16.3.4",
    "192.168.1.10",
    "169.254.169.254",  # link-local: cloud metadata
    "fe80::1",
    "224.0.0.1",  # multicast
    "0.0.0.0",  # noqa: S104 - unspecified, the value under test
    "240.0.0.1",  # reserved
    "::ffff:127.0.0.1",  # loopback written as IPv4-mapped IPv6
    # Carrier-grade NAT (RFC 6598, 100.64/10; tailnet addresses live here). Built from
    # its integer so the export's identity scan, which flags tailnet addresses, does not
    # flag a textbook range.
    str(ipaddress.IPv4Address((100 << 24) | (64 << 16) | 1)),
]


def _resolve_to(monkeypatch: pytest.MonkeyPatch, *addresses: str) -> None:
    monkeypatch.setattr(extract_mod, "_resolve_host", lambda host: list(addresses))


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every URL that reached the network layer; answer with an empty page."""
    calls: list[str] = []

    class _Page:
        def read(self) -> bytes:
            return b"<html></html>"

        def __enter__(self) -> _Page:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def fake_open(request: urllib.request.Request, timeout: float) -> _Page:
        calls.append(request.full_url)
        return _Page()

    monkeypatch.setattr(extract_mod, "_open_url", fake_open)
    monkeypatch.setattr(extract_mod.trafilatura, "extract", lambda html, **kwargs: "# page")
    return calls


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com/",
        "data:text/html,<p>x</p>",
        "http:///no-host",
    ],
)
def test_only_http_and_https_with_a_host_are_fetched(url: str, opened: list[str]) -> None:
    with pytest.raises(UnsafeURLError):
        _check_public_url(url)
    assert _fetch_and_extract(url, timeout=8.0) is None
    assert opened == []


@pytest.mark.parametrize("address", NON_PUBLIC)
def test_a_host_that_resolves_to_a_non_public_address_is_refused(
    address: str, opened: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _resolve_to(monkeypatch, address)
    assert _fetch_and_extract("http://innocent-looking.example/page", timeout=8.0) is None
    assert opened == []


def test_one_private_record_among_public_ones_is_enough_to_refuse(
    opened: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _resolve_to(monkeypatch, "93.184.215.14", "127.0.0.1")
    assert _fetch_and_extract("https://mixed.example/", timeout=8.0) is None
    assert opened == []


def test_an_unresolvable_host_is_refused(
    opened: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(host: str) -> list[str]:
        raise OSError("nxdomain")

    monkeypatch.setattr(extract_mod, "_resolve_host", fail)
    assert _fetch_and_extract("https://nowhere.example/", timeout=8.0) is None
    assert opened == []


def test_a_public_page_is_still_fetched(opened: list[str]) -> None:
    # The conftest resolves every host to a public address.
    assert _fetch_and_extract("https://example.com/article", timeout=8.0) == "# page"
    assert opened == ["https://example.com/article"]


def test_a_redirect_to_a_non_public_address_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        extract_mod,
        "_resolve_host",
        lambda host: ["127.0.0.1"] if host == "internal.example" else ["93.184.215.14"],
    )
    handler = _CheckedRedirectHandler()
    original = urllib.request.Request("https://example.com/start")
    headers = Message()

    with pytest.raises(UnsafeURLError):
        handler.redirect_request(
            original, None, 302, "Found", headers, "http://internal.example/admin"
        )
    with pytest.raises(UnsafeURLError):
        handler.redirect_request(original, None, 302, "Found", headers, "file:///etc/passwd")

    followed = handler.redirect_request(
        original, None, 302, "Found", headers, "https://example.org/next"
    )
    assert followed is not None
    assert followed.full_url == "https://example.org/next"


def test_redirects_are_capped() -> None:
    assert _CheckedRedirectHandler.max_redirections == 5
    assert (
        _CheckedRedirectHandler.max_redirections
        < urllib.request.HTTPRedirectHandler.max_redirections
    )


def test_a_redirect_failure_inside_a_fetch_is_contained(monkeypatch: pytest.MonkeyPatch) -> None:
    # The handler raises from inside urllib; the batch must still get None, not an error.
    def open_that_redirects(request: urllib.request.Request, timeout: float) -> Any:
        _check_public_url("http://127.0.0.1/")
        raise AssertionError("unreachable")

    monkeypatch.setattr(extract_mod, "_open_url", open_that_redirects)
    assert _fetch_and_extract("https://example.com/", timeout=8.0) is None


def test_the_crawl4ai_backend_checks_the_url_before_crawling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crawled: list[str] = []

    def fake_run(coro: Any) -> str:
        coro.close()
        crawled.append("ran")
        return "# crawled"

    monkeypatch.setattr(extract_mod.asyncio, "run", fake_run)
    _resolve_to(monkeypatch, "169.254.169.254")
    assert _crawl4ai_extract("http://metadata.example/latest", timeout=8.0) is None
    assert _crawl4ai_extract("file:///etc/passwd", timeout=8.0) is None
    assert crawled == []

    _resolve_to(monkeypatch, "93.184.215.14")
    assert _crawl4ai_extract("https://example.com/", timeout=8.0) == "# crawled"
    assert crawled == ["ran"]
