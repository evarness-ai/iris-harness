"""The research crawler fetches only public http(s) URLs (security review, 2026-09-26).

Search results are internet input. Before the check, a result URL of ``file:///...``,
``http://localhost:...`` or ``http://169.254.169.254/`` was fetched from inside the
server. Hermetic: DNS is a fake resolver and no socket is opened.

The default backend fetches through the governed client (issue #172), whose address checks,
redirect handling and ledger rows are tested on the real transport in
``test_governed_egress.py``. What is tested here is what stays in this module: the scheme
check every hop goes through, and ``_check_public_url``, the guard the Crawl4AI backend (which
drives its own browser and cannot be governed that way) keeps.
"""

from __future__ import annotations

import ipaddress
from typing import Any

import pytest

import iris_harness.plugins_builtin.research.extract as extract_mod
from iris_harness.plugins_builtin.research.extract import (
    UnsafeURLError,
    _check_public_url,
    _check_scheme,
    _crawl4ai_extract,
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


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com/",
        "data:text/html,<p>x</p>",
    ],
)
def test_only_http_and_https_are_fetched(url: str) -> None:
    with pytest.raises(UnsafeURLError):
        _check_scheme(url)
    with pytest.raises(UnsafeURLError):
        _check_public_url(url)


def test_http_and_https_pass_the_scheme_check() -> None:
    _check_scheme("http://example.com/")
    _check_scheme("HTTPS://example.com/")


def test_a_url_with_no_host_is_refused_by_the_address_check() -> None:
    with pytest.raises(UnsafeURLError):
        _check_public_url("http:///no-host")


@pytest.mark.parametrize("address", NON_PUBLIC)
def test_a_host_that_resolves_to_a_non_public_address_is_refused(
    address: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _resolve_to(monkeypatch, address)
    with pytest.raises(UnsafeURLError):
        _check_public_url("http://innocent-looking.example/page")


def test_one_private_record_among_public_ones_is_enough_to_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resolve_to(monkeypatch, "93.184.215.14", "127.0.0.1")
    with pytest.raises(UnsafeURLError):
        _check_public_url("https://mixed.example/")


def test_an_unresolvable_host_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(host: str) -> list[str]:
        raise OSError("nxdomain")

    monkeypatch.setattr(extract_mod, "_resolve_host", fail)
    with pytest.raises(UnsafeURLError):
        _check_public_url("https://nowhere.example/")


def test_a_public_host_passes_the_address_check() -> None:
    # The conftest resolves every host to a public address.
    _check_public_url("https://example.com/article")


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
