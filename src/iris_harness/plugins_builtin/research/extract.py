"""Content extraction for the research engine.

Crawls result URLs and fills :attr:`SearchResult.content` with clean markdown. Two
backends, selected by ``IRIS_RESEARCH_CRAWLER`` at call time:

* ``trafilatura`` (default) — a fetch through the governed HTTP client + Trafilatura
  extraction.
* ``crawl4ai`` (opt-in) — drives the async ``AsyncWebCrawler``; never a hard dep, so
  a missing/failing install degrades to ``None`` rather than raising.

Either way the work is synchronous from the caller's view (intended for a thread-pool
executor), so the async entry point offloads each page and never blocks the loop.

Failure is always local: one page that 404s, times out, or yields no main content
leaves its ``content`` as ``None`` and never aborts the batch.

Every URL here comes from a search result, which is to say from the internet, so it is
checked before it is fetched: only http/https, and only a host that resolves to public
addresses. Without the check a result pointing at ``file:///etc/passwd``,
``http://localhost:8003/`` or a cloud metadata address (169.254.169.254) was fetched from
inside the server and its text handed to the model.

The default backend fetches through the governed HTTP client (``iris_harness.sdk.http``,
issue #172): the client resolves the name once, refuses a loopback, private, link-local,
metadata or otherwise internal address, and connects to the address it checked (so a DNS
answer that changes between the check and the connect no longer matters), and every
request is checked against the plugin's ``egress`` declaration and recorded in the ledger.
The client does not follow redirects; this module does, one governed request per hop,
re-checking the scheme of each (an https page that redirects to http is refused),
capped at :data:`_MAX_REDIRECTS`. A fetch outside a
governed tool call is refused, never sent ungoverned. The Crawl4AI backend drives its own
browser and cannot be governed that way: its first URL is checked here with
:func:`_check_public_url` (which is why ``socket`` stays on the first-party debt list).
"""

from __future__ import annotations

import asyncio
import contextvars
import ipaddress
import logging
import os
import socket
from collections.abc import Callable
from urllib.parse import urljoin, urlparse

import trafilatura

from iris_harness.plugins_builtin.research.models import SearchResult
from iris_harness.sdk.http import EgressDenied, current_http
from iris_harness.sdk.logging import log_egress

logger = logging.getLogger(__name__)

_USER_AGENT = "Mozilla/5.0 (compatible; iris-research/0.1)"

# Hard cap on extracted content so a single verbose page can't blow the LLM context.
_MAX_CONTENT_CHARS = 6000
_TRUNCATION_MARKER = "… [truncated]"


def _cap(text: str) -> str | None:
    """Strip and length-cap extracted text. Returns ``None`` when nothing remains."""
    stripped = text.strip()
    if not stripped:
        return None
    if len(stripped) > _MAX_CONTENT_CHARS:
        keep = _MAX_CONTENT_CHARS - len(_TRUNCATION_MARKER)
        stripped = stripped[:keep].rstrip() + _TRUNCATION_MARKER
    return stripped


_ALLOWED_SCHEMES = frozenset({"http", "https"})
# Fewer than urllib's default of 10: a page that needs more hops than this is not worth
# the time, and each hop is another address to trust.
_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class UnsafeURLError(ValueError):
    """A URL the crawler refuses to fetch: wrong scheme, or a non-public address."""


def _resolve_host(host: str) -> list[str]:
    """Every address ``host`` resolves to. A seam so tests never touch real DNS."""
    return [str(info[4][0]) for info in socket.getaddrinfo(host, None)]


def _is_public(address: str) -> bool:
    """Whether ``address`` is a routable public address, not this box or its network."""
    ip = ipaddress.ip_address(address.split("%", 1)[0])  # drop an IPv6 zone id
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped  # ::ffff:127.0.0.1 is 127.0.0.1
    # ``is_global`` also rules out ranges none of the flags below name, such as the
    # carrier-grade NAT block (RFC 6598, 100.64/10) that tailnet addresses live in.
    return ip.is_global and not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _check_public_url(url: str) -> None:
    """Raise :exc:`UnsafeURLError` unless ``url`` is http(s) to a public host.

    Every address the host resolves to must be public: a name with one public and one
    private record could otherwise be steered to the private one. The fetch resolves
    the name again, so a DNS server that answers differently the second time is not
    covered (accepted: it needs an attacker-run DNS zone; this narrows the hole, and
    closing it means pinning the address into the connection)."""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UnsafeURLError(f"scheme {parsed.scheme!r} is not fetched")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("no host")
    try:
        addresses = _resolve_host(host)
    except OSError as exc:
        raise UnsafeURLError(f"cannot resolve {host}") from exc
    if not addresses:
        raise UnsafeURLError(f"{host} resolves to nothing")
    for address in addresses:
        if not _is_public(address):
            raise UnsafeURLError(f"{host} resolves to a non-public address")


def _check_scheme(url: str) -> None:
    """Raise :exc:`UnsafeURLError` unless ``url`` is http(s): a redirect to ``file://`` or
    ``ftp://`` is refused before any request is made."""
    scheme = urlparse(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise UnsafeURLError(f"scheme {scheme!r} is not fetched")


def _fetch(url: str, timeout: float) -> bytes | None:
    """The body of ``url`` through the governed client, following at most
    :data:`_MAX_REDIRECTS` redirects; ``None`` for an error status.

    Each hop is its own governed request: the client resolves, checks and pins the address
    (a redirect to loopback, a private range or the metadata address is refused there and
    recorded) and checks the host against the plugin's declaration. The scheme of every hop
    is checked here first. The body is read up to the client's size cap.
    """
    http = current_http()
    for _ in range(_MAX_REDIRECTS + 1):
        _check_scheme(url)
        response = http.get(url, headers={"User-Agent": _USER_AGENT}, timeout=timeout)
        if response.status_code in _REDIRECT_STATUSES:
            location = response.headers.get("location")
            if not location:
                raise UnsafeURLError("a redirect with no location")
            target = urljoin(url, location)
            if (
                urlparse(url).scheme.lower() == "https"
                and urlparse(target).scheme.lower() == "http"
            ):
                raise UnsafeURLError("a redirect from https to http is not followed")
            url = target
            continue
        if response.status_code >= 300:
            return None
        return response.content
    raise UnsafeURLError(f"more than {_MAX_REDIRECTS} redirects")


def _fetch_and_extract(url: str, timeout: float) -> str | None:
    """Fetch ``url`` and return its main content as trimmed markdown, or ``None``.

    Runs synchronously (intended for a thread-pool executor that carries the tool call's
    context, see :func:`extract_into`). The fetch is governed (:func:`_fetch`), then the raw
    HTML goes to Trafilatura. Returns ``None`` on any failure or when extraction yields
    nothing.
    """
    try:
        raw = _fetch(url, timeout)
        if raw is None:
            return None
        html = raw.decode("utf-8", errors="replace")
        markdown = trafilatura.extract(
            html,
            output_format="markdown",
            include_links=False,
            include_comments=False,
            favor_recall=True,
        )
    except EgressDenied as exc:
        logger.warning("content fetch not sent: %s", exc)
        return None
    except Exception:  # extraction must never raise into the batch
        logger.debug("content extraction failed for %s", url, exc_info=True)
        return None

    if not markdown:
        return None

    return _cap(markdown)


def _crawl4ai_extract(url: str, timeout: float) -> str | None:
    """Extract ``url`` via Crawl4AI's ``AsyncWebCrawler``, or ``None`` on any failure.

    Runs synchronously (intended for a thread-pool executor, where there is no running
    loop), so it drives the async crawler with :func:`asyncio.run`. Crawl4AI is never a
    hard dependency: a missing install, crawl error, or timeout degrades to ``None``.
    """

    async def _run() -> str | None:
        from crawl4ai import AsyncWebCrawler  # type: ignore[import-not-found]

        log_egress(
            destination=urlparse(url).netloc,
            method="GET",
            kind="crawl",
            purpose="crawl4ai",
        )
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
        markdown = getattr(result, "markdown", None)
        if not markdown:
            return None
        return _cap(str(markdown))

    try:
        # Crawl4AI drives a browser and follows redirects itself, so only the first
        # URL can be checked here; the redirect check covers the default backend.
        _check_public_url(url)
        return asyncio.run(_run())
    except Exception:  # crawl4ai missing/failure must never raise
        logger.debug("crawl4ai extraction failed for %s", url, exc_info=True)
        return None


def _select_extractor() -> Callable[[str, float], str | None]:
    """Pick the extraction backend from ``IRIS_RESEARCH_CRAWLER`` (read at call time)."""
    if os.environ.get("IRIS_RESEARCH_CRAWLER", "trafilatura").strip().lower() == "crawl4ai":
        return _crawl4ai_extract
    return _fetch_and_extract


async def extract_into(
    results: list[SearchResult], *, max_pages: int = 5, timeout: float = 8.0
) -> None:
    """Fetch + extract content for the first ``max_pages`` results concurrently.

    Mutates each targeted result in place, setting ``content`` to extracted markdown
    (or leaving it ``None`` on failure). Never raises: per-page failures are isolated
    via ``return_exceptions=True``.
    """
    targets = results[:max_pages]
    if not targets:
        return

    extractor = _select_extractor()
    loop = asyncio.get_running_loop()
    # The governed client acts only inside the tool call's scope, which lives in a context
    # variable that run_in_executor does not carry: give each worker its own copy of this
    # call's context (a Context can be entered by one thread at a time).
    tasks = [
        loop.run_in_executor(None, contextvars.copy_context().run, extractor, result.url, timeout)
        for result in targets
    ]
    extracted = await asyncio.gather(*tasks, return_exceptions=True)

    for result, outcome in zip(targets, extracted, strict=True):
        if isinstance(outcome, BaseException):
            logger.debug("content extraction task errored for %s: %r", result.url, outcome)
            continue
        result.content = outcome


def extract_into_sync(
    results: list[SearchResult], *, max_pages: int = 5, timeout: float = 8.0
) -> None:
    """Synchronous convenience wrapper around :func:`extract_into`.

    If no event loop is running, drives :func:`extract_into` via ``asyncio.run``. If a
    loop is already running (so we can't nest ``asyncio.run``), falls back to a simple
    sequential extraction in the current thread. Never raises.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop: safe to spin one up.
        try:
            asyncio.run(extract_into(results, max_pages=max_pages, timeout=timeout))
        except Exception:  # convenience wrapper must never raise
            logger.debug("extract_into_sync failed", exc_info=True)
        return

    # A loop is already running; do the blocking work sequentially as a fallback.
    extractor = _select_extractor()
    for result in results[:max_pages]:
        try:
            result.content = extractor(result.url, timeout)
        except Exception:  # convenience wrapper must never raise
            logger.debug("sequential extraction failed for %s", result.url, exc_info=True)


__all__ = ["extract_into", "extract_into_sync"]
