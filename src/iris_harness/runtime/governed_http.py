"""The governed HTTP client behind ``iris_harness.sdk.http`` (issue #103).

Lives in ``runtime`` rather than ``sdk`` because ``PluginAPI.http`` (in the plugin host,
which ``sdk/__init__`` re-exports) constructs it, and the SDK module re-exports from here
the way ``sdk/logging`` re-exports from ``foundation``. See ``sdk/http.py`` for what a plugin
author reads, and docs/architecture/plugin-egress.md for the design.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
import uuid
import zlib
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, NamedTuple

import httpx

from iris_harness.foundation.ids import new_ulid
from iris_harness.foundation.observability.logging_setup import log_egress
from iris_harness.foundation.process_state import track_globals
from iris_harness.kernel.governance.hooks.tool_payload import CALL_ID, PARENT_CALL_ID
from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.plugin_egress import (
    DEFAULT_RESPONSE_BYTES,
    current_egress_scope,
    egress_kernel,
    egress_policy,
    normalize_host,
)
from iris_harness.kernel.governance.plugins.plugin_egress import PluginEgressHook
from iris_harness.runtime.egress_transport import (
    BlockedAddress,
    DeadlineExceeded,
    PinnedBackend,
    PinnedTransport,
)

#: The whole request's wall-clock budget in seconds (connect, send, headers, body), unless the
#: caller asks for another; a caller's value is clamped to ``_MAX_TIMEOUT``.
_DEFAULT_TIMEOUT = 10.0
_MAX_TIMEOUT = 60.0
#: The most DECODED body bytes a response may have, unless the plugin's manifest declares
#: another (``egress.max_response_bytes``, at most 64 MiB; issue #175).
MAX_RESPONSE_BYTES = DEFAULT_RESPONSE_BYTES


def _too_big(cap: int) -> str:
    size = f"{cap // (1024 * 1024)} MiB" if cap >= 1024 * 1024 else f"{cap} bytes"
    return f"the response is larger than {size}; it was cut off"


_TOO_SLOW = "the request did not finish within its time limit; it was cut off"
_NOT_VALID = "the URL is not valid"
_BAD_HEADER = (
    "a Host, Proxy-*, Connection, Upgrade, TE, Transfer-Encoding or Content-Length header "
    "is not sent"
)
#: Headers the client alone sets: a caller's value is refused, in whatever spelling.
_FORBIDDEN_HEADERS = frozenset(
    {"host", "connection", "upgrade", "te", "transfer-encoding", "content-length"}
)
_BAD_ENCODING = "the response uses a content encoding that is not accepted"
_BAD_BODY = "the response body could not be decoded"
#: How much a bounded decompressor may produce per step (and so the most memory one step adds).
_DECODE_STEP = 64 * 1024
# zlib window bits per accepted Content-Encoding; None = the bytes are the body.
_ENCODINGS: dict[str, int | None] = {
    "": None,
    "identity": None,
    "gzip": 16 + zlib.MAX_WBITS,
    "x-gzip": 16 + zlib.MAX_WBITS,
    "deflate": zlib.MAX_WBITS,
}
_NO_SCOPE = "no governed tool or capability call is running, so a request has no run to be held to"
_OTHER_PLUGIN = "this client belongs to a different plugin than the tool being run"
# What the body no longer matches once it is decoded and complete.
_FRAMING = frozenset({"content-encoding", "content-length", "transfer-encoding"})
_NO_KERNEL = "no governance kernel is bound, so governed requests fail closed"
_NO_HOOK = "the governance kernel has no plugin_egress hook, so governed requests fail closed"

# The transport a test installs (``testing.fake_http``); None in production.
_transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None


class EgressDenied(RuntimeError):
    """A governed request was refused: the host is not declared, or governance said no.

    ``host`` is the host the request was for; the message says why. Nothing was sent (or, for
    a response cut off at its size or time limit, the rest was not read).

    Deliberately not an ``OSError`` (it was a ``PermissionError`` before release): a plugin's
    ``except OSError`` around its network code would otherwise swallow a governance denial as
    if it were a connection failure.
    """

    def __init__(self, message: str, *, host: str = "") -> None:
        super().__init__(message)
        self.host = host


@contextmanager
def _use_transport(transport: httpx.BaseTransport | httpx.AsyncBaseTransport) -> Iterator[None]:
    """Replace the network transport (and only it) inside the block. For ``testing``."""
    global _transport
    previous = _transport
    _transport = transport
    try:
        yield
    finally:
        _transport = previous


def _leaves(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, list | tuple | set | frozenset):
        for item in value:
            yield from _leaves(item)
    elif isinstance(value, int | float) and not isinstance(value, bool):
        yield str(value)


class GovernedHttp:
    """HTTP for one plugin, through governance. Get it from ``api.http``."""

    def __init__(self, plugin: str, *, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._plugin = plugin
        self._timeout = _seconds(timeout, _DEFAULT_TIMEOUT)

    @property
    def plugin(self) -> str:
        """The plugin this client acts for (the harness's stamp)."""
        return self._plugin

    # -- the verbs ----------------------------------------------------------------------
    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        data: Mapping[str, Any] | None = None,
        content: bytes | str | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """Send one request. Raises :class:`EgressDenied` when it is not allowed.

        ``timeout`` is the whole request's budget in seconds (not per operation); ``None``,
        zero or a negative value mean the client's default and anything above 60 is 60. The
        body is read up to 10 MiB decoded; past that, or past the time limit, the request is
        cut off, recorded, and raises :class:`EgressDenied`. The returned response holds the
        decoded body (``Content-Encoding`` and length framing headers are dropped).
        """
        request, egress, ids = self._prepare(method, url, params, headers, json, data, content)
        ctx = self._context(
            HookPoint.PRE_EGRESS, egress, ids, _content_strings(request, json, data)
        )
        request = self._pre(_fire_sync(_kernel(), HookPoint.PRE_EGRESS, ctx), egress, request)
        started = time.monotonic()
        read = _Read()
        try:
            response = self._execute(
                request, _seconds(timeout, self._timeout), read, egress, self._response_cap()
            )
        except Exception as exc:
            self._post_sync(egress, ids, started, request, None, exc, read)
            raise
        self._post_sync(egress, ids, started, request, response, None, read)
        return response

    async def arequest(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        data: Mapping[str, Any] | None = None,
        content: bytes | str | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """:meth:`request` for async code (the transfer runs on a worker thread)."""
        request, egress, ids = self._prepare(method, url, params, headers, json, data, content)
        ctx = self._context(
            HookPoint.PRE_EGRESS, egress, ids, _content_strings(request, json, data)
        )
        kernel = _kernel()
        decision, _ = await kernel.fire(HookPoint.PRE_EGRESS, ctx)
        request = self._pre(decision, egress, request)
        started = time.monotonic()
        read = _Read()
        try:
            response = await asyncio.to_thread(
                self._execute,
                request,
                _seconds(timeout, self._timeout),
                read,
                egress,
                self._response_cap(),
            )
        except Exception as exc:
            await kernel.fire(
                HookPoint.POST_EGRESS,
                self._outcome(egress, ids, started, request, None, exc, read),
            )
            _log(egress, None, exc)
            raise
        await kernel.fire(
            HookPoint.POST_EGRESS,
            self._outcome(egress, ids, started, request, response, None, read),
        )
        _log(egress, response, None)
        return response

    def _response_cap(self) -> int:
        """The cap for this plugin, read from the compiled policy each time. It is passed to the
        transfer as an argument and never read back from the ``egress`` dict, which hooks see:
        the dict is the record of the request, not an input to it."""
        policy = egress_policy()
        return policy.response_cap(self._plugin) if policy is not None else MAX_RESPONSE_BYTES

    @staticmethod
    def _execute(
        request: httpx.Request,
        seconds: float,
        read: _Read,
        egress: dict[str, Any],
        cap: int = MAX_RESPONSE_BYTES,
    ) -> httpx.Response:
        """The transfer: pinned and checked connect, a total deadline, a decoded-size cap."""
        host = str(egress["host"])
        deadline = time.monotonic() + seconds
        backend: PinnedBackend | None = None
        transport: httpx.BaseTransport
        if isinstance(_transport, httpx.BaseTransport):
            transport = _transport  # a test's fake: no socket, no resolution
        else:
            backend = PinnedBackend(seconds)
            transport = PinnedTransport(backend)
        chunks: list[bytes] = []
        try:
            with httpx.Client(
                transport=transport,
                timeout=seconds,
                follow_redirects=False,
                trust_env=False,  # no proxy, netrc or CA bundle from the environment
            ) as client:
                reply = client.send(request, stream=True)
                try:
                    for chunk in _decoded(reply, host, egress):
                        read.bytes_in += len(chunk)
                        if read.bytes_in > cap:
                            egress["aborted"] = "max_bytes"
                            raise EgressDenied(_too_big(cap), host=host)
                        if time.monotonic() > deadline:
                            egress["aborted"] = "deadline"
                            raise EgressDenied(_TOO_SLOW, host=host)
                        chunks.append(chunk)
                finally:
                    reply.close()
        except BlockedAddress as exc:
            egress["aborted"] = "address"
            raise EgressDenied(f"the request was not sent: {exc}", host=host) from None
        except (DeadlineExceeded, httpx.TimeoutException):
            egress["aborted"] = "deadline"
            raise EgressDenied(_TOO_SLOW, host=host) from None
        extensions = {
            k: v for k, v in reply.extensions.items() if k in ("http_version", "reason_phrase")
        }
        return httpx.Response(
            reply.status_code,
            headers=[(k, v) for k, v in reply.headers.multi_items() if k.lower() not in _FRAMING],
            content=b"".join(chunks),
            request=request,
            extensions=extensions,
        )

    # -- the governed parts -------------------------------------------------------------
    def _prepare(
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        json: Any,
        data: Mapping[str, Any] | None,
        content: bytes | str | None,
    ) -> tuple[httpx.Request | None, dict[str, Any], _Ids]:
        request: httpx.Request | None
        try:
            request = httpx.Request(
                method.upper(),
                url,
                params=params,
                headers=headers,
                json=json,
                data=data,
                content=content,
            )
        except (httpx.InvalidURL, UnicodeError):
            request = None  # recorded and denied below; the message never echoes the URL
        scope = current_egress_scope()
        # The request's identity (#134 stage 1). Its own id is a kernel-quality ULID minted
        # here with the same minter as the runner's, never a caller argument, shared by its
        # PRE and POST rows; its parent is the calling tool's runner-minted id, read from
        # the harness's call scope (None for a request made outside a governed call). Both
        # travel in the hook context's METADATA: the kernel's audit write stamps them onto
        # every row from there, so the payload (and the plugin) never names them.
        ids = _Ids(new_ulid(), scope.tool_call_id if scope else None)
        if request is None:
            egress: dict[str, Any] = {
                "plugin": self._plugin,
                "scheme": "",
                "host": "",
                "port": 0,
                "method": str(method).upper()[:16],
                "malformed": _NOT_VALID,
            }
            egress["attempt"], egress["replay_of"] = _attempt(scope, egress, ids.call_id)
            return None, egress, ids
        target = request.url
        host = target.host
        port = target.port or (443 if target.scheme == "https" else 80)
        bad_host = ""
        try:
            shown = normalize_host(host) if host else ""
        except ValueError:
            # Not a host name (an IPv6 literal, stray whitespace, repeated dots): still a
            # record, denied below. Shown ASCII-escaped and bounded, never trusted.
            shown = host.encode("ascii", "backslashreplace").decode()[:253].lower()
            bad_host = "the host is not a valid host name"
        # Only what addresses the host: no path, query, header or body (the ledger's rule).
        egress = {
            "plugin": self._plugin,
            "scheme": target.scheme,
            "host": shown,
            "port": port,
            "method": request.method,
        }
        egress["attempt"], egress["replay_of"] = _attempt(scope, egress, ids.call_id)
        cap = self._response_cap()
        if cap != MAX_RESPONSE_BYTES:
            egress["cap"] = cap  # recorded on the pre row: a plugin that asked for another size
        # A request that cannot be addressed is still a record: the hook denies it, so a
        # refusal is a ledger row and never silence.
        if scope is None or not scope.tool:
            # A request outside the harness's call scope (a thread the tool started, a call
            # made at import) has no run, no data class and no parent call to be held to:
            # not sent. The harness stamps the scope; a plugin cannot supply one.
            egress["malformed"] = _NO_SCOPE
        elif scope.tool_plugin != self._plugin:
            # The client acts for the plugin whose tool is running, whatever name it was
            # constructed with: a plugin cannot borrow another's declaration.
            egress["malformed"] = _OTHER_PLUGIN
        elif not host or target.scheme not in ("http", "https"):
            egress["malformed"] = f"not an http(s) URL with a host ({target.scheme or 'no scheme'})"
        elif bad_host:
            egress["malformed"] = bad_host
        elif target.userinfo:
            # Credentials in the address would be sent where the ledger cannot see them.
            egress["malformed"] = "a URL with credentials in it is not sent"
        elif _bad_headers(headers):
            # A Host header names a different site than the one the policy checked
            # (domain fronting); Proxy-* headers have no meaning on a direct request; the
            # framing and connection headers are the client's own.
            egress["malformed"] = _BAD_HEADER
        else:
            # Only the identity encoding is asked for: the body is decoded here, bounded.
            request.headers["Accept-Encoding"] = "identity"
        return request, egress, ids

    def _context(
        self,
        point: HookPoint,
        egress: dict[str, Any],
        ids: _Ids,
        content: list[str] | None = None,
    ) -> HookContext:
        scope = current_egress_scope()
        payload: dict[str, Any] = {
            # Who owns the running tool: the harness's stamp (the scope), so a client built
            # with another plugin's name is still attributed to the plugin actually running
            # (``egress.plugin``, below, is the name the client was built with).
            "tool_plugin": (scope.tool_plugin if scope and scope.tool_plugin else self._plugin),
            "egress": egress,
        }
        if scope is not None and scope.tool:
            payload["tool_name"] = scope.tool
        if content is not None:
            payload["egress_content"] = content  # read by the PII guards, never audited
        return HookContext(
            hook_point=point,
            run_id=(scope.run_id if scope and scope.run_id else f"egress-{uuid.uuid4().hex[:12]}"),
            agent_type=scope.agent_type if scope else "plugin",
            step_id=scope.step_id if scope else None,
            route=f"egress/{egress['host']}",
            classification=scope.classification if scope else None,
            payload=payload,
            metadata={
                "caller": (scope.caller if scope and scope.caller else None)
                or f"plugin:{self._plugin}",
                CALL_ID: ids.call_id,
                PARENT_CALL_ID: ids.parent_call_id,
            },
        )

    @staticmethod
    def _pre(
        decision: HookDecision, egress: dict[str, Any], request: httpx.Request | None
    ) -> httpx.Request:
        if decision.outcome != "allow":
            raise EgressDenied(decision.reason, host=str(egress["host"]))
        if request is None:  # cannot happen (the hook denies it); fail closed regardless
            raise EgressDenied(f"plugin_egress: {_NOT_VALID}")
        return request

    def _outcome(
        self,
        egress: dict[str, Any],
        ids: _Ids,
        started: float,
        request: httpx.Request,
        response: httpx.Response | None,
        error: BaseException | None,
        read: _Read,
    ) -> HookContext:
        done = dict(egress)
        done["duration_ms"] = round((time.monotonic() - started) * 1000)
        done["bytes_out"] = len(request.content) if request.content else 0
        if response is not None:
            done["status"] = response.status_code
        # Decoded body bytes actually read, also when the transfer was cut off.
        done["bytes_in"] = read.bytes_in
        if error is not None:
            done["error"] = type(error).__name__  # the class only: a message can quote a URL
        return self._context(HookPoint.POST_EGRESS, done, ids)

    def _post_sync(
        self,
        egress: dict[str, Any],
        ids: _Ids,
        started: float,
        request: httpx.Request,
        response: httpx.Response | None,
        error: BaseException | None,
        read: _Read,
    ) -> None:
        _fire_sync(
            _kernel(),
            HookPoint.POST_EGRESS,
            self._outcome(egress, ids, started, request, response, error, read),
        )
        _log(egress, response, error)


def _decoded(reply: httpx.Response, host: str, egress: dict[str, Any]) -> Iterator[bytes]:
    """The response body, decoded here with a bounded decompressor, never by httpx.

    httpx decodes a whole wire chunk in one call, so a small compressed body (layered gzip,
    zstd, brotli, a deep gzip) could expand to gigabytes inside one step, before any size
    check. So the request asks for ``identity``; a server that compresses anyway is accepted
    only for a single ``gzip`` or ``deflate`` layer, decoded at most ``_DECODE_STEP`` bytes at
    a time; any other encoding is refused before its body is read.
    """
    if reply.is_stream_consumed:  # a test's fake response: already complete, no socket
        yield reply.content
        return
    label = reply.headers.get("content-encoding", "").strip().lower()
    if label not in _ENCODINGS:
        egress["aborted"] = "encoding"
        raise EgressDenied(_BAD_ENCODING, host=host)
    wbits = _ENCODINGS[label]
    if wbits is None:
        yield from reply.iter_raw()
        return
    inflater = zlib.decompressobj(wbits)
    first = True
    try:
        for raw in reply.iter_raw():
            if first and label == "deflate":
                first = False
                try:  # some servers send raw deflate without the zlib wrapper
                    probe = zlib.decompressobj(wbits)
                    probe.decompress(raw, 1)
                except zlib.error:
                    inflater = zlib.decompressobj(-zlib.MAX_WBITS)
            data = raw
            while data:
                piece = inflater.decompress(data, _DECODE_STEP)
                if piece:
                    yield piece
                data = inflater.unconsumed_tail
        tail = inflater.flush()
        if tail:
            yield tail
    except zlib.error:
        egress["aborted"] = "decode"
        raise EgressDenied(_BAD_BODY, host=host) from None


class _Read:
    """Decoded response bytes read so far (also when the transfer is cut off)."""

    def __init__(self) -> None:
        self.bytes_in = 0


def _bad_headers(headers: Any) -> bool:
    """Did the caller supply a header the client alone may set? Judged on the normalised
    names (dict, list of pairs, tuples, bytes keys, any case), not on the caller's spelling;
    headers that cannot be normalised are refused."""
    if not headers:
        return False
    try:
        names = {str(name).lower() for name in httpx.Headers(headers).keys()}
    except Exception:  # noqa: BLE001 - whatever httpx cannot read is not sent
        return True
    return bool(names & _FORBIDDEN_HEADERS) or any(n.startswith("proxy-") for n in names)


def _seconds(value: float | None, default: float) -> float:
    """A request's total time budget: the default for None, zero, negative or not-a-number,
    never more than ``_MAX_TIMEOUT``."""
    if isinstance(value, bool):
        return default  # True is not "1 second"
    try:
        seconds = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return default if not seconds > 0 else min(seconds, _MAX_TIMEOUT)


class _Ids(NamedTuple):
    """One request's identity: its own id and the governed call it was made inside."""

    call_id: str
    parent_call_id: str | None


def _attempt(scope: Any, egress: dict[str, Any], call_id: str) -> tuple[int, str | None]:
    """``(attempt, replay_of)``: a repeat of the same method and host inside one governed
    call is attempt 2, 3, ... and names the first request's ``call_id``."""
    if scope is None:
        return 1, None
    key = (egress["method"], egress["host"], egress["port"])
    first = scope.attempts.setdefault(key, [call_id, 0])
    first[1] += 1
    return first[1], (None if first[1] == 1 else first[0])


def _content_strings(request: httpx.Request | None, json: Any, data: Any) -> list[str]:
    """What the request carries, for the PII guards: path segments, query values, body fields."""
    if request is None:
        return []
    out = [seg for seg in request.url.path.split("/") if seg]
    out.extend(v for _, v in request.url.params.multi_items())
    out.extend(_leaves(json))
    out.extend(_leaves(data))
    return out


def _kernel() -> Any:
    kernel = egress_kernel()
    if kernel is None:
        raise EgressDenied(_NO_KERNEL)
    # A kernel that never registered the egress hook would allow everything ("no hooks
    # registered"); a request whose host nobody checked is not sent.
    if PluginEgressHook.name not in kernel.hook_names(HookPoint.PRE_EGRESS):
        raise EgressDenied(_NO_HOOK)
    return kernel


def _fire_sync(kernel: Any, point: HookPoint, ctx: HookContext) -> HookDecision:
    """``kernel.fire_sync``, also from inside a running loop (on a worker thread)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return kernel.fire_sync(point, ctx)[0]  # type: ignore[no-any-return]
    with ThreadPoolExecutor(max_workers=1) as pool:
        run = contextvars.copy_context().run
        return pool.submit(run, kernel.fire_sync, point, ctx).result()[0]  # type: ignore[no-any-return]


def _log(
    egress: dict[str, Any], response: httpx.Response | None, error: BaseException | None
) -> None:
    """The ``iris.egress`` log line, beside the ledger rows: one trail for outbound calls."""
    log_egress(
        destination=f"{egress['host']}:{egress['port']}",
        method=str(egress["method"]),
        kind="plugin",
        purpose=str(egress["plugin"]),
        status=response.status_code if response is not None else type(error).__name__,
    )


def current_http() -> GovernedHttp:
    """The governed client for the plugin whose tool is running, for a declarative plugin.

    A ``flavor: declarative`` plugin has no ``setup(api)``, so its tool function asks for the
    client here. The plugin is the one the harness is running the tool for; called outside a
    governed tool call it raises :class:`EgressDenied` (there is no plugin to act for).
    """
    scope = current_egress_scope()
    if scope is None or not scope.tool_plugin:
        raise EgressDenied("current_http() is only available while a plugin's tool is running")
    return GovernedHttp(scope.tool_plugin)


__all__ = ["EgressDenied", "GovernedHttp", "current_http"]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_transport")
