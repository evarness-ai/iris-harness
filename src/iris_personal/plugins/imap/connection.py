"""One IMAP session: connect, log in, run a few commands, log out -- and log the egress.

Stdlib ``imaplib`` only. Every session writes one ``EGRESS`` line (``log_egress``) naming
the host, the purpose and how the connect/login went -- never the username or password.

The reads never change the mailbox: folders are opened with ``EXAMINE`` (read-only) and
bodies fetched with ``BODY.PEEK``, so syncing does not mark mail read. Writes (keywords,
moves) open the folder with ``SELECT`` and happen only where the provider calls them,
behind its approval gate.
"""

from __future__ import annotations

import imaplib
import re
import ssl
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from iris_harness.sdk.logging import log_egress

from .account import ImapAccount

DEFAULT_TIMEOUT_S = 30.0

SslContextFactory = Callable[[], ssl.SSLContext]


class ImapError(RuntimeError):
    """A server or network failure, in words the owner can act on."""


class ImapAuthError(ImapError):
    """The server refused the login: a wrong or revoked app password."""


class ImapConnectionError(ImapError):
    """The server could not be reached, or TLS could not be set up."""


@dataclass(frozen=True)
class FolderInfo:
    name: str
    uidvalidity: int
    exists: int
    #: ``None`` when the server did not say (then assume keywords are allowed); else the
    #: flags a message in this folder keeps, ``"\\*"`` meaning "any new keyword".
    permanent_flags: frozenset[str] | None

    def accepts_keyword(self, keyword: str) -> bool:
        if self.permanent_flags is None:
            return True
        return "\\*" in self.permanent_flags or keyword in self.permanent_flags


@dataclass(frozen=True)
class FetchedItem:
    uid: int
    flags: tuple[str, ...]
    internaldate: datetime | None
    size: int | None
    body: bytes | None


# ─── Modified UTF-7 (RFC 3501 §5.1.3): folder names on the wire ──────────────


def encode_folder(name: str) -> str:
    """A folder name in IMAP's modified UTF-7 (``Gelöscht`` -> ``Gel&APY-scht``)."""
    out: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            import base64

            raw = "".join(pending).encode("utf-16-be")
            out.append("&" + base64.b64encode(raw).decode().rstrip("=").replace("/", ",") + "-")
            pending.clear()

    for ch in name:
        if 0x20 <= ord(ch) <= 0x7E:
            flush()
            out.append("&-" if ch == "&" else ch)
        else:
            pending.append(ch)
    flush()
    return "".join(out)


def decode_folder(name: str) -> str:
    import base64

    def _one(match: re.Match[str]) -> str:
        chunk = match.group(1)
        if not chunk:
            return "&"
        padded = chunk.replace(",", "/") + "=" * (-len(chunk) % 4)
        return base64.b64decode(padded).decode("utf-16-be")

    return re.sub(r"&([A-Za-z0-9+,]*)-", _one, name)


def quote(value: str) -> str:
    """An IMAP quoted string (``imaplib`` does not quote mailbox names for you)."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def uid_set(uids: Sequence[int]) -> str:
    return ",".join(str(u) for u in sorted(set(uids)))


# ─── Response parsing ─────────────────────────────────────────────────────────

_UID = re.compile(rb"\bUID (\d+)")
_FLAGS = re.compile(rb"\bFLAGS \(([^)]*)\)")
_SIZE = re.compile(rb"\bRFC822\.SIZE (\d+)")
_INTERNALDATE = re.compile(rb'\bINTERNALDATE "([^"]+)"')
_COPYUID = re.compile(r"COPYUID (\d+) ([\d:,]+) ([\d:,]+)")


def _parse_internaldate(raw: bytes) -> datetime | None:
    try:
        return datetime.strptime(raw.decode().strip(), "%d-%b-%Y %H:%M:%S %z")
    except ValueError:
        return None


def parse_fetch(data: Sequence[object]) -> list[FetchedItem]:
    """``imaplib``'s FETCH data (bytes, or ``(prefix, literal)`` tuples) -> items."""
    items: list[FetchedItem] = []
    for entry in data:
        if isinstance(entry, tuple):
            meta, body = bytes(entry[0]), bytes(entry[1])
        elif isinstance(entry, bytes) and _UID.search(entry):
            meta, body = entry, None
        else:
            continue  # the ``)`` that closes a literal-bearing item
        uid = _UID.search(meta)
        if uid is None:
            continue
        flags_m = _FLAGS.search(meta)
        size_m = _SIZE.search(meta)
        date_m = _INTERNALDATE.search(meta)
        items.append(
            FetchedItem(
                uid=int(uid.group(1)),
                flags=tuple(flags_m.group(1).decode().split()) if flags_m else (),
                internaldate=_parse_internaldate(date_m.group(1)) if date_m else None,
                size=int(size_m.group(1)) if size_m else None,
                body=body,
            )
        )
    return items


def _expand(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if ":" in part:
            lo, hi = (int(x) for x in part.split(":", 1))
            out.extend(range(min(lo, hi), max(lo, hi) + 1))
        else:
            out.append(int(part))
    return out


def parse_copyuid(texts: Sequence[str]) -> dict[int, int]:
    """``COPYUID <validity> <old set> <new set>`` (UIDPLUS) -> ``{old uid: new uid}``."""
    for text in texts:
        match = _COPYUID.search(text)
        if match:
            return dict(zip(_expand(match.group(2)), _expand(match.group(3)), strict=False))
    return {}


_LIST_LINE = re.compile(r'\((?P<attrs>[^)]*)\) (?:"(?:[^"\\]|\\.)*"|NIL) (?P<name>.+)$')


def parse_list(data: Sequence[object]) -> list[tuple[frozenset[str], str]]:
    """``LIST`` lines -> ``(attributes, decoded folder name)``."""
    out: list[tuple[frozenset[str], str]] = []
    for entry in data:
        if isinstance(entry, tuple):  # a literal name: ``(prefix, name)``
            line = bytes(entry[0]).decode(errors="replace")
            name = bytes(entry[1]).decode(errors="replace")
            match = _LIST_LINE.match(line.rsplit(" ", 1)[0] + " x")
            if match:
                out.append((frozenset(match.group("attrs").split()), decode_folder(name)))
            continue
        if not isinstance(entry, bytes):
            continue
        match = _LIST_LINE.match(entry.decode(errors="replace"))
        if not match:
            continue
        name = match.group("name").strip()
        if name.startswith('"') and name.endswith('"'):
            name = name[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        out.append((frozenset(match.group("attrs").split()), decode_folder(name)))
    return out


# ─── The session ──────────────────────────────────────────────────────────────


class ImapSession:
    """A logged-in connection. Thin, typed wrappers over the ``imaplib`` calls the
    provider makes; each raises :class:`ImapError` on a ``NO``/``BAD``."""

    def __init__(self, conn: imaplib.IMAP4, account: ImapAccount) -> None:
        self._conn = conn
        self.account = account
        self.selected: FolderInfo | None = None

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset(str(c).upper() for c in self._conn.capabilities)

    def _check(self, typ: str, data: Sequence[object], what: str) -> None:
        if typ != "OK":
            detail = b" ".join(d for d in data if isinstance(d, bytes)).decode(errors="replace")
            raise ImapError(f"IMAP {what} failed on {self.account.host}: {detail or typ}")

    def select(self, folder: str, *, readonly: bool) -> FolderInfo:
        """``EXAMINE`` (``readonly``) or ``SELECT`` a folder; its UIDVALIDITY and flags."""
        typ, data = self._conn.select(quote(encode_folder(folder)), readonly=readonly)
        self._check(typ, data, f"{'EXAMINE' if readonly else 'SELECT'} {folder}")
        _, validity = self._conn.response("UIDVALIDITY")
        _, permanent = self._conn.response("PERMANENTFLAGS")
        raw_validity = validity[0] if validity and validity[0] else b"0"
        flags: frozenset[str] | None = None
        if permanent and isinstance(permanent[0], bytes):
            flags = frozenset(permanent[0].decode().strip("()").split())
        exists = int(data[0]) if data and isinstance(data[0], bytes) else 0
        info = FolderInfo(
            name=folder,
            uidvalidity=int(raw_validity if isinstance(raw_validity, bytes) else b"0"),
            exists=exists,
            permanent_flags=flags,
        )
        self.selected = info
        return info

    def uid_search(self, *criteria: str) -> list[int]:
        typ, data = self._conn.uid("SEARCH", *criteria)
        self._check(typ, data, "UID SEARCH")
        raw = data[0] if data and isinstance(data[0], bytes) else b""
        return sorted(int(x) for x in raw.split())

    def uid_fetch(self, uids: Sequence[int], items: str) -> list[FetchedItem]:
        if not uids:
            return []
        typ, data = self._conn.uid("FETCH", uid_set(uids), items)
        self._check(typ, data, "UID FETCH")
        return parse_fetch(data)

    def uid_store(self, uids: Sequence[int], op: str, flags: Sequence[str]) -> None:
        if not uids or not flags:
            return
        typ, data = self._conn.uid("STORE", uid_set(uids), op, "(" + " ".join(flags) + ")")
        self._check(typ, data, "UID STORE")

    def uid_move(self, uids: Sequence[int], folder: str) -> dict[int, int]:
        """Move messages to ``folder``; ``{old uid: new uid}`` when the server says
        (UIDPLUS). ``MOVE`` (RFC 6851) when offered; else ``COPY`` + ``\\Deleted`` +
        ``UID EXPUNGE`` of exactly those UIDs, which needs UIDPLUS. A server with
        neither is refused: a plain ``EXPUNGE`` would also purge mail the owner had
        marked deleted themselves."""
        if not uids:
            return {}
        target = quote(encode_folder(folder))
        caps = self.capabilities
        if "MOVE" not in caps and "UIDPLUS" not in caps:
            raise PermissionError(
                f"{self.account.host} supports neither MOVE nor UIDPLUS, so IRIS cannot move "
                "a message without risking other mail marked deleted; nothing was moved"
            )
        self._conn.response("COPYUID")  # drop a stale one
        if "MOVE" in caps:
            typ, data = self._conn.uid("MOVE", uid_set(uids), target)
            self._check(typ, data, f"UID MOVE to {folder}")
            return self._copyuid()
        typ, data = self._conn.uid("COPY", uid_set(uids), target)
        self._check(typ, data, f"UID COPY to {folder}")
        mapping = self._copyuid()
        self.uid_store(uids, "+FLAGS.SILENT", ["\\Deleted"])
        typ, data = self._conn.uid("EXPUNGE", uid_set(uids))
        self._check(typ, data, "UID EXPUNGE")
        return mapping

    def _copyuid(self) -> dict[int, int]:
        _, untagged = self._conn.response("COPYUID")
        texts = [f"COPYUID {d.decode()}" for d in untagged or [] if isinstance(d, bytes)]
        return parse_copyuid(texts)

    def list_folders(self) -> list[tuple[frozenset[str], str]]:
        typ, data = self._conn.list('""', '"*"')
        self._check(typ, data, "LIST")
        return parse_list(data)


def _connect(account: ImapAccount, ssl_context: ssl.SSLContext, timeout: float) -> imaplib.IMAP4:
    if account.security == "ssl":
        return imaplib.IMAP4_SSL(
            account.host, account.port, ssl_context=ssl_context, timeout=timeout
        )
    conn = imaplib.IMAP4(account.host, account.port, timeout=timeout)
    if account.security == "starttls":
        try:
            conn.starttls(ssl_context=ssl_context)
        except BaseException:
            conn.shutdown()
            raise
    return conn


def _login(
    account: ImapAccount, factory: SslContextFactory, timeout: float, purpose: str
) -> imaplib.IMAP4:
    """Connect and log in, or raise; one ``EGRESS`` line either way, written once the
    login is decided (``status=ok|auth_failed|connect_failed``), with the host only."""
    status = "connect_failed"
    conn: imaplib.IMAP4 | None = None
    try:
        try:
            conn = _connect(account, factory(), timeout)
            status = "auth_failed"
            conn.login(account.username, account.password)
            status = "ok"
            return conn
        except imaplib.IMAP4.abort as exc:  # the connection dropped mid-login
            status = "connect_failed"
            raise ImapConnectionError(
                f"the IMAP server {account.host}:{account.port} dropped the connection "
                f"({exc.__class__.__name__})"
            ) from None
        except imaplib.IMAP4.error:
            if status == "auth_failed":
                # The server's words are dropped on purpose: some echo the username back.
                raise ImapAuthError(
                    f"{account.host} refused the login for {account.account_id}: check the "
                    f"app password (run `iris auth imap login --user {account.address}` to "
                    "replace it)"
                ) from None
            raise ImapConnectionError(
                f"could not reach the IMAP server {account.host}:{account.port} "
                f"({account.security}): the server refused the connection"
            ) from None
        except (OSError, ssl.SSLError) as exc:
            status = "connect_failed"
            raise ImapConnectionError(
                f"could not reach the IMAP server {account.host}:{account.port} "
                f"({account.security}): {exc.__class__.__name__}"
            ) from None
    finally:
        log_egress(
            destination=account.host,
            method="IMAP",
            purpose=purpose,
            kind="network",
            status=status,
            port=account.port,
        )
        if status != "ok" and conn is not None:
            try:
                conn.shutdown()
            except OSError:
                pass


@contextmanager
def open_session(
    account: ImapAccount,
    *,
    purpose: str,
    ssl_context_factory: SslContextFactory | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> Iterator[ImapSession]:
    """Connect and log in (one ``EGRESS`` line, host only); yield the session; log out."""
    conn = _login(account, ssl_context_factory or ssl.create_default_context, timeout, purpose)
    session = ImapSession(conn, account)
    try:
        yield session
    finally:
        # LOGOUT only, never CLOSE: CLOSE on a folder opened read-write expunges every
        # message marked Deleted, the owner's included.
        try:
            conn.logout()
        except (imaplib.IMAP4.error, OSError):
            pass


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "FetchedItem",
    "FolderInfo",
    "ImapAuthError",
    "ImapConnectionError",
    "ImapError",
    "ImapSession",
    "SslContextFactory",
    "decode_folder",
    "encode_folder",
    "open_session",
    "parse_copyuid",
    "parse_fetch",
    "parse_list",
    "quote",
    "uid_set",
]
