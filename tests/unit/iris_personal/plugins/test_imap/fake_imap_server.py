"""An in-process IMAP4rev1 server for the IMAP provider's tests (L1 exit: "IMAP against a
test server").

Pure Python, threaded, on 127.0.0.1 with an ephemeral port; TLS optional (implicit TLS
like port 993, or STARTTLS like port 143) with a certificate the test makes. It speaks
the subset ``ImapProvider`` uses, the way RFC 3501 / 4315 (UIDPLUS) / 6851 (MOVE) say:

    CAPABILITY  NOOP  LOGIN  STARTTLS  LOGOUT  LIST  CREATE  SELECT  EXAMINE  CLOSE
    UID SEARCH  (ALL, UID <set>, SINCE, HEADER, KEYWORD, TEXT, SUBJECT, CHARSET)
    UID FETCH   (UID FLAGS INTERNALDATE RFC822.SIZE BODY[] BODY.PEEK[] BODY.PEEK[HEADER])
    UID STORE   (+FLAGS / -FLAGS / FLAGS, .SILENT)
    UID COPY / UID MOVE (COPYUID) / UID EXPUNGE

``FakeMailbox`` is the server's state, shared by every connection and readable from the
test (flags, folders, what was written). ``commands`` records each command name in
order, so a test can say "no STORE ever reached the server". A non-PEEK ``BODY[]``
sets ``\\Seen`` the way a real server does, which is how the tests prove a sync never
marks mail read. All data is synthetic.
"""

from __future__ import annotations

import re
import socket
import socketserver
import ssl
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

SYSTEM_FLAGS = ("\\Seen", "\\Answered", "\\Flagged", "\\Deleted", "\\Draft")


@dataclass
class FakeMessage:
    uid: int
    raw: bytes
    flags: set[str]
    internaldate: datetime


@dataclass
class FakeFolder:
    name: str
    uidvalidity: int
    attrs: set[str] = field(default_factory=set)
    uidnext: int = 1
    messages: dict[int, FakeMessage] = field(default_factory=dict)

    def uids(self) -> list[int]:
        return sorted(self.messages)


class FakeMailbox:
    """Server-side state: users, folders, messages. Thread-safe."""

    def __init__(
        self,
        *,
        users: dict[str, str] | None = None,
        keywords_allowed: bool = True,
        capabilities: Iterable[str] = ("UIDPLUS", "MOVE"),
    ) -> None:
        self.users = dict(users or {})
        self.keywords_allowed = keywords_allowed
        self.capabilities = set(capabilities)
        self.lock = threading.RLock()
        self._validity = 1000
        self.folders: dict[str, FakeFolder] = {}
        self.commands: list[str] = []
        self.logins: list[str] = []
        self.add_folder("INBOX")
        self.add_folder("Trash", attrs={"\\Trash", "\\HasNoChildren"})

    def add_folder(self, name: str, *, attrs: Iterable[str] = ("\\HasNoChildren",)) -> FakeFolder:
        with self.lock:
            self._validity += 1
            folder = FakeFolder(name=name, uidvalidity=self._validity, attrs=set(attrs))
            self.folders[name] = folder
            return folder

    def add_message(
        self,
        raw: bytes,
        *,
        folder: str = "INBOX",
        flags: Iterable[str] = (),
        internaldate: datetime | None = None,
    ) -> int:
        with self.lock:
            f = self.folders[folder]
            uid = f.uidnext
            f.uidnext += 1
            f.messages[uid] = FakeMessage(
                uid=uid,
                raw=raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"),
                flags=set(flags),
                internaldate=internaldate or datetime.now(UTC),
            )
            return uid

    def reset_uidvalidity(self, folder: str = "INBOX") -> None:
        """What a server does after a rebuild: new UIDVALIDITY, every UID renumbered."""
        with self.lock:
            f = self.folders[folder]
            self._validity += 1
            f.uidvalidity = self._validity
            old = [f.messages[u] for u in f.uids()]
            f.messages = {}
            f.uidnext = 1
            for message in old:
                message.uid = f.uidnext + 100  # a different numbering than before
                f.messages[message.uid] = message
                f.uidnext = message.uid + 1

    def flags_of(self, folder: str, uid: int) -> set[str]:
        with self.lock:
            return set(self.folders[folder].messages[uid].flags)

    def find(self, needle: bytes, folder: str = "INBOX") -> int | None:
        with self.lock:
            for uid, message in self.folders[folder].messages.items():
                if needle in message.raw:
                    return uid
        return None


# ─── Protocol ────────────────────────────────────────────────────────────────

_TOKEN = re.compile(
    rb'\s*(?:"((?:[^"\\]|\\.)*)"|(\()|(\))|([^\s()"]+(?:\[[^\]]*\])?(?:<[\d.]+>)?))'
)


def _tokens(data: bytes) -> list[Any]:
    """Parse a command line (literals already inlined as ``Lit`` objects) into
    strings and nested lists."""
    stack: list[list[Any]] = [[]]
    pos = 0
    while pos < len(data):
        if data[pos : pos + 1].isspace():
            pos += 1
            continue
        match = _TOKEN.match(data, pos)
        if match is None:
            raise ValueError(f"cannot parse {data[pos:]!r}")
        pos = match.end()
        quoted, open_, close, atom = match.groups()
        if quoted is not None:
            stack[-1].append(re.sub(rb"\\(.)", rb"\1", quoted).decode())
        elif open_:
            stack.append([])
        elif close:
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(atom.decode())
    return stack[0]


def _uid_set(spec: str, uids: list[int]) -> list[int]:
    top = uids[-1] if uids else 0
    wanted: set[int] = set()
    for part in spec.split(","):
        if ":" in part:
            lo_s, hi_s = part.split(":", 1)
            lo = top if lo_s == "*" else int(lo_s)
            hi = top if hi_s == "*" else int(hi_s)
            lo, hi = min(lo, hi), max(lo, hi)
            wanted.update(u for u in uids if lo <= u <= hi)
        else:
            wanted.update(u for u in uids if u == (top if part == "*" else int(part)))
    return sorted(wanted)


def _imap_date(when: datetime) -> str:
    return when.strftime("%d-%b-%Y %H:%M:%S %z")


def _header_value(raw: bytes, name: str) -> str:
    head = raw.split(b"\r\n\r\n", 1)[0].decode(errors="replace")
    head = re.sub(r"\r\n[ \t]+", " ", head)
    for line in head.split("\r\n"):
        key, _, value = line.partition(":")
        if key.strip().lower() == name.lower():
            return value.strip()
    return ""


class _Handler(socketserver.BaseRequestHandler):
    server: _Server

    def setup(self) -> None:
        self.sock: socket.socket = self.request
        if self.server.tls_mode == "ssl":
            self.sock = self.server.ssl_context.wrap_socket(self.sock, server_side=True)
        self.rfile = self.sock.makefile("rb")
        self.tls = self.server.tls_mode == "ssl"
        self.user: str | None = None
        self.folder: FakeFolder | None = None
        self.readonly = True

    def send(self, line: str | bytes) -> None:
        data = line if isinstance(line, bytes) else line.encode()
        self.sock.sendall(data + b"\r\n")

    def read_command(self) -> bytes | None:
        line = self.rfile.readline()
        if not line:
            return None
        out = b""
        while True:
            line = line.rstrip(b"\r\n")
            literal = re.search(rb"\{(\d+)\}$", line)
            if not literal:
                return out + line
            size = int(literal.group(1))
            self.send("+ Ready for literal")
            data = self.rfile.read(size)
            out += line[: literal.start()] + b'"' + data.replace(b'"', b'\\"') + b'"'
            line = self.rfile.readline()

    def caps(self) -> str:
        caps = ["IMAP4rev1", *sorted(self.server.mailbox.capabilities)]
        if self.server.tls_mode == "starttls" and not self.tls:
            caps.append("STARTTLS")
        return " ".join(caps)

    def handle(self) -> None:
        self.send(f"* OK [CAPABILITY {self.caps()}] fake IMAP ready")
        while True:
            try:
                line = self.read_command()
            except (OSError, ValueError):
                return
            if line is None:
                return
            try:
                tokens = _tokens(line)
            except ValueError:
                self.send(b"* BAD cannot parse")
                continue
            if len(tokens) < 2:
                self.send(b"* BAD missing command")
                continue
            tag, command, args = str(tokens[0]), str(tokens[1]).upper(), tokens[2:]
            if command == "UID" and args:
                command, args = f"UID {str(args[0]).upper()}", args[1:]
            with self.server.mailbox.lock:
                self.server.mailbox.commands.append(command)
            try:
                if not self.dispatch(tag, command, args):
                    return
            except (IndexError, KeyError, ValueError) as exc:
                self.send(f"{tag} BAD {exc.__class__.__name__}")

    # -- commands ----------------------------------------------------------------

    def dispatch(self, tag: str, command: str, args: list[Any]) -> bool:
        box = self.server.mailbox
        if command == "CAPABILITY":
            self.send(f"* CAPABILITY {self.caps()}")
        elif command == "NOOP":
            pass
        elif command == "LOGOUT":
            self.send("* BYE logging out")
            self.send(f"{tag} OK LOGOUT completed")
            return False
        elif command == "STARTTLS":
            if self.server.tls_mode != "starttls" or self.tls:
                self.send(f"{tag} BAD STARTTLS not available")
                return True
            self.send(f"{tag} OK begin TLS")
            self.sock = self.server.ssl_context.wrap_socket(self.sock, server_side=True)
            self.rfile = self.sock.makefile("rb")
            self.tls = True
            return True
        elif command == "LOGIN":
            user, password = str(args[0]), str(args[1])
            with box.lock:
                box.logins.append(user)
                ok = box.users.get(user) == password
            if not ok:
                self.send(f"{tag} NO [AUTHENTICATIONFAILED] Invalid credentials")
                return True
            self.user = user
        elif self.user is None:
            self.send(f"{tag} BAD not authenticated")
            return True
        elif command == "LIST":
            with box.lock:
                for folder in box.folders.values():
                    attrs = " ".join(sorted(folder.attrs))
                    self.send(f'* LIST ({attrs}) "/" "{folder.name}"')
        elif command == "CREATE":
            name = str(args[0])
            with box.lock:
                if name in box.folders:
                    self.send(f"{tag} NO [ALREADYEXISTS] exists")
                    return True
                box.add_folder(name)
        elif command in ("SELECT", "EXAMINE"):
            return self.select(tag, command, str(args[0]))
        elif command == "CLOSE":
            if self.folder is not None and not self.readonly:
                with box.lock:
                    for uid in list(self.folder.uids()):
                        if "\\Deleted" in self.folder.messages[uid].flags:
                            del self.folder.messages[uid]
            self.folder = None
        elif self.folder is None:
            self.send(f"{tag} BAD no folder selected")
            return True
        elif command == "UID SEARCH":
            with box.lock:
                found = self.search(args)
            self.send("* SEARCH" + "".join(f" {u}" for u in found))
        elif command == "UID FETCH":
            self.fetch(str(args[0]), args[1] if isinstance(args[1], list) else args[1:])
        elif command == "UID STORE":
            if self.readonly:
                self.send(f"{tag} NO [READ-ONLY] folder is read-only")
                return True
            error = self.store(str(args[0]), str(args[1]), args[2])
            if error:
                self.send(f"{tag} NO {error}")
                return True
        elif command in ("UID COPY", "UID MOVE"):
            if command == "UID MOVE" and "MOVE" not in box.capabilities:
                self.send(f"{tag} BAD MOVE not supported")
                return True
            if self.readonly and command == "UID MOVE":
                self.send(f"{tag} NO [READ-ONLY] folder is read-only")
                return True
            code = self.copy(str(args[0]), str(args[1]), move=command == "UID MOVE")
            if code is None:
                self.send(f"{tag} NO [TRYCREATE] no such folder")
                return True
            if command == "UID MOVE":
                if code:
                    self.send(f"* OK [{code}] moved")
                self.send(f"{tag} OK MOVE completed")
            else:
                self.send(f"{tag} OK [{code}] COPY completed" if code else f"{tag} OK COPY")
            return True
        elif command == "UID EXPUNGE":
            if "UIDPLUS" not in box.capabilities or self.readonly:
                self.send(f"{tag} BAD UID EXPUNGE not available")
                return True
            with box.lock:
                uids = self.folder.uids()
                for uid in _uid_set(str(args[0]), uids):
                    if "\\Deleted" in self.folder.messages[uid].flags:
                        self.send(f"* {self.folder.uids().index(uid) + 1} EXPUNGE")
                        del self.folder.messages[uid]
        else:
            self.send(f"{tag} BAD unknown command {command}")
            return True
        self.send(f"{tag} OK {command} completed")
        return True

    def select(self, tag: str, command: str, name: str) -> bool:
        box = self.server.mailbox
        with box.lock:
            folder = box.folders.get(name)
            if folder is None:
                self.send(f"{tag} NO no such folder")
                self.folder = None
                return True
            self.folder = folder
            self.readonly = command == "EXAMINE"
            self.send("* FLAGS (" + " ".join(SYSTEM_FLAGS) + ")")
            if self.readonly:
                self.send("* OK [PERMANENTFLAGS ()] read-only")
            else:
                extra = " \\*" if box.keywords_allowed else ""
                self.send(f"* OK [PERMANENTFLAGS ({' '.join(SYSTEM_FLAGS)}{extra})] ok")
            self.send(f"* {len(folder.messages)} EXISTS")
            self.send("* 0 RECENT")
            self.send(f"* OK [UIDVALIDITY {folder.uidvalidity}] UIDs valid")
            self.send(f"* OK [UIDNEXT {folder.uidnext}] next")
        mode = "READ-ONLY" if self.readonly else "READ-WRITE"
        self.send(f"{tag} OK [{mode}] {command} completed")
        return True

    def search(self, args: list[Any]) -> list[int]:
        assert self.folder is not None
        uids = self.folder.uids()
        matched = set(uids)
        i = 0
        while i < len(args):
            key = str(args[i]).upper()
            if key == "ALL":
                i += 1
            elif key == "CHARSET":
                i += 2
            elif key == "UID":
                matched &= set(_uid_set(str(args[i + 1]), uids))
                i += 2
            elif key == "SINCE":
                since = datetime.strptime(str(args[i + 1]), "%d-%b-%Y").date()
                matched &= {u for u in uids if self.folder.messages[u].internaldate.date() >= since}
                i += 2
            elif key == "HEADER":
                name, value = str(args[i + 1]), str(args[i + 2]).lower()
                matched &= {
                    u
                    for u in uids
                    if value in _header_value(self.folder.messages[u].raw, name).lower()
                }
                i += 3
            elif key == "KEYWORD":
                word = str(args[i + 1])
                matched &= {u for u in uids if word in self.folder.messages[u].flags}
                i += 2
            elif key in ("TEXT", "SUBJECT"):
                needle = str(args[i + 1]).lower().encode()
                matched &= {
                    u
                    for u in uids
                    if needle
                    in (
                        self.folder.messages[u].raw.lower()
                        if key == "TEXT"
                        else _header_value(self.folder.messages[u].raw, "Subject").lower().encode()
                    )
                }
                i += 2
            else:
                raise ValueError(f"unsupported search key {key}")
        return sorted(matched)

    def fetch(self, spec: str, items: list[Any]) -> None:
        assert self.folder is not None
        names = [str(x).upper() for x in items]
        with self.server.mailbox.lock:
            uids = self.folder.uids()
            for uid in _uid_set(spec, uids):
                message = self.folder.messages[uid]
                seq = uids.index(uid) + 1
                parts: list[str] = [f"UID {uid}"]
                literal: tuple[str, bytes] | None = None
                for name in names:
                    if name == "BODY[]" and not self.readonly:
                        message.flags.add("\\Seen")
                    if name in ("BODY[]", "BODY.PEEK[]"):
                        literal = ("BODY[]", message.raw)
                    elif name in ("BODY.PEEK[HEADER]", "BODY[HEADER]"):
                        head = message.raw.split(b"\r\n\r\n", 1)[0] + b"\r\n\r\n"
                        literal = ("BODY[HEADER]", head)
                for name in names:
                    if name == "FLAGS":
                        parts.append("FLAGS (" + " ".join(sorted(message.flags)) + ")")
                    elif name == "INTERNALDATE":
                        parts.append(f'INTERNALDATE "{_imap_date(message.internaldate)}"')
                    elif name == "RFC822.SIZE":
                        parts.append(f"RFC822.SIZE {len(message.raw)}")
                if literal is None:
                    self.send(f"* {seq} FETCH (" + " ".join(parts) + ")")
                else:
                    head = (
                        f"* {seq} FETCH ("
                        + " ".join(parts)
                        + f" {literal[0]} {{{len(literal[1])}}}"
                    )
                    self.sock.sendall(head.encode() + b"\r\n" + literal[1] + b")\r\n")

    def store(self, spec: str, op: str, flags: Any) -> str | None:
        assert self.folder is not None
        box = self.server.mailbox
        wanted = [str(f) for f in (flags if isinstance(flags, list) else [flags])]
        new = [f for f in wanted if not f.startswith("\\")]
        if new and not box.keywords_allowed and op.upper().startswith(("+", "FLAGS")):
            return "[CANNOT] keywords are not supported"
        silent = op.upper().endswith(".SILENT")
        with box.lock:
            uids = self.folder.uids()
            for uid in _uid_set(spec, uids):
                message = self.folder.messages[uid]
                if op.upper().startswith("+"):
                    message.flags.update(wanted)
                elif op.upper().startswith("-"):
                    message.flags.difference_update(wanted)
                else:
                    message.flags = set(wanted)
                if not silent:
                    flags_text = " ".join(sorted(message.flags))
                    self.send(f"* {uids.index(uid) + 1} FETCH (UID {uid} FLAGS ({flags_text}))")
        return None

    def copy(self, spec: str, target_name: str, *, move: bool) -> str | None:
        assert self.folder is not None
        box = self.server.mailbox
        with box.lock:
            target = box.folders.get(target_name)
            if target is None:
                return None
            uids = self.folder.uids()
            chosen = _uid_set(spec, uids)
            old, new = [], []
            for uid in chosen:
                message = self.folder.messages[uid]
                new_uid = target.uidnext
                target.uidnext += 1
                target.messages[new_uid] = FakeMessage(
                    uid=new_uid,
                    raw=message.raw,
                    flags=set(message.flags),
                    internaldate=message.internaldate,
                )
                old.append(uid)
                new.append(new_uid)
            if move:
                for uid in chosen:
                    self.send(f"* {self.folder.uids().index(uid) + 1} EXPUNGE")
                    del self.folder.messages[uid]
            if not old or "UIDPLUS" not in box.capabilities:
                return ""
            return (
                f"COPYUID {target.uidvalidity} {','.join(map(str, old))} "
                f"{','.join(map(str, new))}"
            )


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self, mailbox: FakeMailbox, tls_mode: str, ssl_context: ssl.SSLContext | None
    ) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.mailbox = mailbox
        self.tls_mode = tls_mode
        self.ssl_context = ssl_context  # type: ignore[assignment]


class FakeImapServer:
    """``with FakeImapServer(mailbox) as server: server.port``. ``tls`` is ``"plain"``,
    ``"ssl"`` or ``"starttls"``; the TLS modes need ``ssl_context`` (server side)."""

    def __init__(
        self,
        mailbox: FakeMailbox,
        *,
        tls: str = "plain",
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if tls != "plain" and ssl_context is None:
            raise ValueError("a TLS mode needs a server ssl_context")
        self.mailbox = mailbox
        self._server = _Server(mailbox, tls, ssl_context)
        self._thread = threading.Thread(
            target=self._server.serve_forever, args=(0.05,), daemon=True
        )

    @property
    def host(self) -> str:
        return "127.0.0.1"

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def __enter__(self) -> FakeImapServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def make_certificate(directory: Any) -> tuple[str, str]:
    """A self-signed certificate for ``localhost`` / ``127.0.0.1`` -> (cert, key) paths."""
    import ipaddress
    from datetime import timedelta
    from pathlib import Path

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    base = Path(directory)
    cert_path, key_path = base / "fake-imap.crt", base / "fake-imap.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return str(cert_path), str(key_path)
