"""The lint half of the egress drift check: plugin source that imports a raw network library.

Stable tier (issue #103). A plugin's outbound requests go through ``api.http`` so that each
is declared, allowed and recorded (docs/architecture/plugin-egress.md). :func:`check_network_imports`
walks Python files and reports every import of a library that opens connections by itself --
``httpx``, ``requests``, ``urllib.request``, ``http.client``, ``socket``, ``smtplib``,
``imaplib`` and the rest of :data:`NETWORK_MODULES` -- so a plugin's CI can fail on one the
way it fails on ``check_stable_imports``::

    def test_my_plugin_makes_no_raw_network_call() -> None:
        assert check_network_imports([Path("src")]) == []

What this proves, and what it does not. It proves the files it read import none of those
modules by name, in a statement an AST shows, and that no file uses one through an
attribute of an imported package (``import urllib`` then ``urllib.request.urlopen``) or
calls ``asyncio.open_connection`` / ``start_server``. It does not see
``importlib.import_module`` with a computed name, a name passed around as a value, an event
loop's own ``create_connection``, a dependency that opens its own connection,
``subprocess``, or any library this list does not name (an SDK that wraps a socket under
another name: ``botocore``, ``psycopg``, ``kafka`` and many more); and an in-process plugin can
always open a socket some other way. A file that does not parse is reported as such rather
than skipped. It is a tripwire on the honest path, not a sandbox.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

#: Modules (and dotted submodules) that open network connections without the harness.
NETWORK_MODULES: tuple[str, ...] = (
    "aiohttp",
    "aiosmtplib",
    "asyncio.open_connection",
    "asyncio.open_unix_connection",
    "asyncio.start_server",
    "asyncio.start_unix_server",
    "boto3",
    "ftplib",
    "googleapiclient",
    "grpc",
    "h11",
    "http.client",
    "http.server",
    "httpcore",
    "httplib2",
    "httpx",
    "imaplib",
    "multiprocessing.connection",
    "openai",
    "paramiko",
    "poplib",
    "pycurl",
    "pymongo",
    "redis",
    "requests",
    "smtplib",
    "socket",
    "socketserver",
    "ssl",
    "telnetlib",
    "urllib.request",
    "urllib3",
    "websocket",
    "websockets",
    "xmlrpc.client",
)


#: The ``module`` of a finding for a file that could not be read or parsed.
UNPARSABLE = "<unparsable"


@dataclass(frozen=True)
class NetworkImportViolation:
    """One import of a library that opens connections on its own."""

    path: Path
    line: int
    module: str

    def __str__(self) -> str:
        if self.module.startswith(UNPARSABLE):
            return f"{self.path}:{self.line}: {self.module}: the file was not checked"
        return (
            f"{self.path}:{self.line}: imports {self.module}: use `api.http` "
            "(iris_harness.sdk.http), which checks the manifest's egress and records the call"
        )


def _matches(module: str) -> str | None:
    for banned in NETWORK_MODULES:
        if module == banned or module.startswith(banned + "."):
            return banned
    return None


def _imported(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """``(line, dotted module)`` of each absolute import; ``from a import b`` names ``a.b``
    too, since ``b`` may be a module (``from urllib import request``)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module
            for alias in node.names:
                if alias.name == "*":
                    # ``from asyncio import *`` brings every name of the module in: report
                    # each listed library under it (an over-approximation, on purpose).
                    for banned in NETWORK_MODULES:
                        if banned.startswith(node.module + "."):
                            yield node.lineno, banned
                else:
                    yield node.lineno, f"{node.module}.{alias.name}"


def _aliases(tree: ast.AST) -> dict[str, str]:
    """What each imported name stands for: ``import a.b`` binds ``a``; ``import a.b as c``
    binds ``c`` to ``a.b``; ``from a import b as c`` binds ``c`` to ``a.b``."""
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    names[alias.asname] = alias.name
                else:
                    names[alias.name.split(".", 1)[0]] = alias.name.split(".", 1)[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return names


def _used(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """``(line, dotted name)`` of each attribute chain rooted at an imported name, resolved
    (``import urllib`` ... ``urllib.request.urlopen`` -> ``urllib.request.urlopen``)."""
    names = _aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        parts = [node.attr]
        base = node.value
        while isinstance(base, ast.Attribute):
            parts.append(base.attr)
            base = base.value
        if isinstance(base, ast.Name) and base.id in names:
            yield node.lineno, ".".join([names[base.id], *reversed(parts)])


def check_network_imports(paths: Iterable[Path]) -> list[NetworkImportViolation]:
    """Every raw network import or use under ``paths`` (files, or directories searched for
    ``.py``); a file that cannot be parsed is a finding of its own."""
    found: list[NetworkImportViolation] = []
    for root in paths:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for file in files:
            try:
                tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
            except (SyntaxError, UnicodeDecodeError, ValueError, OSError) as exc:
                line = getattr(exc, "lineno", None) or 0
                found.append(
                    NetworkImportViolation(file, int(line), f"{UNPARSABLE}: {type(exc).__name__}>")
                )
                continue
            seen: set[tuple[int, str]] = set()
            for line, module in [*_imported(tree), *_used(tree)]:
                banned = _matches(module)
                if banned is not None and (line, banned) not in seen:
                    seen.add((line, banned))
                    found.append(NetworkImportViolation(file, line, banned))
    return found


__all__ = ["NETWORK_MODULES", "NetworkImportViolation", "check_network_imports"]
