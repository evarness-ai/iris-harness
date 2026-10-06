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
modules by name, in a statement an AST shows. It does not see ``importlib.import_module``
with a computed name, a dependency that opens its own connection, ``subprocess``, or a
module this list does not name; and an in-process plugin can always open a socket some other
way. It is a tripwire on the honest path, not a sandbox.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

#: Modules (and dotted submodules) that open network connections without the harness.
NETWORK_MODULES: tuple[str, ...] = (
    "aiohttp",
    "ftplib",
    "googleapiclient",
    "grpc",
    "http.client",
    "httplib2",
    "httpx",
    "imaplib",
    "poplib",
    "pycurl",
    "requests",
    "smtplib",
    "socket",
    "ssl",
    "telnetlib",
    "urllib.request",
    "urllib3",
    "websocket",
    "websockets",
    "xmlrpc.client",
)


@dataclass(frozen=True)
class NetworkImportViolation:
    """One import of a library that opens connections on its own."""

    path: Path
    line: int
    module: str

    def __str__(self) -> str:
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
                yield node.lineno, f"{node.module}.{alias.name}"


def check_network_imports(paths: Iterable[Path]) -> list[NetworkImportViolation]:
    """Every raw network import under ``paths`` (files, or directories searched for ``.py``)."""
    found: list[NetworkImportViolation] = []
    for root in paths:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for file in files:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
            seen: set[tuple[int, str]] = set()
            for line, module in _imported(tree):
                banned = _matches(module)
                if banned is not None and (line, banned) not in seen:
                    seen.add((line, banned))
                    found.append(NetworkImportViolation(file, line, banned))
    return found


__all__ = ["NETWORK_MODULES", "NetworkImportViolation", "check_network_imports"]
