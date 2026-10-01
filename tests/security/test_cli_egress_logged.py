"""Every HTTP call the ``iris`` CLI makes goes through ``cli/api_client.py``.

The standing rule is that every outbound network call is logged on ``iris.egress``.
The CLI's calls to the harness's own IRIS API were not: each command built its own
``httpx`` / ``urllib`` call, the server logged the request as ingress, and the client
side was silent. ``iris_harness.cli.api_client`` is now the one way out
(``harness_api_client`` / ``harness_urlopen``, both logging host-only), and this gate
fails when a CLI module reaches the network any other way.

What it scans: every module of ``iris_harness/cli/`` and the ``iris`` entry point
``iris_harness/main.py``, except the helper itself. What it cannot see, stated plainly:
a CLI module calling into a *non-CLI* module that makes the request (e.g.
``llm/providers.py`` fetching a model catalog) — that module owns its own egress line.
It reads syntax, so a transport reached through ``getattr`` or ``importlib`` is
invisible too.
"""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HARNESS = PROJECT_ROOT / "src" / "iris_harness"
HELPER = HARNESS / "cli" / "api_client.py"

# Calls that open a network connection. A dotted name resolves through the module's
# own import aliases first, so ``import httpx as h; h.get(...)`` and
# ``from urllib.request import urlopen`` are both caught.
_FORBIDDEN_CALLS = frozenset(
    {
        *(
            f"httpx.{name}"
            for name in (
                "get",
                "post",
                "put",
                "patch",
                "delete",
                "head",
                "options",
                "request",
                "stream",
                "Client",
                "AsyncClient",
            )
        ),
        "urllib.request.urlopen",
        "urllib.request.build_opener",
        "urllib.request.OpenerDirector",
        "http.client.HTTPConnection",
        "http.client.HTTPSConnection",
        "socket.create_connection",
        "websockets.connect",
        "websockets.client.connect",
        "aiohttp.ClientSession",
        "aiohttp.request",
    }
)
# Libraries whose every call is a request.
_FORBIDDEN_PREFIXES = ("requests.",)


def _cli_modules() -> list[Path]:
    modules = sorted((HARNESS / "cli").rglob("*.py")) + [HARNESS / "main.py"]
    return [path for path in modules if path != HELPER]


def _aliases(tree: ast.AST) -> dict[str, str]:
    """Local name -> the dotted name it was imported as, for every import in the file."""
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    found[alias.asname] = alias.name
                else:
                    head = alias.name.split(".", 1)[0]
                    found[head] = head
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                found[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return found


def _dotted(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _is_forbidden(name: str) -> bool:
    return name in _FORBIDDEN_CALLS or name.startswith(_FORBIDDEN_PREFIXES)


def unlogged_calls(source: str, label: str) -> list[str]:
    """``label:line: name`` for each call in ``source`` that opens a connection itself."""
    tree = ast.parse(source)
    aliases = _aliases(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted(node.func)
        if dotted is None:
            continue
        head, _, rest = dotted.partition(".")
        resolved = aliases.get(head, head) + (f".{rest}" if rest else "")
        if _is_forbidden(resolved):
            found.append(f"{label}:{node.lineno}: {resolved}")
    return found


def test_no_cli_module_calls_the_network_except_through_the_helper() -> None:
    offenders: list[str] = []
    for path in _cli_modules():
        label = str(path.relative_to(PROJECT_ROOT))
        offenders.extend(unlogged_calls(path.read_text(encoding="utf-8"), label))
    assert not offenders, (
        "CLI HTTP calls that bypass iris_harness.cli.api_client (no egress line); "
        "use harness_api_client(...) or harness_urlopen(...):\n  " + "\n  ".join(offenders)
    )


def test_the_scan_covers_the_cli_and_its_callers_use_the_helper() -> None:
    """A scan that finds no files, or a helper nobody calls, would pass vacuously."""
    modules = _cli_modules()
    assert HARNESS / "main.py" in modules
    assert len(modules) >= 20, modules
    callers = {
        path.name
        for path in modules
        if "harness_api_client(" in path.read_text(encoding="utf-8")
        or "harness_urlopen(" in path.read_text(encoding="utf-8")
    }
    expected = {
        "approvals.py",
        "commands.py",
        "context_health.py",
        "device.py",
        "health.py",
        "learning.py",
        "main.py",
        "modes.py",
        "plugins.py",
        "repl.py",
    }
    assert expected <= callers, sorted(expected - callers)


def test_the_detector_catches_every_spelling() -> None:
    """The gate's own mutation check: each way to open a connection is seen."""
    snippets = {
        "import httpx\nhttpx.get('http://x')\n": "httpx.get",
        "import httpx as h\nh.Client()\n": "httpx.Client",
        "from httpx import post\npost('http://x')\n": "httpx.post",
        "import urllib.request\nurllib.request.urlopen('http://x')\n": "urllib.request.urlopen",
        "from urllib.request import urlopen\nurlopen('http://x')\n": "urllib.request.urlopen",
        "from urllib import request\nrequest.urlopen('http://x')\n": "urllib.request.urlopen",
        "import requests\nrequests.Session().get('x')\n": "requests.Session",
        "import websockets\nwebsockets.connect('ws://x')\n": "websockets.connect",
        "def f():\n    import httpx\n    return httpx.post('http://x')\n": "httpx.post",
    }
    for source, name in snippets.items():
        found = unlogged_calls(source, "snippet")
        assert any(line.endswith(f": {name}") for line in found), (source, found)
    clean = (
        "from iris_harness.cli.api_client import harness_api_client, harness_urlopen\n"
        "import httpx\n"
        "with harness_api_client(purpose='x', timeout=1.0) as c:\n"
        "    c.get('http://x')\n"
        "except_type = httpx.HTTPError\n"
    )
    assert unlogged_calls(clean, "clean") == []
