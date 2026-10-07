"""The lint half of the egress drift check: raw network imports in plugin source (#103)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.testing import NETWORK_MODULES, check_network_imports

_ROOT = Path(__file__).resolve().parents[4]


def _scan(tmp_path: Path, source: str) -> list[tuple[int, str]]:
    path = tmp_path / "plugin.py"
    path.write_text(source, encoding="utf-8")
    return [(v.line, v.module) for v in check_network_imports([path])]


@pytest.mark.parametrize(
    "source, module",
    [
        ("import httpx\n", "httpx"),
        ("import requests\n", "requests"),
        ("import requests.adapters\n", "requests"),
        ("from httpx import Client\n", "httpx"),
        ("import urllib.request\n", "urllib.request"),
        ("from urllib.request import urlopen\n", "urllib.request"),
        ("from urllib import request\n", "urllib.request"),
        ("from http import client\n", "http.client"),
        ("import http.client as h\n", "http.client"),
        ("import socket\n", "socket"),
        ("import smtplib, imaplib\n", "smtplib"),
        ("from googleapiclient.discovery import build\n", "googleapiclient"),
        ("import paramiko\n", "paramiko"),
        ("import boto3\n", "boto3"),
        ("from openai import OpenAI\n", "openai"),
        ("import redis.asyncio as redis\n", "redis"),
        ("from pymongo import MongoClient\n", "pymongo"),
        ("import aiosmtplib\n", "aiosmtplib"),
    ],
)
def test_a_raw_network_import_is_reported_with_its_line(
    tmp_path: Path, source: str, module: str
) -> None:
    found = _scan(tmp_path, "import os\n" + source)
    assert (2, module) in found


@pytest.mark.parametrize(
    "source",
    [
        "import json\nimport urllib.parse\nfrom urllib.parse import quote\n",
        "from http import HTTPStatus\n",
        "from iris_harness.sdk.http import GovernedHttp, EgressDenied\n",
        "from . import httpx\n",  # a relative import is the plugin's own module
        "import asyncio\n",
    ],
)
def test_imports_that_open_no_connection_are_not_reported(tmp_path: Path, source: str) -> None:
    assert _scan(tmp_path, source) == []


def test_a_directory_is_searched_and_the_path_is_named(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "client.py").write_text("import requests\n", encoding="utf-8")
    (tmp_path / "pkg" / "ok.py").write_text("import json\n", encoding="utf-8")
    [violation] = check_network_imports([tmp_path])
    assert violation.path.name == "client.py" and "api.http" in str(violation)


def test_the_list_names_what_it_checks() -> None:
    assert {"httpx", "requests", "socket", "urllib.request", "smtplib"} <= set(NETWORK_MODULES)


# What the project ships as first-party plugins still imports raw network libraries in
# these files. That is a visible debt, not an exemption: a NEW raw import in a first-party
# plugin fails here, and moving a file onto `api.http` deletes its line (then this table must
# shrink with it). Every entry says WHY the file is still pinned (issue #172): a test below
# fails on an entry with no reason, so a pin cannot be added without one.
_DISCOVERY_REASON = (
    "calls the operator's local llama-server (default http://localhost:8090/v1): local model "
    "inference, not egress. The governed client refuses loopback by design; the proper home "
    "is the tier router (issue #217: discovery.py onto the tier router)"
)
_GMAIL_REASON = (
    "the Google API client does its own transport and OAuth; its hosts are declared in the "
    "gmail manifest `egress`, but its calls do not pass through api.http"
)
_FIRST_PARTY_DEBT: dict[tuple[str, str], str] = {
    ("src/iris_harness/plugins_builtin/research/cache.py", "redis"): (
        "the opt-in Redis cache (IRIS_RESEARCH_CACHE=redis) opens a plain TCP connection to the "
        "operator's own Redis (IRIS_REDIS_URL): their infrastructure, and a TCP connection to an "
        "operator-configured server cannot go through api.http"
    ),
    ("src/iris_harness/plugins_builtin/research/extract.py", "socket"): (
        "the Crawl4AI backend drives its own browser and cannot go through api.http; its "
        "first URL is address-checked here (getaddrinfo). The default backend is governed"
    ),
    ("src/iris_harness/plugins_builtin/research/providers/searxng.py", "urllib.request"): (
        "IRIS_SEARXNG_URL is the operator's own SearXNG server, usually on loopback or the "
        "LAN; the governed client refuses internal addresses by design, and widening that "
        "is a security-design decision (issue #216: owner-configured internal-service "
        "egress declaration)"
    ),
    ("src/iris_personal/plugins/email_workflows/demo/run.py", "socket"): (
        "opens nothing: it patches socket.connect to REFUSE network access for the demo run"
    ),
    ("src/iris_personal/plugins/email_workflows/discovery.py", "requests"): _DISCOVERY_REASON,
    ("src/iris_personal/plugins/email_workflows/job_watch.py", "httpx"): (
        "a health probe of the operator's own service roots (root_url(), loopback or LAN); "
        "the governed client refuses internal addresses by design (issue #216: "
        "owner-configured internal-service egress declaration)"
    ),
    ("src/iris_personal/plugins/gmail/gmail_attachments.py", "googleapiclient"): _GMAIL_REASON,
    ("src/iris_personal/plugins/gmail/gmail_fetch.py", "googleapiclient"): _GMAIL_REASON,
    ("src/iris_personal/plugins/gmail/gmail_oauth.py", "googleapiclient"): _GMAIL_REASON,
    ("src/iris_personal/plugins/imap/connection.py", "imaplib"): (
        "a TCP connection (IMAP over TLS) to the owner's own mail server: not an HTTP host, "
        "so no `egress` host list describes it and api.http cannot carry it"
    ),
    ("src/iris_personal/plugins/imap/connection.py", "ssl"): (
        "the TLS context for the same IMAP connection (imaplib)"
    ),
}


def test_every_pinned_file_says_why_it_is_still_pinned() -> None:
    for pair, reason in _FIRST_PARTY_DEBT.items():
        assert len(reason.split()) >= 6, f"{pair} needs a real reason"


def test_the_debt_list_shrank_by_the_four_urllib_pairs_issue_172_moved() -> None:
    """Brave, Exa, Tavily and the page fetch now go through the governed client. What is left
    in the research plugin is SearXNG (the operator's own, usually loopback, server: the
    governed client refuses internal addresses by design) and ``extract.py``'s ``socket``
    (the Crawl4AI backend's address check, since it drives its own browser), plus the opt-in
    Redis cache (#175)."""
    research = sorted(p for p in _FIRST_PARTY_DEBT if "/research/" in p[0])
    assert research == [
        ("src/iris_harness/plugins_builtin/research/cache.py", "redis"),
        ("src/iris_harness/plugins_builtin/research/extract.py", "socket"),
        ("src/iris_harness/plugins_builtin/research/providers/searxng.py", "urllib.request"),
    ]
    assert len(_FIRST_PARTY_DEBT) == 11


def _pairs(paths: list[Path]) -> set[tuple[str, str]]:
    return {(str(v.path.relative_to(_ROOT)), v.module) for v in check_network_imports(paths)}


def test_first_party_plugins_import_raw_network_libraries_only_where_listed() -> None:
    """Pinned by (file, library), so a new library in an already-listed file fails too."""
    found = _pairs(
        [_ROOT / "src/iris_harness/plugins_builtin", _ROOT / "src/iris_personal/plugins"]
    )
    assert found == set(_FIRST_PARTY_DEBT)


def test_a_new_library_in_a_pinned_file_is_not_in_the_pin(tmp_path: Path) -> None:
    pinned = _ROOT / "src/iris_harness/plugins_builtin/research/extract.py"
    copy = tmp_path / "extract.py"
    copy.write_text(pinned.read_text(encoding="utf-8") + "\nimport requests\n", encoding="utf-8")
    modules = {v.module for v in check_network_imports([copy])}
    assert "requests" in modules
    assert ("src/iris_harness/plugins_builtin/research/extract.py", "requests") not in (
        _FIRST_PARTY_DEBT
    )


@pytest.mark.parametrize(
    "source, module",
    [
        ("import urllib\nurllib.request.urlopen('x')\n", "urllib.request"),
        ("import urllib as u\nu.request.urlopen('x')\n", "urllib.request"),
        ("import http\nhttp.client.HTTPConnection('x')\n", "http.client"),
        ("import http\nhttp.server.HTTPServer\n", "http.server"),
        ("import asyncio\nasyncio.open_connection('x', 1)\n", "asyncio.open_connection"),
        ("import asyncio\nasyncio.start_server(f, 'x')\n", "asyncio.start_server"),
        ("from asyncio import open_connection\n", "asyncio.open_connection"),
        ("import httpcore\n", "httpcore"),
        ("import socketserver\n", "socketserver"),
        ("from multiprocessing import connection\n", "multiprocessing.connection"),
        (
            "import multiprocessing\nmultiprocessing.connection.Client(('x', 1))\n",
            "multiprocessing.connection",
        ),
        ("from asyncio import *\n", "asyncio.open_connection"),
        ("from urllib import *\n", "urllib.request"),
        ("import h11\n", "h11"),
        ("from http.server import HTTPServer\n", "http.server"),
    ],
)
def test_attribute_use_and_the_cheap_extra_libraries_are_reported(
    tmp_path: Path, source: str, module: str
) -> None:
    assert module in {m for _, m in _scan(tmp_path, source)}


def test_asyncio_without_a_network_call_is_not_reported(tmp_path: Path) -> None:
    assert _scan(tmp_path, "import asyncio\nasyncio.sleep(1)\nasyncio.run(f())\n") == []


@pytest.mark.parametrize("content", [b"def broken(:\n", b"\xff\xfe not utf-8 \x80\n"])
def test_an_unparsable_file_is_a_finding_not_a_crash(tmp_path: Path, content: bytes) -> None:
    (tmp_path / "bad.py").write_bytes(content)
    (tmp_path / "good.py").write_text("import requests\n", encoding="utf-8")
    found = check_network_imports([tmp_path])
    assert {v.path.name for v in found} == {"bad.py", "good.py"}
    [bad] = [v for v in found if v.path.name == "bad.py"]
    assert bad.module.startswith("<unparsable") and "not checked" in str(bad)


def test_what_the_project_hands_an_outside_author_imports_no_raw_network_library() -> None:
    """The scaffold and the examples are what a third party copies: they import no raw library."""
    sources = [
        path
        for root in (_ROOT / "examples", _ROOT / "src/iris_harness/cli/templates")
        for path in sorted(root.rglob("*.py"))
        if not path.name.startswith("test_")
    ]
    assert check_network_imports(sources) == []
