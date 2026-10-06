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
# plugin fails here, and moving a file onto `api.http` deletes its line (then this set must
# shrink with it). Declared hosts for each are in the plugin's manifest where HTTP is
# involved (research: open_web; gmail: Google's hosts); imap is a raw TCP socket (imaplib),
# which no HTTP host list describes.
_FIRST_PARTY_DEBT = {
    "src/iris_harness/plugins_builtin/research/extract.py",
    "src/iris_harness/plugins_builtin/research/providers/brave.py",
    "src/iris_harness/plugins_builtin/research/providers/exa.py",
    "src/iris_harness/plugins_builtin/research/providers/searxng.py",
    "src/iris_harness/plugins_builtin/research/providers/tavily.py",
    "src/iris_personal/plugins/email_workflows/demo/run.py",
    "src/iris_personal/plugins/email_workflows/discovery.py",
    "src/iris_personal/plugins/email_workflows/job_watch.py",
    "src/iris_personal/plugins/gmail/gmail_attachments.py",
    "src/iris_personal/plugins/gmail/gmail_fetch.py",
    "src/iris_personal/plugins/gmail/gmail_oauth.py",
    "src/iris_personal/plugins/imap/connection.py",
}


def test_first_party_plugins_import_raw_network_libraries_only_where_listed() -> None:
    found = {
        str(v.path.relative_to(_ROOT))
        for v in check_network_imports(
            [_ROOT / "src/iris_harness/plugins_builtin", _ROOT / "src/iris_personal/plugins"]
        )
    }
    assert found == _FIRST_PARTY_DEBT


def test_what_the_project_hands_an_outside_author_imports_no_raw_network_library() -> None:
    """The scaffold and the examples are what a third party copies: they show ``api.http``."""
    sources = [
        path
        for root in (_ROOT / "examples", _ROOT / "src/iris_harness/cli/templates")
        for path in sorted(root.rglob("*.py"))
        if not path.name.startswith("test_")
    ]
    assert check_network_imports(sources) == []
