"""The suite's network guard (tests/conftest.py) fails tests that reach real servers.

Each case runs a tiny probe suite in a subprocess under the REAL conftest, so what
is pinned here is exactly what every test in the repo runs under: a probe that
connects to a model port, an IRIS service port or a remote host fails with the host:port in the report,
even when the code swallowed the error; loopback on any other port still works;
``@pytest.mark.real_llm`` lifts the guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ("pytester",)

_CONFTEST = Path(__file__).resolve().parents[2] / "conftest.py"
_SRC = _CONFTEST.parents[1] / "src"

_PROBES = """
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import urlopen

import pytest


def test_ollama_port_is_blocked():
    socket.create_connection(("127.0.0.1", 11434), timeout=0.2)


def test_lmstudio_port_is_blocked_even_when_the_error_is_swallowed():
    try:
        urlopen("http://localhost:1234/v1/models", timeout=0.2)  # noqa: S310
    except Exception:
        pass  # a "model is down" fallback must not hide the attempt


def test_llama_server_port_is_blocked():
    sock = socket.socket()
    try:
        assert sock.connect_ex(("127.0.0.1", 8090)) != 0
    finally:
        sock.close()


@pytest.fixture(scope="module")
def probed_once_per_module():
    # A module fixture is set up before any function-scoped one; the guard must
    # already be armed. The probe "finds the server down" and skips.
    try:
        socket.create_connection(("127.0.0.1", 11434), timeout=0.2)
    except OSError:
        pytest.skip("model server not reachable")


def test_module_fixture_probe_is_caught(probed_once_per_module):
    pass


def test_iris_service_port_is_blocked():
    try:
        urlopen("http://127.0.0.1:8003/healthz", timeout=0.2)  # noqa: S310
    except Exception:
        pass


def test_remote_host_is_blocked():
    # TEST-NET-3: never routed, so nothing real is contacted even without the guard.
    socket.create_connection(("203.0.113.7", 443), timeout=0.2)


def test_loopback_ephemeral_server_still_works():
    class Ok(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Ok)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/", timeout=2) as resp:  # noqa: S310
            assert resp.read() == b"ok"
    finally:
        server.shutdown()


@pytest.mark.real_llm
def test_real_llm_marker_lifts_the_guard():
    try:
        socket.create_connection(("203.0.113.7", 443), timeout=0.01)
    except OSError as exc:
        assert type(exc).__name__ != "RealNetworkBlocked"
"""


@pytest.fixture()
def probe_run(pytester: pytest.Pytester) -> pytest.RunResult:
    pytester.makeini("""
        [pytest]
        markers =
            real_llm: live model test
            real_embeddings: live embeddings test
        """)
    future = "from __future__ import annotations\n"
    source = _CONFTEST.read_text(encoding="utf-8")
    assert future in source
    # The copy lives in a temp dir, so point it at this checkout's src/ explicitly.
    header = f"{future}import sys\nsys.path.insert(0, {str(_SRC)!r})\n"
    pytester.makeconftest(source.replace(future, header, 1))
    pytester.makepyfile(test_probes=_PROBES)
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", "-rA")


def test_model_ports_and_remote_hosts_fail_the_test(probe_run: pytest.RunResult) -> None:
    probe_run.assert_outcomes(passed=2, failed=5, skipped=1, errors=1)
    out = probe_run.stdout.str()
    for blocked in (
        "test_ollama_port_is_blocked",
        "test_lmstudio_port_is_blocked_even_when_the_error_is_swallowed",
        "test_llama_server_port_is_blocked",
        "test_iris_service_port_is_blocked",
        "test_remote_host_is_blocked",
    ):
        assert f"FAILED test_probes.py::{blocked}" in out
    assert "model server port 127.0.0.1:11434" in out
    assert "model server port 127.0.0.1:8090" in out
    assert "IRIS service port 127.0.0.1:8003" in out
    assert "non-loopback host 203.0.113.7:443" in out
    assert "test_probes.py::test_ollama_port_is_blocked tried to open" in out
    # The skip-after-probe case is reported at teardown, not lost behind the skip.
    assert "ERROR test_probes.py::test_module_fixture_probe_is_caught" in out
    assert "test_probes.py::test_module_fixture_probe_is_caught tried to open" in out


def test_loopback_servers_and_real_llm_tests_are_left_alone(probe_run: pytest.RunResult) -> None:
    out = probe_run.stdout.str()
    assert "PASSED test_probes.py::test_loopback_ephemeral_server_still_works" in out
    assert "PASSED test_probes.py::test_real_llm_marker_lifts_the_guard" in out
