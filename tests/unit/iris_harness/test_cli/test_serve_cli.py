"""``iris serve``: the uvicorn call it makes, loopback by default, a warning off it.

uvicorn.run is replaced in every test: nothing here binds a port.
"""

from __future__ import annotations

from typing import Any

import pytest
import uvicorn
from typer.testing import CliRunner

from iris_harness.cli import serve
from iris_harness.main import app


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    recorded: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: recorded.append((a, kw)))
    monkeypatch.delenv("IRIS_API_HOST", raising=False)
    monkeypatch.delenv("IRIS_API_PORT", raising=False)
    monkeypatch.setenv("IRIS_AUTH_SECRET", "test-secret-for-testing")
    return recorded


def _serve(*args: str) -> Any:
    result = CliRunner().invoke(app, ["serve", *args])
    assert result.exit_code == 0, result.output
    return result


def test_the_default_is_the_iris_api_on_loopback_8003(calls: list[Any]) -> None:
    result = _serve()
    assert calls == [
        (("iris_harness.server.iris_api.main:app",), {"host": "127.0.0.1", "port": 8003})
    ]
    assert "warning" not in result.output


def test_host_and_port_options_reach_uvicorn(calls: list[Any]) -> None:
    _serve("--host", "localhost", "--port", "9100")
    assert calls[0][1] == {"host": "localhost", "port": 9100}


def test_the_start_script_env_vars_set_the_defaults(
    calls: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_API_HOST", "127.0.0.2")
    monkeypatch.setenv("IRIS_API_PORT", "8013")
    result = _serve()
    assert calls[0][1] == {"host": "127.0.0.2", "port": 8013}
    assert "warning" not in result.output  # all of 127.0.0.0/8 is loopback


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "::", "iris.example"])  # noqa: S104
def test_a_non_loopback_host_warns_before_serving(calls: list[Any], host: str) -> None:
    result = _serve("--host", host)
    out = " ".join(result.output.split())
    assert "warning:" in out
    assert "plain HTTP" in out
    assert "IRIS_AUTH_SECRET" in out
    assert calls[0][1]["host"] == host  # it warns; it does not refuse


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]", "127.1.2.3"])
def test_loopback_hosts_do_not_warn(host: str) -> None:
    assert serve.is_loopback(host)
    assert serve.exposure_warning(host, 8003) is None


def test_an_unset_secret_is_called_out(calls: list[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_AUTH_SECRET")
    result = _serve()
    assert "IRIS_AUTH_SECRET is not set" in " ".join(result.output.split())
    assert calls  # the API fails closed on its own; the command still starts it


def test_an_out_of_range_port_is_refused(calls: list[Any]) -> None:
    result = CliRunner().invoke(app, ["serve", "--port", "70000"])
    assert result.exit_code != 0
    assert calls == []
