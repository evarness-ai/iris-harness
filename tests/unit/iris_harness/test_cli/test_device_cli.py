"""``iris device`` is a client of ``/api/v1/devices`` (ADR-0117): it sends the service
secret over HTTP and never opens the devices DB. The HTTP layer is faked here; the
routes themselves are tested against the real app in ``test_iris_api``."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from typing import Any

import pytest
from typer.testing import CliRunner

from iris_harness.main import app

runner = CliRunner()

API = "http://iris.test"
ID_A = "aaaa1111-0000-4000-8000-000000000001"
ID_B = "aaaa2222-0000-4000-8000-000000000002"
ID_C = "cccc3333-0000-4000-8000-000000000003"


def _device(device_id: str, name: str, **over: Any) -> dict[str, Any]:
    return {
        "device_id": device_id,
        "name": name,
        "kind": "app",
        "scope": "control",
        "created_at": "2026-09-20T10:00:00+00:00",
        "last_seen_at": None,
        "revoked_at": None,
        "current": False,
        **over,
    }


DEVICES = [
    _device(ID_A, "Phone"),
    _device(ID_B, "Laptop", kind="browser", scope="read"),
    _device(ID_C, "Old tablet", revoked_at="2026-09-19T08:00:00+00:00"),
]


class _FakeApi:
    """Stands in for ``urllib.request.urlopen``: records requests, answers by route."""

    def __init__(self) -> None:
        self.requests: list[urllib.request.Request] = []
        self.fail_with: Exception | None = None
        self.devices: list[dict[str, Any]] = list(DEVICES)

    def __call__(self, request: urllib.request.Request, timeout: float = 0) -> io.BytesIO:
        self.requests.append(request)
        if self.fail_with is not None:
            raise self.fail_with
        method, path = request.get_method(), request.full_url.removeprefix(API)
        if (method, path) == ("POST", "/api/v1/devices/pair/start"):
            scope = json.loads(request.data or b"{}").get("scope", "control")
            body: dict[str, Any] = {
                "code": "ABCD-EFGH",
                "scope": scope,
                "expires_at": "2026-09-20T12:05:00+00:00",
            }
        elif (method, path) == ("GET", "/api/v1/devices"):
            body = {"devices": self.devices}
        elif method == "DELETE":
            device_id = path.rsplit("/", 1)[1]
            row = next(d for d in self.devices if d["device_id"] == device_id)
            body = {"device": {**row, "revoked_at": "2026-09-20T12:00:00+00:00"}}
        else:  # pragma: no cover - a test asked for a route the CLI must not call
            raise AssertionError(f"unexpected call: {method} {path}")
        return io.BytesIO(json.dumps(body).encode())

    def calls(self) -> list[tuple[str, str]]:
        return [(r.get_method(), r.full_url.removeprefix(API)) for r in self.requests]


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> _FakeApi:
    monkeypatch.setenv("IRIS_AUTH_SECRET", "s3cret")
    fake = _FakeApi()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


def _http_error(code: int, detail: str) -> urllib.error.HTTPError:
    body = io.BytesIO(json.dumps({"detail": detail}).encode())
    return urllib.error.HTTPError(f"{API}/x", code, "err", {}, body)  # type: ignore[arg-type]


# Rich wraps at the terminal width; a wide one keeps each message on one line so the
# assertions read the text rather than the wrapping.
WIDE = {"COLUMNS": "200"}


def _run(*args: str) -> Any:
    return runner.invoke(app, ["device", *args, "--api", API], env=WIDE)


def test_the_cli_never_opens_the_devices_db() -> None:
    """Every capability has an API: the terminal is a client of it. Importing the
    store here would also fork the failed-claim throttle, which lives in the API."""
    import ast
    from pathlib import Path

    import iris_harness.cli.device as device_cli

    tree = ast.parse(Path(device_cli.__file__).read_text())
    imported = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert not {m for m in imported if m and ("kernel" in m or "sqlite" in m)}, imported


def test_pair_posts_the_scope_with_the_secret_and_prints_the_code(api: _FakeApi) -> None:
    result = _run("pair", "--scope", "read")
    assert result.exit_code == 0, result.output
    assert "ABCD-EFGH" in result.output and "read" in result.output
    (request,) = api.requests
    assert api.calls() == [("POST", "/api/v1/devices/pair/start")]
    assert json.loads(request.data or b"") == {"scope": "read"}
    assert request.get_header("Authorization") == "Bearer s3cret"
    assert request.get_header("Content-type") == "application/json"
    assert "s3cret" not in result.output


def test_pair_defaults_to_control(api: _FakeApi) -> None:
    assert _run("pair").exit_code == 0
    assert json.loads(api.requests[0].data or b"") == {"scope": "control"}


def test_pair_refuses_an_unknown_scope_without_calling_the_api(api: _FakeApi) -> None:
    result = _run("pair", "--scope", "admin")
    assert result.exit_code == 1
    assert "--scope must be" in result.output
    assert api.requests == []


def test_pair_json(api: _FakeApi) -> None:
    result = _run("pair", "--json")
    assert json.loads(result.output)["code"] == "ABCD-EFGH"


def test_list_shows_every_device_and_its_state(api: _FakeApi) -> None:
    result = _run("list")
    assert result.exit_code == 0, result.output
    assert api.calls() == [("GET", "/api/v1/devices")]
    for expected in ("aaaa1111", "Phone", "Laptop", "read", "Old tablet", "revoked", "active"):
        assert expected in result.output
    assert "s3cret" not in result.output


def test_list_json_is_the_api_payload(api: _FakeApi) -> None:
    result = _run("list", "--json")
    assert json.loads(result.output) == {"devices": DEVICES}


def test_list_when_nothing_is_paired(api: _FakeApi) -> None:
    api.devices = []
    result = _run("list")
    assert result.exit_code == 0
    assert "No devices paired" in result.output


def test_a_device_name_is_text_not_markup(api: _FakeApi) -> None:
    api.devices = [_device(ID_A, "[bold red]x[/bold red][/nonsense]")]
    result = _run("list")
    assert result.exit_code == 0, result.output
    assert "[/nonsense]" in result.output


def test_revoke_by_full_id(api: _FakeApi) -> None:
    result = _run("revoke", ID_A)
    assert result.exit_code == 0, result.output
    assert api.calls() == [("GET", "/api/v1/devices"), ("DELETE", f"/api/v1/devices/{ID_A}")]
    assert all(r.get_header("Authorization") == "Bearer s3cret" for r in api.requests)
    assert "Revoked Phone" in result.output


def test_revoke_by_unique_prefix(api: _FakeApi) -> None:
    result = _run("revoke", "aaaa2")
    assert result.exit_code == 0, result.output
    assert api.calls()[-1] == ("DELETE", f"/api/v1/devices/{ID_B}")
    assert "Revoked Laptop" in result.output


def test_an_ambiguous_prefix_revokes_nothing(api: _FakeApi) -> None:
    result = _run("revoke", "aaaa")
    assert result.exit_code == 1
    assert "matches 2 devices" in result.output
    assert ID_A in result.output and ID_B in result.output
    assert [m for m, _ in api.calls()] == ["GET"]


def test_an_unknown_prefix_revokes_nothing(api: _FakeApi) -> None:
    result = _run("revoke", "zzzz")
    assert result.exit_code == 1
    assert "no device ID starts with 'zzzz'" in result.output
    assert [m for m, _ in api.calls()] == ["GET"]


def test_a_fragment_from_the_middle_of_an_id_is_not_a_prefix(api: _FakeApi) -> None:
    """`revoke 2222` must not find aaaa2222-…: a revoke is aimed by how the ID STARTS,
    which is what `iris device list` shows."""
    result = _run("revoke", "2222")
    assert result.exit_code == 1
    assert "no device ID starts with '2222'" in result.output
    assert [m for m, _ in api.calls()] == ["GET"]


def test_revoking_an_already_revoked_device_says_so(api: _FakeApi) -> None:
    result = _run("revoke", "cccc")
    assert result.exit_code == 0
    assert "already revoked" in result.output
    assert [m for m, _ in api.calls()] == ["GET"]


@pytest.mark.parametrize("command", [("pair",), ("list",), ("revoke", ID_A)])
def test_api_down_is_a_clear_error(api: _FakeApi, command: tuple[str, ...]) -> None:
    api.fail_with = urllib.error.URLError("connection refused")
    result = _run(*command)
    assert result.exit_code == 1
    assert f"IRIS API unreachable at {API}" in result.output
    assert "Traceback" not in result.output


def test_401_with_a_secret_says_the_secret_is_wrong(api: _FakeApi) -> None:
    api.fail_with = _http_error(401, "missing or invalid bearer token")
    result = _run("list")
    assert result.exit_code == 1
    assert "401" in result.output
    assert "not the secret the API runs with" in result.output
    assert "s3cret" not in result.output


def test_401_without_a_secret_says_it_is_unset(
    api: _FakeApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IRIS_AUTH_SECRET")
    api.fail_with = _http_error(401, "missing or invalid bearer token")
    result = _run("pair")
    assert result.exit_code == 1
    assert "IRIS_AUTH_SECRET is not set" in result.output
    assert api.requests[0].get_header("Authorization") is None


def test_other_http_errors_show_the_api_detail(api: _FakeApi) -> None:
    api.fail_with = _http_error(503, "IRIS_AUTH_SECRET is not set — this service refuses")
    result = _run("list")
    assert result.exit_code == 1
    assert "HTTP 503" in result.output and "this service refuses" in result.output


def test_api_url_comes_from_the_environment(api: _FakeApi, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_API_URL", f"{API}/")
    result = runner.invoke(app, ["device", "list"], env=WIDE)
    assert result.exit_code == 0, result.output
    assert api.requests[0].full_url == f"{API}/api/v1/devices"
